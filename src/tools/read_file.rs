use super::{Parameter, Schema, Tool};
use crate::error::{Result, RoutexError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// ReadFileTool reads files from the local filesystem.
///
/// For security, an optional base_dir restricts reads to a
/// specific directory — agents cannot read arbitrary paths.
///
/// agents.yaml:
///
///   tools:
///     - name: "read_file"
///       base_dir: "./data"  # optional — restrict to this directory
pub struct ReadFileTool {
    /// Optional base directory — all reads are restricted to this path
    /// None means reads are allowed from any path
    base_dir: Option<PathBuf>,

    /// Maximum file size in bytes — prevents reading huge files
    max_bytes: usize,
}

#[derive(Debug, Deserialize)]
struct ReadFileInput {
    /// Path to the file to read
    path: String,

    /// Maximum bytes to read — defaults to tool's max_bytes
    #[serde(default)]
    max_bytes: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ReadFileOutput {
    path: String,
    content: String,
    bytes_read: usize,
    truncated: bool,
}

impl ReadFileTool {
    /// Create a new ReadFileTool.
    /// base_dir is optional — pass None to allow any path.
    pub fn new(base_dir: Option<String>) -> Self {
        Self {
            base_dir: base_dir.map(PathBuf::from),
            max_bytes: 1024 * 1024, // 1MB default
        }
    }

    /// Resolve and validate the requested path.
    ///
    /// If base_dir is set, the resolved path must be within it.
    /// This prevents path traversal attacks — an agent cannot
    /// read ../../etc/passwd if base_dir is ./data
    fn resolve_path(&self, requested: &str) -> Result<PathBuf> {
        let path = Path::new(requested);

        let resolved = if let Some(base) = &self.base_dir {
            // Resolve relative to base_dir
            let full = base.join(path);

            // Canonicalize to resolve .. and symlinks
            // Then verify the result is still within base_dir
            let canonical = full.canonicalize().map_err(|e| RoutexError::ToolFailed {
                name: "read_file".to_string(),
                reason: format!("path '{}' not found: {}", requested, e),
            })?;

            let canonical_base = base.canonicalize().map_err(|e| RoutexError::ToolFailed {
                name: "read_file".to_string(),
                reason: format!("base_dir invalid: {}", e),
            })?;

            // Security check — resolved path must start with base_dir
            if !canonical.starts_with(&canonical_base) {
                return Err(RoutexError::ToolFailed {
                    name: "read_file".to_string(),
                    reason: format!("path '{}' is outside the allowed directory", requested),
                });
            }

            canonical
        } else {
            // No base_dir — resolve from current directory
            path.to_path_buf()
        };

        Ok(resolved)
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn schema(&self) -> Schema {
        Schema {
            description: "Read the contents of a file from the filesystem. \
                Use for reading configuration files, data files, reports, \
                or any text content stored locally."
                .to_string(),
            parameters: HashMap::from([
                (
                    "path".to_string(),
                    Parameter {
                        kind: "string".to_string(),
                        description: "Path to the file to read. \
                            Relative paths are resolved from the \
                            configured base directory."
                            .to_string(),
                        required: true,
                    },
                ),
                (
                    "max_bytes".to_string(),
                    Parameter {
                        kind: "integer".to_string(),
                        description: "Maximum bytes to read. \
                            Defaults to 1MB."
                            .to_string(),
                        required: false,
                    },
                ),
            ]),
        }
    }

    async fn execute(&self, input: Value) -> Result<Value> {
        // parse input
        let params: ReadFileInput =
            serde_json::from_value(input).map_err(|e| RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("invalid input: {}", e),
            })?;

        let max_bytes = params.max_bytes.unwrap_or(self.max_bytes);

        // resolve and validate the path
        let path = self.resolve_path(&params.path)?;

        // read the file
        // tokio::fs::read is the async version of std::fs::read
        // It does not block the Tokio thread pool
        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("could not read '{}': {}", params.path, e),
            })?;

        // truncate if needed
        let (content_bytes, truncated) = if bytes.len() > max_bytes {
            (&bytes[..max_bytes], true)
        } else {
            (bytes.as_slice(), false)
        };

        // convert to UTF-8
        // lossy conversion replaces invalid UTF-8 sequences with
        // the replacement character rather than failing entirely
        let content = String::from_utf8_lossy(content_bytes).into_owned();

        let output = ReadFileOutput {
            path: params.path,
            content,
            bytes_read: content_bytes.len(),
            truncated,
        };

        serde_json::to_value(output).map_err(RoutexError::Json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_temp_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let path = dir.path().join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        write!(file, "{}", content).unwrap();
        path
    }

    #[tokio::test]
    async fn test_read_existing_file() {
        let dir = TempDir::new().unwrap();
        write_temp_file(&dir, "test.txt", "hello routex-rs");

        let tool = ReadFileTool::new(Some(dir.path().to_str().unwrap().to_string()));

        let result = tool
            .execute(json!({
                "path": "test.txt"
            }))
            .await
            .unwrap();

        assert_eq!(result["content"], "hello routex-rs");
        assert_eq!(result["truncated"], false);
    }

    #[tokio::test]
    async fn test_read_truncates_large_file() {
        let dir = TempDir::new().unwrap();
        let large_content = "A".repeat(1000);
        write_temp_file(&dir, "large.txt", &large_content);

        let tool = ReadFileTool::new(Some(dir.path().to_str().unwrap().to_string()));

        let result = tool
            .execute(json!({
                "path": "large.txt",
                "max_bytes": 100
            }))
            .await
            .unwrap();

        assert_eq!(result["content"].as_str().unwrap().len(), 100);
        assert_eq!(result["truncated"], true);
    }

    #[tokio::test]
    async fn test_path_traversal_blocked() {
        let dir = TempDir::new().unwrap();

        let tool = ReadFileTool::new(Some(dir.path().to_str().unwrap().to_string()));

        // Attempt to escape the base directory
        let result = tool
            .execute(json!({
                "path": "../../etc/passwd"
            }))
            .await;

        assert!(result.is_err());
        assert!(
            result
                .as_ref()
                .unwrap_err()
                .to_string()
                .contains("outside the allowed directory")
                || result.unwrap_err().to_string().contains("not found")
        );
    }

    #[tokio::test]
    async fn test_missing_file_returns_error() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool::new(Some(dir.path().to_str().unwrap().to_string()));

        let result = tool
            .execute(json!({
                "path": "nonexistent.txt"
            }))
            .await;

        assert!(result.is_err());
    }

    #[test]
    fn test_name() {
        assert_eq!(ReadFileTool::new(None).name(), "read_file");
    }
}
