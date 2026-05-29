use super::{Parameter, Schema, Tool};
use crate::error::{Result, RoutexError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// WriteFileTool writes content to files on the local filesystem.
///
/// Supports both creating new files and appending to existing ones.
/// Like ReadFileTool, an optional base_dir restricts writes to
/// a specific directory for security.
///
/// agents.yaml:
///
///   tools:
///     - name: "write_file"
///       base_dir: "./output"  # recommended — restrict write location
pub struct WriteFileTool {
    base_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct WriteFileInput {
    /// Path to write to
    path: String,

    /// Content to write
    content: String,

    /// If true, append to existing file instead of overwriting
    #[serde(default)]
    append: bool,
}

#[derive(Debug, Serialize)]
struct WriteFileOutput {
    path: String,
    bytes_written: usize,
    appended: bool,
    created: bool,
}

impl WriteFileTool {
    pub fn new(base_dir: Option<String>) -> Self {
        Self {
            base_dir: base_dir.map(PathBuf::from),
        }
    }

    /// Resolve and validate the write path.
    /// Creates parent directories if they don't exist.
    fn resolve_path(&self, requested: &str) -> Result<PathBuf> {
        let path = Path::new(requested);

        let resolved = if let Some(base) = &self.base_dir {
            // Verify parent is within base_dir
            let canonical_base = base.canonicalize().map_err(|e| RoutexError::ToolFailed {
                name: "write_file".to_string(),
                reason: format!("base_dir invalid: {}", e),
            })?;

            // For write operations the file may not exist yet
            // so we canonicalize the parent (which must exist or be created)
            // and append the filename component
            let normalised = normalise_path(&canonical_base.join(path));

            // Check that the normalised path starts with the base
            if !normalised.starts_with(&canonical_base) {
                return Err(RoutexError::ToolFailed {
                    name: "write_file".to_string(),
                    reason: format!("path '{}' is outside the allowed directory", requested),
                });
            }

            normalised
        } else {
            path.to_path_buf()
        };

        Ok(resolved)
    }
}

/// Normalise a path without requiring it to exist.
/// Resolves .. components manually since canonicalize
/// requires the path to exist on disk.
fn normalise_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::CurDir => {}
            other => components.push(other),
        }
    }
    components.iter().collect()
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn schema(&self) -> Schema {
        Schema {
            description: "Write content to a file on the filesystem. \
                Use for saving reports, storing data, or creating output files. \
                Supports both creating new files and appending to existing ones. \
                Parent directories are created automatically."
                .to_string(),
            parameters: HashMap::from([
                (
                    "path".to_string(),
                    Parameter {
                        kind: "string".to_string(),
                        description: "Path to write to. Parent directories \
                            are created automatically."
                            .to_string(),
                        required: true,
                    },
                ),
                (
                    "content".to_string(),
                    Parameter {
                        kind: "string".to_string(),
                        description: "Content to write to the file.".to_string(),
                        required: true,
                    },
                ),
                (
                    "append".to_string(),
                    Parameter {
                        kind: "boolean".to_string(),
                        description: "If true, append to existing file \
                            instead of overwriting. Defaults to false."
                            .to_string(),
                        required: false,
                    },
                ),
            ]),
        }
    }

    async fn execute(&self, input: Value) -> Result<Value> {
        // parse input
        let params: WriteFileInput =
            serde_json::from_value(input).map_err(|e| RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("invalid input: {}", e),
            })?;

        // resolve path
        let path = self.resolve_path(&params.path)?;

        // create parent directories if needed
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| RoutexError::ToolFailed {
                    name: self.name().to_string(),
                    reason: format!("create directories: {}", e),
                })?;
        }

        // check if file exists before writing
        let existed = path.exists();

        // write the file
        // Use tokio::fs for non-blocking IO
        if params.append {
            // Append mode — add to existing content
            use tokio::io::AsyncWriteExt;
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .await
                .map_err(|e| RoutexError::ToolFailed {
                    name: self.name().to_string(),
                    reason: format!("open file: {}", e),
                })?;

            file.write_all(params.content.as_bytes())
                .await
                .map_err(|e| RoutexError::ToolFailed {
                    name: self.name().to_string(),
                    reason: format!("write file: {}", e),
                })?;

            // Flush ensures data is written to OS buffer
            file.flush().await.map_err(|e| RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("flush file: {}", e),
            })?;
        } else {
            // Overwrite mode — replace existing content
            tokio::fs::write(&path, params.content.as_bytes())
                .await
                .map_err(|e| RoutexError::ToolFailed {
                    name: self.name().to_string(),
                    reason: format!("write file: {}", e),
                })?;
        }

        let output = WriteFileOutput {
            path: params.path,
            bytes_written: params.content.len(),
            appended: params.append,
            created: !existed,
        };

        serde_json::to_value(output).map_err(RoutexError::Json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_write_new_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(Some(dir.path().to_str().unwrap().to_string()));

        let result = tool
            .execute(json!({
                "path": "output.txt",
                "content": "hello from routex-rs"
            }))
            .await
            .unwrap();

        assert_eq!(result["bytes_written"], 20);
        assert_eq!(result["created"], true);
        assert_eq!(result["appended"], false);

        // Verify file was actually written
        let content = std::fs::read_to_string(dir.path().join("output.txt")).unwrap();
        assert_eq!(content, "hello from routex-rs");
    }

    #[tokio::test]
    async fn test_append_to_existing_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(Some(dir.path().to_str().unwrap().to_string()));

        // Write initial content
        tool.execute(json!({
            "path": "log.txt",
            "content": "line 1\n"
        }))
        .await
        .unwrap();

        // Append second line
        let result = tool
            .execute(json!({
                "path": "log.txt",
                "content": "line 2\n",
                "append": true
            }))
            .await
            .unwrap();

        assert_eq!(result["appended"], true);
        assert_eq!(result["created"], false);

        let content = std::fs::read_to_string(dir.path().join("log.txt")).unwrap();
        assert_eq!(content, "line 1\nline 2\n");
    }

    #[tokio::test]
    async fn test_creates_parent_directories() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(Some(dir.path().to_str().unwrap().to_string()));

        tool.execute(json!({
            "path": "reports/2026/summary.txt",
            "content": "report content"
        }))
        .await
        .unwrap();

        assert!(dir.path().join("reports/2026/summary.txt").exists());
    }

    #[tokio::test]
    async fn test_overwrite_existing_file() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(Some(dir.path().to_str().unwrap().to_string()));

        tool.execute(json!({
            "path": "file.txt",
            "content": "original"
        }))
        .await
        .unwrap();

        tool.execute(json!({
            "path": "file.txt",
            "content": "overwritten"
        }))
        .await
        .unwrap();

        let content = std::fs::read_to_string(dir.path().join("file.txt")).unwrap();
        assert_eq!(content, "overwritten");
    }

    #[tokio::test]
    async fn test_path_traversal_blocked() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(Some(dir.path().to_str().unwrap().to_string()));

        let result = tool
            .execute(json!({
                "path": "../../etc/evil.txt",
                "content": "malicious"
            }))
            .await;

        assert!(result.is_err());
    }

    #[test]
    fn test_name() {
        assert_eq!(WriteFileTool::new(None).name(), "write_file");
    }
}
