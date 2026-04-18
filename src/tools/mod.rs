

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::{Result, RoutexError};

/// Parameter describes a single input field a tool accepts.
/// This is sent to the LLM so it knows how to call the tool.
#[derive(Debug, Clone)]
pub struct Parameter {
    /// JSON type: "string", "integer", "boolean", "array", "object"
    pub kind: String,

    /// Human-readable description sent to the LLM
    pub description: String,

    /// Whether the LLM must provide this parameter
    pub required: bool,
}

/// Schema describes the tool to the LLM.
/// The LLM reads this to understand when and how to call the tool.
#[derive(Debug, Clone)]
pub struct Schema {
    /// What this tool does — the LLM reads this to decide when to use it
    pub description: String,
    /// The parameters this tool accepts
    pub parameters: HashMap<String, Parameter>,
}

/// Tool is the core trait every tool must implement.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's registered name — must be unique in the registry.
    /// Agents reference tools by this name in agents.yaml.
    fn name(&self) -> &str;
    /// Describes the tool to the LLM — what it does and what it accepts.
    fn schema(&self) -> Schema;
    /// Execute the tool with the given JSON input.
    ///
    /// The LLM provides input as a JSON object matching the schema.
    /// The tool returns a JSON value the LLM reads as the tool result.
    async fn execute(&self, input: Value) -> Result<Value>;
}

/// ToolInfo is a lightweight snapshot of a tool's metadata.
/// Used by the CLI's `routex tools list` command.
#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
}

/// Registry holds all tools that have been registered with the runtime.
pub struct Registry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl Registry {
    /// create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool.
    /// If a tool with the same name already exists it is replaced.
    pub fn register(&mut self, tool: impl Tool + 'static) {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    pub async fn execute(&self, name: &str, input: Value) -> Result<Value> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| RoutexError::ToolNotFound {
                name: name.to_string(),
            })?;

        tool.execute(input)
            .await
            .map_err(|e| RoutexError::ToolFailed {
                name: name.to_string(),
                reason: e.to_string(),
            })
    }

    /// Check if a tool is registered.
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// List all registered tools with their descriptions.
    pub fn list(&self) -> Vec<ToolInfo> {
        let mut infos: Vec<ToolInfo> = self
            .tools
            .values()
            .map(|t| ToolInfo {
                name: t.name().to_string(),
                description: t.schema().description.clone(),
            })
            .collect();

        infos.sort_by(|a, b| a.name.cmp(&b.name));
        infos
    }

    /// Get a reference to a tool by name.
    /// Returns None if not registered.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Return the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns true if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct RoutexTool;

    #[async_trait]
    impl Tool for RoutexTool {
        fn name(&self) -> &str {
            "routex"
        }

        fn schema(&self) -> Schema {
            Schema {
                description: "Routex the yaml file".to_string(),
                parameters: HashMap::from([(
                    "message".to_string(),
                    Parameter {
                        kind: "string".to_string(),
                        description: "Message to routex".to_string(),
                        required: true,
                    },
                )]),
            }
        }

        async fn execute(&self, input: Value) -> Result<Value> {
            Ok(input)
        }
    }

    #[test]
    fn test_register_and_has() {
        let mut registry = Registry::new();
        registry.register(RoutexTool);

        assert!(registry.has("routex"));
        assert!(!registry.has("nonexistent"));
    }

    #[test]
    fn test_list_returns_sorted() {
        let mut registry = Registry::new();
        registry.register(RoutexTool);

        let list = registry.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "routex");
    }

    #[tokio::test]
    async fn test_execute_known_tool() {
        let mut registry = Registry::new();
        registry.register(RoutexTool);

        let result = registry
            .execute("routex", json!({"message": "hello"}))
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json!({"message": "hello"}))
    }

    #[tokio::test]
    async fn test_execute_unknown_tool() {
        let mut registry = Registry::default();
        registry.register(RoutexTool);

        let result = registry.execute("nonexistent", json!({})).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("nonexistent"))
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut registry = Registry::new();
        assert!(registry.is_empty());
        registry.register(RoutexTool);
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }
}
