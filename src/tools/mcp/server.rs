use super::tool::McpTool;
use super::{McpClient, McpToolDefinition};
use crate::error::Result;
use crate::tools::{Parameter, Registry};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for connecting to an MCP server.
///
/// In agents.yaml:
///
///   tools:
///     - name: "mcp"
///       extra:
///         server_url: "http://localhost:3000"
///         server_name: "github"
///         header_Authorization: "env:GITHUB_TOKEN"
pub struct ServerConfig {
    /// The MCP server URL
    pub server_url: String,

    /// Optional prefix for tool names — prevents collisions
    /// e.g. server_name "github" → tool name "github_create_issue"
    pub server_name: Option<String>,

    /// Extra HTTP headers for authentication
    /// Keys starting with "header_" are treated as headers:
    /// "header_Authorization" → "Authorization: Bearer ..."
    pub headers: Vec<(String, String)>,
}

/// Connect to an MCP server, discover all its tools,
/// and register them in the given Registry.
///
/// This is called automatically by the Runtime when it sees
/// a tool with name "mcp" in agents.yaml.
///
/// Equivalent to tools/mcp/server.go RegisterServer() in Routex Go.
pub async fn register_server(config: ServerConfig, registry: &mut Registry) -> Result<()> {
    // Build the client with auth headers
    let mut client = McpClient::new(&config.server_url);
    for (name, value) in config.headers {
        client = client.with_header(name, value);
    }

    let client = Arc::new(client);

    // Connect and perform the initialisation handshake
    client.connect().await?;

    // Discover all tools the server exposes
    let tools = client.list_tools().await?;

    // Register each tool in our local registry
    for tool_def in tools {
        let local_name = build_tool_name(&tool_def.name, config.server_name.as_deref(), registry);

        let parameters = extract_parameters(&tool_def);
        let description = tool_def
            .description
            .unwrap_or_else(|| format!("MCP tool: {}", tool_def.name));

        let mcp_tool = McpTool::new(
            &local_name,
            &tool_def.name, // remote name stays unchanged
            description,
            parameters,
            Arc::clone(&client),
        );

        registry.register(mcp_tool);
    }

    Ok(())
}

/// Build the local tool name.
///
/// If a server_name is configured, prefix the tool name to avoid
/// collisions when multiple MCP servers expose tools with the same name.
///
/// "create_issue" with server_name "github" → "github_create_issue"
///
/// If the prefixed name is already taken, append a counter:
/// "github_create_issue_2"
fn build_tool_name(remote_name: &str, server_name: Option<&str>, registry: &Registry) -> String {
    let base = match server_name {
        Some(prefix) => format!("{}_{}", prefix, remote_name),
        None => remote_name.to_string(),
    };

    // Check for collisions and append counter if needed
    if !registry.has(&base) {
        return base;
    }

    let mut counter = 2;
    loop {
        let candidate = format!("{}_{}", base, counter);
        if !registry.has(&candidate) {
            return candidate;
        }
        counter += 1;
    }
}

/// Extract parameters from an MCP tool's inputSchema.
///
/// MCP uses JSON Schema for parameter definitions.
/// We convert them to our Parameter type.
fn extract_parameters(tool_def: &McpToolDefinition) -> HashMap<String, Parameter> {
    let mut parameters = HashMap::new();

    let schema = match &tool_def.input_schema {
        Some(s) => s,
        None => return parameters,
    };

    // Extract properties from JSON Schema object
    let properties = match schema.get("properties").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => return parameters,
    };

    // Extract required field list
    let required: Vec<String> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    for (name, prop) in properties {
        let kind = prop
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("string")
            .to_string();

        let description = prop
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();

        parameters.insert(
            name.clone(),
            Parameter {
                kind,
                description,
                required: required.contains(name),
            },
        );
    }

    parameters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Registry;
    use mockito::Server;
    use serde_json::json;

    async fn setup_mock_server(server: &mut mockito::Server) {
        // initialize
        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "test", "version": "1.0" }
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;

        // notifications/initialized
        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .create_async()
            .await;

        // tools/list
        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "tools": [
                            {
                                "name": "create_issue",
                                "description": "Create a GitHub issue",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "title": {
                                            "type": "string",
                                            "description": "Issue title"
                                        },
                                        "body": {
                                            "type": "string",
                                            "description": "Issue body"
                                        }
                                    },
                                    "required": ["title"]
                                }
                            }
                        ]
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;
    }

    #[tokio::test]
    async fn test_register_server_adds_tools() {
        let mut server = Server::new_async().await;
        setup_mock_server(&mut server).await;

        let mut registry = Registry::new();

        let config = ServerConfig {
            server_url: server.url(),
            server_name: Some("github".to_string()),
            headers: vec![],
        };

        register_server(config, &mut registry).await.unwrap();

        // Tool should be registered with server prefix
        assert!(registry.has("github_create_issue"));
    }

    #[tokio::test]
    async fn test_register_server_no_prefix() {
        let mut server = Server::new_async().await;
        setup_mock_server(&mut server).await;

        let mut registry = Registry::new();

        let config = ServerConfig {
            server_url: server.url(),
            server_name: None,
            headers: vec![],
        };

        register_server(config, &mut registry).await.unwrap();

        // Tool registered without prefix
        assert!(registry.has("create_issue"));
    }

    #[tokio::test]
    async fn test_name_collision_gets_counter() {
        let mut server = Server::new_async().await;
        setup_mock_server(&mut server).await;

        let mut registry = Registry::new();

        // Pre-register a tool with the same name
        use crate::error::Result;
        use crate::tools::{Schema, Tool};
        use async_trait::async_trait;
        use serde_json::Value;

        struct DummyTool;
        #[async_trait]
        impl Tool for DummyTool {
            fn name(&self) -> &str {
                "create_issue"
            }
            fn schema(&self) -> Schema {
                Schema {
                    description: "dummy".to_string(),
                    parameters: std::collections::HashMap::new(),
                }
            }
            async fn execute(&self, _: Value) -> Result<Value> {
                Ok(serde_json::json!({}))
            }
        }

        registry.register(DummyTool);
        assert!(registry.has("create_issue"));

        let config = ServerConfig {
            server_url: server.url(),
            server_name: None,
            headers: vec![],
        };

        register_server(config, &mut registry).await.unwrap();

        // Collision resolved with counter
        assert!(registry.has("create_issue_2"));
    }

    #[test]
    fn test_extract_parameters() {
        let tool_def = McpToolDefinition {
            name: "test".to_string(),
            description: None,
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results"
                    }
                },
                "required": ["query"]
            })),
        };

        let params = extract_parameters(&tool_def);
        assert_eq!(params.len(), 2);
        assert!(params["query"].required);
        assert!(!params["limit"].required);
        assert_eq!(params["query"].kind, "string");
        assert_eq!(params["limit"].kind, "integer");
    }
}
