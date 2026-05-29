pub mod server;
pub mod tool;

use crate::error::{Result, RoutexError};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// JSON-RPC 2.0 request — the wire format for all MCP calls.
///
/// Every MCP operation is a JSON-RPC request with:
///   - jsonrpc: always "2.0"
///   - id: unique request identifier (None for notifications)
///   - method: the operation name
///   - params: optional parameters
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,

    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<u64>,

    method: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

/// JSON-RPC 2.0 response — what the server sends back.
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    #[serde(default)]
    id: Option<u64>,

    result: Option<Value>,
    error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    #[allow(dead_code)]
    code: i64,
    message: String,
}

/// Tool definition as returned by tools/list
#[derive(Debug, Clone, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,

    #[serde(default)]
    pub description: Option<String>,

    #[serde(rename = "inputSchema")]
    pub input_schema: Option<Value>,
}

/// tools/list response
#[derive(Debug, Deserialize)]
struct ToolsListResult {
    tools: Vec<McpToolDefinition>,

    #[serde(rename = "nextCursor")]
    next_cursor: Option<String>,
}

/// tools/call response content block
#[derive(Debug, Deserialize)]
struct McpContent {
    #[serde(rename = "type")]
    kind: String,

    #[serde(default)]
    text: Option<String>,
}

/// tools/call response
#[derive(Debug, Deserialize)]
struct ToolCallResult {
    content: Vec<McpContent>,

    #[serde(rename = "isError")]
    #[serde(default)]
    is_error: bool,
}

/// McpClient manages a connection to a single MCP server.
///
/// Handles:
///   - Session initialisation (initialize + notifications/initialized)
///   - Session ID tracking via Mcp-Session-Id header
///   - Tool discovery via tools/list (with pagination)
///   - Tool execution via tools/call
///
/// The session ID is captured from the initialize response header
/// and sent on all subsequent requests — some MCP servers require this
/// for stateful sessions.
///
/// This is equivalent to tools/mcp/client.go in Routex Go.
pub struct McpClient {
    client: Client,
    server_url: String,

    /// Session ID captured from Mcp-Session-Id response header
    /// Wrapped in RwLock because multiple tool calls may run concurrently
    /// and any one of them might need to read the session ID
    session_id: Arc<RwLock<Option<String>>>,

    /// Extra HTTP headers — used for authentication
    /// e.g. Authorization: Bearer <token>
    headers: Vec<(String, String)>,

    /// Request ID counter — incremented for each request
    request_id: Arc<RwLock<u64>>,
}

impl McpClient {
    /// Create a new McpClient for the given server URL.
    pub fn new(server_url: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            server_url: server_url.into(),
            session_id: Arc::new(RwLock::new(None)),
            headers: Vec::new(),
            request_id: Arc::new(RwLock::new(0)),
        }
    }

    /// Add a custom HTTP header — used for authentication.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Connect to the MCP server and perform the initialisation handshake.
    ///
    /// MCP requires two steps before any other calls:
    ///   1. POST initialize — negotiate capabilities
    ///   2. POST notifications/initialized — tell server we're ready
    ///
    /// The session ID is captured from the initialize response header
    /// if the server sends one.
    pub async fn connect(&self) -> Result<()> {
        // Send initialize request
        let id = self.next_id().await;
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id: Some(id),
            method: "initialize".to_string(),
            params: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "routex-rs",
                    "version": "0.5.0"
                }
            })),
        };

        let response = self.send_request(request, true).await?;

        // Verify the server responded with a result
        if response.error.is_some() {
            return Err(RoutexError::ToolFailed {
                name: "mcp".to_string(),
                reason: format!("initialize failed: {}", response.error.unwrap().message),
            });
        }

        // Send notifications/initialized (fire and forget)
        // This is a notification — no id, no response expected
        let notification = JsonRpcRequest {
            jsonrpc: "2.0",
            id: None,
            method: "notifications/initialized".to_string(),
            params: None,
        };

        // Send but don't wait for response — notifications are one-way
        let _ = self.send_request(notification, false).await;

        Ok(())
    }

    /// Discover all tools the server exposes.
    /// Handles pagination via nextCursor automatically.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>> {
        let mut all_tools = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let id = self.next_id().await;

            let params = match &cursor {
                Some(c) => Some(json!({ "cursor": c })),
                None => None,
            };

            let request = JsonRpcRequest {
                jsonrpc: "2.0",
                id: Some(id),
                method: "tools/list".to_string(),
                params,
            };

            let response = self.send_request(request, true).await?;

            if let Some(err) = response.error {
                return Err(RoutexError::ToolFailed {
                    name: "mcp".to_string(),
                    reason: format!("tools/list failed: {}", err.message),
                });
            }

            let result = response.result.ok_or_else(|| RoutexError::ToolFailed {
                name: "mcp".to_string(),
                reason: "tools/list returned no result".to_string(),
            })?;

            let list: ToolsListResult =
                serde_json::from_value(result).map_err(|e| RoutexError::ToolFailed {
                    name: "mcp".to_string(),
                    reason: format!("parse tools/list response: {}", e),
                })?;

            all_tools.extend(list.tools);

            // Check for more pages
            match list.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }

        Ok(all_tools)
    }

    /// Call a tool on the MCP server.
    ///
    /// Returns the tool's output as a JSON Value.
    /// If the server indicates isError=true, returns an error.
    pub async fn call_tool(&self, tool_name: &str, arguments: Value) -> Result<Value> {
        let id = self.next_id().await;

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id: Some(id),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": tool_name,
                "arguments": arguments,
            })),
        };

        let response = self.send_request(request, true).await?;

        if let Some(err) = response.error {
            return Err(RoutexError::ToolFailed {
                name: tool_name.to_string(),
                reason: format!("tools/call failed: {}", err.message),
            });
        }

        let result = response.result.ok_or_else(|| RoutexError::ToolFailed {
            name: tool_name.to_string(),
            reason: "tools/call returned no result".to_string(),
        })?;

        let call_result: ToolCallResult =
            serde_json::from_value(result).map_err(|e| RoutexError::ToolFailed {
                name: tool_name.to_string(),
                reason: format!("parse tools/call response: {}", e),
            })?;

        // Check if the server reported an error
        if call_result.is_error {
            let error_text = call_result
                .content
                .iter()
                .filter_map(|c| c.text.as_ref())
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");

            return Err(RoutexError::ToolFailed {
                name: tool_name.to_string(),
                reason: error_text,
            });
        }

        // Collect all text content blocks into a single JSON value
        let text_parts: Vec<String> = call_result
            .content
            .into_iter()
            .filter(|c| c.kind == "text")
            .filter_map(|c| c.text)
            .collect();

        Ok(json!({ "output": text_parts.join("\n") }))
    }

    /// Send a JSON-RPC request to the server.
    ///
    /// Handles:
    ///   - Attaching the session ID header if we have one
    ///   - Capturing new session IDs from response headers
    ///   - Attaching custom auth headers
    ///
    /// capture_response: false for fire-and-forget notifications
    async fn send_request(
        &self,
        request: JsonRpcRequest,
        capture_response: bool,
    ) -> Result<JsonRpcResponse> {
        let mut req = self
            .client
            .post(&self.server_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&request);

        // Attach session ID if we have one
        let session_id = self.session_id.read().await.clone();
        if let Some(id) = &session_id {
            req = req.header("Mcp-Session-Id", id);
        }

        // Attach custom headers — auth tokens etc
        for (name, value) in &self.headers {
            req = req.header(name.as_str(), value.as_str());
        }

        let response = req.send().await.map_err(|e| RoutexError::ToolFailed {
            name: "mcp".to_string(),
            reason: format!("request failed: {}", e),
        })?;

        // Capture session ID from response header if present
        if let Some(new_id) = response
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|v| v.to_str().ok())
        {
            let mut session = self.session_id.write().await;
            *session = Some(new_id.to_string());
        }

        if !capture_response {
            return Ok(JsonRpcResponse {
                id: None,
                result: None,
                error: None,
            });
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(RoutexError::ToolFailed {
                name: "mcp".to_string(),
                reason: format!("server returned {}: {}", status, body),
            });
        }

        let rpc_response: JsonRpcResponse =
            response.json().await.map_err(|e| RoutexError::ToolFailed {
                name: "mcp".to_string(),
                reason: format!("parse response: {}", e),
            })?;

        Ok(rpc_response)
    }

    /// Get the next request ID — atomically incremented
    async fn next_id(&self) -> u64 {
        let mut id = self.request_id.write().await;
        *id += 1;
        *id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use serde_json::json;

    fn initialize_response() -> String {
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "test-server", "version": "1.0" }
            }
        })
        .to_string()
    }

    fn tools_list_response() -> String {
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get the weather for a location",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ]
            }
        })
        .to_string()
    }

    fn tool_call_response() -> String {
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "Lagos: 32°C, sunny"
                    }
                ],
                "isError": false
            }
        })
        .to_string()
    }

    #[tokio::test]
    async fn test_connect_succeeds() {
        let mut server = Server::new_async().await;

        // Handle initialize
        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(initialize_response())
            .create_async()
            .await;

        // Handle notifications/initialized
        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .create_async()
            .await;

        let client = McpClient::new(server.url());
        let result = client.connect().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_connect_captures_session_id() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_header("Mcp-Session-Id", "test-session-123")
            .with_body(initialize_response())
            .create_async()
            .await;

        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .create_async()
            .await;

        let client = McpClient::new(server.url());
        client.connect().await.unwrap();

        let session = client.session_id.read().await;
        assert_eq!(*session, Some("test-session-123".to_string()));
    }

    #[tokio::test]
    async fn test_list_tools() {
        let mut server = Server::new_async().await;

        // initialize
        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(initialize_response())
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
            .with_body(tools_list_response())
            .create_async()
            .await;

        let client = McpClient::new(server.url());
        client.connect().await.unwrap();

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert!(tools[0].description.is_some());
    }

    #[tokio::test]
    async fn test_call_tool_returns_output() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(initialize_response())
            .create_async()
            .await;

        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .create_async()
            .await;

        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(tool_call_response())
            .create_async()
            .await;

        let client = McpClient::new(server.url());
        client.connect().await.unwrap();

        let result = client
            .call_tool("get_weather", json!({ "location": "Lagos" }))
            .await
            .unwrap();

        assert_eq!(result["output"], "Lagos: 32°C, sunny");
    }

    #[tokio::test]
    async fn test_call_tool_error_response() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(initialize_response())
            .create_async()
            .await;

        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("{}")
            .create_async()
            .await;

        let error_response = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [
                    { "type": "text", "text": "location not found" }
                ],
                "isError": true
            }
        })
        .to_string();

        server
            .mock("POST", "/")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(error_response)
            .create_async()
            .await;

        let client = McpClient::new(server.url());
        client.connect().await.unwrap();

        let result = client
            .call_tool("get_weather", json!({ "location": "unknown" }))
            .await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("location not found")
        );
    }
}
