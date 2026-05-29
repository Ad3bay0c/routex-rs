use super::McpClient;
use crate::error::Result;
use crate::tools::{Parameter, Schema, Tool};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// McpTool wraps a remote MCP tool as a local Tool implementation.
///
/// When an agent calls this tool, McpTool proxies the call to the
/// MCP server via the shared McpClient. From the agent's perspective
/// it looks exactly like any other built-in tool.
///
/// This is equivalent to tools/mcp/tool.go in Routex Go.
pub struct McpTool {
    /// The tool's name — may be prefixed with server name
    /// to avoid collisions: "github_create_issue"
    tool_name: String,

    /// The original name on the MCP server — used in the actual call
    remote_name: String,

    /// Human-readable description from the server's tools/list response
    description: String,

    /// Parameters extracted from the server's inputSchema
    parameters: HashMap<String, Parameter>,

    /// Shared client — Arc because multiple McpTools share one connection
    client: Arc<McpClient>,
}

impl McpTool {
    /// Create a new McpTool.
    ///
    /// tool_name: the name registered in our local Registry
    ///   (may have server prefix e.g. "github_create_issue")
    /// remote_name: the name to use when calling tools/call
    ///   (the original name from the MCP server)
    pub fn new(
        tool_name: impl Into<String>,
        remote_name: impl Into<String>,
        description: impl Into<String>,
        parameters: HashMap<String, Parameter>,
        client: Arc<McpClient>,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            remote_name: remote_name.into(),
            description: description.into(),
            parameters,
            client,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn schema(&self) -> Schema {
        Schema {
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }

    async fn execute(&self, input: Value) -> Result<Value> {
        self.client.call_tool(&self.remote_name, input).await
    }
}
