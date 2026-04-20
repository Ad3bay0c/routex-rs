pub mod anthropic;

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Result, tools::Schema};

/// Role represents who sent a message in the conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// ToolCall represents a single tool call request from the LLM.
///
/// When the LLM decides to use a tool it returns one or more
/// ToolCallRequests instead of a text response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// Unique ID for this tool call — used to match results back
    pub id: String,

    /// The name of the tool to call — matches Tool::name()
    pub tool_name: String,

    /// The input arguments as a JSON value
    pub input: serde_json::Value,
}

/// ToolCallResult is the output of a tool execution.
/// Sent back to the LLM as context for its next response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    /// The ID of the ToolCallRequest this result is for
    pub tool_call_id: String,

    /// The tool name — some APIs require this for matching
    pub tool_name: String,

    /// The tool's output as a JSON value
    pub output: serde_json::Value,

    /// Whether the tool execution failed
    pub is_error: bool,
}

/// Message is a single turn in the conversation history.
///
/// The content field uses an enum to handle the two cases:
///   - Text: the LLM produced a text response
///   - ToolUse: the LLM requested tool calls
///   - ToolResult: we're sending tool results back
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

/// MessageContent is what a message actually contains.
/// Using an enum makes the valid states explicit —
/// a message is either text OR tool calls, never ambiguous.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    /// Plain text — user messages, system prompts, text responses
    Text { text: String },

    /// The LLM wants to call one or more tools
    ToolUse { calls: Vec<ToolCallRequest> },

    /// We're returning results from tool execution
    ToolResult { results: Vec<ToolCallResult> },
}

impl Message {
    /// Create a user text message
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text { text: text.into() },
        }
    }

    /// Create a system message
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text { text: text.into() },
        }
    }

    /// Create a assistant message
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text { text: text.into() },
        }
    }

    /// Create a tool result message — sent after tool execution
    pub fn tool_results(results: Vec<ToolCallResult>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::ToolResult { results },
        }
    }
}

/// ToolDefinition describes a tool to the LLM.
/// Built from our Tool trait's Schema — translated per-provider
/// into the format each API expects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, ParameterDefinition>,
}

/// ParameterDefinition is a single parameter in a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    pub kind: String,
    pub description: String,
    pub required: bool,
}

impl ToolDefinition {
    // Build a ToolDefinition from a tool's Schema.
    /// Called when registering tools with the LLM adapter.
    pub fn from_schema(name: &str, schema: &Schema) -> Self {
        Self {
            name: name.to_string(),
            description: schema.description.clone(),
            parameters: schema
                .parameters
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        ParameterDefinition {
                            kind: v.kind.clone(),
                            description: v.description.clone(),
                            required: v.required,
                        },
                    )
                })
                .collect(),
        }
    }
}

/// Request is what we send to the LLM adapter.
/// the adapter translates this into
/// whatever format the specific LLM API expects.
#[derive(Debug, Clone)]
pub struct Request {
    /// The full conversation history including the latest user message
    pub messages: Vec<Message>,

    /// Tools available to the LLM for this request
    pub tools: Vec<ToolDefinition>,

    /// System prompt — describes the agent's role and goal
    pub system: String,

    /// Maximum tokens to generate
    pub max_tokens: u32,

    /// Model override — if None, uses the adapter's default
    pub model: Option<String>,
}

/// Response is what we get back from the LLM adapter.
///
/// the adapter translates the raw API
/// response into this clean type.
#[derive(Debug, Clone)]
pub struct Response {
    /// The LLM's response — either text or tool calls
    pub content: ResponseContent,

    /// Why the LLM stopped generating
    pub finish_reason: FinishReason,

    /// Token usage for cost tracking
    pub usage: TokenUsage,
}

/// ResponseContent is what the LLM actually produced.
/// Same enum approach as MessageContent — makes valid states explicit.
#[derive(Debug, Clone)]
pub enum ResponseContent {
    /// The LLM produced a text response — the agent is done this turn
    Text(String),

    /// The LLM wants to call one or more tools
    /// We execute them and send results back in the next request
    ToolCalls(Vec<ToolCallRequest>),
}

/// FinishReason explains why the LLM stopped generating.
#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    /// Normal completion — the LLM finished naturally
    Stop,

    /// The LLM wants to use tools
    ToolUse,

    /// Hit the max_tokens limit
    MaxTokens,

    /// Something else — provider-specific value stored as String
    Other(String),
}

/// TokenUsage tracks how many tokens were used.
/// Used for cost monitoring and observability.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl TokenUsage {
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

/// Adapter is the core trait every LLM provider must implement.
#[async_trait]
pub trait Adapter: Send + 'static {
    /// Send a request to the LLM and get a response.
    async fn complete(&self, req: Request) -> Result<Response>;

    /// The model name this adapter is configured to use.
    fn model(&self) -> &str;

    /// The provider name — "anthropic", "openai", "ollama"
    fn provider(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_user() {
        let msg = Message::user("hello");
        assert_eq!(msg.role, Role::User);
        match msg.content {
            MessageContent::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("expected text content"),
        }
    }

    #[test]
    fn test_message_system() {
        let msg = Message::system("you are a researcher");
        assert_eq!(msg.role, Role::System);
    }

    #[test]
    fn test_message_assistant() {
        let msg = Message::assistant("I found the answer");
        assert_eq!(msg.role, Role::Assistant);
    }

    #[test]
    fn test_token_usage_total() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
        };
        assert_eq!(usage.total(), 150);
    }

    #[test]
    fn test_tool_definition_from_schema() {
        use crate::tools::{Parameter, Schema};
        use std::collections::HashMap;

        let schema = Schema {
            description: "Search the web".to_string(),
            parameters: HashMap::from([(
                "query".to_string(),
                Parameter {
                    kind: "string".to_string(),
                    description: "The search query".to_string(),
                    required: true,
                },
            )]),
        };

        let def = ToolDefinition::from_schema("web_search", &schema);
        assert_eq!(def.name, "web_search");
        assert_eq!(def.description, "Search the web");
        assert!(def.parameters.contains_key("query"));
        assert!(def.parameters["query"].required);
    }

    #[test]
    fn test_role_serialises_lowercase() {
        let role = Role::User;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"user\"");

        let role = Role::Assistant;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"assistant\"");
    }

    #[test]
    fn test_finish_reason_equality() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_ne!(FinishReason::Stop, FinishReason::ToolUse);
    }
}
