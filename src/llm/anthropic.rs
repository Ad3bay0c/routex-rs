use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{
    Adapter, FinishReason, Message, MessageContent, Request, Response, ResponseContent, Role,
    TokenUsage, ToolCallRequest,
};
use crate::error::{Result, RoutexError};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// AnthropicAdapter calls the Anthropic Messages API directly over HTTP.
///
/// No SDK. Just reqwest + serde_json
///
/// agents.yaml:
///
///   runtime:
///     llm_provider: "anthropic"
///     model: "claude-haiku-4-5-20251001"
///     api_key: "env:ANTHROPIC_API_KEY"
pub struct AnthropicAdapter {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl AnthropicAdapter {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build HTTP client"),
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

/// The request body sent to POST /v1/messages
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,

    messages: Vec<AnthropicMessage>,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
}

/// A single message in the Anthropic conversation format
#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

/// A content block within an Anthropic message.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContent {
    /// Plain text content
    Text { text: String },

    /// The model wants to call a tool
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },

    /// We're returning a tool result
    ToolResult {
        tool_use_id: String,
        content: String,

        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// Tool definition in Anthropic's format
#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: AnthropicInputSchema,
}

/// JSON Schema for a tool's input parameters
#[derive(Debug, Serialize)]
struct AnthropicInputSchema {
    #[serde(rename = "type")]
    kind: String,
    properties: serde_json::Map<String, Value>,
    required: Vec<String>,
}

/// The response from POST /v1/messages
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    stop_reason: String,
    usage: AnthropicUsage,
}

/// Token usage in Anthropic's format
#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

/// Convert our Message list into Anthropic's message format.
fn translate_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
    let mut results = Vec::new();

    for msg in messages {
        if msg.role == Role::System {
            continue;
        }

        let role = match msg.role {
            Role::User | Role::Tool => "user",
            Role::Assistant => "assistant",
            Role::System => continue,
        };

        let content = match &msg.content {
            MessageContent::Text { text } => {
                vec![AnthropicContent::Text { text: text.clone() }]
            }
            MessageContent::ToolUse { calls } => calls
                .iter()
                .map(|call| AnthropicContent::ToolUse {
                    id: call.id.clone(),
                    name: call.tool_name.clone(),
                    input: call.input.clone(),
                })
                .collect(),
            MessageContent::ToolResult { results } => results
                .iter()
                .map(|r| AnthropicContent::ToolResult {
                    content: r.output.to_string(),
                    tool_use_id: r.tool_call_id.clone(),
                    is_error: if r.is_error { Some(true) } else { None },
                })
                .collect(),
        };

        results.push(AnthropicMessage {
            role: role.to_string(),
            content,
        });
    }

    results
}

/// Convert our ToolDefinitions into Anthropic's tool format.
fn translate_tools(tools: &[super::ToolDefinition]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|tool| {
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();

            for (name, param) in &tool.parameters {
                properties.insert(
                    name.clone(),
                    json!({
                        "type": param.kind,
                        "description": param.description,
                    }),
                );
                if param.required {
                    required.push(name.clone());
                }
            }

            AnthropicTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: AnthropicInputSchema {
                    kind: "object".to_string(),
                    properties,
                    required,
                },
            }
        })
        .collect()
}

/// Extract the system prompt from the messages list.
fn extract_system(messages: &[Message], fallback: &str) -> Option<String> {
    // First check if any message has a system role
    for msg in messages {
        if msg.role == Role::System {
            if let MessageContent::Text { text } = &msg.content {
                return Some(text.clone());
            }
        }
    }
    // Fall back to the request's system field
    if fallback.is_empty() {
        None
    } else {
        Some(fallback.to_string())
    }
}

/// Convert Anthropic's response into our clean Response type.
fn translate_response(raw: AnthropicResponse) -> Response {
    // Collect all content blocks — separate text from tool_use
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCallRequest> = Vec::new();

    for block in raw.content {
        match block {
            AnthropicContent::Text { text } => {
                text_parts.push(text);
            }
            AnthropicContent::ToolUse { id, name, input } => {
                tool_calls.push(ToolCallRequest {
                    id,
                    tool_name: name,
                    input,
                });
            }
            // ToolResult should not appear in responses
            AnthropicContent::ToolResult { .. } => {}
        }
    }

    // Determine the response content
    // If there are tool calls, they take priority
    let content = if !tool_calls.is_empty() {
        ResponseContent::ToolCalls(tool_calls)
    } else {
        ResponseContent::Text(text_parts.join("\n"))
    };

    let finish_reason = match raw.stop_reason.as_str() {
        "end_turn" => FinishReason::Stop,
        "tool_use" => FinishReason::ToolUse,
        "max_tokens" => FinishReason::MaxTokens,
        other => FinishReason::Other(other.to_string()),
    };

    Response {
        content,
        finish_reason,
        usage: TokenUsage {
            input_tokens: raw.usage.input_tokens,
            output_tokens: raw.usage.output_tokens,
        },
    }
}

// Adapter implementation
#[async_trait]
impl Adapter for AnthropicAdapter {
    async fn complete(&self, req: Request) -> Result<Response> {
        let system = extract_system(&req.messages, &req.system);
        let messages = translate_messages(&req.messages);
        let tools = translate_tools(&req.tools);

        let model = req.model.as_deref().unwrap_or(&self.model).to_string();

        let body = AnthropicRequest {
            model,
            max_tokens: req.max_tokens,
            system,
            messages,
            tools,
        };

        // build and send HTTP request
        let url = format!("{}/v1/messages", self.base_url);

        let http_response = self
            .client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| RoutexError::LLM(format!("anthropic: request failed: {}", e)))?;

        let status = http_response.status();

        if !status.is_success() {
            let error_body = http_response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());

            return Err(RoutexError::LLM(format!(
                "anthropic: api returned {}: {}",
                status, error_body
            )));
        }

        // parse response
        let raw: AnthropicResponse = http_response
            .json()
            .await
            .map_err(|e| RoutexError::LLM(format!("anthropic: parse response: {}", e)))?;

        Ok(translate_response(raw))
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &str {
        "anthropic"
    }
}

// TEST
#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use serde_json::json;

    fn make_adapter(url: &str) -> AnthropicAdapter {
        AnthropicAdapter::new("test-api-key", "claude-haiku-4-5-20251001").with_base_url(url)
    }

    fn simple_request() -> Request {
        Request {
            messages: vec![Message::user("What is Rust?")],
            tools: vec![],
            system: "You are a helpful assistant.".to_string(),
            max_tokens: 1024,
            model: None,
        }
    }

    fn text_response_body() -> Value {
        json!({
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Rust is a statically typed compiled language."
                }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        })
    }

    fn tool_use_response_body() -> Value {
        json!({
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "web_search",
                    "input": { "query": "Rust programming language" }
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 10
            }
        })
    }

    #[tokio::test]
    async fn test_text_response() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(text_response_body().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server.url());
        let response = adapter.complete(simple_request()).await.unwrap();

        match response.content {
            ResponseContent::Text(text) => {
                assert!(text.contains("Rust is a statically typed"));
            }
            _ => panic!("expected text response"),
        }

        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 20);
        assert_eq!(response.usage.total(), 30);
    }

    #[tokio::test]
    async fn test_tool_use_response() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(tool_use_response_body().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server.url());
        let response = adapter.complete(simple_request()).await.unwrap();

        match response.content {
            ResponseContent::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].tool_name, "web_search");
                assert_eq!(calls[0].id, "toolu_01");
                assert_eq!(calls[0].input["query"], "Rust programming language");
            }
            _ => panic!("expected tool calls"),
        }

        assert_eq!(response.finish_reason, FinishReason::ToolUse);
    }

    #[tokio::test]
    async fn test_api_error_returns_err() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/v1/messages")
            .with_status(401)
            .with_body(r#"{"error": {"type": "authentication_error"}}"#)
            .create_async()
            .await;

        let adapter = make_adapter(&server.url());
        let result = adapter.complete(simple_request()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("401"));
    }

    #[tokio::test]
    async fn test_sends_correct_headers() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/v1/messages")
            .match_header("x-api-key", "test-api-key")
            .match_header("anthropic-version", ANTHROPIC_VERSION)
            .match_header("content-type", "application/json")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(text_response_body().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server.url());
        let result = adapter.complete(simple_request()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_system_prompt_extracted() {
        let mut server = Server::new_async().await;

        let mock = server
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(text_response_body().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server.url());

        let req = Request {
            messages: vec![
                Message::system("You are a researcher."),
                Message::user("Find information about Rust."),
            ],
            tools: vec![],
            system: String::new(),
            max_tokens: 1024,
            model: None,
        };

        let result = adapter.complete(req).await;
        assert!(result.is_ok());
        mock.assert_async().await;
    }

    #[test]
    fn test_provider_and_model() {
        let adapter = AnthropicAdapter::new("key", "claude-haiku-4-5-20251001");
        assert_eq!(adapter.provider(), "anthropic");
        assert_eq!(adapter.model(), "claude-haiku-4-5-20251001");
    }

    #[test]
    fn test_translate_messages_skips_system() {
        let messages = vec![Message::system("You are helpful."), Message::user("Hello")];
        let translated = translate_messages(&messages);
        // System message should be excluded
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].role, "user");
    }
}
