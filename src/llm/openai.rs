use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{
    Adapter, FinishReason, Message, MessageContent, Request, Response, ResponseContent, Role,
    TokenUsage, ToolCallRequest,
};
use crate::{Result, RoutexError};

/// Default base URL for the OpenAI Chat Completions API.
/// Can be overridden to use any OpenAI-compatible endpoint:
///   - Groq:     https://api.groq.com/openai/v1
///   - Together: https://api.together.xyz/v1
///   - Ollama:   http://localhost:11434/v1
const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// OpenAIAdapter calls the OpenAI Chat Completions API directly over HTTP.
///
/// No SDK — just reqwest + serde_json, same as AnthropicAdapter.
///
/// Because Ollama and many other providers are OpenAI-compatible,
/// this adapter works with all of them by setting a custom base_url.
///
/// agents.yaml:
///
///   runtime:
///     llm_provider: "openai"
///     model: "gpt-4o"
///     api_key: "env:OPENAI_API_KEY"
///
/// With Groq:
///
///   runtime:
///     llm_provider: "openai"
///     model: "llama-3.1-70b-versatile"
///     api_key: "env:GROQ_API_KEY"
///     base_url: "https://api.groq.com/openai/v1"
pub struct OpenAIAdapter {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAIAdapter {
    /// Create a new OpenAIAdapter
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

    /// Override the base URL
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

/// The request body sent to POST /v1/chat/completions
#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool>,

    /// Controls randomness — 0.0 = deterministic, 1.0 = creative
    /// We use 0.7 as a sensible default for agent tasks
    temperature: f32,
}

/// A single message in OpenAI's conversation format.
#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,

    /// Text content — None for tool call messages
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,

    /// Tool calls requested by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,

    /// For role=tool messages — which tool call this is the result for
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,

    /// For role=tool messages — the tool name
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

/// A tool call requested by the assistant
#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,

    /// Always "function" in the current API
    #[serde(rename = "type")]
    kind: String,

    function: OpenAIFunctionCall,
}

/// The function details within a tool call
#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,

    arguments: serde_json::Value,
}

/// Tool definition in OpenAI's format
#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    kind: String,
    function: OpenAIFunction,
}

/// The function details within a tool definition
#[derive(Debug, Serialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: OpenAIParameters,
}

/// JSON Schema for the function's parameters
#[derive(Debug, Serialize)]
struct OpenAIParameters {
    #[serde(rename = "type")]
    kind: String,
    properties: serde_json::Map<String, Value>,
    required: Vec<String>,
}

/// The response from POST /v1/chat/completions
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

/// A single completion choice
#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// converts Message list into OpenAI messages
// System prompt goes first as a "system" role message.
fn translate_messages(messages: &[Message]) -> Vec<OpenAIMessage> {
    let mut result = Vec::new();

    for msg in messages {
        match (&msg.role, &msg.content) {
            // System message — goes into the messages array as role=system
            (Role::System, MessageContent::Text { text }) => {
                result.push(OpenAIMessage {
                    role: "system".to_string(),
                    content: Some(text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }

            // User text message
            (Role::User, MessageContent::Text { text }) => {
                result.push(OpenAIMessage {
                    role: "user".to_string(),
                    content: Some(text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }

            // Assistant text response
            (Role::Assistant, MessageContent::Text { text }) => {
                result.push(OpenAIMessage {
                    role: "assistant".to_string(),
                    content: Some(text.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }

            // Assistant tool use request
            (Role::Assistant, MessageContent::ToolUse { calls }) => {
                let tool_calls = calls
                    .iter()
                    .map(|call| OpenAIToolCall {
                        id: call.id.clone(),
                        kind: "function".to_string(),
                        function: OpenAIFunctionCall {
                            name: call.tool_name.clone(),
                            // OpenAI expects arguments as a JSON string
                            // not a JSON object — we must serialise it
                            arguments: call.input.clone(),
                        },
                    })
                    .collect();

                result.push(OpenAIMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                    name: None,
                });
            }

            // Tool results — one message per result in OpenAI's format
            // Unlike Anthropic which batches them, OpenAI wants
            // individual messages for each tool result
            (_, MessageContent::ToolResult { results }) => {
                for result_item in results {
                    result.push(OpenAIMessage {
                        role: "tool".to_string(),
                        content: Some(result_item.output.to_string()),
                        tool_calls: None,
                        tool_call_id: Some(result_item.tool_call_id.clone()),
                        name: Some(result_item.tool_name.clone()),
                    });
                }
            }

            _ => {}
        }
    }

    result
}

/// Convert our ToolDefinitions into OpenAI's tool format.
fn translate_tools(tools: &[super::ToolDefinition]) -> Vec<OpenAITool> {
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

            OpenAITool {
                kind: "function".to_string(),
                function: OpenAIFunction {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: OpenAIParameters {
                        kind: "object".to_string(),
                        properties,
                        required,
                    },
                },
            }
        })
        .collect()
}

/// Convert OpenAI's response into our clean Response type.
fn translate_response(raw: OpenAIResponse) -> Result<Response> {
    // OpenAI always returns at least one choice
    let choice = raw
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| RoutexError::LLM("openai: response contained no choices".to_string()))?;

    let finish_reason = match choice.finish_reason.as_deref().unwrap_or("stop") {
        "stop" => FinishReason::Stop,
        "tool_calls" => FinishReason::ToolUse,
        "length" => FinishReason::MaxTokens,
        other => FinishReason::Other(other.to_string()),
    };

    // Determine response content
    let content = if let Some(tool_calls) = choice.message.tool_calls {
        // LLM wants to call tools
        let calls: Result<Vec<ToolCallRequest>> = tool_calls
            .into_iter()
            .map(|tc| {
                Ok(ToolCallRequest {
                    id: tc.id,
                    tool_name: tc.function.name,
                    input: tc.function.arguments,
                })
            })
            .collect();

        ResponseContent::ToolCalls(calls?)
    } else {
        // Text response
        let text = choice.message.content.unwrap_or_default();
        ResponseContent::Text(text)
    };

    Ok(Response {
        content,
        finish_reason,
        usage: TokenUsage {
            input_tokens: raw.usage.prompt_tokens,
            output_tokens: raw.usage.completion_tokens,
        },
    })
}

// ── Adapter implementation ──────
#[async_trait]
impl Adapter for OpenAIAdapter {
    fn model(&self) -> &str {
        &self.model
    }

    fn provider(&self) -> &str {
        "openai"
    }

    async fn complete(&self, req: Request) -> Result<Response> {
        let model = req.model.as_deref().unwrap_or(&self.model).to_string();

        let req = OpenAIRequest {
            model,
            max_completion_tokens: Some(req.max_tokens),
            messages: translate_messages(&req.messages),
            tools: translate_tools(&req.tools),
            temperature: 0.7,
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let http_response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&req)
            .send()
            .await
            .map_err(|e| RoutexError::LLM(format!("openai: request failed: {}", e)))?;

        let status = http_response.status();

        if !status.is_success() {
            let error_body = http_response
                .text()
                .await
                .unwrap_or_else(|_| "unknow error".to_string());

            return Err(RoutexError::LLM(format!(
                "openai: api returned {}: {}",
                status, error_body
            )));
        }

        let response: OpenAIResponse = http_response
            .json()
            .await
            .map_err(|e| RoutexError::LLM(format!("open ai: parse response: {}", e)))?;

        translate_response(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use serde_json::json;

    fn make_adapter(url: &str) -> OpenAIAdapter {
        OpenAIAdapter::new("test-api-key", "gpt-4o").with_base_url(url)
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
            "id": "chatcmpl-01",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Rust is a systems programming language."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        })
    }

    fn tool_use_response_body() -> Value {
        json!({
            "id": "chatcmpl-02",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_01",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": "{\"query\": \"Rust programming\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25
            }
        })
    }

    #[tokio::test]
    async fn test_text_response() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(text_response_body().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server.url());
        let response = adapter.complete(simple_request()).await.unwrap();

        match response.content {
            ResponseContent::Text(text) => {
                assert!(text.contains("Rust is a systems"));
            }
            _ => panic!("expected text response"),
        }

        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 20);
    }

    #[tokio::test]
    async fn test_tool_use_response() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/v1/chat/completions")
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
                assert_eq!(calls[0].id, "call_01");
                assert!(calls[0].input.to_string().contains("query"));
                assert!(calls[0].input.to_string().contains("Rust programming"));
            }
            _ => panic!("expected tool calls"),
        }

        assert_eq!(response.finish_reason, FinishReason::ToolUse);
    }

    #[tokio::test]
    async fn test_sends_bearer_auth() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/v1/chat/completions")
            .match_header("Authorization", "Bearer test-api-key")
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
    async fn test_api_error_returns_err() {
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/v1/chat/completions")
            .with_status(401)
            .with_body(r#"{"error": {"message": "Invalid API key"}}"#)
            .create_async()
            .await;

        let adapter = make_adapter(&server.url());
        let result = adapter.complete(simple_request()).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("401"));
    }

    #[tokio::test]
    async fn test_system_message_in_messages_array() {
        // OpenAI puts system messages in the messages array
        // unlike Anthropic which has a top-level system field
        let mut server = Server::new_async().await;

        server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(text_response_body().to_string())
            .create_async()
            .await;

        let adapter = make_adapter(&server.url());

        let req = Request {
            messages: vec![
                Message::system("You are a Rust expert."),
                Message::user("What is ownership?"),
            ],
            tools: vec![],
            system: String::new(),
            max_tokens: 1024,
            model: None,
        };

        let result = adapter.complete(req).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_provider_and_model() {
        let adapter = OpenAIAdapter::new("key", "gpt-4o");
        assert_eq!(adapter.provider(), "openai");
        assert_eq!(adapter.model(), "gpt-4o");
    }

    #[test]
    fn test_tool_arguments_parsed_from_string() {
        // OpenAI sends arguments as a JSON string — verify we parse it correctly
        let raw = OpenAIResponse {
            choices: vec![OpenAIChoice {
                message: OpenAIMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCall {
                        id: "call_01".to_string(),
                        kind: "function".to_string(),
                        function: OpenAIFunctionCall {
                            name: "web_search".to_string(),
                            arguments: json!({"query": "Rust ownership"}),
                        },
                    }]),
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
        };

        let response = translate_response(raw).unwrap();
        match response.content {
            ResponseContent::ToolCalls(calls) => {
                assert_eq!(calls[0].input["query"], "Rust ownership");
            }
            _ => panic!("expected tool calls"),
        }
    }
}
