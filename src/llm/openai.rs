use std::{time::Duration, vec};

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{
    Adapter, FinishReason, Message, MessageContent, Request, Response, ResponseContent, Role,
    TokenUsage, ToolCallRequest,
};
use crate::{Result, RoutexError};

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

pub struct OpenAIAdapter {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAIAdapter {
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

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,

    messages: Vec<OpenAIMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAIToolCall>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,

    #[serde(rename = "type")]
    kind: String,

    function: OpenAIToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCallFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    kind: String,
    function: OpenAIToolFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIToolFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIResponseChoice>,
    usage: OpenAIUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<OpenAIResponseError>,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponseError {
    message: String,
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponseChoice {
    message: OpenAIMessage,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// converts Message list into OpenAI messages
// System prompt goes first as a "system" role message.
fn translate_messages(system_prompt: &str, messages: &[Message]) -> Vec<OpenAIMessage> {
    let mut outputs = Vec::new();

    if !system_prompt.is_empty() {
        outputs.push(OpenAIMessage {
            role: "system".to_string(),
            content: system_prompt.to_string(),
            tool_call_id: None,
            tool_calls: vec![],
        });
    }

    for msg in messages {
        if msg.role == Role::System {
            continue;
        }

        let role = match msg.role {
            Role::User => "user",
            Role::Tool => "tool",
            Role::Assistant => "assistant",
            Role::System => continue,
        };

        match &msg.content {
            MessageContent::Text { text } => {
                outputs.push(OpenAIMessage {
                    content: text.to_string(),
                    role: role.to_string(),
                    tool_call_id: None,
                    tool_calls: vec![],
                });
            }
            MessageContent::ToolUse { calls } => {
                let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();

                for tc in calls {
                    tool_calls.push(OpenAIToolCall {
                        function: OpenAIToolCallFunction {
                            arguments: tc.input.clone(),
                            name: tc.tool_name.clone(),
                        },
                        id: tc.id.clone(),
                        kind: "function".to_string(),
                    });
                }

                outputs.push(OpenAIMessage {
                    content: "".to_string(),
                    role: role.to_string(),
                    tool_call_id: None,
                    tool_calls,
                });
            }
            MessageContent::ToolResult { results } => {
                let mut results: Vec<OpenAIMessage> = results
                    .iter()
                    .map(|r| OpenAIMessage {
                        content: r.output.to_string(),
                        role: role.to_string(),
                        tool_call_id: Some(r.tool_call_id.clone()),
                        tool_calls: vec![],
                    })
                    .collect();

                outputs.append(&mut results);
            }
        };
    }

    outputs
}

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

            let mut params_schema: Map<String, Value> = [
                ("type".to_string(), json!("object")),
                ("properties".to_string(), json!(properties)),
            ]
            .into_iter()
            .collect();
            if !required.is_empty() {
                params_schema.insert("required".to_string(), json!(required));
            }

            OpenAITool {
                kind: "function".to_string(),
                function: OpenAIToolFunction {
                    name: tool.name.clone(),
                    description: Some(tool.description.clone()),
                    parameters: json!(params_schema),
                },
            }
        })
        .collect()
}

fn translate_response(response: OpenAIResponse) -> Response {
    let usage = TokenUsage {
        input_tokens: response.usage.prompt_tokens,
        output_tokens: response.usage.completion_tokens,
    };

    if response.choices.is_empty() {
        return Response {
            content: ResponseContent::Text("".to_string()),
            finish_reason: FinishReason::Other("no_choices".to_string()),
            usage,
        };
    }

    let choice = &response.choices[0];

    let tool_calls: Vec<ToolCallRequest> = choice
        .message
        .tool_calls
        .iter()
        .map(|tc| ToolCallRequest {
            id: tc.id.clone(),
            tool_name: tc.function.name.clone(),
            input: tc.function.arguments.clone(),
        })
        .collect();

    let finish_reason = match choice.finish_reason.as_str() {
        "end_turn" => FinishReason::Stop,
        "tool" => FinishReason::ToolUse,
        "max_tokens" => FinishReason::MaxTokens,
        other => FinishReason::Other(other.to_string()),
    };

    let content = if !tool_calls.is_empty() {
        ResponseContent::ToolCalls(tool_calls)
    } else {
        ResponseContent::Text(choice.message.content.clone())
    };

    Response {
        content,
        finish_reason,
        usage,
    }
}

#[async_trait]
impl Adapter for OpenAIAdapter {
    fn model(&self) -> &str {
        "openai"
    }

    fn provider(&self) -> &str {
        &self.model
    }

    async fn complete(&self, req: Request) -> Result<Response> {
        let model = req.model.as_deref().unwrap_or(&self.model).to_string();

        let req = OpenAiRequest {
            model,
            max_completion_tokens: Some(req.max_tokens),
            messages: translate_messages(&req.system, &req.messages),
            tools: translate_tools(&req.tools),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);

        let http_response = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.api_key)
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
                "openai: returned {}: {}",
                status, error_body
            )));
        }

        let response: OpenAIResponse = http_response
            .json()
            .await
            .map_err(|e| RoutexError::LLM(format!("open ai: parse response: {}", e)))?;

        if let Some(e) = response.error {
            return Err(RoutexError::LLM(format!(
                "openai: response error: {}: {}",
                e.kind, e.message
            )));
        }

        Ok(translate_response(response))
    }
}
