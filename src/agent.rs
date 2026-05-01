use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::{
    Result, RoutexError,
    config::AgentConfig,
    llm::{self, Adapter, Message, Request, ResponseContent, ToolCallResult, ToolDefinition},
    tools::Registry,
};

/// AgentStatus represents what an agent is currently doing.
/// Sent through the output channel so the runtime can track progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Agent has started its thinking loop
    Started,

    /// Agent is calling the LLM
    Thinking,

    /// Agent is executing tools — includes which tools are running
    ExecutingTools(Vec<String>),

    /// Agent has finished — includes the final output
    Completed(String),

    /// Agent failed — includes the error message
    Failed(String),
}

/// AgentMessage is sent through the output channel.
/// The runtime receives these to track the crew's progress.
#[derive(Debug, Clone)]
pub struct AgentMessage {
    /// Which agent sent this message
    pub agent_id: String,
    /// What state the agent is in
    pub status: AgentStatus,
}

/// Agent is a single AI agent in the crew.
///
/// Each agent runs in its own Tokio task — the Rust equivalent of
/// a goroutine. It receives a task via the inbox channel, runs its
/// thinking loop until it produces a final answer, then sends the
/// result through the output channel.
///
/// The thinking loop:
///   1. Call the LLM with the current conversation history
///   2. If the LLM returns text → done, send the output
///   3. If the LLM returns tool calls → execute them concurrently
///   4. Add tool results to history and go back to step 1
pub struct Agent {
    /// Configuration from agents.yaml
    pub config: AgentConfig,

    /// The LLM adapter — Anthropic, OpenAI, or Ollama
    adapter: Arc<dyn Adapter + Send + Sync>,

    /// The tool registry — shared across all agents in the crew
    registry: Arc<Registry>,
}

impl Agent {
    /// Create a new Agent.
    ///
    /// Takes Arc references because multiple agents share the same
    /// registry and potentially the same adapter.
    pub fn new(
        config: AgentConfig,
        adapter: Arc<dyn Adapter + Send + Sync>,
        registry: Arc<Registry>,
    ) -> Self {
        Self {
            config,
            adapter,
            registry,
        }
    }

    /// Run the agent's thinking loop.
    ///
    /// Receives a task via `inbox`, runs until completion or failure,
    /// sends status updates via `output`.
    ///
    /// Parameters:
    ///   inbox  — receives the task input from the scheduler
    ///   output — sends status updates back to the runtime
    pub async fn run(
        &self,
        mut inbox: mpsc::Receiver<String>,
        output: mpsc::Sender<AgentMessage>,
    ) -> Result<String> {
        // wait for the task from the scheduler
        // recv() returns None if the sender is dropped - that means
        // the scheduler cancelled this agent before it started
        let task = inbox.recv().await.ok_or_else(|| RoutexError::AgentFailed {
            id: self.config.id.clone(),
            reason: "inbox closed before task arrived".to_string(),
        })?;

        // Notify the runtime that we have started
        self.send_status(&output, AgentStatus::Started).await;

        let system = self.build_system_prompt();

        // Build the tool definitions for this agent
        // Only include tools this agent is allowed to use
        let tool_defs = self.build_tool_definitions();

        // Initial conversation: just the task as a user message
        let mut history: Vec<Message> = vec![Message::user(task)];

        // Runs until the LLM produces a text response or we hit limits
        let mut tool_call_count = 0u32;
        let max_tool_calls = self.config.max_tool_calls;

        loop {
            // Notify: we are calling the LLM
            self.send_status(&output, AgentStatus::Thinking).await;

            let request = Request {
                messages: history.clone(),
                tools: tool_defs.clone(),
                system: system.clone(),
                max_tokens: 4096,
                model: self.config.llm.as_ref().map(|l| l.model.clone()),
            };

            let response =
                self.adapter
                    .complete(request)
                    .await
                    .map_err(|e| RoutexError::AgentFailed {
                        id: self.config.id.clone(),
                        reason: e.to_string(),
                    })?;

            match response.content {
                // LLM produced a text response - we're done
                ResponseContent::Text(text) => {
                    history.push(Message::assistant(&text));

                    self.send_status(&output, AgentStatus::Completed(text.clone()))
                        .await;

                    return Ok(text);
                }
                // LLM wants to call tools
                ResponseContent::ToolCalls(calls) => {
                    tool_call_count += calls.len() as u32;
                    if tool_call_count > max_tool_calls {
                        let redirect = format!(
                            "You have made {} tool calls which exceeds \
                            the budget of {}. Stop calling tools and \
                            produce your final answer now using the \
                            information already in your history.",
                            tool_call_count, max_tool_calls
                        );
                        history.push(Message::user(redirect));
                        continue;
                    }

                    let tool_names: Vec<String> =
                        calls.iter().map(|c| c.tool_name.clone()).collect();

                    self.send_status(&output, AgentStatus::ExecutingTools(tool_names.clone()))
                        .await;
                    history.push(Message {
                        role: crate::llm::Role::Assistant,
                        content: crate::llm::MessageContent::ToolUse {
                            calls: calls.clone(),
                        },
                    });

                    // execute all tool calls concurrently
                    let results = self.execute_tools_concurrent(calls).await;

                    history.push(Message::tool_results(results));
                }
            }
        }
    }

    /// Send a status update through the output channel.
    /// Uses fire-and-forget — if the receiver is gone we don't panic.
    async fn send_status(&self, output: &mpsc::Sender<AgentMessage>, status: AgentStatus) {
        let _ = output
            .send(AgentMessage {
                agent_id: self.config.id.clone(),
                status,
            })
            .await;
    }

    /// Build the system prompt from the agent's role and goal.
    fn build_system_prompt(&self) -> String {
        let mut propmt = format!(
            "{}\n\nYour specific goal for this task: {}",
            self.config.role.system_prompt().clone(),
            self.config.goal,
        );

        if let Some(backstory) = &self.config.backstory {
            propmt.push_str(&format!("\n\nBackground: {}", backstory));
        }

        propmt
    }

    /// Build tool definitions for this agent.
    /// Only includes tools the agent is configured to use.
    fn build_tool_definitions(&self) -> Vec<ToolDefinition> {
        self.config
            .tools
            .iter()
            .filter_map(|name| {
                self.registry
                    .get(name)
                    .map(|tool| ToolDefinition::from_schema(name, &tool.schema()))
            })
            .collect()
    }

    /// Execute multiple tool calls concurrently.
    async fn execute_tools_concurrent(
        &self,
        calls: Vec<llm::ToolCallRequest>,
    ) -> Vec<llm::ToolCallResult> {
        // clone the registry Arc so each task can hold a reference
        let registry = Arc::clone(&self.registry);

        // spawn a future for each tool call
        let futures: Vec<_> = calls
            .into_iter()
            .map(|call| {
                let registry = Arc::clone(&registry);
                let call = call.clone();

                async move {
                    let (output, is_error) =
                        match registry.execute(&call.tool_name, call.input.clone()).await {
                            Ok(result) => (result, false),
                            Err(e) => (serde_json::json!({"error": e.to_string()}), true),
                        };

                    ToolCallResult {
                        tool_call_id: call.id,
                        tool_name: call.tool_name,
                        output,
                        is_error,
                    }
                }
            })
            .collect();

        futures::future::join_all(futures).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::llm::{FinishReason, Response, ResponseContent, TokenUsage};
    use crate::tools::{Parameter, Schema, Tool};
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Mock LLM adapter — returns predefined responses
    struct MockAdapter {
        /// Responses to return in order
        /// Mutex because the mock is shared across async calls
        responses: Mutex<Vec<Response>>,
    }

    impl MockAdapter {
        fn new(responses: Vec<Response>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }

        fn text_response(text: &str) -> Response {
            Response {
                content: ResponseContent::Text(text.to_string()),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 20,
                },
            }
        }

        fn tool_response(tool_name: &str, input: Value) -> Response {
            Response {
                content: ResponseContent::ToolCalls(vec![crate::llm::ToolCallRequest {
                    id: "test-call-id".to_string(),
                    tool_name: tool_name.to_string(),
                    input,
                }]),
                finish_reason: FinishReason::ToolUse,
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
            }
        }
    }

    #[async_trait]
    impl Adapter for MockAdapter {
        async fn complete(&self, _req: Request) -> Result<Response> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                return Err(RoutexError::LLM(
                    "mock adapter: no more responses".to_string(),
                ));
            }
            Ok(responses.remove(0))
        }

        fn model(&self) -> &str {
            "mock-model"
        }
        fn provider(&self) -> &str {
            "mock"
        }
    }

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn schema(&self) -> Schema {
            Schema {
                description: "Echoes input".to_string(),
                parameters: HashMap::from([(
                    "message".to_string(),
                    Parameter {
                        kind: "string".to_string(),
                        description: "Message to echo".to_string(),
                        required: true,
                    },
                )]),
            }
        }

        async fn execute(&self, input: Value) -> Result<Value> {
            Ok(input)
        }
    }

    fn make_config() -> AgentConfig {
        AgentConfig {
            id: "test-agent".to_string(),
            role: crate::config::Role::Researcher,
            goal: "research topics thoroughly".to_string(),
            backstory: None,
            tools: vec!["echo".to_string()],
            depends: vec![],
            restart: "one_for_one".to_string(),
            llm: None,
            max_tool_calls: 20,
        }
    }

    fn make_agent(adapter: Arc<dyn Adapter + Send + Sync>) -> Agent {
        let mut registry = Registry::new();
        registry.register(EchoTool);
        Agent::new(make_config(), adapter, Arc::new(registry))
    }

    #[tokio::test]
    async fn test_agent_completes_with_text_response() {
        let adapter = Arc::new(MockAdapter::new(vec![MockAdapter::text_response(
            "The research is complete.",
        )]));
        let agent = make_agent(adapter);

        let (tx_in, rx_in) = mpsc::channel(1);
        let (tx_out, mut rx_out) = mpsc::channel(10);

        tx_in
            .send("Research Rust frameworks".to_string())
            .await
            .unwrap();

        let result = agent.run(rx_in, tx_out).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "The research is complete.");

        // Verify status updates were sent
        let started = rx_out.recv().await.unwrap();
        assert!(matches!(started.status, AgentStatus::Started));
    }

    #[tokio::test]
    async fn test_agent_executes_tool_then_completes() {
        let adapter = Arc::new(MockAdapter::new(vec![
            // First call: LLM requests a tool
            MockAdapter::tool_response("echo", json!({"message": "hello"})),
            // Second call: LLM produces final text
            MockAdapter::text_response("Done after using echo tool."),
        ]));
        let agent = make_agent(adapter);

        let (tx_in, rx_in) = mpsc::channel(1);
        let (tx_out, _rx_out) = mpsc::channel(10);

        tx_in.send("Do something".to_string()).await.unwrap();

        let result = agent.run(rx_in, tx_out).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Done after using echo tool.");
    }

    #[tokio::test]
    async fn test_agent_fails_if_inbox_closed() {
        let adapter = Arc::new(MockAdapter::new(vec![]));
        let agent = make_agent(adapter);

        let (_tx_in, rx_in) = mpsc::channel::<String>(1);
        let (tx_out, _rx_out) = mpsc::channel(10);

        // Drop tx_in immediately — inbox is closed before task arrives
        drop(_tx_in);

        let result = agent.run(rx_in, tx_out).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("inbox closed"));
    }

    #[tokio::test]
    async fn test_agent_respects_tool_call_budget() {
        // Return tool calls until budget exceeded, then a text response
        let mut responses = Vec::new();
        for _ in 0..25 {
            responses.push(MockAdapter::tool_response(
                "echo",
                json!({"message": "test"}),
            ));
        }
        responses.push(MockAdapter::text_response("Final answer."));

        let adapter = Arc::new(MockAdapter::new(responses));
        let agent = make_agent(adapter);

        let (tx_in, rx_in) = mpsc::channel(1);
        let (tx_out, _rx_out) = mpsc::channel(100);

        tx_in.send("Do many tool calls".to_string()).await.unwrap();

        let result = agent.run(rx_in, tx_out).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_system_prompt_includes_role_and_goal() {
        let adapter = Arc::new(MockAdapter::new(vec![]));
        let agent = make_agent(adapter);
        let prompt = agent.build_system_prompt();
        assert!(prompt.contains("research agent"));
        assert!(prompt.contains("research topics thoroughly"));
    }

    #[test]
    fn test_build_tool_definitions_only_includes_agent_tools() {
        let adapter = Arc::new(MockAdapter::new(vec![]));
        let agent = make_agent(adapter);
        let defs = agent.build_tool_definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
    }
}
