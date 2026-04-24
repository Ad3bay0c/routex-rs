use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::{
    Result, RoutexError,
    config::AgentConfig,
    llm::{
        self, Adapter, Message, Request, ResponseContent, ToolCallRequest, ToolCallResult,
        ToolDefinition,
    },
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
    adapter: Arc<dyn Adapter>,

    /// The tool registry — shared across all agents in the crew
    registry: Arc<Registry>,
}

impl Agent {
    /// Create a new Agent.
    ///
    /// Takes Arc references because multiple agents share the same
    /// registry and potentially the same adapter.
    pub fn new(config: AgentConfig, adapter: Arc<dyn Adapter>, registry: Arc<Registry>) -> Self {
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
