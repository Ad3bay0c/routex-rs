use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

use tokio::sync::mpsc;

use crate::{
    Result, RoutexError,
    agent::{Agent, AgentMessage},
    config::Config,
    llm::{Adapter, anthropic::AnthropicAdapter, openai::OpenAIAdapter},
    tools::Registry,
};

/// RunResult is what you get back after a crew completes.
/// Contains the final output from the last agent in the graph
/// and all individual agent outputs for inspection.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// The final output — from the last agent in the dependency graph
    pub output: String,

    /// Individual outputs from every agent keyed by agent ID
    pub agent_outputs: HashMap<String, String>,

    /// Total token usage across all agents
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
}

/// Runtime is the top-level entry point for routex-rs.
///
/// It owns the tool registry, the LLM adapter, and the agent configs.
/// It is responsible for scheduling agents, running them in the correct
/// order, and collecting results.
///
/// Library usage:
///
///   let runtime = Runtime::from_config("agents.yaml")?;
///   let result = runtime.run().await?;
///   println!("{}", result.output);
///
/// Programmatic usage:
///
///   let mut runtime = Runtime::new(config);
///   runtime.register_tool(MyTool::new());
///   let result = runtime.run().await?;
pub struct Runtime {
    config: Config,
    registry: Arc<Registry>,
    adapter: Option<Arc<dyn Adapter + Send + Sync>>,
}

impl Runtime {
    /// Load a config file and create a Runtime.
    /// This is the primary entry point for most users.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let config = Config::from_file(path)?;
        Self::from_config(config)
    }

    /// Create a Runtime from an already-parsed Config.
    pub fn from_config(config: Config) -> Result<Self> {
        let mut registry = Registry::new();

        // Auto-register built-in tools declared in config
        for tool_cfg in &config.tools {
            match tool_cfg.name.as_str() {
                "web_search" => {
                    registry.register(crate::tools::web_search::WebSearchTool::new());
                }
                unknown => {
                    return Err(RoutexError::ToolNotFound {
                        name: unknown.to_string(),
                    });
                }
            }
        }

        Ok(Self {
            config,
            registry: Arc::new(registry),
            adapter: None,
        })
    }

    /// Register a tool with the runtime.
    /// Must be called before run().
    pub fn register_tool(&mut self, tool: impl crate::tools::Tool + 'static) {
        if let Some(registry) = Arc::get_mut(&mut self.registry) {
            registry.register(tool);
        }
    }

    /// Run the crew and return when all agents complete.
    ///
    /// It:
    ///   1. Builds the dependency graph from config
    ///   2. Runs agents in topological order — independent agents in parallel
    ///   3. Passes results from upstream agents to downstream agents
    ///   4. Returns the final output when all agents complete
    pub async fn run(&self) -> Result<RunResult> {
        // Validate tool references before starting
        self.validate_tool_references()?;

        let adapter = build_adapter(&self.config)?;

        let agent_count = self.config.agents.len();

        let (status_tx, mut status_rx) = mpsc::channel::<AgentMessage>(agent_count * 10);

        // Track outputs from completed agents
        let mut agent_outputs: HashMap<String, String> = HashMap::new();

        let waves = build_execution_waves(&self.config)?;

        // Execute wave by wave
        // Each wave contains agents that can run in parallel
        for wave in waves {
            // Spawn all agents in this wave concurrently
            let mut handles = Vec::new();

            for agent_id in &wave {
                let agent_config = self
                    .config
                    .agents
                    .iter()
                    .find(|a| &a.id == agent_id)
                    .expect("agent in wave must exist in config")
                    .clone();

                // Build the task input for this agent
                // Include the original task + outputs from dependencies
                let task = build_agent_task(
                    &self.config.task.input,
                    &agent_config.depends,
                    &agent_outputs,
                );

                let agent = Agent::new(
                    agent_config,
                    Arc::clone(&adapter),
                    Arc::clone(&self.registry),
                );

                // Create channels for this agent
                let (inbox_tx, inbox_rx) = mpsc::channel::<String>(1);
                let status_tx = status_tx.clone();

                // Send the task to the agent's inbox
                inbox_tx
                    .send(task)
                    .await
                    .map_err(|e| RoutexError::AgentFailed {
                        id: agent_id.clone(),
                        reason: format!("failed to send task: {}", e),
                    })?;

                // Spawn the agent as an independent Tokio task
                let handle = tokio::spawn(async move { agent.run(inbox_rx, status_tx).await });

                handles.push((agent_id.clone(), handle));
            }

            // Wait for all agents in this wave to complete
            // Same as wg.Wait() in Go — blocks until the wave is done
            for (agent_id, handle) in handles {
                match handle.await {
                    Ok(Ok(output)) => {
                        agent_outputs.insert(agent_id, output);
                    }
                    Ok(Err(e)) => {
                        return Err(RoutexError::AgentFailed {
                            id: agent_id,
                            reason: e.to_string(),
                        });
                    }
                    Err(e) => {
                        // JoinError — the task panicked
                        return Err(RoutexError::AgentFailed {
                            id: agent_id,
                            reason: format!("task panicked: {}", e),
                        });
                    }
                }
            }
        }

        // Drain remaining status messages
        drop(status_tx);
        while status_rx.try_recv().is_ok() {}

        // The final output is from the last agent in the dependency graph
        // — the agent with no dependents
        let final_output = find_final_output(&self.config, &agent_outputs)?;

        Ok(RunResult {
            output: final_output,
            agent_outputs,
            total_input_tokens: 0, // TODO: collect from agent status
            total_output_tokens: 0,
        })
    }

    /// Validate that all tool references in agent configs exist
    /// in the registry. Catches configuration mistakes early.
    fn validate_tool_references(&self) -> Result<()> {
        for agent in &self.config.agents {
            for tool_name in &agent.tools {
                if !self.registry.has(tool_name) {
                    return Err(RoutexError::ToolNotFound {
                        name: tool_name.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// List all registered tools.
    /// Used by the CLI's `routex tools list` command.
    pub fn list_tools(&self) -> Vec<crate::tools::ToolInfo> {
        self.registry.list()
    }
}

/// Find the final output — from the agent with no dependents.
/// In a linear crew (A → B → C), C is the final agent.
/// In a fan-in crew (A, B → C), C is the final agent.
fn find_final_output(config: &Config, outputs: &HashMap<String, String>) -> Result<String> {
    // Find agent IDs that no other agent depends on
    let all_deps: HashSet<String> = config
        .agents
        .iter()
        .flat_map(|a| a.depends.iter().cloned())
        .collect();

    let final_agents: Vec<&str> = config
        .agents
        .iter()
        .filter(|a| !all_deps.contains(&a.id))
        .map(|a| a.id.as_str())
        .collect();

    match final_agents.len() {
        0 => Err(RoutexError::Config(
            "could not determine final agent".to_string(),
        )),
        1 => {
            let id = final_agents[0];
            outputs
                .get(id)
                .cloned()
                .ok_or_else(|| RoutexError::AgentFailed {
                    id: id.to_string(),
                    reason: "no output recorded".to_string(),
                })
        }
        _ => {
            // Multiple final agents — concatenate their outputs
            let combined = final_agents
                .iter()
                .filter_map(|id| outputs.get(*id))
                .cloned()
                .collect::<Vec<_>>()
                .join("\n\n");
            Ok(combined)
        }
    }
}

/// Build the task input for an agent.
/// Includes the original task and outputs from dependency agents.
fn build_agent_task(
    original_task: &str,
    depends: &[String],
    outputs: &HashMap<String, String>,
) -> String {
    if depends.is_empty() {
        return original_task.to_string();
    }

    // Build context from dependency outputs
    let mut context = format!("Task: {}\n\nContext from previous agents:\n", original_task);

    for dep_id in depends {
        if let Some(output) = outputs.get(dep_id) {
            context.push_str(&format!("\n[{}]:\n{}\n", dep_id, output));
        }
    }

    context
}

/// Build the execution waves using Kahn's algorithm.
///
/// Returns a Vec of waves where each wave contains agent IDs
/// that can run in parallel. Wave N only starts after wave N-1 completes.
///
/// The algorithm:
///   1. Count in-degrees (how many dependencies each agent has)
///   2. Start with agents that have zero dependencies (wave 1)
///   3. When an agent completes, decrement in-degrees of its dependents
///   4. Add newly zero-degree agents to the next wave
///   5. Repeat until all agents are scheduled
fn build_execution_waves(config: &Config) -> Result<Vec<Vec<String>>> {
    // Build how many dependencies each agent has
    let mut in_degree: HashMap<String, usize> = HashMap::new();

    // Build dependencies map - which agents depend on each agent
    let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

    for agent in &config.agents {
        in_degree.entry(agent.id.clone()).or_insert(0);
        dependents.entry(agent.id.clone()).or_default();

        for dep in &agent.depends {
            *in_degree.entry(agent.id.clone()).or_insert(0) += 1;
            dependents
                .entry(dep.clone())
                .or_default()
                .push(agent.id.clone());
        }
    }

    let mut queue: VecDeque<String> = in_degree
        .iter()
        .filter(|&(_, &degree)| degree == 0)
        .map(|(id, _)| id.clone())
        .collect();

    if queue.is_empty() && !config.agents.is_empty() {
        return Err(RoutexError::Config(
            "all agents have dependencies — possible cycle".to_string(),
        ));
    }

    let mut waves: Vec<Vec<String>> = Vec::new();
    let mut scheduled: HashSet<String> = HashSet::new();

    // Process wave by wave
    while !queue.is_empty() {
        // Current wave = everything in the queue right now
        let wave: Vec<String> = queue.drain(..).collect();

        for id in &wave {
            scheduled.insert(id.clone());

            // Decrement in-degree of dependents
            if let Some(deps) = dependents.get(id) {
                for dependent in deps {
                    let degree = in_degree.get_mut(dependent).unwrap();
                    *degree -= 1;
                    // If all dependencies are satisfied, add to next wave
                    if *degree == 0 {
                        queue.push_back(dependent.clone());
                    }
                }
            }
        }

        waves.push(wave);
    }

    // If not all agents were scheduled, there's a cycle
    if scheduled.len() != config.agents.len() {
        let unscheduled: Vec<String> = config
            .agents
            .iter()
            .filter(|a| !scheduled.contains(&a.id))
            .map(|a| a.id.clone())
            .collect();

        return Err(RoutexError::CyclicDependency {
            id: unscheduled.first().cloned().unwrap_or_default(),
        });
    }

    Ok(waves)
}

/// Build an LLM adapter from the runtime config.
/// Currently supports Anthropic — OpenAI and Ollama come next.
fn build_adapter(config: &Config) -> Result<Arc<dyn Adapter + Send + Sync>> {
    match config.runtime.llm_provider.as_str() {
        "anthropic" => {
            if config.runtime.api_key.is_empty() {
                return Err(RoutexError::Config(
                    "anthropic provider require an api_key".to_string(),
                ));
            }
            Ok(Arc::new(AnthropicAdapter::new(
                &config.runtime.api_key,
                &config.runtime.model,
            )))
        }
        "openai" => {
            if config.runtime.api_key.is_empty() {
                return Err(RoutexError::Config(
                    "openai provider require an api_key".to_string(),
                ));
            }
            Ok(Arc::new(OpenAIAdapter::new(
                &config.runtime.api_key,
                &config.runtime.model,
            )))
        }
        other => Err(RoutexError::Config(format!(
            "unknown llm_provider '{}' - supported: anthropic",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, Config, RuntimeConfig, TaskConfig};

    fn make_config(agents: Vec<AgentConfig>) -> Config {
        Config {
            runtime: RuntimeConfig {
                name: "test".to_string(),
                llm_provider: "anthropic".to_string(),
                model: "claude-haiku-4-5-20251001".to_string(),
                api_key: "test-key".to_string(),
                base_url: None,
                log_level: "info".to_string(),
                max_tokens: 4096,
            },
            task: TaskConfig {
                input: "Research Go frameworks".to_string(),
            },
            agents,
            tools: vec![],
        }
    }

    fn simple_agent(id: &str, depends: Vec<&str>) -> AgentConfig {
        AgentConfig {
            id: id.to_string(),
            role: crate::config::Role::Researcher,
            goal: "research".to_string(),
            backstory: None,
            tools: vec![],
            depends: depends.iter().map(|s| s.to_string()).collect(),
            restart: "one_for_one".to_string(),
            llm: None,
            max_tool_calls: 20,
        }
    }

    #[test]
    fn test_single_agent_wave() {
        let config = make_config(vec![simple_agent("researcher", vec![])]);
        let waves = build_execution_waves(&config).unwrap();
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0], vec!["researcher"]);
    }

    #[test]
    fn test_sequential_agents_two_waves() {
        let config = make_config(vec![
            simple_agent("researcher", vec![]),
            simple_agent("writer", vec!["researcher"]),
        ]);
        let waves = build_execution_waves(&config).unwrap();
        assert_eq!(waves.len(), 2);
        assert!(waves[0].contains(&"researcher".to_string()));
        assert!(waves[1].contains(&"writer".to_string()));
    }

    #[test]
    fn test_parallel_agents_one_wave() {
        let config = make_config(vec![
            simple_agent("researcher-1", vec![]),
            simple_agent("researcher-2", vec![]),
            simple_agent("researcher-3", vec![]),
        ]);
        let waves = build_execution_waves(&config).unwrap();
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].len(), 3);
    }

    #[test]
    fn test_fan_in_pattern() {
        // researcher-1, researcher-2 → writer
        let config = make_config(vec![
            simple_agent("researcher-1", vec![]),
            simple_agent("researcher-2", vec![]),
            simple_agent("writer", vec!["researcher-1", "researcher-2"]),
        ]);
        let waves = build_execution_waves(&config).unwrap();
        assert_eq!(waves.len(), 2);
        assert_eq!(waves[0].len(), 2); // two parallel researchers
        assert_eq!(waves[1].len(), 1); // one writer
        assert!(waves[1].contains(&"writer".to_string()));
    }

    #[test]
    fn test_cyclic_dependency_detected() {
        // A depends on B, B depends on A — cycle
        let agent_a = simple_agent("a", vec!["b"]);
        let agent_b = simple_agent("b", vec!["a"]);
        let config = make_config(vec![agent_a, agent_b]);
        let result = build_execution_waves(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_cyclic_dependency_detected2() {
        // A depends on B, B depends on A — cycle
        let agent_a = simple_agent("a", vec!["b"]);
        let agent_b = simple_agent("b", vec!["c"]);
        let agent_c = simple_agent("c", vec!["b"]);
        let config = make_config(vec![agent_a, agent_b, agent_c]);
        let result = build_execution_waves(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_agent_task_no_deps() {
        let outputs = HashMap::new();
        let task = build_agent_task("Research Go", &[], &outputs);
        assert_eq!(task, "Research Go");
    }

    #[test]
    fn test_build_agent_task_with_deps() {
        let mut outputs = HashMap::new();
        outputs.insert(
            "researcher".to_string(),
            "Go is fast and concurrent.".to_string(),
        );
        let task = build_agent_task("Write a report", &["researcher".to_string()], &outputs);
        assert!(task.contains("Write a report"));
        assert!(task.contains("Go is fast and concurrent."));
        assert!(task.contains("[researcher]"));
    }

    #[test]
    fn test_find_final_output_single() {
        let config = make_config(vec![
            simple_agent("researcher", vec![]),
            simple_agent("writer", vec!["researcher"]),
        ]);
        let mut outputs = HashMap::new();
        outputs.insert("researcher".to_string(), "research done".to_string());
        outputs.insert("writer".to_string(), "report written".to_string());

        let result = find_final_output(&config, &outputs).unwrap();
        assert_eq!(result, "report written");
    }

    #[test]
    fn test_find_final_output_multiple() {
        // Two agents with no dependents — both are final
        let config = make_config(vec![
            simple_agent("agent-a", vec![]),
            simple_agent("agent-b", vec![]),
        ]);
        let mut outputs = HashMap::new();
        outputs.insert("agent-a".to_string(), "output a".to_string());
        outputs.insert("agent-b".to_string(), "output b".to_string());

        let result = find_final_output(&config, &outputs).unwrap();
        assert!(result.contains("output a") || result.contains("output b"));
    }
}
