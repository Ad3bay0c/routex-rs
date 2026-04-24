use crate::{Result, RoutexError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Config is the top-level structure that maps directly to agents.yaml.
///
/// A user's agents.yaml looks like:
///
/// runtime:
///   name: "research-crew"
///   llm_provider: "anthropic"
///   model: "claude-haiku-4-5-20251001"
///   api_key: "env:ANTHROPIC_API_KEY"
///
/// agents:
///   - id: "researcher"
///     role: "researcher"
///     goal: "Find information about the topic"
///     tools: ["web_search"]
///
///   - id: "writer"
///     role: "writer"
///     goal: "Write a report from the research"
///     depends: ["researcher"]
///
/// task:
///   input: "Compare Go web frameworks"
///
/// tools:
///   - name: "web_search"
///     api_key: "env:BRAVE_API_KEY"

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub runtime: RuntimeConfig,
    pub task: TaskConfig,

    #[serde(default)]
    pub agents: Vec<AgentConfig>,

    #[serde(default)]
    pub tools: Vec<ToolConfig>,
}

/// RuntimeConfig holds global settings that apply to the entire crew.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RuntimeConfig {
    /// Human-readable name for this crew
    #[serde(default = "default_runtime_name")]
    pub name: String,

    /// LLM provider: "anthropic", "openai", or "ollama"
    pub llm_provider: String,

    /// Model name — e.g. "claude-haiku-4-5-20251001", "gpt-4o"
    pub model: String,

    /// API key — supports "env:VAR_NAME" syntax to read from environment
    #[serde(default)]
    pub api_key: String,

    /// Optional base URL override for OpenAI-compatible endpoints
    /// e.g. Ollama at "http://localhost:11434/v1"
    #[serde(default)]
    pub base_url: Option<String>,

    /// Log level: "debug", "info", "warn", "error"
    /// Defaults to "info" if not set
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Maximum tokens per LLM response
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Planner,
    Writer,
    Critic,
    Executor,
    Researcher,
}

impl Role {
    pub fn system_prompt(&self) -> String {
        match self {
            Role::Planner => "You are a planning agent. Your only job is to read the task \
			and break it down into a clear, numbered list of steps for other \
			agents to follow. Do not do the work yourself — only plan it. \
			Be specific and actionable."
                .to_string(),
            Role::Writer => "You are a writing agent. You receive a plan and execute it \
			by researching, thinking, and producing well-structured written output. \
			Use your tools when you need information from the web or files. \
			Be thorough and cite your sources."
                .to_owned(),
            Role::Critic => "You are a critic agent. You receive a piece of work and review it \
			for quality, accuracy, completeness, and clarity. \
			Be constructive. Point out what is good, what is missing, \
			and what could be improved. Give a score out of 10."
                .to_string(),
            Role::Executor => "You are an executor agent. You carry out specific actions \
			using the tools available to you. Follow instructions precisely. \
			Report back exactly what happened — success or failure."
                .to_string(),
            Role::Researcher => "You are a research agent. Your job is to find, read, and \
			summarise information relevant to the task. \
			Do not produce long reports — produce concise, factual summaries \
			that other agents can use."
                .to_string(),
        }
    }
}

/// AgentConfig defines a single agent in the crew.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Unique identifier for this agent within the crew
    pub id: String,

    /// The agent's role — used in the system prompt
    /// e.g. "researcher", "writer", "critic"
    pub role: Role,

    /// What this agent is trying to achieve — used in the system prompt
    pub goal: String,

    /// Optional backstory for richer agent persona
    #[serde(default)]
    pub backstory: Option<String>,

    /// Tool names this agent can use
    /// Must match names registered in the tool registry
    #[serde(default)]
    pub tools: Vec<String>,

    /// Agent IDs this agent depends on
    /// This agent will not start until all dependencies complete
    #[serde(default)]
    pub depends: Vec<String>,

    /// Restart policy if this agent fails
    /// "one_for_one" | "one_for_all" | "rest_for_one" | "never"
    #[serde(default = "default_restart_policy")]
    pub restart: String,

    /// Per-agent LLM override — if not set, uses runtime defaults
    #[serde(default)]
    pub llm: Option<AgentLlmConfig>,

    /// Maximum number of tool calls per thinking turn
    #[serde(default = "default_max_tool_calls")]
    pub max_tool_calls: u32,
}

/// AgentLlmConfig allows per-agent LLM overrides.
/// An agent can use a different provider or model than the runtime default.
///
/// agents:
///   - id: "researcher"
///     llm:
///       provider: "anthropic"
///       model: "claude-haiku-4-5-20251001"  # cheap for data gathering
///
///   - id: "writer"
///     llm:
///       provider: "openai"
///       model: "gpt-4o"                      # better for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLlmConfig {
    pub provider: String,
    pub model: String,

    #[serde(default)]
    pub api_key: Option<String>,

    #[serde(default)]
    pub base_url: Option<String>,
}

/// TaskConfig holds the task that the crew is working on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    /// The input prompt or question for the crew
    pub input: String,
}

/// ToolConfig holds configuration for a single tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Tool name — must match a registered tool
    /// e.g. "web_search", "wikipedia", "gcs"
    pub name: String,

    /// Optional API key for tools that need one
    #[serde(default)]
    pub api_key: Option<String>,

    /// Optional base directory for file tools
    #[serde(default)]
    pub base_dir: Option<String>,

    /// Maximum results for search tools
    #[serde(default)]
    pub max_results: Option<u32>,

    /// Arbitrary extra settings — same as Go's Extra map[string]string
    /// Catches any tool-specific keys not covered above
    #[serde(default)]
    pub extra: HashMap<String, String>,
}

fn default_runtime_name() -> String {
    "routex".to_owned()
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_restart_policy() -> String {
    "one_for_one".to_string()
}

fn default_max_tool_calls() -> u32 {
    20
}

impl Config {
    /// Load and parse a config file from disk.
    ///
    /// This is the primary entry point for loading agents.yaml.
    /// It reads the file, parses the YAML, resolves env: references,
    /// and validates the result.
    ///
    /// Usage:
    ///   let config = Config::from_file("agents.yaml")?;
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            RoutexError::Config(format!("Could not read {}: {}", path.as_ref().display(), e))
        })?;

        // parse YAML into config struct
        let mut config: Config = serde_yaml::from_str(&content)?;

        // Resolve env: references throughout the config
        config.resolve_env();

        // Validate the config before returning
        config.validate()?;

        Ok(config)
    }

    /// Resolve "env:VAR_NAME" syntax in string fields.
    ///
    /// Wherever a string starts with "env:", we read the named
    /// environment variable instead of using the literal value.
    fn resolve_env(&mut self) {
        self.runtime.api_key = resolve_env_value(&self.runtime.api_key);

        for agent in &mut self.agents {
            if let Some(llm) = &mut agent.llm {
                if let Some(key) = &llm.api_key {
                    llm.api_key = Some(resolve_env_value(key))
                }

                llm.provider = resolve_env_value(&llm.provider);
                llm.model = resolve_env_value(&llm.model);

                if let Some(base_url) = &llm.base_url {
                    llm.base_url = Some(resolve_env_value(base_url));
                }
            }
        }

        for tool in &mut self.tools {
            tool.name = resolve_env_value(&tool.name);

            if let Some(api_key) = &tool.api_key {
                tool.api_key = Some(resolve_env_value(&api_key));
            }
            if let Some(base_dir) = &tool.base_dir {
                tool.base_dir = Some(resolve_env_value(&base_dir));
            }

            for extra in tool.extra.values_mut() {
                *extra = resolve_env_value(extra);
            }
        }
    }

    /// Validate the config for common mistakes.
    ///
    /// Catches errors early — before any agents run — so the user
    /// gets a clear error message rather than a cryptic runtime failure.
    ///
    /// Checks:
    ///   - At least one agent is declared
    ///   - No duplicate agent IDs
    ///   - All dependency references point to real agents
    ///   - No cyclic dependencies
    fn validate(&self) -> Result<()> {
        if self.agents.is_empty() {
            return Err(RoutexError::Config(
                "agents.yaml must declare at least one agent".to_string(),
            ));
        }

        let agents = &self.agents;

        let mut agent_ids = HashSet::new();
        for agent in agents {
            if agent.id.is_empty() {
                return Err(RoutexError::Config(
                    "every agent must have a non-empty id".to_string(),
                ));
            }

            if !agent_ids.insert(&agent.id) {
                return Err(RoutexError::Config(format!(
                    "duplicate agent id: '{}'",
                    agent.id
                )));
            }
        }

        // Every dependency must reference a real agent
        for agent in agents {
            for dep in &agent.depends {
                if !agent_ids.contains(dep) {
                    return Err(RoutexError::UnknownDependency {
                        id: agent.id.clone(),
                        dep: dep.clone(),
                    });
                }
            }
        }

        Ok(())
    }
}

/// Resolve the "env:VAR_NAME" syntax.
/// If the value starts with "env:", read that environment variable.
/// Otherwise return the value unchanged.
fn resolve_env_value(value: &str) -> String {
    if let Some(var_name) = value.strip_prefix("env:") {
        std::env::var(var_name).unwrap_or_default()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_yaml(contents: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(file, "{}", contents).unwrap();
        file
    }

    #[test]
    fn test_parse_minimal_config() {
        let yaml = r#"
runtime:
  name: "test-crew"
  llm_provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
  api_key: "test-key"

task:
  input: "test task"

agents:
  - id: "researcher"
    role: "researcher"
    goal: "research the topic"
"#;

        let config: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.runtime.name, "test-crew");
        assert_eq!(config.runtime.api_key, "test-key");
        assert_eq!(config.agents.len(), 1);
        assert_eq!(config.agents[0].id, "researcher");
    }

    #[test]
    fn test_defaults_applied() {
        let yaml = r#"
runtime:
  llm_provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
  api_key: "key"
task:
  input: "test"
agents:
  - id: "a"
    role: "researcher"
    goal: "research"
"#;
        let config: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.runtime.name, default_runtime_name());
        assert_eq!(config.runtime.log_level, default_log_level());
        assert_eq!(config.runtime.max_tokens, default_max_tokens());
        assert_eq!(config.agents[0].restart, default_restart_policy());
        assert_eq!(config.agents[0].max_tool_calls, default_max_tool_calls());
    }

    #[test]
    fn test_env_resolution() {
        unsafe {
            std::env::set_var("TEST_ROUTEX_KEY", "resolved-value");
        }
        let yaml = r#"
runtime:
  name: "test"
  llm_provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
  api_key: "env:TEST_ROUTEX_KEY"
task:
  input: "test"
agents:
  - id: "a"
    role: "researcher"
    goal: "research"
"#;
        let config: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.runtime.api_key, "env:TEST_ROUTEX_KEY");

        unsafe {
            std::env::remove_var("TEST_ROUTEX_KEY");
        }
    }

    #[test]
    fn test_validation_rejects_duplicate_ids() {
        let yaml = r#"
runtime:
  name: "test"
  llm_provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
  api_key: "key"
task:
  input: "test"
agents:
  - id: "researcher"
    role: "researcher"
    goal: "research"
  - id: "researcher"
    role: "writer"
    goal: "write"
"#;
        let config = Config::from_file(write_temp_yaml(yaml));
        assert!(config.is_err());
        assert!(config.unwrap_err().to_string().contains("duplicate"));
    }

    #[test]
    fn test_reject_unknown_dependency() {
        let yaml = r#"
runtime:
  name: "test"
  llm_provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
  api_key: "key"
task:
  input: "test"
agents:
  - id: "writer"
    role: "writer"
    goal: "write"
    depends: ["researcher"]
"#;
        let config = Config::from_file(write_temp_yaml(yaml));
        assert!(config.is_err());
        assert!(config.unwrap_err().to_string().contains("researcher"));
    }
}
