/// RoutexError is the canonical error type for the entire library.
///
/// Every operation in routex-rs that can fail returns either:
///   - Result<T, RoutexError>  — for library code
///   - anyhow::Result<T>       — for CLI code in bin/routex.rs
#[derive(Debug, thiserror::Error)]
pub enum RoutexError {
    /// Config file could not be read or parsed
    #[error("config error: {0}")]
    Config(String),

    /// A tool was referenced in agents.yaml but is not registered
    #[error("tool '{name}' is not registered")]
    ToolNotFound { name: String },

    /// A tool failed during execution
    #[error("tool '{name}' failed: {reason}")]
    ToolFailed { name: String, reason: String },

    /// LLM API call error
    #[error("llm error: {0}")]
    LLM(String),

    /// Agent failed during its thinking loop
    #[error("agent '{id}' failed: {reason}")]
    AgentFailed { id: String, reason: String },

    /// Dependency cycle detected in agents.yaml
    /// e.g. agent A depends on B, B depends on A
    #[error("dependency cycle detected involving agent '{id}'")]
    CyclicDependency { id: String },

    /// An agent declared a dependency on an agent that doesn't exist
    #[error("agent '{id}' depends on '{dep}' which does not exist")]
    UnknownDependency { id: String, dep: String },

    /// HTTP request failed — wraps reqwest errors
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization failed
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// YAML parsing failed
    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// IO error — reading config files etc
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, RoutexError>;
