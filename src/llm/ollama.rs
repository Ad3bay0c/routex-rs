use super::openai::OpenAIAdapter;
use super::{Adapter, Request, Response};
use crate::error::Result;
use async_trait::async_trait;

/// Default Ollama base URL — the local server address
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// OllamaAdapter runs local LLMs via Ollama.
///
/// Ollama exposes an OpenAI-compatible API so this adapter
/// is a thin wrapper around OpenAIAdapter with a different base URL.
/// No API key is required — Ollama runs locally.
///
/// agents.yaml:
///
///   runtime:
///     llm_provider: "ollama"
///     model: "llama3"
///
/// Make sure Ollama is running:
///   ollama serve
///   ollama pull llama3
pub struct OllamaAdapter {
    inner: OpenAIAdapter,
}

impl OllamaAdapter {
    /// Create a new OllamaAdapter connecting to the default local URL.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            inner: OpenAIAdapter::new("ollama", model).with_base_url(DEFAULT_OLLAMA_URL),
        }
    }

    /// Connect to a custom Ollama URL — for remote or Docker deployments.
    pub fn with_url(model: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            inner: OpenAIAdapter::new("ollama", model).with_base_url(url),
        }
    }
}

#[async_trait]
impl Adapter for OllamaAdapter {
    async fn complete(&self, req: Request) -> Result<Response> {
        // Delegate entirely to OpenAIAdapter
        // Ollama's API is OpenAI-compatible so no translation needed
        self.inner.complete(req).await
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider(&self) -> &str {
        "ollama"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_and_model() {
        let adapter = OllamaAdapter::new("llama3");
        assert_eq!(adapter.provider(), "ollama");
        assert_eq!(adapter.model(), "llama3");
    }

    #[test]
    fn test_custom_url() {
        let adapter = OllamaAdapter::with_url("llama3", "http://remote-server:11434");
        assert_eq!(adapter.model(), "llama3");
    }
}
