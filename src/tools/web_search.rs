use std::{collections::HashMap, time::Duration};

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    Result, RoutexError,
    tools::{Parameter, Schema, Tool},
};

/// WebSearchTool searches the web using the DuckDuckGo Instant Answer API.
///
/// DuckDuckGo is used as the default because it requires no API key —
/// making it the perfect zero-config tool for getting started quickly.
/// For production workloads, swap to BraveSearch which has richer results.
///
/// agents.yaml:
///
///   tools:
///     - name: "web_search"
///
/// No api_key needed for DuckDuckGo.
pub struct WebSearchTool {
    client: Client,
    max_results: usize,
    base_url: String,
}

/// The JSON the LLM sends when calling this tool
#[derive(Debug, Deserialize)]
struct WebSearchInput {
    query: String,

    #[serde(default = "default_max_results")]
    max_result: usize,
}

/// A single search result returned to the LLM
#[derive(Debug, Serialize)]
struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

/// The full response returned to the LLM
#[derive(Debug, Serialize)]
struct WebSearchOutput {
    query: String,
    results: Vec<SearchResult>,
    total: usize,
}

/// DuckDuckGo API response — only the fields we use
#[derive(Debug, Deserialize)]
struct DuckDuckGoResponse {
    #[serde(rename = "AbstractText")]
    abstract_text: String,

    #[serde(rename = "AbstractURL")]
    abstract_url: String,

    #[serde(rename = "AbstractSource")]
    abstract_source: String,

    #[serde(rename = "RelatedTopics")]
    related_topics: Vec<DuckDuckGoTopic>,
}

#[derive(Debug, Deserialize)]
struct DuckDuckGoTopic {
    #[serde(rename = "Text")]
    text: String,

    #[serde(rename = "FirstURL")]
    first_url: String,

    #[serde(rename = "Result")]
    result: Option<String>,
}

fn default_max_results() -> usize {
    5
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(15))
                .user_agent("routex-rs/0.1.0")
                .build()
                .expect("failed to build HTTP client"),
            max_results: default_max_results(),
            base_url: "https://api.duckduckgo.com".to_string(),
        }
    }

    /// Create with a custom base URL — used in tests.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            ..Self::new()
        }
    }

    /// Create with a custom max_results setting.
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn schema(&self) -> Schema {
        Schema {
            description: "Search the web for current information about a topic. \
			Use this when you need facts, recent events, or data you do not already know. \
			Returns a list of relevant results with titles, URLs, and snippets."
                .to_string(),
            parameters: HashMap::from([
                (
                    "query".to_string(),
                    Parameter {
                        kind: "string".to_string(),
                        description: "The search query. Be specific for \
                            best results. Example: 'Go 1.24 release notes'"
                            .to_string(),
                        required: true,
                    },
                ),
                (
                    "max_results".to_string(),
                    Parameter {
                        kind: "string".to_string(),
                        description: "Maximum number of results to return. \
                            Defaults to 5."
                            .to_string(),
                        required: false,
                    },
                ),
            ]),
        }
    }

    async fn execute(&self, input: Value) -> Result<Value> {
        let params: WebSearchInput =
            serde_json::from_value(input).map_err(|e| RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("invalid input: {}", e),
            })?;
        if params.query.is_empty() {
            return Err(RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: "empty query".to_string(),
            });
        }

        let max = params.max_result.min(self.max_results);

        let url = format!(
            "{}/?q={}&format=json&no_redirect=1&no_html=1&skip_disambig=1",
            self.base_url,
            urlencoding::encode(&params.query)
        );

        // make HTTP request
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("request failed: {}", e),
            })?;

        // check status code
        if !response.status().is_success() {
            return Err(RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("DuckDuckGo returned status {}", response.status()),
            });
        }

        // parse response
        let ddg: DuckDuckGoResponse =
            response.json().await.map_err(|e| RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("parse response error: {}", e),
            })?;

        let mut results: Vec<SearchResult> = Vec::new();

        // The abstract is the top summary result — add it first if present
        if !ddg.abstract_text.is_empty() && !ddg.abstract_url.is_empty() {
            results.push(SearchResult {
                title: ddg.abstract_source.clone(),
                url: ddg.abstract_url.clone(),
                snippet: ddg.abstract_text.clone(),
            });
        }

        // Add related topics up to max_results
        for topic in ddg
            .related_topics
            .iter()
            .take(max.saturating_sub(results.len()))
        {
            let (text, url) = (&topic.text, &topic.first_url);
            if !text.is_empty() && !url.is_empty() {
                let (title, snippet) = match text.split_once(" - ") {
                    Some((t, s)) => (t.to_string(), s.to_string()),
                    None => (text.clone(), text.clone()),
                };

                results.push(SearchResult {
                    title,
                    url: url.clone(),
                    snippet,
                });
            }
        }

        let output = WebSearchOutput {
            query: params.query,
            total: results.len(),
            results,
        };

        serde_json::to_value(output).map_err(RoutexError::Json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use serde_json::json;

    fn fake_ddg_response() -> serde_json::Value {
        json!({
            "AbstractText": "Go is a statically typed compiled language.",
            "AbstractURL": "https://go.dev",
            "AbstractSource": "Go.dev",
            "RelatedTopics": [
                {
                    "Text": "Golang Tutorial - Learn Go programming",
                    "FirstURL": "https://go.dev/tour"
                },
                {
                    "Text": "Go Packages - Standard library documentation",
                    "FirstURL": "https://pkg.go.dev"
                }
            ]
        })
    }

    #[tokio::test]
    async fn test_execute_return_results() {
        let mut server = Server::new_async().await;

        let mock = server
            .mock("GET", mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(fake_ddg_response().to_string())
            .create_async()
            .await;

        let tool = WebSearchTool::with_base_url(server.url());
        let result = tool.execute(json!({ "query": "golang"})).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output["results"].is_array());
        assert!(output["total"].as_u64().unwrap() > 0);
        assert_eq!(output["query"], "golang");

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_execute_invalid_input_returns_error() {
        let tool = WebSearchTool::new();
        // Missing required "query" field
        let result = tool.execute(json!({ "max_results": 5 })).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid input"));
    }

    #[tokio::test]
    async fn test_execute_server_error_returns_error() {
        let mut server = Server::new_async().await;

        server
            .mock("GET", mockito::Matcher::Any)
            .with_status(500)
            .create_async()
            .await;

        let tool = WebSearchTool::with_base_url(server.url());
        let result = tool.execute(json!({ "query": "golang" })).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_name() {
        let tool = WebSearchTool::new();
        assert_eq!(tool.name(), "web_search");
    }

     #[test]
    fn test_schema_has_required_fields() {
        let tool = WebSearchTool::new();
        let schema = tool.schema();
        assert!(!schema.description.is_empty());
        assert!(schema.parameters.contains_key("query"));
        assert!(schema.parameters["query"].required);
        assert!(schema.parameters.contains_key("max_results"));
        assert!(!schema.parameters["max_results"].required);
    }
}
