use super::{Parameter, Schema, Tool};
use crate::error::{Result, RoutexError};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::Duration;

/// WikipediaTool fetches article summaries from the Wikipedia REST API.
///
/// Uses the Wikipedia API's summary endpoint — no API key required.
/// Returns the article extract, URL, and basic metadata.
///
/// agents.yaml:
///
///   tools:
///     - name: "wikipedia"
pub struct WikipediaTool {
    client: Client,
    base_url: String,
}

/// The JSON the LLM sends when calling this tool
#[derive(Debug, Deserialize)]
struct WikipediaInput {
    /// The article title or search term
    query: String,

    /// Maximum length of the extract in characters
    /// Defaults to 1000
    #[serde(default = "default_max_length")]
    max_length: usize,
}

/// Wikipedia REST API summary response
#[derive(Debug, Deserialize)]
struct WikipediaSummary {
    #[serde(default)]
    title: String,

    #[serde(default)]
    extract: String,

    #[serde(default)]
    description: Option<String>,

    content_urls: Option<ContentUrls>,
}

#[derive(Debug, Deserialize)]
struct ContentUrls {
    desktop: Option<DesktopUrl>,
}

#[derive(Debug, Deserialize)]
struct DesktopUrl {
    page: Option<String>,
}

/// What we return to the LLM
#[derive(Debug, Serialize)]
struct WikipediaOutput {
    title: String,
    extract: String,
    description: Option<String>,
    url: Option<String>,
    truncated: bool,
}

fn default_max_length() -> usize {
    1000
}

impl WikipediaTool {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(15))
                .user_agent("routex-rs/0.3.0")
                .build()
                .expect("failed to build HTTP client"),
            base_url: "https://en.wikipedia.org".to_string(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            ..Self::new()
        }
    }
}

impl Default for WikipediaTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WikipediaTool {
    fn name(&self) -> &str {
        "wikipedia"
    }

    fn schema(&self) -> Schema {
        Schema {
            description: "Fetch article summaries from Wikipedia. \
                Use for factual information, definitions, historical events, \
                and general knowledge. Provide the article title or topic \
                as the query."
                .to_string(),
            parameters: HashMap::from([
                (
                    "query".to_string(),
                    Parameter {
                        kind: "string".to_string(),
                        description: "Article title or topic to search. \
                            More specific titles give better results. \
                            Example: 'Rust programming language'"
                            .to_string(),
                        required: true,
                    },
                ),
                (
                    "max_length".to_string(),
                    Parameter {
                        kind: "integer".to_string(),
                        description: "Maximum extract length in characters. \
                            Defaults to 1000."
                            .to_string(),
                        required: false,
                    },
                ),
            ]),
        }
    }

    async fn execute(&self, input: Value) -> Result<Value> {
        let params: WikipediaInput =
            serde_json::from_value(input).map_err(|e| RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("invalid input: {}", e),
            })?;

        let title = urlencoding::encode(&params.query);
        let url = format!("{}/api/rest_v1/page/summary/{}", self.base_url, title);

        let response = self
            .client
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("request failed: {}", e),
            })?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(json!({
                "error": format!(
                    "No Wikipedia article found for '{}'. \
                    Try a more specific or differently worded title.",
                    params.query
                )
            }));
        }

        if !response.status().is_success() {
            return Err(RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("Wikipedia returned status {}", response.status()),
            });
        }

        let summary: WikipediaSummary =
            response.json().await.map_err(|e| RoutexError::ToolFailed {
                name: self.name().to_string(),
                reason: format!("parse response: {}", e),
            })?;

        let (extract, truncated) = if summary.extract.len() > params.max_length {
            let truncated_text = summary
                .extract
                .chars()
                .take(params.max_length)
                .collect::<String>();
            (truncated_text, true)
        } else {
            (summary.extract.clone(), false)
        };

        let url = summary
            .content_urls
            .and_then(|u| u.desktop)
            .and_then(|d| d.page);

        let output = WikipediaOutput {
            title: summary.title,
            extract,
            description: summary.description,
            url,
            truncated,
        };

        serde_json::to_value(output).map_err(RoutexError::Json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;
    use serde_json::json;

    fn fake_wikipedia_response() -> Value {
        json!({
            "title": "Rust (programming language)",
            "extract": "Rust is a multi-paradigm, general-purpose \
                programming language that emphasizes performance, \
                type safety, and concurrency.",
            "description": "General-purpose programming language",
            "content_urls": {
                "desktop": {
                    "page": "https://en.wikipedia.org/wiki/Rust_(programming_language)"
                }
            }
        })
    }

    #[tokio::test]
    async fn test_execute_returns_summary() {
        let mut server = Server::new_async().await;

        server
            .mock("GET", mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(fake_wikipedia_response().to_string())
            .create_async()
            .await;

        let tool = WikipediaTool::with_base_url(server.url());
        let result = tool
            .execute(json!({
                "query": "Rust programming language"
            }))
            .await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output["title"], "Rust (programming language)");
        assert!(!output["extract"].as_str().unwrap().is_empty());
        assert!(output["url"].is_string());
        assert_eq!(output["truncated"], false);
    }

    #[tokio::test]
    async fn test_not_found_returns_helpful_message() {
        let mut server = Server::new_async().await;

        server
            .mock("GET", mockito::Matcher::Any)
            .with_status(404)
            .create_async()
            .await;

        let tool = WikipediaTool::with_base_url(server.url());
        let result = tool
            .execute(json!({
                "query": "nonexistent article xyz123"
            }))
            .await;

        assert!(result.is_ok()); // not an error — graceful handling
        let output = result.unwrap();
        assert!(output["error"].is_string());
        assert!(
            output["error"]
                .as_str()
                .unwrap()
                .contains("No Wikipedia article")
        );
    }

    #[tokio::test]
    async fn test_extract_truncated_at_max_length() {
        let mut server = Server::new_async().await;

        let long_extract = "A".repeat(2000);
        let response = json!({
            "title": "Test",
            "extract": long_extract,
            "content_urls": {
                "desktop": { "page": "https://en.wikipedia.org/wiki/Test" }
            }
        });

        server
            .mock("GET", mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(response.to_string())
            .create_async()
            .await;

        let tool = WikipediaTool::with_base_url(server.url());
        let result = tool
            .execute(json!({
                "query": "Test",
                "max_length": 500
            }))
            .await
            .unwrap();

        assert_eq!(result["extract"].as_str().unwrap().len(), 500);
        assert_eq!(result["truncated"], true);
    }

    #[test]
    fn test_name() {
        assert_eq!(WikipediaTool::new().name(), "wikipedia");
    }

    #[test]
    fn test_schema_has_required_query() {
        let schema = WikipediaTool::new().schema();
        assert!(schema.parameters["query"].required);
        assert!(!schema.parameters["max_length"].required);
    }
}
