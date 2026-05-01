use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use routex::Runtime;

const BANNER: &str = r#"
  ██████╗  ██████╗ ██╗   ██╗████████╗███████╗██╗  ██╗
  ██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝╚██╗██╔╝
  ██████╔╝██║   ██║██║   ██║   ██║   █████╗   ╚███╔╝
  ██╔══██╗██║   ██║██║   ██║   ██║   ██╔══╝   ██╔██╗
  ██║  ██║╚██████╔╝╚██████╔╝   ██║   ███████╗██╔╝ ██╗
  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝

  lightweight AI agent runtime for Go
"#;

/// routex - a YAML-driven multi-agent AI runtime
#[derive(Parser)]
#[command(
    name = "routex",
    version = "0.1.0",
    about = "A YAML-driven multi-agent AI runtime",
    long_about = BANNER
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a crew from a YAML config file
    Run {
        /// Path to the agents.yaml file
        #[arg(default_value = "agents.yaml")]
        config: String,
    },

    /// Validate a config file without running it
    Validate {
        /// Path to the agents.yaml file
        #[arg(default_value = "agents.yaml")]
        config: String,
    },

    /// Manage tools
    Tools {
        #[command(subcommand)]
        command: ToolsCommand,
    },
}

#[derive(Subcommand)]
enum ToolsCommand {
    /// List all registered built-in tools
    List,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    match cli.command {
        Command::Run { config } => run_command(&config).await,
        Command::Validate { config } => validate_command(&config),
        Command::Tools { command: _ } => tools_list_command(),
    }
}

/// routex run agents.yaml
///
/// Loads the config, builds the runtime, runs the crew,
/// and prints the final output.
async fn run_command(config_path: &str) -> Result<()> {
    println!("routex › loading {}", config_path);

    let runtime = Runtime::from_file(config_path)
        .with_context(|| format!("failed to load config: {}", config_path))?;

    println!("routex › starting crew\n");

    let result = runtime.run().await.with_context(|| "crew run failed")?;

    println!("\n─────────────────────────────────────────");
    println!("routex › crew completed\n");
    println!("{}", result.output);
    println!("─────────────────────────────────────────");
    println!(
        "\nagents: {}  |  output: {} chars",
        result.agent_outputs.len(),
        result.output.len(),
    );

    Ok(())
}

/// routex validate agents.yaml
///
/// Parses and validates the config without running any agents.
/// Useful for catching mistakes in CI before deployment.
fn validate_command(config_path: &str) -> Result<()> {
    print!("routex › validating {} ... ", config_path);

    Runtime::from_file(config_path)
        .with_context(|| format!("validation failed: {}", config_path))?;

    println!("✓ valid");
    Ok(())
}

/// routex tools list
///
/// Lists all built-in tools available in this build.
fn tools_list_command() -> Result<()> {
    // Build a temporary runtime with an empty config to access the registry
    // This is a lightweight operation — no LLM calls, no agents
    use routex::config::{Config, RuntimeConfig, TaskConfig};

    let config = Config {
        runtime: RuntimeConfig {
            name: "tools-list".to_string(),
            llm_provider: "anthropic".to_string(),
            model: String::new(),
            api_key: String::new(),
            base_url: None,
            log_level: "info".to_string(),
            max_tokens: 4096,
        },
        task: TaskConfig {
            input: String::new(),
        },
        agents: vec![],
        tools: vec![routex::config::ToolConfig {
            name: "web_search".to_string(),
            api_key: None,
            base_dir: None,
            max_results: None,
            extra: std::collections::HashMap::new(),
        }],
    };

    let runtime = Runtime::from_config(config).context("failed to initialise runtime")?;

    let tools = runtime.list_tools();

    if tools.is_empty() {
        println!("no tools registered");
        return Ok(());
    }

    println!("\nBuilt-in tools ({})\n", tools.len());
    println!("  {:<25}  {}", "NAME", "DESCRIPTION");
    println!(
        "  {:<25}  {}",
        "────────────────────────", "──────────────────────────────────────────"
    );

    for tool in tools {
        // Truncate description at 55 chars for clean alignment
        let desc = if tool.description.len() > 55 {
            format!("{}...", &tool.description[..52])
        } else {
            tool.description.clone()
        };
        println!("  {:<25}  {}", tool.name, desc);
    }

    println!();
    Ok(())
}
