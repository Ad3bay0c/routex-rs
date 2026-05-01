[![Repo](https://img.shields.io/badge/github-routex--rs-181717?logo=github&logoColor=white)](https://github.com/Ad3bay0c/routex-rs)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-DEA584?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status: WIP](https://img.shields.io/badge/status-WIP-yellow)](#)

```
  ██████╗  ██████╗ ██╗   ██╗████████╗███████╗██╗  ██╗      ██████╗ ███████╗
  ██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝╚██╗██╔╝      ██╔══██╗██╔════╝
  ██████╔╝██║   ██║██║   ██║   ██║   █████╗   ╚███╔╝ █████╗██████╔╝███████╗
  ██╔══██╗██║   ██║██║   ██║   ██║   ██╔══╝   ██╔██╗ ╚════╝██╔══██╗╚════██║
  ██║  ██║╚██████╔╝╚██████╔╝   ██║   ███████╗██╔╝ ██╗      ██║  ██║███████║
  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝      ╚═╝  ╚═╝╚══════╝

  Routex-rs — lightweight AI agent runtime for Rust
```


Routex is a small Rust crate + CLI for running **multi-agent** workflows defined in an `agents.yaml`.
Agents form a dependency graph; the runtime executes independent agents **in parallel** and passes upstream outputs into downstream prompts.

This repository currently includes:

- **runtime**: builds execution “waves” from agent dependencies and runs them on Tokio
- **agent loop**: calls an LLM, optionally executes tool calls concurrently, then continues until a final text response
- **LLM adapters**: `anthropic` and `openai` (HTTP via `reqwest`)
- **tools**: a registry plus one built-in tool, `web_search` (DuckDuckGo Instant Answer API)

## Install

Routex is early-stage. For now, the simplest way to try it is from source.

```bash
git clone https://github.com/Ad3bay0c/routex-rs.git
cd routex-rs
cargo build
```

## Quickstart (CLI)

1) Create an `agents.yaml` in the repo root.

```yaml
runtime:
  name: "demo"
  llm_provider: "anthropic" # or "openai"
  model: "claude-haiku-4-5-20251001"
  api_key: "env:ANTHROPIC_API_KEY"

task:
  input: "Compare three Rust web frameworks in a short table."

tools:
  - name: "web_search"

agents:
  - id: "researcher"
    role: "researcher"
    goal: "Gather key facts and links."
    tools: ["web_search"]

  - id: "writer"
    role: "writer"
    goal: "Write a concise comparison using the research."
    depends: ["researcher"]
```

2) Export your API key and run.

```bash
export ANTHROPIC_API_KEY="..."
cargo run --bin routex -- run agents.yaml
```

You can also validate the config without running any agents:

```bash
cargo run --bin routex -- validate agents.yaml
```

## Using as a library

The public entry points are `routex::Runtime` and `routex::Config`.

```rust
use routex::Runtime;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let runtime = Runtime::from_file("agents.yaml")?;
    let result = runtime.run().await?;
    println!("{}", result.output);
    Ok(())
}
```

## Configuration

Routex loads configuration via `serde_yaml` into `routex::config::Config`.

### `env:` secrets

String fields like `runtime.api_key` support `env:VAR_NAME` syntax.
At load time, `env:ANTHROPIC_API_KEY` is replaced with the value of `$ANTHROPIC_API_KEY` (or an empty string if unset).

### Agents and dependencies

- **`agents[*].id`** must be unique and non-empty.
- **`agents[*].depends`** lists upstream agent IDs.
- The runtime constructs a DAG and executes agents **wave-by-wave** (topological order).

When an agent runs, its input prompt is:

- the original `task.input`, plus
- a “Context from previous agents” section containing outputs from its dependencies (if any).

### Roles

`agents[*].role` is one of: `planner`, `writer`, `critic`, `executor`, `researcher`.
The role selects a built-in system prompt template; `agents[*].goal` is appended to that prompt.

## Tools

Tools implement `routex::tools::Tool` and are executed with JSON input.

### Built-in tool: `web_search`

- **name**: `web_search`
- **backend**: DuckDuckGo Instant Answer API
- **input**: `{ "query": "..." }` (optionally `max_results`)
- **output**: JSON containing `results[]` with `title`, `url`, and `snippet`

Enable it in `agents.yaml` by listing it under `tools` and then allowing it per-agent:

```yaml
tools:
  - name: "web_search"

agents:
  - id: "researcher"
    role: "researcher"
    goal: "Find sources."
    tools: ["web_search"]
```

## LLM providers

The runtime currently supports:

- **`anthropic`**: calls the Anthropic Messages API
- **`openai`**: calls `/v1/chat/completions`

Select the provider via `runtime.llm_provider` and set `runtime.model` + `runtime.api_key`.

## Current limitations (by design / not implemented yet)

This project is intentionally small; some configuration fields exist but are not wired through everywhere yet.

- **Per-agent LLM overrides**: `agents[*].llm` exists in config, but the runtime currently builds a single adapter from `runtime.*` and does not switch adapters per agent.
- **`runtime.base_url`**: present in config (for OpenAI-compatible endpoints), but not currently applied when constructing adapters.
- **Tool configuration**: `tools[*].api_key`, `base_dir`, `max_results`, and `extra` are parsed but not currently used by the built-in `web_search` tool.
- **Token usage totals**: `RunResult.total_input_tokens` and `total_output_tokens` are returned as `0` for now.
- **Restart policies**: `agents[*].restart` is parsed but not currently enforced by the scheduler.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
