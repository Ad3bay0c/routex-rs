# Contributing to routex-rs

Thanks for your interest in contributing. This repository is a small Rust crate + CLI for running multi-agent workflows from `agents.yaml`. The goal of these guidelines is to keep changes reviewable, tested, and consistent with the current scope of the codebase.

## Code of conduct

Be respectful and constructive in issues, PRs, and reviews.

## Development setup

### Prerequisites

- Rust toolchain with Cargo (edition 2024)

### Build and test

```bash
cargo build
cargo test
```

### Format

```bash
cargo fmt
```

If you add new public APIs or change behavior, prefer adding/adjusting tests in the same PR.

## Project structure (high-level)

- `src/lib.rs`: crate entry point; re-exports `Runtime`, `Config`, and error types
- `bin/routex.rs`: CLI (`routex run`, `routex validate`, `routex tools list`)
- `src/config.rs`: YAML config types + `env:` resolution + validation
- `src/runtime.rs`: scheduler (dependency waves) + runtime orchestration
- `src/agent.rs`: agent thinking loop (LLM call → optional tool calls → repeat)
- `src/llm/`: provider adapters (`anthropic`, `openai`) and shared request/response types
- `src/tools/`: tool trait + registry + built-in tools (currently `web_search`)

## How to contribute

### 1) Fork and branch

```bash
git clone https://github.com/<your-username>/routex-rs.git
cd routex-rs
git checkout -b feat/<short-description>
```

### 2) Make focused changes

Keep PRs small and single-purpose when possible:

- bugfixes should include a minimal repro (test or clear steps)
- refactors should avoid behavior changes unless explicitly intended
- new features should match the existing direction: YAML-driven runtime + small core abstractions

### 3) Run the checks locally

```bash
cargo fmt
cargo test
```

### 4) Open a PR

When opening a PR, include:

- **what changed** and **why**
- configuration examples if you changed YAML behavior
- any relevant trade-offs or follow-ups

## Adding or changing tools

Tools are defined by the `routex::tools::Tool` trait and executed with JSON input.

### Where to implement

- Add a new tool module under `src/tools/` (e.g. `src/tools/my_tool.rs`)
- Export it from `src/tools/mod.rs`

### Registering built-in tools

The runtime auto-registers built-in tools based on `config.tools[*].name` in `Runtime::from_config` (`src/runtime.rs`). If you add a new built-in tool:

- implement the tool type
- add a match arm in `Runtime::from_config` to register it by name
- add tests for:
  - tool schema shape (required fields)
  - tool execution success and failure modes (use `mockito` if HTTP is involved)

### Tool input/output stability

Because tools are called by an LLM, small changes to a tool’s JSON schema can materially change runtime behavior. If you modify a tool schema:

- keep parameter names stable when possible
- update any examples in `README.md`
- add a test that asserts the schema contains the expected required parameters

## Adding or changing LLM providers (adapters)

LLM providers implement `routex::llm::Adapter` (`src/llm/mod.rs`) and translate between:

- routex request types (`Request`, `Message`, `ToolDefinition`)
- the provider’s HTTP API format

### Where to implement

- Add a module under `src/llm/` (e.g. `src/llm/my_provider.rs`)
- Export it from `src/llm/mod.rs`
- Wire it into adapter construction in `build_adapter` (`src/runtime.rs`)

### Testing adapters

Adapters should be tested without making real network calls:

- use `mockito` to stand up a fake HTTP server
- ensure you test:
  - successful text completions
  - tool call responses (if supported)
  - non-2xx responses and error bodies

## Configuration changes (`agents.yaml`)

Configuration is parsed in `src/config.rs`. If you add or change config fields:

- update the YAML examples in `README.md` (only if the feature is implemented end-to-end)
- add tests that parse a minimal YAML snippet and assert defaults/validation behavior
- avoid adding “future” fields without wiring them through, unless they’re explicitly marked as not-yet-used in docs

## Documentation

Keep docs aligned with what the code does today:

- `README.md` should remain a reliable “what works now” guide
- if you add a new tool/provider/CLI flag, update the relevant section with a minimal example

## PR checklist

- [ ] `cargo fmt` passes
- [ ] `cargo test` passes
- [ ] New behavior is covered by tests (or clearly justified if not)
- [ ] `README.md` updated if user-facing behavior changed
- [ ] No secrets committed (API keys, `.env`, credentials)

