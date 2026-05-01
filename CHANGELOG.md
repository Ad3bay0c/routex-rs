# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.1.1] - 2026-05-01

### Added

- **First public release** of `routex-rs` (`routex` library + `routex` CLI), tagged as [`v0.1.1`][0.1.1].
- YAML-driven crew configuration (`agents.yaml`) with dependency-aware scheduling (“waves”) for parallel execution where possible.
- Agent runtime loop: LLM completions plus concurrent tool execution when the model requests tool calls (with a configurable tool-call budget per agent).
- LLM adapters via HTTP:
  - Anthropic Messages API (`runtime.llm_provider: anthropic`)
  - OpenAI Chat Completions API (`runtime.llm_provider: openai`)
- Built-in tool registry plus `web_search` (DuckDuckGo Instant Answer API).
- CLI commands:
  - `routex run` to execute a crew from a YAML config
  - `routex validate` to parse/validate config without running agents
  - `routex tools list` to show registered tools
- CLI loads a local `.env` file when present (via `dotenvy`) to simplify local API keys during development.

### Changed

- Documentation updates for `README.md`, `CONTRIBUTING.md`, and repository metadata.

### Notes

- Some configuration fields are parsed but not fully wired through yet; see `README.md` (“Current limitations”).
- Copy/paste release text for GitHub Releases lives in [`RELEASE_NOTES_v0.1.1.md`](RELEASE_NOTES_v0.1.1.md).

[0.1.1]: https://github.com/Ad3bay0c/routex-rs/releases/tag/v0.1.1
