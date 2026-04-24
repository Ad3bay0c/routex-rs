```
  ██████╗  ██████╗ ██╗   ██╗████████╗███████╗██╗  ██╗      ██████╗ ███████╗
  ██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝╚██╗██╔╝      ██╔══██╗██╔════╝
  ██████╔╝██║   ██║██║   ██║   ██║   █████╗   ╚███╔╝ █████╗██████╔╝███████╗
  ██╔══██╗██║   ██║██║   ██║   ██║   ██╔══╝   ██╔██╗ ╚════╝██╔══██╗╚════██║
  ██║  ██║╚██████╔╝╚██████╔╝   ██║   ███████╗██╔╝ ██╗      ██║  ██║███████║
  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝      ╚═╝  ╚═╝╚══════╝

  Routex-rs — lightweight AI agent runtime for Rust
```

**A lightweight AI agent runtime for Rust.**

Routex lets you build, run, and supervise multi-agent AI crews. Define your crew in a YAML file or pure Rust code, wire in any LLM provider and tools, and let the runtime handle scheduling, parallelism, retries, memory, and observability.

## Contributing

Contributions are welcome.

### 1) Fork the repository

Click **Fork** on the top right of the [Routex GitHub page](https://github.com/Ad3bay0c/routex-rs) to create a copy under your own account. This is important — you do not have write access to the main repository, so all changes must come through your fork.

### 2) Clone your fork

```bash
git clone https://github.com/<your-username>/routex-rs.git
cd routex-rs

# Add the original repo as upstream so you can sync changes later
git remote add upstream https://github.com/Ad3bay0c/routex-rs.git
git remote -v
```

### 3) Create a feature branch

```bash
git checkout -b feat/<short-description>
```

### 4) Make your changes and run tests

```bash
cargo fmt
cargo test
```

### 5) Commit and push to your fork

```bash
git add -A
git commit -m "Describe your change"
git push -u origin HEAD
```

### 6) Open a Pull Request

Open a PR from your fork/branch to `main` in `Ad3bay0c/routex-rs`.

### Standard guidelines

- Keep PRs focused and reasonably small.
- Describe the **why** and **what** (include repro steps for bugs).
- Add/update tests when behavior changes.
