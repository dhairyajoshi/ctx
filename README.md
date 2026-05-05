# ctx

`ctx` is a local repo knowledge graph CLI and MCP server for code agents. It indexes a repository into a compact SQLite graph, builds optional embedding vectors, and gives humans or agents fast ways to ask:

- Where is this feature implemented?
- What depends on this file or symbol?
- Which tests are related to this change?
- What files are conceptually relevant to this task?
- What compact context should an agent read before editing?

The default design is local-first:

- No server required
- SQLite embedded storage
- `sqlite-vec` embedded vector search
- Minimal runtime dependencies managed with `uv`
- Optional embedding providers
- MCP tools for agent integrations

## Installation

Install globally with `uv`:

```bash
uv tool install /Users/dhairyajoshi/workspace/personal/ctx
ctx --help
```

Install globally in editable mode while developing:

```bash
uv tool install --editable /Users/dhairyajoshi/workspace/personal/ctx
ctx --help
```

Run directly from this checkout without installing:

```bash
./ctx --help
./ctx --repo /path/to/repo index
```

Set up the development environment:

```bash
uv sync
uv run python tests/test_smoke.py
```

## Quick Start

From inside a target repo:

```bash
ctx init --storage central --update commit
ctx index
ctx status
ctx search billing
ctx semantic "where are invoices created"
ctx symbol calculateTotal
ctx impact src/billing/calculate_total.py
ctx tests src/billing/calculate_total.py
ctx explain "checkout flow"
```

From outside a target repo:

```bash
ctx --repo /path/to/repo init --storage central --update commit
ctx --repo /path/to/repo index
ctx --repo /path/to/repo semantic "auth session lifecycle"
```

## Storage

`ctx` stores graph data in SQLite and vector data in `sqlite-vec` virtual tables inside the same database.

Central storage:

```bash
ctx init --storage central
```

Graph path:

```text
~/.ctx/repos/<repo-key>/graph.sqlite
```

Repo-local storage:

```bash
ctx init --storage repo
```

Graph path:

```text
.ctx/graph.sqlite
```

Use central storage for personal/local usage. Use repo-local storage only if a team intentionally wants to share graph artifacts or use a stable repo-relative graph location.

## Config

`ctx init` writes `ctx.config.json`:

```json
{
  "storage": "central",
  "update": "manual",
  "include_extensions": [".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".md"],
  "ignore": [".git", ".ctx", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "dist", "build", ".next", ".turbo", "coverage", "vendor", "ctx.config.json"],
  "features": {}
}
```

Feature groups are optional:

```json
{
  "features": {
    "billing": ["src/billing/**", "src/payments/**"],
    "auth": ["src/auth/**", "src/session/**"]
  }
}
```

## Indexing

Build or rebuild the graph:

```bash
ctx index
```

Show graph status:

```bash
ctx status
```

What is indexed today:

- Files
- Python imports, functions, classes, calls, test files
- JavaScript/TypeScript imports, functions, arrow functions, classes, simple routes, calls, test files
- Package imports
- Feature group edges
- Test-to-file relationships
- Compact terms for semantic fallback

The graph is intentionally compact. It is meant to guide agents to the right files, symbols, and tests before they spend tokens reading code.

## Search Commands

Exact-ish search over names, paths, and metadata:

```bash
ctx search invoice
```

Symbol lookup:

```bash
ctx symbol createInvoice
```

Semantic retrieval:

```bash
ctx semantic "where are invoices created"
```

Impact analysis:

```bash
ctx impact src/billing/invoice.py
ctx impact createInvoice
```

Test suggestions:

```bash
ctx tests src/billing/invoice.py
```

Compact topic brief:

```bash
ctx explain "checkout flow"
```

JSON output:

```bash
ctx semantic "auth session lifecycle" --json
ctx impact src/auth/session.py --json
```

## Semantic Search

`ctx semantic` has two modes.

Term-vector fallback:

```bash
ctx semantic "where are invoices created" --term-only
```

Embedding-backed search:

```bash
ctx embed
ctx semantic "where are invoices created"
```

`ctx embed` stores vectors in the same SQLite database and mirrors them into `sqlite-vec` vector tables for nearest-neighbor search. It skips unchanged nodes on later runs. Rebuild everything with:

```bash
ctx embed --force
```

The search command uses embeddings when vectors exist for the selected provider and model. If embeddings are unavailable, it falls back to local term-vector ranking.

If `sqlite-vec` is unavailable in a source checkout, `ctx` falls back to exact JSON-vector cosine search so commands still work. Installed `uv` environments install `sqlite-vec` and use the real vector backend.

## Embedding Providers

OpenAI:

```bash
OPENAI_API_KEY=... ctx embed --provider openai --model text-embedding-3-small
ctx semantic --provider openai --model text-embedding-3-small "where invoices are created"
```

OpenAI-compatible providers:

```bash
CTX_EMBED_BASE_URL=https://provider.example/v1 \
CTX_EMBED_API_KEY=... \
ctx embed --provider openai-compatible --model your-embedding-model
```

Local Ollama:

```bash
ollama pull embeddinggemma
ctx embed --provider ollama --model embeddinggemma
ctx semantic --provider ollama --model embeddinggemma "auth session lifecycle"
```

Voyage AI:

```bash
VOYAGE_API_KEY=... ctx embed --provider voyage --model voyage-code-3
ctx semantic --provider voyage --model voyage-code-3 "where invoices are created"
```

Claude/Anthropic:

```bash
ctx embed --provider claude
```

Claude/Anthropic does not provide embedding models directly. That command fails with a clear message and recommends `--provider voyage`.

Environment variables:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
CTX_EMBED_PROVIDER
CTX_EMBED_MODEL
CTX_EMBED_DIMENSIONS
CTX_EMBED_BASE_URL
CTX_EMBED_API_KEY
OLLAMA_BASE_URL
CTX_OLLAMA_URL
VOYAGE_API_KEY
CTX_VOYAGE_API_KEY
```

## Update Policies

Manual:

```bash
ctx init --update manual
ctx index
```

Commit-based:

```bash
ctx init --update commit
ctx update
```

Watch-style:

```bash
ctx init --update watch
ctx watch
ctx watch --once
```

`ctx update` only reindexes when the configured policy says the graph is stale.

## MCP Server

Run MCP over stdio:

```bash
ctx mcp --ensure-index
```

Install into a generic MCP JSON config:

```bash
ctx --repo /path/to/repo install-mcp --config /path/to/mcp-config.json
```

Use this checkout's local executable in the MCP config:

```bash
./ctx --repo /path/to/repo install-mcp --config /path/to/mcp-config.json --local
```

Customize the MCP server name or command:

```bash
ctx --repo /path/to/repo install-mcp \
  --config /path/to/mcp-config.json \
  --name ctx-local \
  --command /absolute/path/to/ctx
```

Generated config shape:

```json
{
  "mcpServers": {
    "ctx": {
      "command": "ctx",
      "args": ["--repo", "/path/to/repo", "mcp", "--ensure-index"]
    }
  }
}
```

MCP tools exposed:

```text
ctx_search
ctx_semantic
ctx_symbol
ctx_impact
ctx_tests
ctx_explain
ctx_status
```

## Recommended Agent Workflow

For planning:

```text
1. ctx_status
2. ctx_semantic for the task description
3. ctx_impact on the highest-confidence files/symbols
4. ctx_tests for changed files
5. Read only the top relevant source files
```

For implementation:

```text
1. Run ctx semantic "<task>"
2. Inspect top files/symbols
3. Run ctx impact before editing shared code
4. Run ctx tests for touched files
5. Re-run ctx index after structural changes
```

For multi-agent councils:

```text
1. Coordinator queries ctx_semantic and ctx_impact
2. All agents receive the same compact graph brief
3. Agents debate plans from shared evidence
4. Workers edit bounded file sets
5. Reviewer runs impact/tests before final answer
```

## Development

Run tests:

```bash
uv run python tests/test_smoke.py
```

Compile check:

```bash
python -m compileall -q src tests
```

Run the local executable:

```bash
./ctx --help
```

Regenerate the lockfile:

```bash
uv lock
```

## Current Limits

This is still an MVP.

Known limitations:

- JS/TS parsing uses conservative regexes, not a compiler
- Python call edges are name-based
- Import resolution is basic
- Embedding vectors are mirrored into `sqlite-vec`; JSON vectors are retained for portability/fallback
- Large-scale ANN tuning still needs evaluation
- No incremental per-file reindexing yet
- No Tree-sitter integration yet

Good next upgrades:

- Tree-sitter parsers for Python, JS, TS, Go, Rust
- Real import/module resolution
- Incremental indexing from git diff
- Kuzu graph backend evaluation
- Richer symbol summaries
- PR/CI impact comments
- Agent-facing graph briefs with token budgets
