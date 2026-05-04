# ctx

`ctx` is a local repo knowledge graph CLI for developers and code agents. It indexes a repository into a SQLite graph, answers compact impact/search/test questions, and exposes the same graph through an MCP stdio server.

The MVP is intentionally local-first and dependency-light.

## Install

From this checkout:

```bash
uv tool install /Users/dhairyajoshi/workspace/personal/ctx
```

For development:

```bash
uv sync
./ctx --help
./ctx index
uv run python tests/test_smoke.py
```

To install a global `ctx` executable from this checkout:

```bash
uv tool install --editable /Users/dhairyajoshi/workspace/personal/ctx
ctx --help
```

## Quick Start

```bash
cd /path/to/repo
ctx init --storage central --update commit
ctx index
ctx status
ctx search billing
ctx symbol calculateTotal
ctx impact src/billing/calculate_total.py
ctx tests src/billing/calculate_total.py
ctx explain "checkout flow"
```

Storage modes:

- `central`: graph lives under `~/.ctx/repos/<repo-key>/graph.sqlite`
- `repo`: graph lives under `.ctx/graph.sqlite` in the target repo

`ctx.config.json` is shareable. `.ctx/` is ignored by default when using repo storage, but teams can choose to commit graph artifacts if they want.

## MCP

Run the MCP stdio server from inside a repo:

```bash
ctx mcp --ensure-index
```

Add `ctx` to an MCP JSON config file:

```bash
ctx --repo /path/to/repo install-mcp --config /path/to/mcp-config.json
```

From this checkout, use the local executable:

```bash
./ctx --repo /path/to/repo install-mcp --config /path/to/mcp-config.json --local
```

Tools exposed:

- `ctx_search`
- `ctx_symbol`
- `ctx_impact`
- `ctx_tests`
- `ctx_explain`
- `ctx_status`

Example MCP server config:

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

## Update Policy

`ctx.config.json` controls freshness:

```json
{
  "storage": "central",
  "include_extensions": [".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".json"],
  "update": "manual"
}
```

Modes:

- `manual`: only update when `ctx index` is run
- `commit`: `ctx update` reindexes when `git rev-parse HEAD` changes
- `watch`: `ctx update` and `ctx watch` reindex whenever invoked

Commands:

```bash
ctx update
ctx watch
ctx watch --once
```

## What It Indexes Today

Python:

- imports
- functions
- methods
- classes
- call names
- test files

JavaScript/TypeScript:

- imports
- function declarations
- arrow-function assignments
- classes
- route declarations like `router.get("/path", ...)`
- call names
- test files

Text/config files:

- compact term metadata for search

## Current Limits

This MVP is not a full compiler-grade graph. JS/TS parsing uses conservative regexes, and cross-file call edges are name-based. That is good enough for compact agent context and impact hints, but deeper accuracy should come next from Tree-sitter or language server adapters.

Good next improvements:

- Tree-sitter parsers for JS/TS/Python
- import resolution to concrete files
- Git diff-aware partial indexing
- embeddings for docs/comments
- PR comment mode
- graph export to Kuzu/Neo4j
- richer MCP prompts/resources
