# ctx

`ctx` is a local repo knowledge graph for code agents. It indexes a codebase into a compact SQLite graph, embeds every node, and exposes the result as a CLI and an MCP server so any agent can answer:

- Where is this feature implemented?
- What depends on this file or symbol?
- Which tests are related to this change?
- What files are conceptually relevant to this task?
- What compact context should I read before editing?

Local-first by design: SQLite + `sqlite-vec`, no server, no account, optional cloud embeddings.

## 60-second quickstart

```bash
# 1. install
uv tool install git+https://github.com/your-org/ctx        # or: uv tool install /path/to/checkout

# 2. index a repo (auto-embeds with the built-in local provider)
cd /path/to/your/repo
ctx init
ctx index

# 3. try it
ctx semantic "where invoices are created"
ctx impact src/billing/invoice.py
```

That's it. No API keys, no extra services. `ctx index` builds the graph **and** embeds every node into `sqlite-vec` so semantic queries are fast from the first run.

## Plug into your agent

`ctx` ships an MCP server. **One install, every repo** — a single MCP entry serves any repo you've indexed. The server auto-resolves which repo to query from `CTX_REPO`, the agent's working directory, or an explicit `repo` arg.

### Claude Code

```bash
ctx install-mcp --config ~/.claude.json --name ctx
```

Restart Claude Code. The tools `ctx_search`, `ctx_semantic`, `ctx_symbol`, `ctx_impact`, `ctx_tests`, `ctx_explain`, `ctx_status`, and `ctx_repos` will appear and work in any indexed repo.

### Cursor / VS Code (workspace-aware)

```bash
ctx install-mcp --config ~/.cursor/mcp.json --name ctx --workspace-env
```

`--workspace-env` adds `CTX_REPO=${workspaceFolder}` so the server always queries the current workspace.

### Any MCP client (generic JSON config)

```bash
ctx install-mcp --config /path/to/mcp.json
```

The generated entry:

```json
{
  "mcpServers": {
    "ctx": {
      "command": "ctx",
      "args": ["mcp", "--multi"]
    }
  }
}
```

### Indexing repos for the multi-repo server

The server can only query repos it knows about. Add a repo to the registry by indexing it once:

```bash
cd /path/to/your/repo
ctx index
```

`ctx index` (and `ctx init`) automatically register the repo into `~/.ctx/repos.json`. List, change default, or remove entries:

```bash
ctx repos                                # list registered repos
ctx repos --set-default <name>           # change the default
ctx register /path/to/repo --name myrepo # register without indexing
ctx unregister myrepo
```

### How the server picks a repo

For each tool call, in order:

1. Explicit `repo` arg in the tool input (path or registered nickname).
2. `CTX_REPO` environment variable.
3. The agent's working directory, walked up to the nearest registered repo.
4. The registry default.

Agents can call `ctx_repos` once to enumerate available repos, then pass `repo` in subsequent calls for cross-repo work.

### Pinning to a single repo (legacy)

If you want the server hard-bound to one repo (the previous behavior):

```bash
ctx --repo /path/to/your/repo install-mcp \
  --config ~/.claude.json --name ctx --single-repo
```

That writes `["--repo", "/path/to/your/repo", "mcp", "--ensure-index"]`, which still works unchanged.

### From a checkout (no install)

```bash
./ctx install-mcp --config /path/to/mcp.json --local
```

`--local` writes the absolute path of the checkout's `./ctx` script as the command, useful while developing on `ctx` itself.

## Recommended agent prompt

Add this to your agent's system prompt or project-level instructions:

```
You have a `ctx` MCP server. It serves any repo you've indexed; the server
auto-detects the current workspace, but you can target another repo by passing
`repo: "<nickname-or-path>"` in any tool call.

Before reading or editing code, prefer:
  1. ctx_repos — list available repos if you might need a different one.
  2. ctx_status — confirm the graph is fresh.
  3. ctx_semantic("<task description>") — find conceptually relevant files.
  4. ctx_impact(<file or symbol>) — understand blast radius before edits.
  5. ctx_tests(<file>) — find tests to update.
Only read full source files for the top-ranked results.
```

## CLI reference

```bash
ctx init [--storage central|repo] [--update manual|commit|watch]
ctx index [--no-embed] [--embed-provider <name>] [--embed-model <name>]
ctx update                    # re-index when the policy says it's stale
ctx watch [--once]            # poll loop
ctx status

ctx search <term>
ctx symbol <name>
ctx semantic "<query>" [--provider <name>] [--model <name>] [--term-only]
ctx impact <file-or-symbol>
ctx tests <path>
ctx explain "<topic>"

ctx embed [--provider <name>] [--model <name>] [--force]
ctx mcp [--multi] [--ensure-index]
ctx install-mcp --config <path> [--name ctx] [--local] [--single-repo] [--workspace-env]

ctx repos [--set-default <name>]
ctx register [<path>] [--name <nick>]
ctx unregister <name-or-path>
```

JSON output on any command:

```bash
ctx semantic "auth session lifecycle" --json
```

## Storage

```bash
ctx init --storage central   # ~/.ctx/repos/<key>/graph.sqlite  (default)
ctx init --storage repo      # .ctx/graph.sqlite                (committed/shared)
```

Use `central` for personal use. Use `repo` only if a team intentionally shares the graph artifact.

## Embeddings

Out of the box, `ctx index` runs a deterministic local hashing embedder (provider `local`, model `hash-256-v1`). It produces real 256-dim dense vectors stored in `sqlite-vec`, with no network calls, no API keys, and no extra dependencies. This means `ctx semantic` does ANN search from the first index, not a slow per-node scan.

To upgrade to a transformer-grade provider, set environment variables before running `ctx index`. The indexer will automatically pick the best available provider, and fall back to `local` if the chosen provider errors.

### Better results with Voyage or OpenAI

For better conceptual ranking in `ctx semantic` and `ctx_explain`, set a hosted embedding key and re-embed:

```bash
export VOYAGE_API_KEY=...
ctx index --embed-force
```

When `VOYAGE_API_KEY` is set, `ctx` prefers Voyage's `voyage-code-3` model by default, even if `OPENAI_API_KEY` is also present. To use OpenAI instead, pin it explicitly:

```bash
export OPENAI_API_KEY=sk-...
ctx index --embed-provider openai --embed-model text-embedding-3-small
```

Voyage (Anthropic's recommended embeddings):

```bash
export VOYAGE_API_KEY=...
ctx index                                              # auto-detects voyage
# or pin explicitly:
ctx index --embed-provider voyage --embed-model voyage-code-3
```

Ollama (fully local, larger model):

```bash
ollama pull embeddinggemma
ctx index --embed-provider ollama --embed-model embeddinggemma
```

OpenAI-compatible providers (Together, DeepInfra, etc.):

```bash
export CTX_EMBED_BASE_URL=https://provider.example/v1
export CTX_EMBED_API_KEY=...
ctx index --embed-provider openai-compatible --embed-model your-model
```

To re-embed an existing graph without re-indexing:

```bash
ctx embed --provider voyage --model voyage-code-3 [--force]
```

`ctx semantic` automatically uses whatever provider was last used for embedding (stored in graph metadata), so queries don't require flags.

### Skipping embedding

```bash
ctx index --no-embed         # graph only
```

You can also set `embed.auto: false` in `ctx.config.json`.

### Recognized environment variables

```text
OPENAI_API_KEY, OPENAI_BASE_URL
VOYAGE_API_KEY, CTX_VOYAGE_API_KEY
CTX_EMBED_PROVIDER, CTX_EMBED_MODEL, CTX_EMBED_DIMENSIONS
CTX_EMBED_BASE_URL, CTX_EMBED_API_KEY
OLLAMA_BASE_URL, CTX_OLLAMA_URL
CTX_HOME    # overrides ~/.ctx for central storage and the repo registry
CTX_REPO    # default repo (nickname or path) for the multi-repo MCP server
```

## Config file

`ctx init` writes `ctx.config.json` in the repo root:

```json
{
  "storage": "central",
  "update": "manual",
  "include_extensions": [".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".md"],
  "ignore": [".git", ".ctx", "__pycache__", "node_modules", "dist", "build", ".next", ".turbo", "coverage", "vendor", "ctx.config.json"],
  "embed": {
    "auto": true,
    "provider": null,
    "model": null,
    "dimensions": null,
    "batch_size": 64
  },
  "features": {}
}
```

Optional feature groups let you ask "what's in billing?":

```json
{
  "features": {
    "billing": ["src/billing/**", "src/payments/**"],
    "auth": ["src/auth/**", "src/session/**"]
  }
}
```

## What gets indexed

- Files (with content hash, size, term summary)
- Python: imports, functions, classes, FastAPI/Flask-style routes, calls, test files
- JavaScript/TypeScript: imports, functions, arrow functions, classes, simple Express-style routes, calls, test files
- Package imports
- Feature group → file edges
- Test → source file edges
- Embedding vectors for every node, stored in `sqlite-vec`

## Update policies

```bash
ctx init --update manual   # rebuild only when you run ctx index
ctx init --update commit   # ctx update reindexes if HEAD changed
ctx init --update watch    # ctx watch polls and reindexes
```

## MCP tools

```text
ctx_search     exact-ish search over names, paths, metadata
ctx_semantic   hybrid BM25 + local-hash ranker over rich code context
ctx_symbol     find functions/classes/components by name
ctx_impact     dependents and dependencies for a path or symbol
ctx_callers    one-hop callers of a symbol or file
ctx_callees    one-hop callees of a symbol or file
ctx_trace      bounded call paths from one symbol/file, optionally to a target
ctx_tests      tests related to a path
ctx_explain    compact graph brief for a topic
ctx_status     graph counts, last index, last embed (resolved repo included)
ctx_repos      list registered repos and the currently-resolved repo
```

Every tool accepts an optional `repo: "<nickname-or-path>"` argument. Omit it to use `CTX_REPO` / current workspace / registry default.

## Development

```bash
uv sync
uv run --with pytest pytest tests/
./ctx --help
```

## Limits (MVP)

- JS/TS parsing uses conservative regexes, not a compiler
- Python call edges are name-based
- Import resolution is basic
- No incremental per-file reindexing yet
- Local hash embedder is great for keyword/code retrieval; for cross-language synonym search, prefer `voyage` or `openai`

## Roadmap

- Tree-sitter parsers for Python, JS, TS, Go, Rust
- Real import/module resolution
- Incremental indexing from git diff
- Token-budgeted graph briefs for agent prompts
- PR/CI impact comments
