from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .config import (
    load_config,
    load_registry,
    register_repo,
    save_registry,
    unregister_repo,
    write_config,
)
from .embeddings import provider_from_env
from .indexer import embed_index, index_repo, should_reindex
from .mcp import run_mcp_server
from .store import GraphStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ctx", description="Local repo knowledge graph CLI and MCP server.")
    parser.add_argument("--repo", type=Path, default=None, help="Repository path. Defaults to the nearest git/config root.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_json_flag(command: argparse.ArgumentParser) -> argparse.ArgumentParser:
        command.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
        return command

    init_cmd = add_json_flag(sub.add_parser("init", help="Create ctx.config.json."))
    init_cmd.add_argument("--storage", choices=["central", "repo"], default="central")
    init_cmd.add_argument("--update", choices=["manual", "commit", "watch"], default="manual")
    init_cmd.add_argument("--force", action="store_true")

    index_cmd = add_json_flag(sub.add_parser("index", help="Build or rebuild the graph."))
    index_cmd.add_argument("--no-reset", action="store_true", help="Do not clear existing graph before indexing.")
    embed_group = index_cmd.add_mutually_exclusive_group()
    embed_group.add_argument("--no-embed", dest="auto_embed", action="store_false", help="Skip auto-embedding after indexing.")
    embed_group.add_argument("--embed", dest="auto_embed", action="store_true", help="Force auto-embedding after indexing.")
    index_cmd.set_defaults(auto_embed=None)
    index_cmd.add_argument("--embed-provider", help="Embedding provider for auto-embed (defaults to config or env).")
    index_cmd.add_argument("--embed-model", help="Embedding model for auto-embed.")
    index_cmd.add_argument("--embed-dimensions", type=int, help="Optional embedding dimensions override.")
    index_cmd.add_argument("--embed-batch-size", type=int, help="Embedding batch size.")
    index_cmd.add_argument("--embed-force", action="store_true", help="Re-embed even when content hashes match.")

    add_json_flag(sub.add_parser("status", help="Show graph status."))
    add_json_flag(sub.add_parser("update", help="Reindex only if update policy says the graph is stale."))

    watch_cmd = add_json_flag(sub.add_parser("watch", help="Poll and reindex according to update.interval_seconds."))
    watch_cmd.add_argument("--once", action="store_true", help="Run one poll cycle and exit.")

    search_cmd = add_json_flag(sub.add_parser("search", help="Search nodes by name, path, or metadata."))
    search_cmd.add_argument("term")
    search_cmd.add_argument("--limit", type=int, default=20)

    semantic_cmd = add_json_flag(sub.add_parser("semantic", help="Rank graph nodes by hybrid BM25 + local hash search over indexed code context."))
    semantic_cmd.add_argument("query")
    semantic_cmd.add_argument("--limit", type=int, default=8)
    semantic_cmd.add_argument("--kind", choices=["symbol", "route", "file", "test", "package", "feature"], help="Restrict results to one node kind.")
    semantic_cmd.add_argument("--path-glob", help="Restrict results to paths matching a Python fnmatch glob, e.g. 'src/*'.")
    semantic_cmd.add_argument("--no-group", dest="group_by_file", action="store_false", help="Do not group co-located symbol hits from the same file.")
    semantic_cmd.set_defaults(group_by_file=True)
    semantic_cmd.add_argument("--provider", help="Embedding provider to use when embedded vectors exist.")
    semantic_cmd.add_argument("--model", help="Embedding model to use when embedded vectors exist.")
    semantic_cmd.add_argument("--term-only", action="store_true", help="Skip hosted embedding search and use the local hybrid ranker.")

    embed_cmd = add_json_flag(sub.add_parser("embed", help="Build embedding vectors for indexed graph nodes."))
    embed_cmd.add_argument("--provider")
    embed_cmd.add_argument("--model")
    embed_cmd.add_argument("--dimensions", type=int)
    embed_cmd.add_argument("--batch-size", type=int, default=64)
    embed_cmd.add_argument("--force", action="store_true")

    symbol_cmd = add_json_flag(sub.add_parser("symbol", help="Find symbols by name."))
    symbol_cmd.add_argument("name")
    symbol_cmd.add_argument("--limit", type=int, default=20)

    impact_cmd = add_json_flag(sub.add_parser("impact", help="Show dependents and dependencies for a path or symbol."))
    impact_cmd.add_argument("target")
    impact_cmd.add_argument("--limit", type=int, default=50)
    impact_cmd.add_argument("--include-vendor", action="store_true", help="Include .venv/vendor/node_modules paths in graph results.")

    callers_cmd = add_json_flag(sub.add_parser("callers", help="One-hop callers of a symbol or file."))
    callers_cmd.add_argument("target")
    callers_cmd.add_argument("--limit", type=int, default=50)
    callers_cmd.add_argument("--include-vendor", action="store_true", help="Include .venv/vendor/node_modules paths in graph results.")

    callees_cmd = add_json_flag(sub.add_parser("callees", help="One-hop callees of a symbol or file."))
    callees_cmd.add_argument("target")
    callees_cmd.add_argument("--limit", type=int, default=50)
    callees_cmd.add_argument("--include-vendor", action="store_true", help="Include .venv/vendor/node_modules paths in graph results.")

    tests_cmd = add_json_flag(sub.add_parser("tests", help="Suggest tests related to a path."))
    tests_cmd.add_argument("path")
    tests_cmd.add_argument("--limit", type=int, default=50)

    explain_cmd = add_json_flag(sub.add_parser("explain", help="Compact repo brief for a topic."))
    explain_cmd.add_argument("topic")
    explain_cmd.add_argument("--limit", type=int, default=12)

    mcp_cmd = add_json_flag(sub.add_parser("mcp", help="Run an MCP stdio server exposing graph query tools."))
    mcp_cmd.add_argument("--ensure-index", action="store_true", help="Build the graph first if it does not exist (single-repo mode only).")
    mcp_cmd.add_argument("--multi", action="store_true", help="Run in multi-repo mode; resolve the repo per tool call from CTX_REPO/cwd/registry.")

    install_mcp_cmd = add_json_flag(sub.add_parser("install-mcp", help="Add ctx to an MCP JSON config file."))
    install_mcp_cmd.add_argument("--config", type=Path, required=True, help="Path to an MCP config JSON file.")
    install_mcp_cmd.add_argument("--name", default="ctx", help="MCP server name to write.")
    install_mcp_cmd.add_argument("--command", dest="mcp_command", default="ctx", help="Executable command agents should run.")
    install_mcp_cmd.add_argument("--local", action="store_true", help="Use this checkout's ./ctx executable as the command.")
    install_mcp_cmd.add_argument("--single-repo", action="store_true", help="Pin the entry to a single repo (legacy form).")
    install_mcp_cmd.add_argument("--workspace-env", action="store_true", help="Add CTX_REPO=${workspaceFolder} to the entry env (Cursor/VS Code).")

    repos_cmd = add_json_flag(sub.add_parser("repos", help="List indexed repos in the registry."))
    repos_cmd.add_argument("--set-default", help="Set the default repo nickname.")

    register_cmd = add_json_flag(sub.add_parser("register", help="Add a repo to the registry without indexing."))
    register_cmd.add_argument("path", type=Path, nargs="?", default=None, help="Repo path. Defaults to --repo or cwd.")
    register_cmd.add_argument("--name", help="Nickname; defaults to the repo's directory name.")

    unregister_cmd = add_json_flag(sub.add_parser("unregister", help="Remove a repo from the registry."))
    unregister_cmd.add_argument("identifier", help="Repo nickname or path.")

    args = parser.parse_args(argv)

    if args.command == "repos":
        return command_repos(args)
    if args.command == "register":
        return command_register(args)
    if args.command == "unregister":
        return command_unregister(args)
    if args.command == "mcp" and getattr(args, "multi", False) and args.repo is None:
        run_mcp_server(None)
        return 0

    config = load_config(args.repo)

    if args.command == "init":
        return command_init(args, config.repo)
    if args.command == "index":
        counts = index_repo(
            config,
            reset=not args.no_reset,
            embed=args.auto_embed,
            embed_provider=args.embed_provider,
            embed_model=args.embed_model,
            embed_dimensions=args.embed_dimensions,
            embed_batch_size=args.embed_batch_size,
            embed_force=args.embed_force,
        )
        return emit(args, {"db": str(config.db_path), **counts})
    if args.command == "status":
        return command_status(args, config)
    if args.command == "update":
        stale, reason = should_reindex(config)
        if stale:
            counts = index_repo(config)
            return emit(args, {"updated": True, "reason": reason, "db": str(config.db_path), **counts})
        return emit(args, {"updated": False, "reason": reason, "db": str(config.db_path)})
    if args.command == "watch":
        return command_watch(args, config)
    if args.command == "search":
        return with_store(args, config, lambda store: store.search(args.term, args.limit))
    if args.command == "semantic":
        return with_store(args, config, lambda store: semantic_query(store, args))
    if args.command == "embed":
        return with_store(args, config, lambda store: embed_graph(store, args, config))
    if args.command == "symbol":
        return with_store(args, config, lambda store: store.symbols(args.name, args.limit))
    if args.command == "impact":
        return with_store(args, config, lambda store: store.impact(args.target, args.limit, args.include_vendor))
    if args.command == "callers":
        return with_store(args, config, lambda store: store.callers(args.target, args.limit, args.include_vendor))
    if args.command == "callees":
        return with_store(args, config, lambda store: store.callees(args.target, args.limit, args.include_vendor))
    if args.command == "tests":
        return with_store(args, config, lambda store: store.tests_for_path(args.path, args.limit))
    if args.command == "explain":
        return command_explain(args, config)
    if args.command == "mcp":
        if args.multi and args.repo is None:
            run_mcp_server(None)
            return 0
        if args.ensure_index and not config.db_path.exists():
            index_repo(config)
        run_mcp_server(config)
        return 0
    if args.command == "install-mcp":
        return command_install_mcp(args, config)
    return 1


def command_init(args: argparse.Namespace, repo: Path) -> int:
    path = repo / "ctx.config.json"
    if path.exists() and not args.force:
        print(f"{path} already exists. Use --force to overwrite.", file=sys.stderr)
        return 2
    created = write_config(repo, args.storage, args.update)
    if args.storage == "repo":
        ctxignore = repo / ".gitignore"
        if ctxignore.exists() and ".ctx/" not in ctxignore.read_text(encoding="utf-8", errors="ignore"):
            with ctxignore.open("a", encoding="utf-8") as handle:
                handle.write("\n.ctx/\n")
    try:
        register_repo(repo)
    except Exception:
        pass
    print(f"created {created}")
    return 0


def command_repos(args: argparse.Namespace) -> int:
    registry = load_registry()
    if getattr(args, "set_default", None):
        if args.set_default not in registry.entries:
            print(f"unknown repo '{args.set_default}'", file=sys.stderr)
            return 2
        registry.default = args.set_default
        save_registry(registry)
    payload = {
        "default": registry.default,
        "repos": [
            {
                "name": entry.name,
                "path": str(entry.path),
                "db": str(entry.db),
                "indexed": entry.db.exists(),
                "indexed_at": entry.indexed_at,
            }
            for entry in sorted(registry.entries.values(), key=lambda item: item.name)
        ],
    }
    return emit(args, payload)


def command_register(args: argparse.Namespace) -> int:
    repo = (args.path or args.repo or Path.cwd()).resolve()
    if not repo.is_dir():
        print(f"not a directory: {repo}", file=sys.stderr)
        return 2
    entry = register_repo(repo, name=args.name)
    return emit(args, {"registered": entry.name, "path": str(entry.path), "db": str(entry.db)})


def command_unregister(args: argparse.Namespace) -> int:
    if unregister_repo(args.identifier):
        return emit(args, {"unregistered": args.identifier})
    print(f"no registry entry for '{args.identifier}'", file=sys.stderr)
    return 2


def command_status(args: argparse.Namespace, config) -> int:
    exists = config.db_path.exists()
    payload: dict[str, Any] = {
        "repo": str(config.repo),
        "storage": config.storage,
        "db": str(config.db_path),
        "exists": exists,
    }
    if exists:
        store = GraphStore(config.db_path)
        payload.update(store.counts())
        fts = store.fts_status()
        payload["fts"] = fts
        if fts.get("warnings"):
            payload["warnings"] = fts["warnings"]
        payload["embeddings"] = store.embedding_count()
        payload["last_index"] = store.get_meta("last_index", {})
        payload["last_embed"] = store.get_meta("last_embed", {})
        store.close()
    return emit(args, payload)


def embed_graph(store: GraphStore, args: argparse.Namespace, config) -> dict[str, Any]:
    return embed_index(
        store,
        config,
        provider=args.provider,
        model=args.model,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        force=args.force,
    )


def semantic_query(store: GraphStore, args: argparse.Namespace) -> list[dict]:
    if not args.term_only:
        provider = resolve_query_provider(store, args.provider, args.model)
        if provider.provider != "local" and store.embedding_count(provider.provider, provider.model):
            try:
                query_vector = provider.embed([args.query], input_type="query")[0]
                vector_results = store.vector_search(query_vector, provider.provider, provider.model, max(args.limit * 4, 25))
                return store.shape_semantic_results(vector_results, args.limit, kind=args.kind, path_glob=args.path_glob, group_by_file=args.group_by_file)
            except Exception as exc:
                fallback = store.semantic_search(args.query, args.limit, kind=args.kind, path_glob=args.path_glob, group_by_file=args.group_by_file)
                for item in fallback:
                    item["score_source"] = "bm25_fallback"
                    item["embedding_error"] = str(exc)
                return fallback
    results = store.semantic_search(args.query, args.limit, kind=args.kind, path_glob=args.path_glob, group_by_file=args.group_by_file)
    for item in results:
        item.setdefault("score_source", "bm25")
    return results


def resolve_query_provider(store: GraphStore, provider: str | None, model: str | None):
    if provider or model:
        return provider_from_env(provider, model)
    last = store.get_meta("last_embed", {}) or {}
    return provider_from_env(last.get("provider") or None, last.get("model") or None)


def command_install_mcp(args: argparse.Namespace, config) -> int:
    config_path = args.config.expanduser()
    command = str(Path(__file__).resolve().parents[2] / "ctx") if args.local else args.mcp_command
    if args.single_repo:
        entry: dict[str, Any] = {
            "command": command,
            "args": ["--repo", str(config.repo), "mcp", "--ensure-index"],
        }
    else:
        entry = {
            "command": command,
            "args": ["mcp", "--multi"],
        }
    if args.workspace_env:
        entry["env"] = {"CTX_REPO": "${workspaceFolder}"}
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        data = {}
        config_path.parent.mkdir(parents=True, exist_ok=True)
    servers = data.setdefault("mcpServers", {})
    servers[args.name] = entry
    config_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return emit(args, {"config": str(config_path), "server": args.name, "entry": entry})


def command_watch(args: argparse.Namespace, config) -> int:
    interval = int(config.update.get("interval_seconds", 300)) if isinstance(config.update, dict) else 300
    while True:
        stale, reason = should_reindex(config)
        if stale:
            counts = index_repo(config)
            print(json.dumps({"updated": True, "reason": reason, **counts}, indent=2))
        else:
            print(json.dumps({"updated": False, "reason": reason}, indent=2))
        if args.once:
            return 0
        time.sleep(interval)


def command_explain(args: argparse.Namespace, config) -> int:
    def query(store: GraphStore):
        hits = store.search(args.topic, args.limit)
        impact = store.impact(args.topic, args.limit)
        files = [item for item in hits if item["kind"] == "file"][:5]
        symbols = [item for item in hits if item["kind"] in {"symbol", "function", "method", "class", "component"}][:8]
        return {
            "topic": args.topic,
            "brief": {
                "primary_files": compact_nodes(files),
                "primary_symbols": compact_nodes(symbols),
                "dependents": compact_nodes(impact.get("dependents", [])[:8]),
                "dependencies": compact_nodes(impact.get("dependencies", [])[:8]),
            },
        }

    return with_store(args, config, query)


def with_store(args: argparse.Namespace, config, callback) -> int:
    if not config.db_path.exists():
        print(f"graph does not exist at {config.db_path}. Run `ctx index` first.", file=sys.stderr)
        return 2
    store = GraphStore(config.db_path)
    try:
        payload = callback(store)
    finally:
        store.close()
    return emit(args, payload)


def emit(args: argparse.Namespace, payload: Any) -> int:
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(humanize(payload))
    return 0


def humanize(payload: Any) -> str:
    if isinstance(payload, list):
        if not payload:
            return "No results."
        return "\n".join(format_node(item) if isinstance(item, dict) and "kind" in item else json.dumps(item, indent=2) for item in payload)
    if isinstance(payload, dict):
        if payload.get("warnings"):
            body = dict(payload)
            warnings = body.pop("warnings")
            return "\n".join([*(f"Warning: {warning}" for warning in warnings), json.dumps(body, indent=2)])
        if "target" in payload and "dependents" in payload:
            lines = [f"Target: {format_node(payload['target'])}"]
            lines.append("\nDependents:")
            lines.extend(format_related(item) for item in payload.get("dependents", [])[:20] or [{"name": "none", "kind": "info"}])
            lines.append("\nDependencies:")
            lines.extend(format_related(item) for item in payload.get("dependencies", [])[:20] or [{"name": "none", "kind": "info"}])
            return "\n".join(lines)
        return json.dumps(payload, indent=2)
    return str(payload)


def compact_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "kind": node.get("kind"),
            "name": node.get("name"),
            "path": node.get("path"),
            "line": node.get("line"),
            "edge": node.get("edge_kind"),
        }
        for node in nodes
    ]


def format_node(node: dict[str, Any]) -> str:
    location = f" {node['path']}" if node.get("path") else ""
    if node.get("line"):
        location += f":{node['line']}"
    return f"{node.get('kind')} {node.get('name')}{location}"


def format_related(node: dict[str, Any]) -> str:
    if node.get("kind") == "info":
        return "- none"
    location = f" {node['path']}" if node.get("path") else ""
    if node.get("line"):
        location += f":{node['line']}"
    edge = f" via {node.get('edge_kind')}" if node.get("edge_kind") else ""
    return f"- {node.get('kind')} {node.get('name')}{location}{edge}"


if __name__ == "__main__":
    raise SystemExit(main())
