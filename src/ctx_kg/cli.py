from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config, write_config
from .embeddings import provider_from_env
from .indexer import index_repo, should_reindex
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

    add_json_flag(sub.add_parser("status", help="Show graph status."))
    add_json_flag(sub.add_parser("update", help="Reindex only if update policy says the graph is stale."))

    watch_cmd = add_json_flag(sub.add_parser("watch", help="Poll and reindex according to update.interval_seconds."))
    watch_cmd.add_argument("--once", action="store_true", help="Run one poll cycle and exit.")

    search_cmd = add_json_flag(sub.add_parser("search", help="Search nodes by name, path, or metadata."))
    search_cmd.add_argument("term")
    search_cmd.add_argument("--limit", type=int, default=20)

    semantic_cmd = add_json_flag(sub.add_parser("semantic", help="Rank graph nodes by embeddings, falling back to term-vector similarity."))
    semantic_cmd.add_argument("query")
    semantic_cmd.add_argument("--limit", type=int, default=20)
    semantic_cmd.add_argument("--provider", help="Embedding provider to use when embedded vectors exist.")
    semantic_cmd.add_argument("--model", help="Embedding model to use when embedded vectors exist.")
    semantic_cmd.add_argument("--term-only", action="store_true", help="Skip embedding search and use local term vectors.")

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

    tests_cmd = add_json_flag(sub.add_parser("tests", help="Suggest tests related to a path."))
    tests_cmd.add_argument("path")
    tests_cmd.add_argument("--limit", type=int, default=50)

    explain_cmd = add_json_flag(sub.add_parser("explain", help="Compact repo brief for a topic."))
    explain_cmd.add_argument("topic")
    explain_cmd.add_argument("--limit", type=int, default=12)

    mcp_cmd = add_json_flag(sub.add_parser("mcp", help="Run an MCP stdio server exposing graph query tools."))
    mcp_cmd.add_argument("--ensure-index", action="store_true", help="Build the graph first if it does not exist.")

    install_mcp_cmd = add_json_flag(sub.add_parser("install-mcp", help="Add ctx to an MCP JSON config file."))
    install_mcp_cmd.add_argument("--config", type=Path, required=True, help="Path to an MCP config JSON file.")
    install_mcp_cmd.add_argument("--name", default="ctx", help="MCP server name to write.")
    install_mcp_cmd.add_argument("--command", dest="mcp_command", default="ctx", help="Executable command agents should run.")
    install_mcp_cmd.add_argument("--local", action="store_true", help="Use this checkout's ./ctx executable as the command.")

    args = parser.parse_args(argv)
    config = load_config(args.repo)

    if args.command == "init":
        return command_init(args, config.repo)
    if args.command == "index":
        counts = index_repo(config, reset=not args.no_reset)
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
        return with_store(args, config, lambda store: embed_graph(store, args))
    if args.command == "symbol":
        return with_store(args, config, lambda store: store.symbols(args.name, args.limit))
    if args.command == "impact":
        return with_store(args, config, lambda store: store.impact(args.target, args.limit))
    if args.command == "tests":
        return with_store(args, config, lambda store: store.tests_for_path(args.path, args.limit))
    if args.command == "explain":
        return command_explain(args, config)
    if args.command == "mcp":
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
    print(f"created {created}")
    return 0


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
        payload["embeddings"] = store.embedding_count()
        payload["last_index"] = store.get_meta("last_index", {})
        store.close()
    return emit(args, payload)


def embed_graph(store: GraphStore, args: argparse.Namespace) -> dict[str, Any]:
    provider = provider_from_env(args.provider, args.model, args.dimensions)
    documents = store.semantic_documents()
    existing = store.existing_embedding_hashes(provider.provider, provider.model)
    pending = [doc for doc in documents if args.force or existing.get(doc["node"]["id"]) != doc["sha1"]]
    embedded = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        vectors = provider.embed([doc["text"] for doc in batch], input_type="document")
        for doc, vector in zip(batch, vectors):
            store.upsert_embedding(doc["node"]["id"], provider.provider, provider.model, vector, doc["sha1"])
            embedded += 1
        store.commit()
    return {
        "provider": provider.provider,
        "model": provider.model,
        "documents": len(documents),
        "embedded": embedded,
        "skipped": len(documents) - embedded,
    }


def semantic_query(store: GraphStore, args: argparse.Namespace) -> list[dict]:
    if not args.term_only:
        provider = provider_from_env(args.provider, args.model)
        if store.embedding_count(provider.provider, provider.model):
            try:
                query_vector = provider.embed([args.query], input_type="query")[0]
                return store.vector_search(query_vector, provider.provider, provider.model, args.limit)
            except Exception as exc:
                fallback = store.semantic_search(args.query, args.limit)
                for item in fallback:
                    item["score_source"] = "term_fallback"
                    item["embedding_error"] = str(exc)
                return fallback
    results = store.semantic_search(args.query, args.limit)
    for item in results:
        item["score_source"] = "term"
    return results


def command_install_mcp(args: argparse.Namespace, config) -> int:
    config_path = args.config.expanduser()
    command = str(Path(__file__).resolve().parents[2] / "ctx") if args.local else args.mcp_command
    entry = {
        "command": command,
        "args": ["--repo", str(config.repo), "mcp", "--ensure-index"],
    }
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
