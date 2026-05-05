from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import CtxConfig, find_repo_for_cwd, load_config, load_registry
from .embeddings import provider_from_env
from .store import GraphStore


PROTOCOL_VERSION = "2024-11-05"


def run_mcp_server(config: CtxConfig | None = None) -> None:
    server = McpServer(config)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = server.handle(request)
        except Exception as exc:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(exc)}}
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


class RepoRouter:
    def __init__(self, default_config: CtxConfig | None = None):
        self.pinned = default_config
        self.spawn_cwd = Path.cwd()

    def resolve(self, repo_arg: str | None) -> tuple[CtxConfig | None, str | None]:
        if repo_arg:
            config = self._config_from_identifier(str(repo_arg))
            if config is None:
                return None, f"unknown repo '{repo_arg}'. Call ctx_repos to list available repos, or pass an absolute path."
            return config, None
        if self.pinned is not None:
            return self.pinned, None
        env_repo = os.environ.get("CTX_REPO")
        if env_repo:
            config = self._config_from_identifier(env_repo)
            if config is not None:
                return config, None
        cwd_entry = find_repo_for_cwd(self.spawn_cwd)
        if cwd_entry:
            return load_config(cwd_entry.path), None
        registry = load_registry()
        if registry.default and registry.default in registry.entries:
            return load_config(registry.entries[registry.default].path), None
        return None, "no repo specified and none could be auto-detected. Call ctx_repos, set CTX_REPO, or pass `repo` in the tool call."

    def _config_from_identifier(self, identifier: str) -> CtxConfig | None:
        registry = load_registry()
        entry = registry.resolve(identifier)
        if entry is not None:
            return load_config(entry.path)
        candidate = Path(identifier).expanduser()
        if candidate.is_dir():
            return load_config(candidate)
        return None


class McpServer:
    def __init__(self, config: CtxConfig | None = None):
        self.config = config
        self.router = RepoRouter(config)
        self.tools: dict[str, Callable[[dict[str, Any]], Any]] = {
            "ctx_search": self.tool_search,
            "ctx_semantic": self.tool_semantic,
            "ctx_symbol": self.tool_symbol,
            "ctx_impact": self.tool_impact,
            "ctx_callers": self.tool_callers,
            "ctx_callees": self.tool_callees,
            "ctx_tests": self.tool_tests,
            "ctx_explain": self.tool_explain,
            "ctx_status": self.tool_status,
            "ctx_repos": self.tool_repos,
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        if method == "initialize":
            return self.result(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ctx-kg", "version": "0.1.0"},
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return self.result(request_id, {"tools": tool_definitions()})
        if method == "tools/call":
            params = request.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            if name not in self.tools:
                return self.error(request_id, -32602, f"unknown tool: {name}")
            payload = self.tools[name](args)
            return self.result(request_id, {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]})
        return self.error(request_id, -32601, f"unknown method: {method}")

    def result(self, request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def error(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def with_store(self, args: dict[str, Any], callback):
        config, error = self.router.resolve(args.get("repo") if isinstance(args, dict) else None)
        if error:
            return {"error": error}
        if not config.db_path.exists():
            return {
                "error": f"graph does not exist at {config.db_path}",
                "repo": str(config.repo),
                "hint": f"run: ctx --repo {config.repo} index",
            }
        store = GraphStore(config.db_path)
        try:
            payload = callback(store, config)
        finally:
            store.close()
        if isinstance(payload, dict) and "repo" not in payload:
            payload["repo"] = str(config.repo)
        return payload

    def tool_search(self, args: dict[str, Any]):
        query = str(args.get("query", ""))
        limit = int(args.get("limit", 20))
        fallback = bool(args.get("fallback", True))

        def search(store: GraphStore, config: CtxConfig):
            hits = store.search(query, limit)
            if hits or not fallback:
                return {"query": query, "source": "lexical", "results": hits}
            semantic = self._semantic_query(store, query, limit)
            return {"query": query, "source": "semantic_fallback", "results": semantic}

        return self.with_store(args, search)

    def tool_semantic(self, args: dict[str, Any]):
        query = str(args.get("query", ""))
        limit = int(args.get("limit", 8))
        provider_name = args.get("provider")
        model = args.get("model")
        kind = args.get("kind")
        path_glob = args.get("path_glob")
        group_by_file = bool(args.get("group_by_file", True))
        return self.with_store(args, lambda store, config: self._semantic_query(store, query, limit, provider_name, model, kind, path_glob, group_by_file))

    def _semantic_query(
        self,
        store: GraphStore,
        query: str,
        limit: int,
        provider_name: str | None = None,
        model: str | None = None,
        kind: str | None = None,
        path_glob: str | None = None,
        group_by_file: bool = True,
    ):
        if provider_name or model:
            provider = provider_from_env(str(provider_name) if provider_name else None, str(model) if model else None)
        else:
            last = store.get_meta("last_embed", {}) or {}
            provider = provider_from_env(last.get("provider") or None, last.get("model") or None)
        if provider.provider != "local" and store.embedding_count(provider.provider, provider.model):
            try:
                query_vector = provider.embed([query], input_type="query")[0]
                vector_results = store.vector_search(query_vector, provider.provider, provider.model, max(limit * 4, 25))
                return store.shape_semantic_results(vector_results, limit, kind=kind, path_glob=path_glob, group_by_file=group_by_file)
            except Exception as exc:
                fallback = store.semantic_search(query, limit, kind=kind, path_glob=path_glob, group_by_file=group_by_file)
                for item in fallback:
                    item["score_source"] = "bm25_fallback"
                    item["embedding_error"] = str(exc)
                return fallback
        results = store.semantic_search(query, limit, kind=kind, path_glob=path_glob, group_by_file=group_by_file)
        for item in results:
            item.setdefault("score_source", "bm25")
        return results

    def tool_symbol(self, args: dict[str, Any]):
        return self.with_store(args, lambda store, config: store.symbols(str(args.get("name", "")), int(args.get("limit", 20))))

    def tool_impact(self, args: dict[str, Any]):
        return self.with_store(args, lambda store, config: store.impact(str(args.get("target", "")), int(args.get("limit", 50)), bool(args.get("include_vendor", False))))

    def tool_callers(self, args: dict[str, Any]):
        return self.with_store(args, lambda store, config: store.callers(str(args.get("target", "")), int(args.get("limit", 50)), bool(args.get("include_vendor", False))))

    def tool_callees(self, args: dict[str, Any]):
        return self.with_store(args, lambda store, config: store.callees(str(args.get("target", "")), int(args.get("limit", 50)), bool(args.get("include_vendor", False))))

    def tool_tests(self, args: dict[str, Any]):
        return self.with_store(args, lambda store, config: store.tests_for_path(str(args.get("path", "")), int(args.get("limit", 50))))

    def tool_explain(self, args: dict[str, Any]):
        topic = str(args.get("topic", ""))
        limit = int(args.get("limit", 12))

        def query(store: GraphStore, config: CtxConfig):
            lexical = store.search(topic, limit)
            semantic = self._semantic_query(store, topic, limit)
            seen: set[str] = set()
            primary: list[dict] = []
            for node in lexical + semantic:
                node_id = node.get("id")
                if not node_id or node_id in seen:
                    continue
                seen.add(node_id)
                primary.append(node)
                if len(primary) >= limit:
                    break
            files = [item for item in primary if item.get("kind") == "file"][:5]
            symbols = [item for item in primary if item.get("kind") == "symbol"][:8]
            routes = [item for item in primary if item.get("kind") == "route"][:5]
            best = _choose_anchor(topic, symbols, files, primary)
            dependents: list[dict] = []
            dependencies: list[dict] = []
            callers: list[dict] = []
            callees: list[dict] = []
            if best is not None:
                impact = store.impact(best["id"], limit)
                dependents = impact.get("dependents", [])
                dependencies = impact.get("dependencies", [])
                if best.get("kind") == "symbol":
                    callers = store.callers(best["id"], limit).get("callers", [])
                    callees = store.callees(best["id"], limit).get("callees", [])
            return {
                "topic": topic,
                "summary": _summarize(topic, primary, files, symbols, routes, best),
                "anchor": best,
                "files": _compact(files),
                "symbols": _compact(symbols),
                "routes": _compact(routes),
                "callers": _compact(callers),
                "callees": _compact(callees),
                "dependents": _compact(dependents),
                "dependencies": _compact(dependencies),
            }

        return self.with_store(args, query)

    def tool_status(self, args: dict[str, Any]):
        def query(store: GraphStore, config: CtxConfig):
            last_index = store.get_meta("last_index", {}) or {}
            last_embed = store.get_meta("last_embed", {}) or {}
            indexed_at_raw = store.get_meta("indexed_at")
            indexed_at = _coerce_int(indexed_at_raw)
            indexed_commit = store.get_meta("git_commit")
            current = _current_commit_safe(config.repo)
            counts = store.counts()
            fts = store.fts_status()
            return {
                "repo": str(config.repo),
                "db": str(config.db_path),
                "counts": counts,
                "fts": fts,
                "warnings": fts.get("warnings", []),
                "last_index": last_index,
                "last_embed": last_embed,
                "indexed_at": indexed_at,
                "indexed_at_iso": _iso(indexed_at),
                "age_seconds": int(time.time()) - indexed_at if indexed_at else None,
                "indexed_commit": indexed_commit,
                "current_commit": current,
                "stale": bool(current and indexed_commit and current != indexed_commit),
            }

        return self.with_store(args, query)

    def tool_repos(self, args: dict[str, Any]):
        registry = load_registry()
        config, _ = self.router.resolve(args.get("repo") if isinstance(args, dict) else None)
        repos = []
        for name, entry in registry.entries.items():
            repos.append({
                "name": name,
                "path": str(entry.path),
                "db": str(entry.db),
                "indexed_at": entry.indexed_at,
                "indexed": entry.db.exists(),
            })
        repos.sort(key=lambda item: item["name"])
        return {
            "repos": repos,
            "default": registry.default,
            "current": str(config.repo) if config else None,
            "pinned": str(self.config.repo) if self.config else None,
            "spawn_cwd": str(self.router.spawn_cwd),
        }


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iso(ts: int | None) -> str | None:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _current_commit_safe(repo: Path) -> str | None:
    try:
        from .indexer import current_commit

        return current_commit(repo)
    except Exception:
        return None


def _compact(nodes: list[dict]) -> list[dict]:
    out = []
    for node in nodes:
        out.append(
            {
                "kind": node.get("kind"),
                "name": node.get("name"),
                "path": node.get("path"),
                "line": node.get("line"),
                "id": node.get("id"),
                "edge": node.get("edge_kind"),
                "score": node.get("score"),
            }
        )
    return out


def _summarize(topic: str, primary: list[dict], files: list[dict], symbols: list[dict], routes: list[dict], anchor: dict | None = None) -> str:
    if not primary:
        return f"No graph nodes match '{topic}'. The repo may not be indexed or the topic may not appear in symbol/path/route names — try a more concrete identifier."
    bits = []
    if symbols:
        bits.append(f"{len(symbols)} symbol(s)")
    if files:
        bits.append(f"{len(files)} file(s)")
    if routes:
        bits.append(f"{len(routes)} route(s)")
    head = ", ".join(bits) or f"{len(primary)} node(s)"
    if anchor:
        location = f" ({anchor.get('path')}:{anchor.get('line')})" if anchor.get("path") and anchor.get("line") else ""
        return f"'{topic}' resolves to {head}; primary anchor is {anchor.get('kind')} {anchor.get('name')}{location}."
    return f"'{topic}' resolves to {head}; primary anchor is selected from the strongest symbol/path matches."


def _choose_anchor(topic: str, symbols: list[dict], files: list[dict], primary: list[dict]) -> dict | None:
    if symbols:
        return sorted(symbols, key=lambda item: _anchor_key(topic, item))[0]
    if files:
        return sorted(files, key=lambda item: _anchor_key(topic, item))[0]
    return primary[0] if primary else None


def _anchor_key(topic: str, node: dict) -> tuple:
    topic_terms = _name_terms(topic)
    node_terms = _name_terms(" ".join(str(node.get(key) or "") for key in ["name", "path", "id"]))
    overlap = len(topic_terms & node_terms)
    score = float(node.get("score") or 0)
    conversion_penalty = 1 if {"convert", "conversion"} & node_terms else 0
    entry_bonus = 1 if ENTRY_VERBS & node_terms else 0
    test_penalty = 1 if _is_test_node(node) else 0
    return (test_penalty, -overlap, conversion_penalty, -entry_bonus, -score, node.get("path") or "", node.get("line") or 0, node.get("name") or "")


def _is_test_node(node: dict) -> bool:
    if node.get("kind") == "test":
        return True
    path = (node.get("path") or "").lower()
    if not path:
        return False
    return any(token in path for token in ("/test/", "/tests/", "test_", "_test.", ".spec.", "/spec/", "/specs/"))


def _name_terms(value: str) -> set[str]:
    expanded = value.replace("_", " ").replace("-", " ").replace("/", " ")
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", expanded)
    terms = {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", expanded)}
    return terms - {"and", "for", "from", "the", "with", "file", "symbol"}


REPO_FIELD_DESC = "Optional repo nickname or absolute path. Defaults to CTX_REPO env, then the current workspace."

ENTRY_VERBS = {
    "handle", "initiate", "create", "process", "route", "dispatch",
    "terminate", "run", "execute", "start", "stop", "register",
    "listen", "accept", "serve", "consume", "publish", "send",
}


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "ctx_search",
            "description": "Lexical search over node names, paths, and ids. Falls through to semantic search if nothing matches.",
            "inputSchema": object_schema({"query": "string", "limit": "number", "fallback": "boolean", "repo": "string"}, ["query"]),
        },
        {
            "name": "ctx_semantic",
            "description": "Rank files, symbols, routes, and tests by hybrid BM25 + local hash search over indexed code context. path_glob uses Python fnmatch syntax.",
            "inputSchema": object_schema(
                {"query": "string", "limit": "number", "provider": "string", "model": "string", "kind": "string", "path_glob": "string", "group_by_file": "boolean", "repo": "string"},
                ["query"],
            ),
        },
        {
            "name": "ctx_symbol",
            "description": "Find functions, methods, classes, or components by symbol name.",
            "inputSchema": object_schema({"name": "string", "limit": "number", "repo": "string"}, ["name"]),
        },
        {
            "name": "ctx_impact",
            "description": "Return dependents (callers/users) and dependencies (callees/imports) for a path or symbol. Resolves bare symbol names, path:symbol, or full ids.",
            "inputSchema": object_schema({"target": "string", "limit": "number", "include_vendor": "boolean", "repo": "string"}, ["target"]),
        },
        {
            "name": "ctx_callers",
            "description": "Return one-hop callers of a symbol or file (incoming 'calls' edges).",
            "inputSchema": object_schema({"target": "string", "limit": "number", "include_vendor": "boolean", "repo": "string"}, ["target"]),
        },
        {
            "name": "ctx_callees",
            "description": "Return one-hop callees of a symbol or file (outgoing 'calls' edges).",
            "inputSchema": object_schema({"target": "string", "limit": "number", "include_vendor": "boolean", "repo": "string"}, ["target"]),
        },
        {
            "name": "ctx_tests",
            "description": "Suggest tests related to a file path.",
            "inputSchema": object_schema({"path": "string", "limit": "number", "repo": "string"}, ["path"]),
        },
        {
            "name": "ctx_explain",
            "description": "Compact graph brief for a topic. Combines lexical + semantic hits, anchors on the strongest match, and synthesizes one-hop callers/callees and dependencies.",
            "inputSchema": object_schema({"topic": "string", "limit": "number", "repo": "string"}, ["topic"]),
        },
        {
            "name": "ctx_status",
            "description": "Graph status with indexed_at, current vs indexed git commit, and a stale flag.",
            "inputSchema": object_schema({"repo": "string"}, []),
        },
        {
            "name": "ctx_repos",
            "description": "List indexed repos this server can query, plus the default and the currently-resolved repo.",
            "inputSchema": object_schema({}, []),
        },
    ]


def object_schema(properties: dict[str, str], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {key: {"type": value} for key, value in properties.items()},
        "required": required,
        "additionalProperties": False,
    }
