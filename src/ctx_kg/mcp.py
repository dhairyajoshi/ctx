from __future__ import annotations

import json
import sys
from typing import Any, Callable

from .config import CtxConfig
from .embeddings import provider_from_env
from .store import GraphStore


PROTOCOL_VERSION = "2024-11-05"


def run_mcp_server(config: CtxConfig) -> None:
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


class McpServer:
    def __init__(self, config: CtxConfig):
        self.config = config
        self.tools: dict[str, Callable[[dict[str, Any]], Any]] = {
            "ctx_search": self.tool_search,
            "ctx_semantic": self.tool_semantic,
            "ctx_symbol": self.tool_symbol,
            "ctx_impact": self.tool_impact,
            "ctx_tests": self.tool_tests,
            "ctx_explain": self.tool_explain,
            "ctx_status": self.tool_status,
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

    def with_store(self, callback):
        if not self.config.db_path.exists():
            return {"error": f"graph does not exist at {self.config.db_path}", "hint": "run ctx index first"}
        store = GraphStore(self.config.db_path)
        try:
            return callback(store)
        finally:
            store.close()

    def tool_search(self, args: dict[str, Any]):
        return self.with_store(lambda store: store.search(str(args.get("query", "")), int(args.get("limit", 20))))

    def tool_semantic(self, args: dict[str, Any]):
        query = str(args.get("query", ""))
        limit = int(args.get("limit", 20))
        provider_name = args.get("provider")
        model = args.get("model")

        def search(store: GraphStore):
            provider = provider_from_env(str(provider_name) if provider_name else None, str(model) if model else None)
            if store.embedding_count(provider.provider, provider.model):
                try:
                    query_vector = provider.embed([query], input_type="query")[0]
                    return store.vector_search(query_vector, provider.provider, provider.model, limit)
                except Exception as exc:
                    fallback = store.semantic_search(query, limit)
                    for item in fallback:
                        item["score_source"] = "term_fallback"
                        item["embedding_error"] = str(exc)
                    return fallback
            results = store.semantic_search(query, limit)
            for item in results:
                item["score_source"] = "term"
            return results

        return self.with_store(search)

    def tool_symbol(self, args: dict[str, Any]):
        return self.with_store(lambda store: store.symbols(str(args.get("name", "")), int(args.get("limit", 20))))

    def tool_impact(self, args: dict[str, Any]):
        return self.with_store(lambda store: store.impact(str(args.get("target", "")), int(args.get("limit", 50))))

    def tool_tests(self, args: dict[str, Any]):
        return self.with_store(lambda store: store.tests_for_path(str(args.get("path", "")), int(args.get("limit", 50))))

    def tool_explain(self, args: dict[str, Any]):
        topic = str(args.get("topic", ""))
        limit = int(args.get("limit", 12))

        def query(store: GraphStore):
            hits = store.search(topic, limit)
            impact = store.impact(topic, limit)
            return {
                "topic": topic,
                "matches": hits,
                "dependents": impact.get("dependents", [])[:limit],
                "dependencies": impact.get("dependencies", [])[:limit],
            }

        return self.with_store(query)

    def tool_status(self, args: dict[str, Any]):
        def query(store: GraphStore):
            return {
                "repo": str(self.config.repo),
                "db": str(self.config.db_path),
                "counts": store.counts(),
                "last_index": store.get_meta("last_index", {}),
            }

        return self.with_store(query)


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "ctx_search",
            "description": "Search the repo knowledge graph by text, symbol, path, or metadata.",
            "inputSchema": object_schema({"query": "string", "limit": "number"}, ["query"]),
        },
        {
            "name": "ctx_semantic",
            "description": "Rank files, symbols, routes, and tests by embedding vectors when available, falling back to local term-vector similarity.",
            "inputSchema": object_schema({"query": "string", "limit": "number", "provider": "string", "model": "string"}, ["query"]),
        },
        {
            "name": "ctx_symbol",
            "description": "Find functions, methods, classes, or components by symbol name.",
            "inputSchema": object_schema({"name": "string", "limit": "number"}, ["name"]),
        },
        {
            "name": "ctx_impact",
            "description": "Return compact dependents and dependencies for a file path or symbol.",
            "inputSchema": object_schema({"target": "string", "limit": "number"}, ["target"]),
        },
        {
            "name": "ctx_tests",
            "description": "Suggest tests related to a file path.",
            "inputSchema": object_schema({"path": "string", "limit": "number"}, ["path"]),
        },
        {
            "name": "ctx_explain",
            "description": "Return a compact graph brief for a topic.",
            "inputSchema": object_schema({"topic": "string", "limit": "number"}, ["topic"]),
        },
        {
            "name": "ctx_status",
            "description": "Return graph database status and index metadata.",
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
