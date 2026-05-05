from __future__ import annotations

import ast
import fnmatch
import hashlib
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import CtxConfig
from .embeddings import EmbeddingProvider, provider_from_env
from .store import GraphStore


IMPORT_RE = re.compile(r"^\s*import\s+(?:type\s+)?(?:[\w*{},\s]+from\s+)?['\"]([^'\"]+)['\"]", re.M)
REQUIRE_RE = re.compile(r"require\(['\"]([^'\"]+)['\"]\)")
EXPORT_FROM_RE = re.compile(r"export\s+.*?\s+from\s+['\"]([^'\"]+)['\"]")
FUNC_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", re.M)
ARROW_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>", re.M)
CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", re.M)
ROUTE_RE = re.compile(r"\b(?:app|router)\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]")
CALL_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")

PARALLEL_THRESHOLD = 200
MAX_FILE_BYTES = 2_000_000  # skip files larger than 2 MB to keep parsing predictable


@dataclass
class IndexStats:
    files: int = 0
    symbols: int = 0
    edges: int = 0


CallRecord = tuple[str | None, str, int]

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python
except Exception:  # pragma: no cover - optional fallback path
    Language = None
    Parser = None
    tree_sitter_python = None


_PYTHON_PARSER = None


def index_repo(
    repo_or_config,
    config: CtxConfig | None = None,
    store: GraphStore | None = None,
    reset: bool = True,
    embed: bool | None = None,
    embed_provider: str | None = None,
    embed_model: str | None = None,
    embed_dimensions: int | None = None,
    embed_batch_size: int | None = None,
    embed_force: bool = False,
):
    if isinstance(repo_or_config, CtxConfig):
        cfg = repo_or_config
        repo = cfg.repo.resolve()
        owned_store = store or GraphStore(cfg.db_path)
        stats = _index(repo, cfg, owned_store, reset)
        counts = owned_store.stats()
        owned_store.set_meta("last_index", {"files": stats.files, "symbols": stats.symbols, "edges": counts.get("edges", stats.edges)})
        owned_store.commit()
        embedding_summary = None
        if embed is None:
            embed = bool(cfg.embed.get("auto", True))
        if embed:
            embedding_summary = embed_index(
                owned_store,
                cfg,
                provider=embed_provider,
                model=embed_model,
                dimensions=embed_dimensions,
                batch_size=embed_batch_size,
                force=embed_force,
            )
            counts = owned_store.stats()
        if store is None:
            owned_store.close()
        result = {"files": stats.files, "symbols": stats.symbols, "edges": stats.edges, **counts}
        if embedding_summary is not None:
            result["embed"] = embedding_summary
        return result
    repo = Path(repo_or_config).resolve()
    cfg = config or CtxConfig(repo=repo)
    owned_store = store or GraphStore(cfg.db_path)
    stats = _index(repo, cfg, owned_store, reset)
    if store is None:
        owned_store.close()
    return stats


def embed_index(
    store: GraphStore,
    config: CtxConfig,
    provider: str | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    batch_size: int | None = None,
    force: bool = False,
) -> dict:
    embed_cfg = config.embed if isinstance(config.embed, dict) else {}
    selected_provider = provider or embed_cfg.get("provider")
    selected_model = model or embed_cfg.get("model")
    selected_dimensions = dimensions if dimensions is not None else embed_cfg.get("dimensions")
    selected_batch = batch_size or embed_cfg.get("batch_size") or 64
    fallback_used = False
    fallback_error: str | None = None
    try:
        active = provider_from_env(selected_provider, selected_model, selected_dimensions)
        if active.provider != "local":
            _probe_provider(active)
    except Exception as exc:
        fallback_used = True
        fallback_error = str(exc)
        active = provider_from_env("local")
    summary = _run_embedding(store, active, selected_batch, force)
    summary["fallback"] = fallback_used
    if fallback_error:
        summary["fallback_reason"] = fallback_error
    store.set_meta(
        "last_embed",
        {
            "provider": summary["provider"],
            "model": summary["model"],
            "documents": summary["documents"],
            "embedded": summary["embedded"],
            "fallback": fallback_used,
        },
    )
    store.commit()
    return summary


def _probe_provider(provider: EmbeddingProvider) -> None:
    provider.embed(["ctx-index probe"], input_type="document")


def _run_embedding(store: GraphStore, provider: EmbeddingProvider, batch_size: int, force: bool) -> dict:
    documents = store.semantic_documents()
    existing = store.existing_embedding_hashes(provider.provider, provider.model)
    pending = [doc for doc in documents if force or existing.get(doc["node"]["id"]) != doc["sha1"]]
    embedded = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
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


def should_reindex(config: CtxConfig) -> tuple[bool, str]:
    if not config.db_path.exists():
        return True, "missing graph"
    policy = config.update if isinstance(config.update, str) else config.update.get("mode", "manual")
    if policy == "manual":
        return False, "manual"
    if policy == "commit":
        store = GraphStore(config.db_path)
        try:
            old = store.get_meta("git_commit")
        finally:
            store.close()
        changed = old != current_commit(config.repo)
        return changed, "commit changed" if changed else "commit unchanged"
    if policy in {"interval", "watch"}:
        interval = 300
        if isinstance(config.update, dict):
            interval = int(config.update.get("interval_seconds", interval))
        store = GraphStore(config.db_path)
        try:
            indexed_at = store.get_meta("indexed_at", 0)
            if isinstance(indexed_at, str):
                indexed_at = float(indexed_at)
        finally:
            store.close()
        import time

        stale = time.time() - float(indexed_at or 0) >= interval
        return stale, "interval elapsed" if stale else "interval not elapsed"
    return False, f"unknown policy: {policy}"


def _index(repo: Path, config: CtxConfig, store: GraphStore, reset: bool = True) -> IndexStats:
    stats = IndexStats()
    files = iter_files(repo, config)
    rels = [path.relative_to(repo).as_posix() for path in files]
    known_files = set(rels)
    if reset:
        store.reset(repo)
    commit = current_commit(repo)
    if commit:
        store.set_meta("git_commit", commit)

    parsed = _extract_all(repo, rels)

    file_rows: list[tuple] = []
    symbol_rows: list[tuple] = []
    package_rows: list[tuple] = []
    route_rows: list[tuple] = []
    edge_rows: list[tuple] = []

    symbols_by_name: dict[str, list[str]] = {}
    file_symbols: dict[str, list[str]] = {}

    import json as _json
    for rel, info in parsed.items():
        if info is None:
            continue
        kind = "test" if is_test_file(rel) else "file"
        meta = {"sha1": info["sha"], "size": info["size"], "terms": info["terms"]}
        file_rows.append((file_id(rel), kind, Path(rel).name, rel, None, _json.dumps(meta, sort_keys=True)))
        stats.files += 1

        for name, line, subkind in info["symbols"]:
            sid = symbol_id(rel, name)
            symbol_rows.append((sid, "symbol", name, rel, line, _json.dumps({"symbol_kind": subkind}, sort_keys=True)))
            edge_rows.append((file_id(rel), sid, "defines", "{}"))
            symbols_by_name.setdefault(name, []).append(sid)
            file_symbols.setdefault(rel, []).append(sid)
            stats.symbols += 1

        for import_path in info["imports"]:
            dst = resolve_import(import_path, rel, known_files)
            if dst:
                if dst.startswith("package:"):
                    package_rows.append((dst, "package", dst.removeprefix("package:"), None, None, "{}"))
                edge_rows.append((file_id(rel), dst, "imports", _json.dumps({"import": import_path}, sort_keys=True)))
                stats.edges += 1

        for method, route_path in info["routes"]:
            rid = route_id(method, route_path)
            route_meta = {"method": method.upper(), "route": route_path}
            route_rows.append((rid, "route", f"{method.upper()} {route_path}", rel, None, _json.dumps(route_meta, sort_keys=True)))
            edge_rows.append((rid, file_id(rel), "handled_by", "{}"))
            stats.edges += 1

    for rel, info in parsed.items():
        if info is None:
            continue
        defined_in_file = {sid.rsplit(":", 1)[-1]: sid for sid in file_symbols.get(rel, [])}
        for scope, call, line in info["calls"]:
            src = defined_in_file.get(scope) if scope else None
            if src is None:
                src = file_id(rel)
            for dst in symbols_by_name.get(call, [])[:10]:
                if src != dst:
                    edge_rows.append((src, dst, "calls", _json.dumps({"name": call, "line": line}, sort_keys=True)))
                    stats.edges += 1

    store.bulk_add_nodes(file_rows + symbol_rows + package_rows + route_rows)
    store.bulk_add_edges(edge_rows)

    add_feature_nodes(config, store, known_files)
    link_tests(store, known_files)
    store.commit()
    return stats


def _extract_all(repo: Path, rels: list[str]) -> dict[str, dict | None]:
    if not rels:
        return {}
    if len(rels) < PARALLEL_THRESHOLD:
        return {rel: _extract_one((str(repo), rel)) for rel in rels}
    workers = min(os.cpu_count() or 2, 8)
    chunksize = max(1, len(rels) // (workers * 4))
    out: dict[str, dict | None] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rel, info in zip(rels, pool.map(_extract_one, [(str(repo), rel) for rel in rels], chunksize=chunksize)):
            out[rel] = info
    return out


def _extract_one(args: tuple[str, str]) -> dict | None:
    repo_str, rel = args
    path = Path(repo_str) / rel
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
    except Exception:
        return None
    symbols, imports, calls, routes = extract(rel, text)
    return {
        "sha": hash_text(text),
        "size": len(text),
        "terms": extract_terms(text),
        "symbols": symbols,
        "imports": imports,
        "calls": calls,
        "routes": routes,
    }


def file_id(rel: str) -> str:
    return f"file:{rel}"


def symbol_id(rel: str, name: str) -> str:
    return f"symbol:{rel}:{name}"


def route_id(method: str, path: str) -> str:
    return f"route:{method.upper()} {path}"


def iter_files(repo: Path, config: CtxConfig) -> list[Path]:
    files = []
    extensions = set(config.include_extensions)
    for path in repo.rglob("*"):
        if path.is_file() and not should_ignore(path, repo, config) and path.suffix in extensions:
            files.append(path)
    return files


def should_ignore(path: Path, repo: Path, config: CtxConfig) -> bool:
    rel = path.relative_to(repo).as_posix()
    parts = set(path.relative_to(repo).parts)
    return any(part in config.ignore for part in parts) or any(fnmatch.fnmatch(rel, pattern) for pattern in config.ignore)


def resolve_import(import_path: str, source_rel: str, known_files: set[str]) -> str | None:
    if not import_path.startswith("."):
        return f"package:{import_path.split('/')[0]}"
    base = Path(source_rel).parent
    candidate = (base / import_path).as_posix()
    variants = [candidate, *[candidate + ext for ext in [".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]]]
    variants.extend((Path(candidate) / f"index{ext}").as_posix() for ext in [".js", ".jsx", ".ts", ".tsx"])
    for variant in variants:
        if str(Path(variant)).replace("\\", "/") in known_files:
            return file_id(str(Path(variant)).replace("\\", "/"))
    return None


class PythonVisitor(ast.NodeVisitor):
    def __init__(self):
        self.symbols: list[tuple[str, int, str]] = []
        self.imports: list[str] = []
        self.calls: list[CallRecord] = []
        self.scope: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend(alias.name for alias in node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append("." * node.level + node.module)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.symbols.append((node.name, node.lineno, "function"))
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.symbols.append((node.name, node.lineno, "function"))
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append((node.name, node.lineno, "class"))
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name:
            self.calls.append((self.scope[-1] if self.scope else None, name, node.lineno))
        self.generic_visit(node)


def extract(rel: str, text: str) -> tuple[list[tuple[str, int, str]], list[str], list[CallRecord], list[tuple[str, str]]]:
    if rel.endswith(".py"):
        parsed = extract_python_tree_sitter(text)
        if parsed is not None:
            return parsed
        return extract_python_ast(text)
    imports = IMPORT_RE.findall(text) + REQUIRE_RE.findall(text) + EXPORT_FROM_RE.findall(text)
    symbols = []
    for regex, subkind in [(FUNC_RE, "function"), (ARROW_RE, "function"), (CLASS_RE, "class")]:
        for match in regex.finditer(text):
            symbols.append((match.group(1), text.count("\n", 0, match.start()) + 1, subkind))
    calls = scoped_text_calls(text, symbols)
    return symbols, imports, calls, ROUTE_RE.findall(text)


def extract_python_ast(text: str) -> tuple[list[tuple[str, int, str]], list[str], list[CallRecord], list[tuple[str, str]]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], [], [], []
    visitor = PythonVisitor()
    visitor.visit(tree)
    return visitor.symbols, visitor.imports, visitor.calls, []


def extract_python_tree_sitter(text: str) -> tuple[list[tuple[str, int, str]], list[str], list[CallRecord], list[tuple[str, str]]] | None:
    parser = python_parser()
    if parser is None:
        return None
    source = text.encode("utf-8", errors="ignore")
    try:
        tree = parser.parse(source)
    except Exception:
        return None
    symbols: list[tuple[str, int, str]] = []
    imports: list[str] = []
    calls: list[CallRecord] = []
    walk_python_tree(tree.root_node, source, [], symbols, imports, calls)
    return symbols, imports, calls, []


def python_parser():
    global _PYTHON_PARSER
    if _PYTHON_PARSER is not None:
        return _PYTHON_PARSER
    if Parser is None or Language is None or tree_sitter_python is None:
        return None
    try:
        parser = Parser()
        parser.language = Language(tree_sitter_python.language())
    except Exception:
        return None
    _PYTHON_PARSER = parser
    return parser


def walk_python_tree(
    node,
    source: bytes,
    scope: list[str],
    symbols: list[tuple[str, int, str]],
    imports: list[str],
    calls: list[CallRecord],
) -> None:
    node_type = node.type
    if node_type == "function_definition":
        name_node = node.child_by_field_name("name")
        name = node_text(name_node, source) if name_node else None
        if name:
            symbols.append((name, node.start_point.row + 1, "function"))
            scope.append(name)
            body = node.child_by_field_name("body")
            if body is not None:
                walk_python_tree(body, source, scope, symbols, imports, calls)
            scope.pop()
            return
    if node_type == "class_definition":
        name_node = node.child_by_field_name("name")
        name = node_text(name_node, source) if name_node else None
        if name:
            symbols.append((name, node.start_point.row + 1, "class"))
            scope.append(name)
            body = node.child_by_field_name("body")
            if body is not None:
                walk_python_tree(body, source, scope, symbols, imports, calls)
            scope.pop()
            return
    if node_type in {"import_statement", "import_from_statement"}:
        imports.extend(python_imports_from_node(node, source))
    elif node_type == "call":
        name = python_call_name(node, source)
        if name:
            calls.append((scope[-1] if scope else None, name, node.start_point.row + 1))
    for child in node.children:
        walk_python_tree(child, source, scope, symbols, imports, calls)


def python_imports_from_node(node, source: bytes) -> list[str]:
    if node.type == "import_statement":
        return [
            node_text(child, source)
            for child in node.children
            if child.type in {"dotted_name", "aliased_import"}
        ]
    module = node.child_by_field_name("module_name")
    module_name = node_text(module, source) if module else None
    return [module_name] if module_name else []


def python_call_name(node, source: bytes) -> str | None:
    function = node.child_by_field_name("function")
    if function is None:
        return None
    if function.type == "identifier":
        return node_text(function, source)
    if function.type == "attribute":
        attribute = function.child_by_field_name("attribute")
        return node_text(attribute, source) if attribute is not None else None
    return None


def node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def scoped_text_calls(text: str, symbols: list[tuple[str, int, str]]) -> list[CallRecord]:
    ignored = {"if", "for", "while", "switch", "function"}
    symbol_ranges = sorted((line, name) for name, line, _ in symbols)
    calls: list[CallRecord] = []
    current_index = 0
    for match in CALL_RE.finditer(text):
        name = match.group(1)
        if name in ignored:
            continue
        line = text.count("\n", 0, match.start()) + 1
        while current_index + 1 < len(symbol_ranges) and symbol_ranges[current_index + 1][0] <= line:
            current_index += 1
        scope = symbol_ranges[current_index][1] if symbol_ranges and symbol_ranges[current_index][0] <= line else None
        calls.append((scope, name, line))
    return calls


def is_test_file(rel: str) -> bool:
    lower = rel.lower()
    return any(part in lower for part in ["test.", "spec.", "_test.", "test_", "/tests/", "/test/"])


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


_TERM_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
_TERM_SPLIT_RE = re.compile(r"([a-z0-9])([A-Z])")
_TERM_STOPWORDS = frozenset({"and", "are", "for", "from", "import", "not", "return", "the", "this", "true", "with"})


def extract_terms(text: str, limit: int = 80) -> list[str]:
    expanded = _TERM_SPLIT_RE.sub(r"\1 \2", text.replace("_", " "))
    counts: dict[str, int] = {}
    for term in _TERM_TOKEN_RE.findall(expanded):
        lowered = term.lower()
        if lowered in _TERM_STOPWORDS:
            continue
        counts[lowered] = counts.get(lowered, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def current_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def add_feature_nodes(config: CtxConfig, store: GraphStore, known_files: set[str]) -> None:
    import json as _json
    feature_rows: list[tuple] = []
    edge_rows: list[tuple] = []
    for feature, patterns in config.features.items():
        fid = f"feature:{feature}"
        feature_rows.append((fid, "feature", feature, None, None, _json.dumps({"patterns": patterns}, sort_keys=True)))
        for rel in known_files:
            if any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
                edge_rows.append((fid, file_id(rel), "contains", "{}"))
    if feature_rows:
        store.bulk_add_nodes(feature_rows)
    if edge_rows:
        store.bulk_add_edges(edge_rows)


def link_tests(store: GraphStore, known_files: set[str]) -> None:
    edge_rows: list[tuple] = []
    by_stem: dict[str, list[str]] = {}
    for target in known_files:
        by_stem.setdefault(Path(target).stem, []).append(target)
    for rel in known_files:
        if not is_test_file(rel):
            continue
        stem = Path(rel).name
        base = stem.replace(".test", "").replace(".spec", "").replace("_test", "").removeprefix("test_")
        candidate_stem = Path(base).stem
        for target in by_stem.get(candidate_stem, []):
            if target != rel:
                edge_rows.append((file_id(rel), file_id(target), "tests", "{}"))
    if edge_rows:
        store.bulk_add_edges(edge_rows)
