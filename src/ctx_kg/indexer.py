from __future__ import annotations

import ast
import fnmatch
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import CtxConfig
from .store import GraphStore


IMPORT_RE = re.compile(r"^\s*import\s+(?:type\s+)?(?:[\w*{},\s]+from\s+)?['\"]([^'\"]+)['\"]", re.M)
REQUIRE_RE = re.compile(r"require\(['\"]([^'\"]+)['\"]\)")
EXPORT_FROM_RE = re.compile(r"export\s+.*?\s+from\s+['\"]([^'\"]+)['\"]")
FUNC_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", re.M)
ARROW_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>", re.M)
CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", re.M)
ROUTE_RE = re.compile(r"\b(?:app|router)\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]")
CALL_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")


@dataclass
class IndexStats:
    files: int = 0
    symbols: int = 0
    edges: int = 0


def index_repo(repo_or_config, config: CtxConfig | None = None, store: GraphStore | None = None, reset: bool = True):
    if isinstance(repo_or_config, CtxConfig):
        cfg = repo_or_config
        repo = cfg.repo.resolve()
        owned_store = store or GraphStore(cfg.db_path)
        stats = _index(repo, cfg, owned_store, reset)
        counts = owned_store.stats()
        owned_store.set_meta("last_index", {"files": stats.files, "symbols": stats.symbols, "edges": counts.get("edges", stats.edges)})
        owned_store.commit()
        if store is None:
            owned_store.close()
        return {"files": stats.files, "symbols": stats.symbols, "edges": stats.edges, **counts}
    repo = Path(repo_or_config).resolve()
    cfg = config or CtxConfig(repo=repo)
    owned_store = store or GraphStore(cfg.db_path)
    stats = _index(repo, cfg, owned_store, reset)
    if store is None:
        owned_store.close()
    return stats


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
    known_files = {path.relative_to(repo).as_posix() for path in files}
    if reset:
        store.reset(repo)
    commit = current_commit(repo)
    if commit:
        store.set_meta("git_commit", commit)

    for path in files:
        rel = path.relative_to(repo).as_posix()
        text = read_text(path)
        kind = "test" if is_test_file(rel) else "file"
        store.add_node(file_id(rel), kind, Path(rel).name, rel, meta={"sha1": hash_text(text), "size": len(text)})
        stats.files += 1

    symbols_by_name: dict[str, list[str]] = {}
    file_symbols: dict[str, list[str]] = {}
    parsed: dict[str, list[str]] = {}

    for path in files:
        rel = path.relative_to(repo).as_posix()
        text = read_text(path)
        symbols, imports, calls, routes = extract(rel, text)
        parsed[rel] = calls
        for name, line, subkind in symbols:
            sid = symbol_id(rel, name)
            store.add_node(sid, "symbol", name, rel, line, {"symbol_kind": subkind})
            store.add_edge(file_id(rel), sid, "defines")
            symbols_by_name.setdefault(name, []).append(sid)
            file_symbols.setdefault(rel, []).append(sid)
            stats.symbols += 1
        for import_path in imports:
            dst = resolve_import(import_path, rel, known_files)
            if dst:
                if dst.startswith("package:"):
                    store.add_node(dst, "package", dst.removeprefix("package:"), meta={})
                store.add_edge(file_id(rel), dst, "imports", {"import": import_path})
                stats.edges += 1
        for method, route_path in routes:
            rid = route_id(method, route_path)
            store.add_node(rid, "route", f"{method.upper()} {route_path}", rel, meta={"method": method.upper(), "route": route_path})
            store.add_edge(rid, file_id(rel), "handled_by")
            stats.edges += 1

    for rel, calls in parsed.items():
        src_candidates = file_symbols.get(rel) or [file_id(rel)]
        for call in calls:
            for dst in symbols_by_name.get(call, [])[:10]:
                src = src_candidates[0]
                if src != dst:
                    store.add_edge(src, dst, "calls", {"name": call})
                    stats.edges += 1

    add_feature_nodes(config, store, known_files)
    link_tests(store, known_files)
    store.commit()
    return stats


def file_id(rel: str) -> str:
    return f"file:{rel}"


def symbol_id(rel: str, name: str) -> str:
    return f"symbol:{rel}:{name}"


def route_id(method: str, path: str) -> str:
    return f"route:{method.upper()} {path}"


def iter_files(repo: Path, config: CtxConfig) -> list[Path]:
    files = []
    for path in repo.rglob("*"):
        if path.is_file() and not should_ignore(path, repo, config) and path.suffix in config.include_extensions:
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
        self.calls: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend(alias.name for alias in node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append("." * node.level + node.module)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.symbols.append((node.name, node.lineno, "function"))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.symbols.append((node.name, node.lineno, "function"))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append((node.name, node.lineno, "class"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        self.generic_visit(node)


def extract(rel: str, text: str) -> tuple[list[tuple[str, int, str]], list[str], list[str], list[tuple[str, str]]]:
    if rel.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [], [], [], []
        visitor = PythonVisitor()
        visitor.visit(tree)
        return visitor.symbols, visitor.imports, visitor.calls, []
    imports = IMPORT_RE.findall(text) + REQUIRE_RE.findall(text) + EXPORT_FROM_RE.findall(text)
    symbols = []
    for regex, subkind in [(FUNC_RE, "function"), (ARROW_RE, "function"), (CLASS_RE, "class")]:
        for match in regex.finditer(text):
            symbols.append((match.group(1), text.count("\n", 0, match.start()) + 1, subkind))
    calls = [name for name in CALL_RE.findall(text) if name not in {"if", "for", "while", "switch", "function"}]
    return symbols, imports, calls, ROUTE_RE.findall(text)


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


def current_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def add_feature_nodes(config: CtxConfig, store: GraphStore, known_files: set[str]) -> None:
    for feature, patterns in config.features.items():
        fid = f"feature:{feature}"
        store.add_node(fid, "feature", feature, meta={"patterns": patterns})
        for rel in known_files:
            if any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
                store.add_edge(fid, file_id(rel), "contains")


def link_tests(store: GraphStore, known_files: set[str]) -> None:
    for rel in known_files:
        if not is_test_file(rel):
            continue
        stem = Path(rel).name
        base = stem.replace(".test", "").replace(".spec", "").replace("_test", "").removeprefix("test_")
        for target in known_files:
            if target != rel and Path(target).stem == Path(base).stem:
                store.add_edge(file_id(rel), file_id(target), "tests")
