from __future__ import annotations

import json
import hashlib
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    import sqlite_vec
except Exception:  # pragma: no cover - optional in source checkout without uv sync
    sqlite_vec = None


SCHEMA = """
pragma journal_mode = wal;
create table if not exists metadata (key text primary key, value text not null);
create table if not exists nodes (
  id text primary key,
  kind text not null,
  name text not null,
  path text,
  line integer,
  meta text not null default '{}'
);
create table if not exists edges (
  src text not null,
  dst text not null,
  kind text not null,
  meta text not null default '{}',
  primary key (src, dst, kind)
);
create index if not exists idx_nodes_kind on nodes(kind);
create index if not exists idx_nodes_name on nodes(name);
create index if not exists idx_nodes_path on nodes(path);
create index if not exists idx_edges_src on edges(src);
create index if not exists idx_edges_dst on edges(dst);
create table if not exists embedding_rowids (
  rowid integer primary key autoincrement,
  node_id text not null,
  provider text not null,
  model text not null,
  dimensions integer not null,
  unique(node_id, provider, model)
);
create table if not exists embeddings (
  node_id text not null,
  provider text not null,
  model text not null,
  dimensions integer not null,
  content_sha1 text not null,
  vector text not null,
  updated_at integer not null,
  primary key (node_id, provider, model)
);
create index if not exists idx_embeddings_provider_model on embeddings(provider, model);
"""


class GraphStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute("pragma synchronous = normal")
        self.conn.execute("pragma temp_store = memory")
        self.conn.execute("pragma cache_size = -65536")
        self.vector_backend = self._load_vector_backend()

    def close(self) -> None:
        self.conn.close()

    def reset(self, repo: Path) -> None:
        self.conn.execute("delete from edges")
        self.conn.execute("delete from nodes")
        self.conn.execute("delete from embeddings")
        self.conn.execute("delete from embedding_rowids")
        self.set_meta("repo_root", str(repo.resolve()))
        self.set_meta("indexed_at", str(int(time.time())))
        self.conn.commit()

    def _load_vector_backend(self) -> str:
        if sqlite_vec is None:
            return "json"
        try:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            return "sqlite-vec"
        except Exception:
            try:
                self.conn.enable_load_extension(False)
            except Exception:
                pass
            return "json"

    def set_meta(self, key: str, value) -> None:
        if not isinstance(value, str):
            value = json.dumps(value, sort_keys=True)
        self.conn.execute(
            "insert into metadata(key, value) values(?, ?) on conflict(key) do update set value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str, default=None):
        row = self.conn.execute("select value from metadata where key = ?", (key,)).fetchone()
        if not row:
            return default
        value = row["value"]
        try:
            return json.loads(value)
        except Exception:
            return value

    def add_node(self, node_id: str, kind: str, name: str, path: str | None = None, line: int | None = None, meta: dict | None = None) -> None:
        self.conn.execute(
            """
            insert into nodes(id, kind, name, path, line, meta) values(?, ?, ?, ?, ?, ?)
            on conflict(id) do update set kind=excluded.kind, name=excluded.name, path=excluded.path, line=excluded.line, meta=excluded.meta
            """,
            (node_id, kind, name, path, line, json.dumps(meta or {}, sort_keys=True)),
        )

    def add_edge(self, src: str, dst: str, kind: str, meta: dict | None = None) -> None:
        self.conn.execute("insert or ignore into edges(src, dst, kind, meta) values(?, ?, ?, ?)", (src, dst, kind, json.dumps(meta or {}, sort_keys=True)))

    def bulk_add_nodes(self, rows: Iterable[tuple]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.conn.executemany(
            """
            insert into nodes(id, kind, name, path, line, meta) values(?, ?, ?, ?, ?, ?)
            on conflict(id) do update set kind=excluded.kind, name=excluded.name, path=excluded.path, line=excluded.line, meta=excluded.meta
            """,
            rows,
        )

    def bulk_add_edges(self, rows: Iterable[tuple]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.conn.executemany(
            "insert or ignore into edges(src, dst, kind, meta) values(?, ?, ?, ?)",
            rows,
        )

    def commit(self) -> None:
        self.conn.commit()

    def search(self, query: str, limit: int = 20) -> list[dict]:
        needle = f"%{query}%"
        rows = self.conn.execute(
            """
            select * from nodes
            where name like ? or path like ? or id like ?
            order by case kind when 'symbol' then 0 when 'route' then 1 when 'file' then 2 else 3 end, name
            limit ?
            """,
            (needle, needle, needle, limit),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    def semantic_search(self, query: str, limit: int = 20) -> list[dict]:
        query_vector = term_vector(query)
        if not query_vector:
            return []
        rows = self.conn.execute("select * from nodes").fetchall()
        scored = []
        for row in rows:
            node = row_to_dict(row, keep_terms=True)
            text = semantic_text(node)
            vector = term_vector(text)
            score = cosine(query_vector, vector)
            if score > 0:
                node = row_to_dict(row)
                node["score"] = round(score, 4)
                scored.append(node)
        scored.sort(key=lambda item: (-item["score"], kind_rank(item["kind"]), item.get("path") or "", item["name"]))
        return scored[:limit]

    def semantic_documents(self) -> list[dict]:
        rows = self.conn.execute("select * from nodes order by kind, path, name").fetchall()
        documents = []
        for row in rows:
            node = row_to_dict(row, keep_terms=True)
            text = semantic_text(node)
            if text.strip():
                documents.append({"node": row_to_dict(row), "text": text, "sha1": content_sha1(text)})
        return documents

    def embedding_count(self, provider: str | None = None, model: str | None = None) -> int:
        if provider and model:
            return self.conn.execute("select count(*) from embeddings where provider = ? and model = ?", (provider, model)).fetchone()[0]
        return self.conn.execute("select count(*) from embeddings").fetchone()[0]

    def existing_embedding_hashes(self, provider: str, model: str) -> dict[str, str]:
        rows = self.conn.execute("select node_id, content_sha1 from embeddings where provider = ? and model = ?", (provider, model)).fetchall()
        return {row["node_id"]: row["content_sha1"] for row in rows}

    def upsert_embedding(self, node_id: str, provider: str, model: str, vector: list[float], content_sha: str) -> None:
        self.conn.execute(
            """
            insert into embeddings(node_id, provider, model, dimensions, content_sha1, vector, updated_at)
            values(?, ?, ?, ?, ?, ?, ?)
            on conflict(node_id, provider, model) do update set
              dimensions=excluded.dimensions,
              content_sha1=excluded.content_sha1,
              vector=excluded.vector,
              updated_at=excluded.updated_at
            """,
            (node_id, provider, model, len(vector), content_sha, json.dumps(vector), int(time.time())),
        )
        self._upsert_vector_embedding(node_id, provider, model, vector)

    def vector_search(self, query_vector: list[float], provider: str, model: str, limit: int = 20) -> list[dict]:
        if self.vector_backend == "sqlite-vec":
            results = self._sqlite_vec_search(query_vector, provider, model, limit)
            if results:
                return results
        rows = self.conn.execute(
            """
            select e.vector, n.*
            from embeddings e join nodes n on n.id = e.node_id
            where e.provider = ? and e.model = ?
            """,
            (provider, model),
        ).fetchall()
        scored = []
        for row in rows:
            vector = json.loads(row["vector"])
            score = dense_cosine(query_vector, vector)
            if score > 0:
                node = row_to_dict(row)
                node["score"] = round(score, 4)
                node["score_source"] = "json-vector"
                scored.append(node)
        scored.sort(key=lambda item: (-item["score"], kind_rank(item["kind"]), item.get("path") or "", item["name"]))
        return scored[:limit]

    def _upsert_vector_embedding(self, node_id: str, provider: str, model: str, vector: list[float]) -> None:
        if self.vector_backend != "sqlite-vec":
            return
        table = vector_table_name(provider, model, len(vector))
        self.conn.execute(f"create virtual table if not exists {table} using vec0(embedding float[{len(vector)}])")
        self.conn.execute(
            """
            insert into embedding_rowids(node_id, provider, model, dimensions)
            values(?, ?, ?, ?)
            on conflict(node_id, provider, model) do update set dimensions=excluded.dimensions
            """,
            (node_id, provider, model, len(vector)),
        )
        rowid = self.conn.execute(
            "select rowid from embedding_rowids where node_id = ? and provider = ? and model = ?",
            (node_id, provider, model),
        ).fetchone()["rowid"]
        self.conn.execute(f"delete from {table} where rowid = ?", (rowid,))
        self.conn.execute(f"insert into {table}(rowid, embedding) values(?, ?)", (rowid, serialize_float32(vector)))

    def _sqlite_vec_search(self, query_vector: list[float], provider: str, model: str, limit: int) -> list[dict]:
        table = vector_table_name(provider, model, len(query_vector))
        exists = self.conn.execute("select name from sqlite_master where type = 'table' and name = ?", (table,)).fetchone()
        if not exists:
            return []
        rows = self.conn.execute(
            f"""
            select r.node_id, v.distance
            from {table} v
            join embedding_rowids r on r.rowid = v.rowid
            where v.embedding match ? and k = ?
            order by v.distance
            """,
            (serialize_float32(query_vector), limit),
        ).fetchall()
        if not rows:
            return []
        distances = {row["node_id"]: float(row["distance"]) for row in rows}
        placeholders = ",".join("?" for _ in distances)
        nodes = self.conn.execute(f"select * from nodes where id in ({placeholders})", tuple(distances)).fetchall()
        results = []
        for row in nodes:
            node = row_to_dict(row)
            distance = distances[node["id"]]
            node["distance"] = round(distance, 6)
            node["score"] = round(1.0 / (1.0 + distance), 4)
            node["score_source"] = "sqlite-vec"
            results.append(node)
        results.sort(key=lambda item: (item["distance"], kind_rank(item["kind"]), item.get("path") or "", item["name"]))
        return results[:limit]

    def resolve_targets(self, value: str, limit: int = 25) -> list[dict]:
        """Resolve a free-form target (node id, symbol name, file path, or fragment) to nodes."""
        if not value:
            return []
        rows = self.conn.execute(
            "select * from nodes where id = ? or path = ? or name = ?",
            (value, value, value),
        ).fetchall()
        if rows:
            return [row_to_dict(row) for row in rows]
        # Symbol-id with explicit prefix variants the caller might have dropped.
        for prefix in ("symbol:", "file:", "route:"):
            if not value.startswith(prefix):
                rows = self.conn.execute("select * from nodes where id = ?", (prefix + value,)).fetchall()
                if rows:
                    return [row_to_dict(row) for row in rows]
        # Path:symbol form, e.g. "workflow/service.py:handle_request".
        if ":" in value and not value.startswith(("symbol:", "file:", "route:", "package:", "feature:")):
            head, tail = value.rsplit(":", 1)
            rows = self.conn.execute(
                "select * from nodes where kind = 'symbol' and path = ? and name = ?",
                (head, tail),
            ).fetchall()
            if rows:
                return [row_to_dict(row) for row in rows]
        like = f"%{value}%"
        rows = self.conn.execute(
            """
            select * from nodes
            where name like ? or path like ? or id like ?
            order by case kind when 'symbol' then 0 when 'route' then 1 when 'file' then 2 else 3 end, name
            limit ?
            """,
            (like, like, like, limit),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    def nodes_by_path_or_name(self, value: str) -> list[dict]:
        return self.resolve_targets(value)

    def dependents(self, node_ids: Iterable[str], limit: int = 50) -> list[dict]:
        ids = list(node_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            select e.kind edge_kind, n.*
            from edges e join nodes n on n.id = e.src
            where e.dst in ({placeholders}) and e.kind != 'defines'
            order by n.kind, n.path, n.name
            limit ?
            """,
            (*ids, limit),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    def callers(self, target: str, limit: int = 50) -> dict:
        """Return symbols/files that call into the target via 'calls' edges."""
        nodes = self.resolve_targets(target)
        if not nodes:
            return {"target": None, "callers": []}
        match = _best_match(nodes)
        ids = [node["id"] for node in nodes if node["id"].startswith("symbol:") or node["id"] == match["id"]] or [match["id"]]
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            select e.kind edge_kind, e.dst dst_id, n.*
            from edges e join nodes n on n.id = e.src
            where e.dst in ({placeholders}) and e.kind = 'calls'
            order by n.kind, n.path, n.name
            limit ?
            """,
            (*ids, limit),
        ).fetchall()
        return {"target": match, "matches": nodes[:10], "callers": [row_to_dict(row) for row in rows]}

    def callees(self, target: str, limit: int = 50) -> dict:
        nodes = self.resolve_targets(target)
        if not nodes:
            return {"target": None, "callees": []}
        match = _best_match(nodes)
        ids = [node["id"] for node in nodes if node["id"].startswith("symbol:") or node["id"] == match["id"]] or [match["id"]]
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            select e.kind edge_kind, e.src src_id, n.*
            from edges e join nodes n on n.id = e.dst
            where e.src in ({placeholders}) and e.kind = 'calls'
            order by n.kind, n.path, n.name
            limit ?
            """,
            (*ids, limit),
        ).fetchall()
        return {"target": match, "matches": nodes[:10], "callees": [row_to_dict(row) for row in rows]}

    def related_tests(self, node_ids: Iterable[str], limit: int = 50) -> list[dict]:
        ids = list(node_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            select distinct n.*
            from edges e join nodes n on n.id = e.src
            where e.dst in ({placeholders}) and (e.kind = 'tests' or n.kind = 'test')
            order by n.path, n.name
            limit ?
            """,
            (*ids, limit),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute("select kind, count(*) count from nodes group by kind").fetchall()
        edge_count = self.conn.execute("select count(*) count from edges").fetchone()["count"]
        out = {row["kind"]: row["count"] for row in rows}
        out["edges"] = edge_count
        out["nodes"] = sum(value for key, value in out.items() if key != "edges")
        out["embeddings"] = self.embedding_count()
        out["vector_backend"] = self.vector_backend
        return out

    def counts(self) -> dict[str, int]:
        return self.stats()

    def symbols(self, name: str, limit: int = 20) -> list[dict]:
        needle = f"%{name}%"
        rows = self.conn.execute("select * from nodes where kind = 'symbol' and name like ? order by name limit ?", (needle, limit)).fetchall()
        return [row_to_dict(row) for row in rows]

    def tests_for_path(self, path: str, limit: int = 50) -> list[dict]:
        nodes = self.resolve_targets(path)
        return self.related_tests([node["id"] for node in nodes], limit)

    def impact(self, target: str, limit: int = 50) -> dict:
        nodes = self.resolve_targets(target)
        if not nodes:
            return {"target": None, "matches": [], "dependents": [], "dependencies": []}
        target_node = _best_match(nodes)
        ids = [node["id"] for node in nodes if node["id"] == target_node["id"] or node["kind"] == target_node["kind"]]
        if not ids:
            ids = [target_node["id"]]
        placeholders = ",".join("?" for _ in ids)
        deps = self.conn.execute(
            f"""
            select e.kind edge_kind, n.*
            from edges e join nodes n on n.id = e.dst
            where e.src in ({placeholders})
            order by n.kind, n.path, n.name
            limit ?
            """,
            (*ids, limit),
        ).fetchall()
        return {
            "target": target_node,
            "matches": nodes[:10],
            "dependents": self.dependents(ids, limit),
            "dependencies": [row_to_dict(row) for row in deps],
        }


def _best_match(nodes: list[dict]) -> dict:
    """Pick the most useful node from a candidate list (symbols beat files beat packages)."""
    return sorted(nodes, key=lambda node: (kind_rank(node.get("kind", "")), node.get("path") or "", node.get("name") or ""))[0]


def row_to_dict(row: sqlite3.Row | dict, keep_terms: bool = False) -> dict:
    data = dict(row)
    data.pop("vector", None)
    if "meta" in data and isinstance(data["meta"], str):
        try:
            data["meta"] = json.loads(data["meta"])
        except Exception:
            pass
    if not keep_terms and isinstance(data.get("meta"), dict) and "terms" in data["meta"]:
        meta = dict(data["meta"])
        meta.pop("terms", None)
        data["meta"] = meta
    return data


def semantic_text(node: dict) -> str:
    meta = node.get("meta") or {}
    parts = [node.get("kind", ""), node.get("name", ""), node.get("path") or ""]
    for key in ["symbol_kind", "route", "method", "import"]:
        value = meta.get(key)
        if value:
            parts.append(str(value))
    terms = meta.get("terms")
    if isinstance(terms, list):
        parts.extend(str(term) for term in terms)
    return " ".join(parts)


def term_vector(text: str) -> Counter[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text.replace("_", " ").replace("-", " ").replace("/", " "))
    terms = [term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", expanded)]
    return Counter(term for term in terms if term not in STOPWORDS)


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = set(left) & set(right)
    numerator = sum(left[term] * right[term] for term in overlap)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def dense_cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def content_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def vector_table_name(provider: str, model: str, dimensions: int) -> str:
    digest = hashlib.sha1(f"{provider}:{model}:{dimensions}".encode("utf-8")).hexdigest()[:16]
    return f"vec_embeddings_{digest}"


def serialize_float32(vector: list[float]):
    if sqlite_vec is not None and hasattr(sqlite_vec, "serialize_float32"):
        return sqlite_vec.serialize_float32(vector)
    return json.dumps(vector)


def kind_rank(kind: str) -> int:
    return {"symbol": 0, "route": 1, "file": 2, "test": 3, "package": 4}.get(kind, 9)


STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "import",
    "into",
    "not",
    "the",
    "this",
    "true",
    "with",
}
