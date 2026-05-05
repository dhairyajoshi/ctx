from __future__ import annotations

import json
import hashlib
import fnmatch
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

try:
    import sqlite_vec
except Exception:  # pragma: no cover - optional in source checkout without uv sync
    sqlite_vec = None

from .embeddings import DEFAULT_LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_DIMENSIONS, embed_local


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
create table if not exists nodes_fts_rowids (
  rowid integer primary key autoincrement,
  node_id text not null unique
);
create virtual table if not exists nodes_fts using fts5(
  node_id unindexed,
  kind unindexed,
  name,
  signature,
  docstring,
  decorators,
  path,
  body,
  neighbors,
  morph,
  -- FTS5 parses this tokenizer string; the nested single quotes around _ are intentional.
  tokenize = "unicode61 tokenchars '_'"
);
"""


class GraphStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate_schema()
        self.conn.execute("pragma synchronous = normal")
        self.conn.execute("pragma temp_store = memory")
        self.conn.execute("pragma cache_size = -65536")
        self.vector_backend = self._load_vector_backend()

    def close(self) -> None:
        self.conn.close()

    def _migrate_schema(self) -> None:
        migrated = False
        rowid_columns = {row["name"]: dict(row) for row in self.conn.execute("pragma table_info(nodes_fts_rowids)").fetchall()}
        if rowid_columns and rowid_columns.get("rowid", {}).get("pk") != 1:
            self.conn.execute("drop table nodes_fts_rowids")
            self.conn.execute(
                """
                create table nodes_fts_rowids (
                  rowid integer primary key autoincrement,
                  node_id text not null unique
                )
                """
            )
            migrated = True
        fts_columns = {row["name"] for row in self.conn.execute("pragma table_info(nodes_fts)").fetchall()}
        if fts_columns and "morph" not in fts_columns:
            self.conn.execute("drop table nodes_fts")
            self.conn.execute("delete from nodes_fts_rowids")
            self.conn.execute(
                """
                create virtual table nodes_fts using fts5(
                  node_id unindexed,
                  kind unindexed,
                  name,
                  signature,
                  docstring,
                  decorators,
                  path,
                  body,
                  neighbors,
                  morph,
                  tokenize = "unicode61 tokenchars '_'"
                )
                """
            )
            migrated = True
        if migrated:
            self.set_meta("fts_rebuild_required", "1")

    def reset(self, repo: Path) -> None:
        self.conn.execute("delete from edges")
        self.conn.execute("delete from nodes")
        self.conn.execute("delete from nodes_fts")
        self.conn.execute("delete from nodes_fts_rowids")
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
        encoded_meta = json.dumps(meta or {}, sort_keys=True)
        self.conn.execute(
            """
            insert into nodes(id, kind, name, path, line, meta) values(?, ?, ?, ?, ?, ?)
            on conflict(id) do update set kind=excluded.kind, name=excluded.name, path=excluded.path, line=excluded.line, meta=excluded.meta
            """,
            (node_id, kind, name, path, line, encoded_meta),
        )
        self._upsert_fts_row(node_id, kind, name, path, encoded_meta)
        self._mark_nodes_modified()

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
        self._upsert_fts_rows(rows)
        self._mark_nodes_modified()

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

    def _mark_nodes_modified(self) -> None:
        self.set_meta("nodes_modified_at", str(int(time.time_ns())))

    def search(self, query: str, limit: int = 20) -> list[dict]:
        needle = f"%{query}%"
        rows = self.conn.execute(
            """
            select id, kind, name, path, line, meta from nodes
            where name like ? or path like ? or id like ?
            order by case kind when 'symbol' then 0 when 'route' then 1 when 'file' then 2 else 3 end, name
            limit ?
            """,
            (needle, needle, needle, limit),
        ).fetchall()
        seen: set[str] = set()
        results: list[dict] = []
        for row in rows:
            seen.add(row["id"])
            node = row_to_dict(row)
            node["match"] = "name"
            results.append(node)
        if len(results) >= limit:
            return results
        # Reference search: find nodes whose body/decorators/neighbors contain the term.
        # Surfaces non-symbol references (calls, imports, config wiring) that LIKE on
        # name/path/id misses.
        match_query = lexical_reference_query(query)
        if match_query:
            self._ensure_fts_populated()
            extra = self.conn.execute(
                """
                select n.id, n.kind, n.name, n.path, n.line, n.meta
                from nodes_fts
                join nodes n on n.id = nodes_fts.node_id
                where nodes_fts match ?
                order by case n.kind when 'symbol' then 0 when 'route' then 1 when 'file' then 2 else 3 end, n.path, n.name
                limit ?
                """,
                (match_query, (limit - len(results)) * 2),
            ).fetchall()
            for row in extra:
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                node = row_to_dict(row)
                node["match"] = "reference"
                results.append(node)
                if len(results) >= limit:
                    break
        return results

    def semantic_search(
        self,
        query: str,
        limit: int = 8,
        *,
        kind: str | None = None,
        path_glob: str | None = None,
        group_by_file: bool = True,
    ) -> list[dict]:
        match_query = fts_query(query)
        if not match_query:
            return []
        self._ensure_fts_populated()
        candidate_limit = max(limit * (20 if path_glob else 4), 500 if path_glob else 25)
        fts_results = self._fts_search(match_query, candidate_limit, kind, path_glob)
        hash_results = self._local_hash_search(query, candidate_limit, kind, path_glob)
        results = rrf_fuse(fts_results, hash_results, limit * 3)
        if group_by_file:
            results = group_same_file_symbols(results, limit)
        return results[:limit]

    def _fts_search(self, match_query: str, limit: int, kind: str | None, path_glob: str | None) -> list[dict]:
        filters, params = semantic_filters(kind, "n")
        rows = self.conn.execute(
            f"""
            select n.*, (-bm25(nodes_fts, 0.0, 0.0, 10.0, 5.0, 4.0, 5.0, 2.0, 1.0, 1.0, 1.0) * 1000000.0) score,
                   snippet(nodes_fts, 7, '', '', ' ... ', 18) body_snippet,
                   snippet(nodes_fts, 4, '', '', ' ... ', 18) docstring_snippet,
                   snippet(nodes_fts, 3, '', '', ' ... ', 18) signature_snippet
            from nodes_fts
            join nodes n on n.id = nodes_fts.node_id
            where nodes_fts match ?
              {filters}
            order by score desc, case n.kind when 'symbol' then 0 when 'route' then 1 when 'file' then 2 else 3 end, n.path, n.name
            limit ?
            """,
            (match_query, *params, limit),
        ).fetchall()
        results = []
        for index, row in enumerate(rows, start=1):
            if path_glob and not path_matches(row["path"], path_glob):
                continue
            node = semantic_result_from_row(row)
            node["score"] = round(float(row["score"]), 4)
            node["score_source"] = "bm25"
            node["_bm25_rank"] = index
            results.append(node)
        return results

    def _local_hash_search(self, query: str, limit: int, kind: str | None, path_glob: str | None) -> list[dict]:
        self._ensure_local_embeddings()
        query_vector = embed_local([query], DEFAULT_LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_DIMENSIONS)[0]
        candidates = self.vector_search(query_vector, "local", DEFAULT_LOCAL_EMBEDDING_MODEL, max(limit * 8, 100))
        scored = self.shape_semantic_results(candidates, max(limit * 4, 25), kind=kind, path_glob=path_glob, group_by_file=False)
        for index, item in enumerate(scored[:limit], start=1):
            item["score_source"] = "local-hash"
            item["_hash_rank"] = index
        return scored[:limit]

    def _ensure_local_embeddings(self, batch_size: int = 128) -> None:
        node_count = self.conn.execute("select count(*) from nodes").fetchone()[0]
        if node_count and self.embedding_count("local", DEFAULT_LOCAL_EMBEDDING_MODEL) == node_count:
            modified_at = str(self.get_meta("nodes_modified_at", ""))
            complete_at = str(self.get_meta("local_embed_complete_nodes_modified_at", ""))
            if complete_at == modified_at:
                return
        documents = self.semantic_documents()
        existing = self.existing_embedding_hashes("local", DEFAULT_LOCAL_EMBEDDING_MODEL)
        pending = [doc for doc in documents if existing.get(doc["node"]["id"]) != doc["sha1"]]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            vectors = embed_local([doc["text"] for doc in batch], DEFAULT_LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_DIMENSIONS)
            for doc, vector in zip(batch, vectors):
                self.upsert_embedding(doc["node"]["id"], "local", DEFAULT_LOCAL_EMBEDDING_MODEL, vector, doc["sha1"])
            self.commit()
        self.set_meta("local_embed_complete_nodes_modified_at", self.get_meta("nodes_modified_at", ""))
        self.commit()

    def semantic_documents(self) -> list[dict]:
        rows = self.conn.execute("select * from nodes order by kind, path, name").fetchall()
        documents = []
        for row in rows:
            node = row_to_dict(row, keep_terms=True)
            text = semantic_text(node)
            if text.strip():
                documents.append({"node": row_to_dict(row), "text": text, "sha1": content_sha1(text)})
        return documents

    def _upsert_fts_rows(self, rows: Iterable[tuple]) -> None:
        rows = list(rows)
        if not rows:
            return
        rows = list({row[0]: row for row in rows}.values())
        rowids = self._ensure_fts_rowids([row[0] for row in rows])
        self.conn.executemany("delete from nodes_fts where rowid = ?", [(rowids[row[0]],) for row in rows])
        self.conn.executemany(
            """
            insert into nodes_fts(rowid, node_id, kind, name, signature, docstring, decorators, path, body, neighbors, morph)
            values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(rowids[row[0]], *fts_row(row[0], row[1], row[2], row[3], row[5])) for row in rows],
        )

    def _upsert_fts_row(self, node_id: str, kind: str, name: str, path: str | None, encoded_meta: str) -> None:
        rowid = self._ensure_fts_rowids([node_id])[node_id]
        self.conn.execute("delete from nodes_fts where rowid = ?", (rowid,))
        self.conn.execute(
            """
            insert into nodes_fts(rowid, node_id, kind, name, signature, docstring, decorators, path, body, neighbors, morph)
            values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (rowid, *fts_row(node_id, kind, name, path, encoded_meta)),
        )

    def _ensure_fts_rowids(self, node_ids: Iterable[str]) -> dict[str, int]:
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return {}
        self.conn.executemany("insert or ignore into nodes_fts_rowids(node_id) values(?)", [(node_id,) for node_id in ids])
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(f"select node_id, rowid from nodes_fts_rowids where node_id in ({placeholders})", ids).fetchall()
        return {row["node_id"]: int(row["rowid"]) for row in rows}

    def _ensure_fts_populated(self) -> None:
        fts_count = self.conn.execute("select count(*) from nodes_fts").fetchone()[0]
        if fts_count:
            return
        rows = self.conn.execute("select id, kind, name, path, line, meta from nodes").fetchall()
        if rows:
            self._upsert_fts_rows([tuple(row) for row in rows])
            self.set_meta("fts_rebuild_required", "0")
            self.conn.commit()

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
                raw_node = row_to_dict(row, keep_terms=True)
                node = compact_node(raw_node)
                node["score"] = round(score, 4)
                node["score_source"] = "json-vector"
                snippet = semantic_snippet(raw_node)
                if snippet:
                    node["snippet"] = snippet
                scored.append(node)
        scored.sort(key=lambda item: (-item["score"], kind_rank(item["kind"]), item.get("path") or "", item["name"]))
        return scored[:limit]

    def shape_semantic_results(
        self,
        results: list[dict],
        limit: int,
        *,
        kind: str | None = None,
        path_glob: str | None = None,
        group_by_file: bool = True,
    ) -> list[dict]:
        out = []
        for item in results:
            if kind and item.get("kind") != kind:
                continue
            if path_glob and not path_matches(item.get("path"), path_glob):
                continue
            out.append(compact_node(item))
        if group_by_file:
            out = group_same_file_symbols(out, limit)
        return out[:limit]

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
            raw_node = row_to_dict(row, keep_terms=True)
            node = compact_node(raw_node)
            distance = distances[node["id"]]
            node["distance"] = round(distance, 6)
            node["score"] = round(1.0 / (1.0 + distance), 4)
            node["score_source"] = "sqlite-vec"
            snippet = semantic_snippet(raw_node)
            if snippet:
                node["snippet"] = snippet
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

    def dependents(self, node_ids: Iterable[str], limit: int = 50, include_vendor: bool = False) -> list[dict]:
        ids = list(node_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            select e.kind edge_kind, n.*
            from edges e join nodes n on n.id = e.src
            where e.dst in ({placeholders}) and e.kind != 'defines'
              and (? or n.path is null or not ({vendor_path_predicate('n.path')}))
            order by n.kind, n.path, n.name
            limit ?
            """,
            (*ids, int(include_vendor), limit),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    def callers(self, target: str, limit: int = 50, include_vendor: bool = False) -> dict:
        """Return symbols/files that call into the target via 'calls' edges."""
        nodes = self.resolve_targets(target)
        if not nodes:
            return {"target": None, "callers": []}
        match = _best_match(nodes)
        ids = [node["id"] for node in nodes if node["id"].startswith("symbol:") or node["id"] == match["id"]] or [match["id"]]
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            select e.kind edge_kind, e.dst dst_id, e.meta edge_meta, n.*
            from edges e join nodes n on n.id = e.src
            where e.dst in ({placeholders}) and e.kind = 'calls'
              and (? or n.path is null or not ({vendor_path_predicate('n.path')}))
            order by n.kind, n.path, cast(json_extract(e.meta, '$.line') as integer), n.name
            limit ?
            """,
            (*ids, int(include_vendor), limit),
        ).fetchall()
        return {"target": match, "matches": nodes[:10], "callers": [edge_row_to_dict(row) for row in rows]}

    def callees(self, target: str, limit: int = 50, include_vendor: bool = False) -> dict:
        nodes = self.resolve_targets(target)
        if not nodes:
            return {"target": None, "callees": []}
        match = _best_match(nodes)
        ids = [node["id"] for node in nodes if node["id"].startswith("symbol:") or node["id"] == match["id"]] or [match["id"]]
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            select e.kind edge_kind, e.src src_id, e.meta edge_meta, n.*
            from edges e join nodes n on n.id = e.dst
            where e.src in ({placeholders}) and e.kind = 'calls'
              and (? or n.path is null or not ({vendor_path_predicate('n.path')}))
            order by cast(json_extract(e.meta, '$.line') as integer), n.kind, n.path, n.name
            limit ?
            """,
            (*ids, int(include_vendor), limit),
        ).fetchall()
        return {"target": match, "matches": nodes[:10], "callees": [edge_row_to_dict(row) for row in rows]}

    def trace(self, source: str, target: str | None = None, max_hops: int = 3, limit: int = 100, include_vendor: bool = False) -> dict:
        """Return ordered call paths from a source symbol/file, optionally stopping at a target."""
        max_hops = max(1, min(int(max_hops), 10))
        limit = max(1, int(limit))
        source_nodes = self.resolve_targets(source)
        if not source_nodes:
            return {"source": None, "target": None, "paths": []}
        source_match = _best_match(source_nodes)
        target_nodes = self.resolve_targets(target) if target else []
        target_match = _best_match(target_nodes) if target_nodes else None
        target_ids = {node["id"] for node in target_nodes} if target_nodes else set()
        target_paths = {node["path"] for node in target_nodes if node.get("kind") == "file" and node.get("path")}
        start_ids = [node["id"] for node in source_nodes if node["id"].startswith("symbol:") or node["id"] == source_match["id"]] or [source_match["id"]]

        nodes: dict[str, dict] = {source_match["id"]: source_match}
        if target_match:
            nodes[target_match["id"]] = target_match
        paths: list[dict] = []
        paths_by_hop: dict[int, int] = defaultdict(int)
        paths_remaining = 0
        paths_explored = 0
        truncated = False
        frontier = deque((node_id, [], {node_id}) for node_id in start_ids)
        while frontier:
            node_id, edges, seen = frontier.popleft()
            if len(edges) >= max_hops:
                continue
            for edge in self._call_edges_from(node_id, None, include_vendor):
                dst = edge["to"]["id"]
                if dst in seen:
                    continue
                paths_explored += 1
                next_edges = [*edges, edge]
                hops = len(next_edges)
                reaches_target = dst in target_ids or edge["to"].get("path") in target_paths
                if (not target_ids and not target_paths) or reaches_target:
                    if paths_by_hop[hops] < limit:
                        for path_edge in next_edges:
                            nodes[path_edge["from"]["id"]] = path_edge["from"]
                            nodes[path_edge["to"]["id"]] = path_edge["to"]
                        paths.append({"hops": hops, "edges": [compact_trace_edge(path_edge) for path_edge in next_edges]})
                        paths_by_hop[hops] += 1
                    else:
                        paths_remaining += 1
                        truncated = True
                if len(next_edges) < max_hops:
                    frontier.append((dst, next_edges, {*seen, dst}))
        result = {
            "source": source_match,
            "target": target_match,
            "max_hops": max_hops,
            "limit_per_hop": limit,
            "truncated": truncated,
            "paths_explored": paths_explored,
            "paths_remaining": paths_remaining,
            "nodes": {node_id: nodes[node_id] for node_id in sorted(nodes)},
            "paths": paths,
        }
        if len(source_nodes) != 1:
            result["source_matches"] = source_nodes[:10]
        if target and len(target_nodes) != 1:
            result["target_matches"] = target_nodes[:10]
        return result

    def _call_edges_from(self, node_id: str, limit: int | None, include_vendor: bool) -> list[dict]:
        limit_clause = "limit ?" if limit is not None else ""
        params: tuple = (node_id, int(include_vendor), limit) if limit is not None else (node_id, int(include_vendor))
        rows = self.conn.execute(
            f"""
            select e.src, e.dst, e.kind edge_kind, e.meta edge_meta,
                   src.kind src_kind, src.name src_name, src.path src_path, src.line src_line, src.meta src_meta,
                   dst.kind dst_kind, dst.name dst_name, dst.path dst_path, dst.line dst_line, dst.meta dst_meta
            from edges e
            join nodes src on src.id = e.src
            join nodes dst on dst.id = e.dst
            where e.src = ? and e.kind = 'calls'
              and (? or dst.path is null or not ({vendor_path_predicate('dst.path')}))
            order by cast(json_extract(e.meta, '$.line') as integer), dst.kind, dst.path, dst.name
            {limit_clause}
            """,
            params,
        ).fetchall()
        out = []
        for row in rows:
            edge_meta = parse_json_object(row["edge_meta"])
            out.append(
                {
                    "from": node_from_prefixed_row(row, "src"),
                    "to": node_from_prefixed_row(row, "dst"),
                    "edge": row["edge_kind"],
                    "call_line": edge_meta.get("line"),
                    "call_name": edge_meta.get("name"),
                    "call_qualifier": edge_meta.get("qualifier"),
                }
            )
        return out

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

    def fts_status(self) -> dict:
        node_count = self.conn.execute("select count(*) from nodes").fetchone()[0]
        fts_count = self.conn.execute("select count(*) from nodes_fts").fetchone()[0]
        sample_rows = self.conn.execute("select meta from nodes limit 200").fetchall()
        rich_count = sum(1 for row in sample_rows if meta_has_rich_fields(row["meta"]))
        warnings = []
        if self.get_meta("fts_rebuild_required") == "1":
            warnings.append("FTS schema changed and needs a rebuild; it will be rebuilt on the next semantic query.")
        if node_count and not fts_count:
            warnings.append("FTS is empty; it will be rebuilt on the next semantic query.")
        if node_count and fts_count and not rich_count:
            warnings.append("FTS was rebuilt from legacy metadata without rich symbol documents. Run `ctx index` for better BM25 quality.")
        return {"nodes": node_count, "rows": fts_count, "rich_sample_rows": rich_count, "warnings": warnings}

    def symbols(self, name: str, limit: int = 20) -> list[dict]:
        needle = f"%{name}%"
        rows = self.conn.execute("select * from nodes where kind = 'symbol' and name like ? order by name limit ?", (needle, limit)).fetchall()
        return [row_to_dict(row) for row in rows]

    def tests_for_path(self, path: str, limit: int = 50) -> list[dict]:
        nodes = self.resolve_targets(path)
        return self.related_tests([node["id"] for node in nodes], limit)

    def impact(self, target: str, limit: int = 50, include_vendor: bool = False) -> dict:
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
              and (? or n.path is null or not ({vendor_path_predicate('n.path')}))
            order by n.kind, n.path, n.name
            limit ?
            """,
            (*ids, int(include_vendor), limit),
        ).fetchall()
        return {
            "target": target_node,
            "matches": nodes[:10],
            "dependents": self.dependents(ids, limit, include_vendor),
            "dependencies": [row_to_dict(row) for row in deps],
        }


def _best_match(nodes: list[dict]) -> dict:
    """Pick the most useful node from a candidate list (symbols beat files beat packages)."""
    return sorted(nodes, key=lambda node: (kind_rank(node.get("kind", "")), node.get("path") or "", node.get("name") or ""))[0]


def vendor_path_predicate(column: str) -> str:
    return (
        f"{column} like '.venv/%' or {column} like 'venv/%' or "
        f"{column} like 'vendor/%' or {column} like 'node_modules/%' or "
        f"{column} like '%/.venv/%' or {column} like '%/venv/%' or "
        f"{column} like '%/vendor/%' or {column} like '%/node_modules/%'"
    )


def query_terms(query: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", query.replace("_", " ").replace(".", " "))
    terms = [term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9]{1,}", expanded)]
    return [term for term in terms if term not in STOPWORDS]


def meta_has_rich_fields(encoded_meta: str) -> bool:
    try:
        meta = json.loads(encoded_meta)
    except Exception:
        return False
    return any(meta.get(key) for key in ("signature", "docstring", "body_preview", "routes", "exports", "neighbors"))


def semantic_filters(kind: str | None, alias: str) -> tuple[str, list[str]]:
    filters = []
    params = []
    if kind:
        filters.append(f"and {alias}.kind = ?")
        params.append(kind)
    return "\n              ".join(filters), params


def path_matches(path: str | None, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path or "", pattern)


def semantic_result_from_row(row) -> dict:
    node = compact_node(row_to_dict(row))
    snippet = first_nonempty(row["docstring_snippet"], row["signature_snippet"], row["body_snippet"])
    if snippet:
        node["snippet"] = snippet
    return node


def rrf_fuse(primary: list[dict], secondary: list[dict], limit: int, k: int = 60) -> list[dict]:
    fused: dict[str, dict] = {}
    for source, rank_key in [(primary, "_bm25_rank"), (secondary, "_hash_rank")]:
        for index, item in enumerate(source, start=1):
            node_id = item["id"]
            rank = int(item.get(rank_key) or index)
            existing = fused.setdefault(node_id, {key: value for key, value in item.items() if not key.startswith("_")})
            existing["_rrf"] = existing.get("_rrf", 0.0) + 1.0 / (k + rank)
            sources = set(existing.get("score_sources", []))
            sources.add(str(item.get("score_source", "")))
            existing["score_sources"] = sorted(source for source in sources if source)
            if item.get("snippet") and not existing.get("snippet"):
                existing["snippet"] = item["snippet"]
    results = list(fused.values())
    max_score = max((float(item.get("_rrf", 0.0)) for item in results), default=0.0)
    for item in results:
        raw_score = float(item.pop("_rrf", 0.0))
        item["score"] = round(raw_score / max_score, 4) if max_score else 0.0
        item["score_source"] = "hybrid" if len(item.get("score_sources", [])) > 1 else (item.get("score_sources") or ["hybrid"])[0]
    results.sort(key=lambda item: (-item["score"], kind_rank(item["kind"]), item.get("path") or "", item["name"]))
    return results[:limit]


def group_same_file_symbols(results: list[dict], limit: int) -> list[dict]:
    grouped: list[dict] = []
    seen: set[str] = set()
    by_path: dict[str, list[dict]] = {}
    for item in results:
        if item.get("kind") == "symbol" and item.get("path"):
            by_path.setdefault(item["path"], []).append(item)
    for item in results:
        node_id = item["id"]
        if node_id in seen:
            continue
        if item.get("kind") == "symbol" and item.get("path") and len(by_path.get(item["path"], [])) > 1:
            all_siblings = [compact_node(sibling) for sibling in by_path[item["path"]]]
            siblings = all_siblings[:5]
            merged = dict(item)
            merged["co_located_symbols"] = [
                {"name": sibling["name"], "line": sibling.get("line"), "id": sibling["id"], "score": sibling.get("score")}
                for sibling in siblings
                if sibling["id"] != node_id
            ]
            merged["co_located_total"] = len(all_siblings) - 1
            if len(all_siblings) > len(siblings):
                merged["co_located_truncated"] = True
            for sibling in siblings:
                seen.add(sibling["id"])
            grouped.append(merged)
        else:
            seen.add(node_id)
            grouped.append(item)
        if len(grouped) >= limit:
            break
    return grouped


def compact_node(node: dict) -> dict:
    node = dict(node)
    meta = dict(node.get("meta") or {}) if isinstance(node.get("meta"), dict) else {}
    meta.setdefault("_node_kind", node.get("kind"))
    node["meta"] = compact_meta(meta)
    return node


def compact_meta(meta) -> dict:
    if not isinstance(meta, dict):
        return {}
    kind = meta.get("_node_kind")
    keep = {}
    if kind in {"file", "test"}:
        for key, max_items in [("routes", 8), ("exports", 12), ("neighbors", 8)]:
            value = meta.get(key)
            if isinstance(value, list) and value:
                keep[key] = value[:max_items]
                if len(value) > max_items:
                    keep[f"{key}_truncated"] = True
        for key in ("sha1", "size"):
            if meta.get(key) is not None:
                keep[key] = meta[key]
        if meta.get("docstring"):
            keep["docstring"] = first_line(str(meta["docstring"]))
    elif kind == "symbol":
        for key in ("symbol_kind", "parent", "signature"):
            if meta.get(key):
                keep[key] = first_line(str(meta[key])) if key == "signature" else meta[key]
    elif kind == "route":
        for key in ("method", "route", "handler", "signature", "docstring"):
            if meta.get(key):
                keep[key] = first_line(str(meta[key])) if key == "signature" else meta[key]
    else:
        for key in ("symbol_kind", "method", "route", "handler", "parent", "import", "patterns"):
            if meta.get(key):
                keep[key] = meta[key]
    return keep


def first_line(value: str) -> str:
    return value.strip().splitlines()[0] if value.strip() else ""


def fts_row(node_id: str, kind: str, name: str, path: str | None, encoded_meta: str) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    try:
        meta = json.loads(encoded_meta) if encoded_meta else {}
    except Exception:
        meta = {}
    signature = meta_text(meta, "signature")
    docstring = meta_text(meta, "docstring")
    decorators = " ".join(str(item) for item in meta.get("decorators", []) if item)
    body_parts = [meta_text(meta, "body_preview")]
    # File-level term frequency list — top-N identifiers per file. Indexing these here
    # lets ctx_search find file-level references that aren't captured by the first-10-lines
    # body_preview window (e.g., a class used 200 lines into a 500-line file).
    body_parts.extend(str(term) for term in meta_list(meta, "terms") if term)
    body = " ".join(part for part in body_parts if part)
    neighbors = " ".join(
        str(item)
        for key in ("neighbors", "exports", "routes", "imports", "parent")
        for item in meta_list(meta, key)
        if item
    )
    return (
        node_id,
        kind,
        fts_text(name),
        fts_text(signature),
        fts_text(docstring),
        fts_text(decorators),
        fts_text(path or ""),
        fts_text(body),
        fts_text(neighbors),
        morph_text(" ".join([name, signature, docstring, decorators, path or "", body, neighbors])),
    )


def meta_text(meta: dict, key: str) -> str:
    value = meta.get(key)
    return value if isinstance(value, str) else ""


def meta_list(meta: dict, key: str) -> list:
    value = meta.get(key)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def fts_text(value: str) -> str:
    value = value.replace("/", " ").replace(".", " ").replace("-", " ")
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value.replace("_", " "))
    if split == value:
        return value
    return f"{value} {split}"


def morph_text(value: str) -> str:
    variants: list[str] = []
    seen: set[str] = set()
    for token in tokenize_identifier_text(value):
        for variant in morphology_variants(token):
            if variant not in seen:
                seen.add(variant)
                variants.append(variant)
    return " ".join(variants)


def fts_query(query: str) -> str:
    terms = expand_query_terms(query)
    if not terms:
        return ""
    return " OR ".join(f'"{term}"' for term in terms)


def lexical_reference_query(query: str) -> str:
    """FTS5 query for finding *literal* references — used by ctx_search to locate
    nodes whose body/decorators/neighbors mention the term, even when the node's
    own name doesn't match. AND the camel/snake-split tokens so all parts must
    appear, no morphology / no thesaurus.
    """
    tokens = [token for token in tokenize_identifier_text(query) if len(token) >= 2 and token not in STOPWORDS]
    if not tokens:
        return ""
    return " AND ".join(f'"{token}"' for token in tokens)


def expand_query_terms(query: str) -> list[str]:
    raw_terms = tokenize_identifier_text(query)
    out: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        if term in STOPWORDS:
            continue
        for candidate in [*morphology_variants(term), *CODE_THESAURUS.get(term, [])]:
            cleaned = re.sub(r'["\s]+', " ", candidate.strip().lower())
            if len(cleaned) >= 2 and cleaned not in STOPWORDS and cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
    return out


def tokenize_identifier_text(value: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " "))
    return [term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9]{1,}", expanded)]


def morphology_variants(term: str) -> list[str]:
    irregular = MORPHOLOGY.get(term, [])
    bases = {term, *irregular}
    for suffix, replacement in [
        ("ations", "ate"),
        ("ation", "ate"),
        ("itions", "it"),
        ("ition", "it"),
        ("ions", ""),
        ("ion", ""),
        ("ing", ""),
        ("ed", ""),
        ("es", ""),
        ("s", ""),
    ]:
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            bases.add(term[: -len(suffix)] + replacement)
    out: list[str] = []
    seen: set[str] = set()
    for base in bases:
        for variant in [base, f"{base}s", f"{base}es", f"{base}ed", f"{base}ing", f"{base}er", f"{base}ion", f"{base}tion"]:
            normalized = normalize_doubled_suffix(variant)
            if len(normalized) >= 2 and normalized not in seen:
                seen.add(normalized)
                out.append(normalized)
    return out


def normalize_doubled_suffix(value: str) -> str:
    return value.replace("eeing", "eing").replace("eion", "ion").replace("etion", "ation")


def first_nonempty(*values) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def semantic_snippet(node: dict) -> str:
    meta = node.get("meta") or {}
    return first_nonempty(meta.get("docstring"), meta.get("signature"), meta.get("body_preview"))


def row_to_dict(row: sqlite3.Row | dict, keep_terms: bool = False) -> dict:
    data = dict(row)
    data.pop("vector", None)
    data.pop("body_snippet", None)
    data.pop("docstring_snippet", None)
    data.pop("signature_snippet", None)
    if "meta" in data and isinstance(data["meta"], str):
        try:
            data["meta"] = json.loads(data["meta"])
        except Exception:
            pass
    if not keep_terms and isinstance(data.get("meta"), dict):
        meta = dict(data["meta"])
        meta["_node_kind"] = data.get("kind")
        data["meta"] = compact_meta(meta)
    return data


def edge_row_to_dict(row: sqlite3.Row | dict) -> dict:
    data = row_to_dict(row)
    edge_meta = parse_json_object(data.pop("edge_meta", None))
    if edge_meta:
        data["edge_meta"] = edge_meta
    if "line" in edge_meta:
        data["call_line"] = edge_meta["line"]
    if "name" in edge_meta:
        data["call_name"] = edge_meta["name"]
    if "qualifier" in edge_meta:
        data["call_qualifier"] = edge_meta["qualifier"]
    return data


def compact_trace_edge(edge: dict) -> dict:
    return {
        "from": edge["from"]["id"],
        "to": edge["to"]["id"],
        "edge": edge["edge"],
        "call_line": edge.get("call_line"),
        "call_name": edge.get("call_name"),
        "call_qualifier": edge.get("call_qualifier"),
    }


def parse_json_object(value) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def node_from_prefixed_row(row: sqlite3.Row | dict, prefix: str) -> dict:
    data = dict(row)
    meta = parse_json_object(data.get(f"{prefix}_meta"))
    meta["_node_kind"] = data.get(f"{prefix}_kind")
    return {
        "id": data.get(prefix),
        "kind": data.get(f"{prefix}_kind"),
        "name": data.get(f"{prefix}_name"),
        "path": data.get(f"{prefix}_path"),
        "line": data.get(f"{prefix}_line"),
        "meta": compact_meta(meta),
    }


def semantic_text(node: dict) -> str:
    meta = node.get("meta") or {}
    parts = [node.get("kind", ""), node.get("name", ""), node.get("path") or ""]
    for key in ["symbol_kind", "route", "method", "import", "signature", "docstring", "body_preview", "handler", "parent"]:
        value = meta.get(key)
        if value:
            parts.append(str(value))
    for key in ["decorators", "exports", "routes", "neighbors"]:
        value = meta.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
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
    "where",
    "with",
}

CODE_THESAURUS = {
    "auth": ["authentication", "authorize", "authorization", "login"],
    "authentication": ["auth", "login"],
    "login": ["auth", "authentication", "signin"],
    "signin": ["login", "auth"],
    "db": ["database", "storage", "store"],
    "database": ["db", "storage", "store"],
    "req": ["request"],
    "request": ["req"],
    "res": ["response"],
    "response": ["res"],
    "cfg": ["config", "configuration"],
    "config": ["cfg", "configuration"],
    "configuration": ["config", "cfg"],
    "id": ["identifier"],
    "identifier": ["id"],
    "del": ["delete", "remove"],
    "delete": ["del", "remove", "destroy"],
    "remove": ["delete", "del"],
    "create": ["add", "insert", "new"],
    "add": ["create", "insert"],
    "update": ["edit", "modify", "patch"],
    "edit": ["update", "modify"],
    "list": ["search", "find", "query"],
    "find": ["search", "lookup", "query"],
    "search": ["find", "lookup", "query"],
    "error": ["exception", "failure"],
    "exception": ["error", "failure"],
    "cache": ["memo", "memoize"],
    "env": ["environment"],
    "environment": ["env"],
    "msg": ["message"],
    "message": ["msg"],
    "repo": ["repository"],
    "repository": ["repo"],
    "route": ["endpoint", "handler"],
    "endpoint": ["route", "handler"],
    "handler": ["route", "endpoint"],
    "user": ["account", "profile"],
    "account": ["user", "profile"],
}

MORPHOLOGY = {
    "create": ["creates", "created", "creating", "creation"],
    "creates": ["create", "created", "creating", "creation"],
    "created": ["create", "creates", "creating", "creation"],
    "creating": ["create", "creates", "created", "creation"],
    "creation": ["create", "creates", "created", "creating"],
    "delete": ["deletes", "deleted", "deleting", "deletion"],
    "deletes": ["delete", "deleted", "deleting", "deletion"],
    "deleted": ["delete", "deletes", "deleting", "deletion"],
    "deleting": ["delete", "deletes", "deleted", "deletion"],
    "deletion": ["delete", "deletes", "deleted", "deleting"],
    "remove": ["removes", "removed", "removing", "removal"],
    "removes": ["remove", "removed", "removing", "removal"],
    "removed": ["remove", "removes", "removing", "removal"],
    "removing": ["remove", "removes", "removed", "removal"],
    "removal": ["remove", "removes", "removed", "removing"],
    "update": ["updates", "updated", "updating"],
    "updates": ["update", "updated", "updating"],
    "updated": ["update", "updates", "updating"],
    "updating": ["update", "updates", "updated"],
    "configure": ["configures", "configured", "configuring", "configuration"],
    "configuration": ["configure", "configures", "configured", "configuring", "config"],
    "patch": ["patches", "patched", "patching"],
    "patches": ["patch", "patched", "patching"],
    "patched": ["patch", "patches", "patching"],
    "patching": ["patch", "patches", "patched"],
    "fetch": ["fetches", "fetched", "fetching"],
    "fetches": ["fetch", "fetched", "fetching"],
    "fetched": ["fetch", "fetches", "fetching"],
    "fetching": ["fetch", "fetches", "fetched"],
    "send": ["sends", "sent", "sending"],
    "sends": ["send", "sent", "sending"],
    "sent": ["send", "sends", "sending"],
    "sending": ["send", "sends", "sent"],
    "get": ["gets", "got", "getting"],
    "gets": ["get", "got", "getting"],
    "got": ["get", "gets", "getting"],
    "getting": ["get", "gets", "got"],
    "set": ["sets", "setting"],
    "sets": ["set", "setting"],
    "setting": ["set", "sets"],
    "load": ["loads", "loaded", "loading"],
    "loads": ["load", "loaded", "loading"],
    "loaded": ["load", "loads", "loading"],
    "loading": ["load", "loads", "loaded"],
    "save": ["saves", "saved", "saving"],
    "saves": ["save", "saved", "saving"],
    "saved": ["save", "saves", "saving"],
    "saving": ["save", "saves", "saved"],
    "parse": ["parses", "parsed", "parsing"],
    "parses": ["parse", "parsed", "parsing"],
    "parsed": ["parse", "parses", "parsing"],
    "parsing": ["parse", "parses", "parsed"],
    "validate": ["validates", "validated", "validating", "validation"],
    "validation": ["validate", "validates", "validated", "validating"],
}
