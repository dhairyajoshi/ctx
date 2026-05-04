from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable


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
"""


class GraphStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def reset(self, repo: Path) -> None:
        self.conn.execute("delete from edges")
        self.conn.execute("delete from nodes")
        self.set_meta("repo_root", str(repo.resolve()))
        self.set_meta("indexed_at", str(int(time.time())))
        self.conn.commit()

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

    def commit(self) -> None:
        self.conn.commit()

    def search(self, query: str, limit: int = 20) -> list[dict]:
        needle = f"%{query}%"
        rows = self.conn.execute(
            """
            select * from nodes
            where name like ? or path like ? or meta like ?
            order by case kind when 'symbol' then 0 when 'file' then 1 else 2 end, name
            limit ?
            """,
            (needle, needle, needle, limit),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    def nodes_by_path_or_name(self, value: str) -> list[dict]:
        rows = self.conn.execute(
            "select * from nodes where id = ? or path = ? or name = ? or path like ? order by kind",
            (value, value, value, f"%{value}%"),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

    def dependents(self, node_ids: Iterable[str], limit: int = 50) -> list[dict]:
        ids = list(node_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            select e.kind edge_kind, n.*
            from edges e join nodes n on n.id = e.src
            where e.dst in ({placeholders})
            order by n.kind, n.path, n.name
            limit ?
            """,
            (*ids, limit),
        ).fetchall()
        return [row_to_dict(row) for row in rows]

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
        return out

    def counts(self) -> dict[str, int]:
        return self.stats()

    def symbols(self, name: str, limit: int = 20) -> list[dict]:
        needle = f"%{name}%"
        rows = self.conn.execute("select * from nodes where kind = 'symbol' and name like ? order by name limit ?", (needle, limit)).fetchall()
        return [row_to_dict(row) for row in rows]

    def tests_for_path(self, path: str, limit: int = 50) -> list[dict]:
        nodes = self.nodes_by_path_or_name(path)
        return self.related_tests([node["id"] for node in nodes], limit)

    def impact(self, target: str, limit: int = 50) -> dict:
        nodes = self.nodes_by_path_or_name(target)
        if not nodes:
            matches = self.search(target, 1)
            nodes = matches
        if not nodes:
            return {"target": None, "dependents": [], "dependencies": []}
        target_node = nodes[0]
        deps = self.conn.execute(
            """
            select e.kind edge_kind, n.*
            from edges e join nodes n on n.id = e.dst
            where e.src = ?
            order by n.kind, n.path, n.name
            limit ?
            """,
            (target_node["id"], limit),
        ).fetchall()
        return {
            "target": target_node,
            "dependents": self.dependents([target_node["id"]], limit),
            "dependencies": [row_to_dict(row) for row in deps],
        }


def row_to_dict(row: sqlite3.Row | dict) -> dict:
    data = dict(row)
    if "meta" in data and isinstance(data["meta"], str):
        try:
            data["meta"] = json.loads(data["meta"])
        except Exception:
            pass
    return data
