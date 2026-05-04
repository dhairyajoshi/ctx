from __future__ import annotations

import json
import sqlite3


def row_to_dict(row: sqlite3.Row | dict) -> dict:
    data = dict(row)
    if "meta" in data:
        try:
            data["meta"] = json.loads(data["meta"])
        except Exception:
            pass
    return data


def compact_node(row: sqlite3.Row | dict) -> str:
    path = row["path"] or ""
    line = f":{row['line']}" if row["line"] else ""
    location = f" {path}{line}" if path else ""
    prefix = f"[{row['kind']}]"
    return f"{prefix} {row['name']}{location}"
