from __future__ import annotations

from pathlib import Path

from ctx_kg.config import CtxConfig
from ctx_kg.indexer import index_repo
from ctx_kg.store import GraphStore


def test_indexes_python_symbols_and_tests(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "calc.py"
    test = repo / "calc_test.py"
    src.write_text("def add(a, b):\n    return a + b\n")
    test.write_text("from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    store = GraphStore(tmp_path / "graph.sqlite")

    stats = index_repo(repo, CtxConfig(storage="central"), store)

    assert stats.files == 2
    assert stats.symbols >= 2
    rows = store.search("add")
    assert any(row["kind"] == "symbol" for row in rows)
    tests = store.related_tests(["file:calc.py"])
    assert any(row["path"] == "calc_test.py" for row in tests)


def test_indexes_typescript_imports_and_routes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "server.ts").write_text(
        "import { createInvoice } from './invoice'\n"
        "export function boot() { router.post('/invoices', createInvoice) }\n"
    )
    (repo / "invoice.ts").write_text("export const createInvoice = () => true\n")
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    routes = store.search("POST /invoices")
    assert routes
    impacts = store.dependents(["file:invoice.ts"])
    assert any(row["path"] == "server.ts" for row in impacts)
