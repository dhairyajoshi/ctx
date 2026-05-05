from __future__ import annotations

from pathlib import Path

from ctx_kg.config import CtxConfig
from ctx_kg.indexer import index_repo
from ctx_kg.mcp import _choose_anchor
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


def test_python_call_edges_are_scoped_to_enclosing_function(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        "class Earlier:\n"
        "    pass\n\n"
        "def target():\n"
        "    return True\n\n"
        "def actual_caller():\n"
        "    return target()\n",
        encoding="utf-8",
    )
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    callers = store.callers("target")["callers"]
    assert any(row["name"] == "actual_caller" for row in callers)
    assert not any(row["name"] == "Earlier" for row in callers)

    callees = store.callees("actual_caller")["callees"]
    assert any(row["name"] == "target" for row in callees)


def test_python_tree_sitter_indexes_syntax_ast_cannot_parse(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "modern.py").write_text(
        "def consume(value):\n"
        "    return value\n\n"
        "def route[T](item: T):\n"
        "    return consume(item)\n",
        encoding="utf-8",
    )
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    assert any(row["name"] == "route" for row in store.symbols("route"))
    assert any(row["name"] == "consume" for row in store.callees("route")["callees"])


def test_explain_anchor_prefers_flow_entry_symbol_over_conversion_helper() -> None:
    symbols = [
        {"kind": "symbol", "name": "convert_source_to_target", "path": "workflow/service.py", "line": 120, "score": 0.91},
        {"kind": "symbol", "name": "handle_example_workflow", "path": "workflow/service.py", "line": 2100, "score": 0.83},
    ]

    anchor = _choose_anchor("workflow lifecycle", symbols, [], symbols)

    assert anchor is not None
    assert anchor["name"] == "handle_example_workflow"


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
