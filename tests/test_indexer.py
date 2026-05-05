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


def test_python_callees_resolve_imports_without_bare_name_fanout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "helpers.py").write_text(
        "def select(value):\n"
        "    return value\n\n"
        "def unique_helper():\n"
        "    return True\n",
        encoding="utf-8",
    )
    (repo / "vendorish.py").write_text("def select(value):\n    return value\n", encoding="utf-8")
    (repo / "service.py").write_text(
        "from helpers import select\n\n"
        "def handle():\n"
        "    return select(str(1))\n",
        encoding="utf-8",
    )
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    callees = store.callees("handle")["callees"]
    assert [row["path"] for row in callees] == ["helpers.py"]
    assert [row["name"] for row in callees] == ["select"]


def test_python_import_resolution_survives_ast_parse_failure(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "helpers.py").write_text("def target():\n    return True\n", encoding="utf-8")
    (repo / "legacy_helpers.py").write_text("def target():\n    return True\n", encoding="utf-8")
    (repo / "service.py").write_text(
        "from helpers import target\n\n"
        "def handle[T]():\n"
        "    return target()\n",
        encoding="utf-8",
    )
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    callers = store.callers("target")["callers"]
    assert any(row["path"] == "service.py" and row["name"] == "handle" for row in callers)
    callees = store.callees("handle")["callees"]
    assert [row["path"] for row in callees] == ["helpers.py"]


def test_python_relative_import_resolution_disambiguates_calls(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    package = repo / "workflow"
    package.mkdir(parents=True)
    (package / "task_helper.py").write_text("def run_task_helper():\n    return True\n", encoding="utf-8")
    (package / "legacy_task_helper.py").write_text("def run_task_helper():\n    return True\n", encoding="utf-8")
    (package / "service.py").write_text(
        "from .task_helper import run_task_helper\n\n"
        "def handle_example_workflow():\n"
        "    return run_task_helper()\n",
        encoding="utf-8",
    )
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    callers = store.callers("run_task_helper")["callers"]
    assert any(row["path"] == "workflow/service.py" and row["name"] == "handle_example_workflow" for row in callers)
    callees = store.callees("handle_example_workflow")["callees"]
    assert [row["path"] for row in callees] == ["workflow/task_helper.py"]


def test_python_imported_route_wrapper_is_reported_as_caller(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    package = repo / "workflow"
    package.mkdir(parents=True)
    (package / "service.py").write_text(
        "def handle_example_workflow(payload):\n"
        "    return payload\n",
        encoding="utf-8",
    )
    (package / "router.py").write_text(
        "from workflow.service import handle_example_workflow\n\n"
        "async def route_example_workflow(payload):\n"
        "    return await handle_example_workflow(payload)\n",
        encoding="utf-8",
    )
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    callers = store.callers("handle_example_workflow")["callers"]
    assert any(row["path"] == "workflow/router.py" and row["name"] == "route_example_workflow" for row in callers)


def test_indexes_python_decorator_routes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "api.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health')\n"
        "def health():\n"
        "    return {'ok': True}\n\n"
        "@app.post('/invoices')\n"
        "async def create_invoice():\n"
        "    return {}\n",
        encoding="utf-8",
    )
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    assert any(row["name"] == "GET /health" for row in store.search("GET /health"))
    assert any(row["name"] == "POST /invoices" for row in store.search("POST /invoices"))


def test_indexes_python_generic_route_methods(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "api.py").write_text(
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n\n"
        "@app.route('/checkout', methods=['POST', 'PUT'])\n"
        "def checkout():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    assert any(row["name"] == "POST /checkout" for row in store.search("POST /checkout"))
    assert any(row["name"] == "PUT /checkout" for row in store.search("PUT /checkout"))


def test_python_ambiguous_bare_calls_are_dropped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "left.py").write_text("def shared():\n    return True\n", encoding="utf-8")
    (repo / "right.py").write_text("def shared():\n    return True\n", encoding="utf-8")
    (repo / "service.py").write_text("def handle():\n    return shared()\n", encoding="utf-8")
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    assert store.callees("handle")["callees"] == []


def test_markdown_is_not_used_for_call_edges(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        "def run_task_helper():\n"
        "    return True\n\n"
        "def handle_example_workflow():\n"
        "    return run_task_helper()\n",
        encoding="utf-8",
    )
    (repo / "notes.md").write_text("Notes mention run_task_helper() but are not code.\n", encoding="utf-8")
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    callers = store.callers("run_task_helper")["callers"]
    assert any(row["name"] == "handle_example_workflow" for row in callers)
    assert not any(row["path"] == "notes.md" for row in callers)


def test_search_identifier_member_uses_lexical_terms_before_semantic_fallback(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        "class TaskStatus:\n"
        "    registered = 'registered'\n\n"
        "def handle(task):\n"
        "    task.status = TaskStatus.registered\n",
        encoding="utf-8",
    )
    store = GraphStore(tmp_path / "graph.sqlite")

    index_repo(repo, CtxConfig(storage="central"), store)

    hits = store.search("TaskStatus.registered")
    assert hits
    assert hits[0]["path"] == "service.py"


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
