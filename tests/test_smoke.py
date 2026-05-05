from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ctx_kg.config import CtxConfig, load_registry, register_repo
from ctx_kg.embeddings import provider_from_env
from ctx_kg.indexer import index_repo
from ctx_kg.mcp import McpServer
from ctx_kg.store import GraphStore


class SmokeTest(unittest.TestCase):
    def test_indexes_python_repo(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            os.environ["CTX_HOME"] = str(Path(temp) / "home")
            repo = Path(temp) / "repo"
            repo.mkdir()
            (repo / "billing.py").write_text(
                """
def calculate_total(items):
    return sum(items)

class Invoice:
    def create(self, items):
        return calculate_total(items)
""",
                encoding="utf-8",
            )
            (repo / "test_billing.py").write_text(
                """
from billing import calculate_total

def test_calculate_total():
    assert calculate_total([1, 2]) == 3
""",
                encoding="utf-8",
            )

            config = CtxConfig(repo=repo)
            counts = index_repo(config)

            self.assertEqual(counts["files"], 2)
            self.assertGreater(counts["nodes"], 2)

            store = GraphStore(config.db_path)
            try:
                symbols = store.symbols("calculate_total")
                tests = store.tests_for_path("billing.py")
                impact = store.impact("calculate_total")
            finally:
                store.close()

            self.assertTrue(symbols)
            self.assertTrue(tests)
            self.assertEqual(impact["target"]["name"], "calculate_total")

    def test_index_caches_local_embeddings_for_hybrid_search(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            os.environ["CTX_HOME"] = str(Path(temp) / "home")
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("VOYAGE_API_KEY", None)
            os.environ.pop("CTX_VOYAGE_API_KEY", None)
            os.environ.pop("CTX_EMBED_PROVIDER", None)
            os.environ.pop("CTX_EMBED_API_KEY", None)
            os.environ.pop("CTX_EMBED_BASE_URL", None)
            repo = Path(temp) / "repo"
            repo.mkdir()
            (repo / "search.py").write_text("def find_user(query):\n    return query\n", encoding="utf-8")

            config = CtxConfig(repo=repo)
            counts = index_repo(config)

            self.assertGreater(counts.get("embeddings", 0), 0)
            self.assertEqual(counts["embed"]["provider"], "local")
            self.assertGreater(counts["embed"]["embedded"], 0)
            self.assertTrue(counts["embed"]["cached_for_hybrid"])

            store = GraphStore(config.db_path)
            try:
                last_embed = store.get_meta("last_embed", {})
                self.assertEqual(last_embed.get("provider"), "local")
                self.assertTrue(last_embed.get("cached_for_hybrid"))
                self.assertGreater(store.embedding_count("local", last_embed["model"]), 0)
            finally:
                store.close()

    def test_vector_search_uses_stored_embeddings(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            os.environ["CTX_HOME"] = str(Path(temp) / "home")
            repo = Path(temp) / "repo"
            repo.mkdir()
            (repo / "invoice.py").write_text("def create_invoice(order):\n    return order\n", encoding="utf-8")

            config = CtxConfig(repo=repo)
            index_repo(config)
            store = GraphStore(config.db_path)
            try:
                for doc in store.semantic_documents():
                    vector = [1.0, 0.0] if "invoice" in doc["text"] else [0.0, 1.0]
                    store.upsert_embedding(doc["node"]["id"], "test", "fake", vector, doc["sha1"])
                store.commit()
                results = store.vector_search([1.0, 0.0], "test", "fake", 5)
            finally:
                store.close()

            self.assertTrue(results)
            self.assertIn(results[0]["score_source"], {"sqlite-vec", "json-vector"})

    def test_semantic_search_uses_rich_bm25_documents(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            os.environ["CTX_HOME"] = str(Path(temp) / "home")
            repo = Path(temp) / "repo"
            repo.mkdir()
            (repo / "billing.py").write_text(
                'def create_invoice(order):\n'
                '    """Create a new invoice for a user order."""\n'
                "    total = order.total\n"
                "    return {'invoice_total': total}\n",
                encoding="utf-8",
            )

            config = CtxConfig(repo=repo)
            index_repo(config, embed=False)
            store = GraphStore(config.db_path)
            try:
                results = store.semantic_search("where invoices are created", 5)
            finally:
                store.close()

            self.assertTrue(results)
            self.assertEqual(results[0]["name"], "create_invoice")
            self.assertEqual(results[0]["score_source"], "hybrid")
            self.assertIn("snippet", results[0])

    def test_semantic_search_expands_morphology_and_typos(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            os.environ["CTX_HOME"] = str(Path(temp) / "home")
            repo = Path(temp) / "repo"
            repo.mkdir()
            (repo / "billing.py").write_text(
                'def create_invoice(order):\n'
                '    """Create a new invoice for a user order."""\n'
                "    return order\n",
                encoding="utf-8",
            )

            config = CtxConfig(repo=repo)
            index_repo(config, embed=False)
            store = GraphStore(config.db_path)
            try:
                creating = store.semantic_search("creating invoices", 5, group_by_file=False)
                creation = store.semantic_search("invoice creation", 5, group_by_file=False)
                typo = store.semantic_search("cretae invioce", 5, group_by_file=False)
            finally:
                store.close()

            self.assertTrue(any(row["name"] == "create_invoice" for row in creating))
            self.assertTrue(any(row["name"] == "create_invoice" for row in creation))
            self.assertTrue(any(row["name"] == "create_invoice" for row in typo))

    def test_semantic_search_filters_and_compacts_meta(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            os.environ["CTX_HOME"] = str(Path(temp) / "home")
            repo = Path(temp) / "repo"
            repo.mkdir()
            (repo / "billing.py").write_text("def create_invoice(order):\n    return order\n", encoding="utf-8")
            (repo / "users.py").write_text("def create_user(payload):\n    return payload\n", encoding="utf-8")

            config = CtxConfig(repo=repo)
            index_repo(config, embed=False)
            store = GraphStore(config.db_path)
            try:
                results = store.semantic_search("create", 5, kind="symbol", path_glob="billing.py", group_by_file=False)
                lexical = store.search("billing.py", 5)
            finally:
                store.close()

            self.assertTrue(results)
            self.assertTrue(all(row["kind"] == "symbol" and row["path"] == "billing.py" for row in results))
            self.assertNotIn("terms", lexical[0].get("meta", {}))
            file_hit = next(row for row in lexical if row["kind"] == "file")
            self.assertIn("exports", file_hit.get("meta", {}))

    def test_index_registers_repo_and_router_resolves_it(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            os.environ["CTX_HOME"] = str(Path(temp) / "home")
            os.environ.pop("CTX_REPO", None)
            repo_a = Path(temp) / "repo_a"
            repo_b = Path(temp) / "repo_b"
            repo_a.mkdir()
            repo_b.mkdir()
            (repo_a / "billing.py").write_text(
                'def create_invoice(order):\n    """Create a new invoice."""\n    return order\n',
                encoding="utf-8",
            )
            (repo_b / "users.py").write_text(
                'def create_user(payload):\n    """Create a user."""\n    return payload\n',
                encoding="utf-8",
            )

            index_repo(CtxConfig(repo=repo_a), embed=False)
            index_repo(CtxConfig(repo=repo_b), embed=False)

            registry = load_registry()
            self.assertEqual(set(registry.entries), {"repo_a", "repo_b"})
            self.assertIn(registry.default, {"repo_a", "repo_b"})

            server = McpServer(None)
            payload = server.tool_repos({})
            self.assertEqual({entry["name"] for entry in payload["repos"]}, {"repo_a", "repo_b"})

            response_a = server.tool_semantic({"query": "invoice", "repo": "repo_a", "limit": 3, "group_by_file": False})
            response_b = server.tool_semantic({"query": "user", "repo": str(repo_b), "limit": 3, "group_by_file": False})

            names_a = {row["name"] for row in response_a if isinstance(row, dict) and "name" in row}
            names_b = {row["name"] for row in response_b if isinstance(row, dict) and "name" in row}
            self.assertIn("create_invoice", names_a)
            self.assertIn("create_user", names_b)

            unknown = server.tool_semantic({"query": "anything", "repo": "no-such-repo"})
            self.assertIsInstance(unknown, dict)
            self.assertIn("error", unknown)

    def test_router_uses_ctx_repo_env_when_no_explicit_repo(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temp:
            os.environ["CTX_HOME"] = str(Path(temp) / "home")
            repo = Path(temp) / "envrepo"
            repo.mkdir()
            (repo / "auth.py").write_text(
                'def login(user):\n    """Authenticate a user."""\n    return user\n',
                encoding="utf-8",
            )
            index_repo(CtxConfig(repo=repo), embed=False)

            os.environ["CTX_REPO"] = str(repo)
            try:
                server = McpServer(None)
                response = server.tool_semantic({"query": "login", "limit": 3, "group_by_file": False})
            finally:
                os.environ.pop("CTX_REPO", None)

            names = {row["name"] for row in response if isinstance(row, dict) and "name" in row}
            self.assertIn("login", names)

    def test_voyage_key_wins_default_provider_detection(self) -> None:
        old_env = {key: os.environ.get(key) for key in ["OPENAI_API_KEY", "VOYAGE_API_KEY", "CTX_VOYAGE_API_KEY", "CTX_EMBED_PROVIDER"]}
        try:
            os.environ["OPENAI_API_KEY"] = "test-openai"
            os.environ["VOYAGE_API_KEY"] = "test-voyage"
            os.environ.pop("CTX_VOYAGE_API_KEY", None)
            os.environ.pop("CTX_EMBED_PROVIDER", None)

            provider = provider_from_env()

            self.assertEqual(provider.provider, "voyage")
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
