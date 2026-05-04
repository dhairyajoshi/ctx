from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ctx_kg.config import CtxConfig
from ctx_kg.indexer import index_repo
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


if __name__ == "__main__":
    unittest.main()

