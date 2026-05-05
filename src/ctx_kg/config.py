from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_FILE = "ctx.config.json"
LOCAL_DIR = ".ctx"
DEFAULT_IGNORES = [
    ".git",
    ".ctx",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".turbo",
    "coverage",
    "vendor",
    CONFIG_FILE,
]
DEFAULT_EXTENSIONS = [".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".md"]


DEFAULT_EMBED: dict[str, Any] = {
    "auto": True,
    "provider": None,
    "model": None,
    "dimensions": None,
    "batch_size": 64,
}


@dataclass
class CtxConfig:
    repo: Path = Path(".")
    storage: str = "central"
    update: str | dict[str, Any] = "manual"
    include_extensions: list[str] = field(default_factory=lambda: DEFAULT_EXTENSIONS.copy())
    ignore: list[str] = field(default_factory=lambda: DEFAULT_IGNORES.copy())
    features: dict[str, list[str]] = field(default_factory=dict)
    embed: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_EMBED))

    @property
    def db_path(self) -> Path:
        return graph_path(self.repo, self)


def default_config() -> dict[str, Any]:
    return {
        "storage": "central",
        "update": {"mode": "manual", "interval_seconds": 300},
        "include_extensions": DEFAULT_EXTENSIONS,
        "ignore": DEFAULT_IGNORES,
        "features": {},
        "embed": dict(DEFAULT_EMBED),
    }


def load_config(repo: Path | None = None) -> CtxConfig:
    root = find_repo_root((repo or Path(".")).resolve())
    data = default_config()
    path = root / CONFIG_FILE
    if path.exists():
        data.update(json.loads(path.read_text(encoding="utf-8")))
    embed_data = data.get("embed", DEFAULT_EMBED)
    if not isinstance(embed_data, dict):
        embed_data = dict(DEFAULT_EMBED)
    embed_cfg = dict(DEFAULT_EMBED)
    embed_cfg.update(embed_data)
    return CtxConfig(
        repo=root,
        storage=data.get("storage", "central"),
        update=data.get("update", "manual"),
        include_extensions=list(data.get("include_extensions", data.get("extensions", DEFAULT_EXTENSIONS))),
        ignore=list(data.get("ignore", DEFAULT_IGNORES)),
        features=dict(data.get("features", {})),
        embed=embed_cfg,
    )


def write_config(repo: Path, storage: str = "central", update: str = "manual") -> Path:
    path = repo / CONFIG_FILE
    data = default_config()
    data["storage"] = storage
    data["update"] = update
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def write_default_config(repo: Path, storage: str = "central") -> Path:
    return write_config(repo, storage=storage)


def find_repo_root(path: Path) -> Path:
    current = path if path.is_dir() else path.parent
    for parent in [current, *current.parents]:
        if (parent / CONFIG_FILE).exists() or (parent / ".git").exists():
            return parent
    return current


def repo_key(repo: Path) -> str:
    return str(repo.resolve()).replace("/", "_").replace(":", "_").strip("_") or "repo"


def central_data_dir() -> Path:
    root = os.environ.get("CTX_HOME")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".ctx"


def graph_path(repo: Path, config: CtxConfig | None = None, storage: str | None = None) -> Path:
    cfg = config or load_config(repo)
    mode = storage or cfg.storage
    if mode == "repo":
        return repo / LOCAL_DIR / "graph.sqlite"
    if mode != "central":
        raise ValueError(f"unknown storage mode: {mode}")
    return central_data_dir() / "repos" / repo_key(repo) / "graph.sqlite"
