from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_FILE = "ctx.config.json"
LOCAL_DIR = ".ctx"
REGISTRY_FILE = "repos.json"
DEFAULT_IGNORES = [
    ".git",
    ".ctx",
    ".venv",
    "venv",
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
    configured_ignore = list(data.get("ignore", DEFAULT_IGNORES))
    for pattern in DEFAULT_IGNORES:
        if pattern not in configured_ignore:
            configured_ignore.append(pattern)
    return CtxConfig(
        repo=root,
        storage=data.get("storage", "central"),
        update=data.get("update", "manual"),
        include_extensions=list(data.get("include_extensions", data.get("extensions", DEFAULT_EXTENSIONS))),
        ignore=configured_ignore,
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


@dataclass
class RegistryEntry:
    name: str
    path: Path
    db: Path
    indexed_at: int | None = None


@dataclass
class Registry:
    entries: dict[str, RegistryEntry] = field(default_factory=dict)
    default: str | None = None

    def by_path(self, path: Path) -> RegistryEntry | None:
        try:
            target = path.resolve()
        except Exception:
            return None
        for entry in self.entries.values():
            try:
                if entry.path.resolve() == target:
                    return entry
            except Exception:
                continue
        return None

    def resolve(self, identifier: str) -> RegistryEntry | None:
        if not identifier:
            return None
        if identifier in self.entries:
            return self.entries[identifier]
        try:
            return self.by_path(Path(identifier).expanduser())
        except Exception:
            return None


def registry_path() -> Path:
    return central_data_dir() / REGISTRY_FILE


def load_registry() -> Registry:
    path = registry_path()
    if not path.exists():
        return Registry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return Registry()
    entries: dict[str, RegistryEntry] = {}
    for name, raw in (data.get("repos") or {}).items():
        path_value = raw.get("path") if isinstance(raw, dict) else None
        db_value = raw.get("db") if isinstance(raw, dict) else None
        if not path_value or not db_value:
            continue
        entries[name] = RegistryEntry(
            name=name,
            path=Path(path_value),
            db=Path(db_value),
            indexed_at=raw.get("indexed_at") if isinstance(raw, dict) else None,
        )
    default = data.get("default") if isinstance(data, dict) else None
    if default not in entries:
        default = next(iter(entries), None)
    return Registry(entries=entries, default=default)


def save_registry(registry: Registry) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repos": {
            name: {
                "path": str(entry.path),
                "db": str(entry.db),
                **({"indexed_at": entry.indexed_at} if entry.indexed_at is not None else {}),
            }
            for name, entry in registry.entries.items()
        },
        "default": registry.default,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def register_repo(repo: Path, name: str | None = None, indexed_at: int | None = None) -> RegistryEntry:
    repo_path = repo.resolve()
    config = load_config(repo_path)
    db = config.db_path
    registry = load_registry()
    existing = registry.by_path(repo_path)
    chosen_name = name or (existing.name if existing else suggest_repo_name(repo_path, registry))
    for old_name in [n for n, e in registry.entries.items() if e.path.resolve() == repo_path and n != chosen_name]:
        del registry.entries[old_name]
    registry.entries[chosen_name] = RegistryEntry(
        name=chosen_name,
        path=repo_path,
        db=db,
        indexed_at=indexed_at if indexed_at is not None else (existing.indexed_at if existing else None),
    )
    if registry.default is None or registry.default not in registry.entries:
        registry.default = chosen_name
    save_registry(registry)
    return registry.entries[chosen_name]


def suggest_repo_name(repo: Path, registry: Registry) -> str:
    base = repo.name or "repo"
    candidate = base
    suffix = 2
    while candidate in registry.entries and registry.entries[candidate].path.resolve() != repo.resolve():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def unregister_repo(identifier: str) -> bool:
    registry = load_registry()
    entry = registry.resolve(identifier)
    if not entry:
        return False
    del registry.entries[entry.name]
    if registry.default == entry.name:
        registry.default = next(iter(registry.entries), None)
    save_registry(registry)
    return True


def find_repo_for_cwd(cwd: Path | None = None) -> RegistryEntry | None:
    target = (cwd or Path.cwd()).resolve()
    registry = load_registry()
    for parent in [target, *target.parents]:
        entry = registry.by_path(parent)
        if entry:
            return entry
    root = find_repo_root(target)
    if root != target:
        entry = registry.by_path(root)
        if entry:
            return entry
    return None
