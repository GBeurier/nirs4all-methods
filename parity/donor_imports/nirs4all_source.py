"""Utilities for resolving the external nirs4all reference package.

Parity generators load selected nirs4all modules by source file to avoid
executing the heavy package ``__init__``. The package root can be provided with
``NIRS4ALL_ROOT``, discovered from a sibling checkout during local development,
or discovered from an installed ``nirs4all`` wheel in CI.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


def _package_root(path: Path) -> Path | None:
    path = path.expanduser()
    if (path / "__init__.py").exists() and path.name == "nirs4all":
        return path
    nested = path / "nirs4all"
    if (nested / "__init__.py").exists():
        return nested
    return None


def find_nirs4all_root() -> Path:
    candidates: list[Path] = []

    env_root = os.environ.get("NIRS4ALL_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root.parent / "nirs4all" / "nirs4all")

    spec = importlib.util.find_spec("nirs4all")
    if spec is not None:
        if spec.submodule_search_locations:
            candidates.extend(Path(p) for p in spec.submodule_search_locations)
        elif spec.origin:
            candidates.append(Path(spec.origin).parent)

    for candidate in candidates:
        root = _package_root(candidate)
        if root is not None:
            parent = str(root.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return root.resolve()

    raise FileNotFoundError(
        "nirs4all package root not found; set NIRS4ALL_ROOT or install "
        "nirs4all==0.9.1"
    )


def get_nirs4all_version(root: Path | None = None) -> str:
    init_py = (root or find_nirs4all_root()) / "__init__.py"
    if not init_py.exists():
        return "unknown"
    for line in init_py.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "unknown"


def load_nirs4all_source(root: Path, relative: str, name: str) -> Any:
    path = root / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
