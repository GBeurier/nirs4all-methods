"""Catalog Python binding integrity checks."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CATALOG_METHODS = _REPO_ROOT / "catalog" / "methods"


def _yaml_scalar(line: str) -> str:
    stripped = line.strip()
    value = stripped[2:].strip() if stripped.startswith("- ") else line.split(":", 1)[1].strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _method_id(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("method_id:"):
            return _yaml_scalar(line)
    return None


def _python_binding(text: str) -> dict[str, object] | None:
    in_bindings = False
    in_python = False
    in_aliases = False
    module = None
    class_name = None
    aliases: list[str] = []
    for line in text.splitlines():
        if line.startswith("bindings:"):
            in_bindings = True
            in_python = False
            in_aliases = False
            continue
        if in_bindings and line and not line.startswith(" "):
            break
        if not in_bindings:
            continue
        if line.startswith("  python:"):
            in_python = True
            in_aliases = False
            continue
        if in_python and line.startswith("  ") and not line.startswith("    ") and line.strip():
            break
        if in_python and line.startswith("    module:"):
            module = _yaml_scalar(line)
            in_aliases = False
        elif in_python and line.startswith("    class:"):
            class_name = _yaml_scalar(line)
            in_aliases = False
        elif in_python and line.startswith("    legacy_aliases:"):
            in_aliases = True
        elif in_python and in_aliases and line.startswith("      - "):
            aliases.append(_yaml_scalar(line))
    if module and class_name:
        return {"module": module, "class": class_name, "legacy_aliases": tuple(aliases)}
    return None


def _bench_registry_entry(text: str) -> str | None:
    in_bench = False
    for line in text.splitlines():
        if line.startswith("bench:"):
            in_bench = True
            continue
        if in_bench and line and not line.startswith(" "):
            break
        if in_bench and line.startswith("  registry_entry:"):
            value = _yaml_scalar(line)
            return None if value == "null" else value
    return None


def test_catalog_python_bindings_resolve_to_real_exports():
    checked = 0
    for path in sorted(_CATALOG_METHODS.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        binding = _python_binding(text)
        if binding is None:
            continue
        checked += 1
        method_id = _method_id(text)
        module = importlib.import_module(str(binding["module"]))
        class_name = str(binding["class"])
        assert hasattr(module, class_name), (method_id, path.name, binding)
        for alias in binding["legacy_aliases"]:
            assert hasattr(module, alias), (method_id, path.name, alias)
    assert checked > 0


def test_methods_index_total_matches_catalog_file_count():
    index = (_REPO_ROOT / "docs/methods/index.md").read_text(encoding="utf-8")
    match = re.search(r"_Total catalogued native methods_: \*\*(\d+)\*\*", index)
    assert match is not None
    assert int(match.group(1)) == len(list(_CATALOG_METHODS.glob("*.yaml")))


def test_legacy_catalog_validator_accepts_current_catalog_schema():
    script = _REPO_ROOT / "catalog" / "scripts" / "validate_catalog.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_direct_moment_heads_advertise_the_shared_timing_benchmark():
    expected = "benchmarks/cross_binding/bench_direct_moment_heads_timing.py"
    method_ids = {
        "models.regularized.ridge",
        "models.pls.pls_fit_simple",
        "models.pls.pcr",
        "models.pls.cppls",
        "models.regularized.weighted_pls",
        "models.regularized.robust_pls",
        "models.regularized.ridge_pls",
        "models.regularized.continuum_regression",
        "models.specialized.ecr",
    }
    for method_id in method_ids:
        path = _CATALOG_METHODS / f"{method_id}.yaml"
        text = path.read_text(encoding="utf-8")
        assert _method_id(text) == method_id
        assert _bench_registry_entry(text) == expected
        assert (_REPO_ROOT / expected).is_file()


def test_aom_pls_selectors_advertise_the_selector_timing_benchmark():
    expected = "benchmarks/cross_binding/bench_aom_selector_timing.py"
    for method_id in {"aom_pop.aom_pls", "aom_pop.pop_pls"}:
        path = _CATALOG_METHODS / f"{method_id}.yaml"
        text = path.read_text(encoding="utf-8")
        assert _method_id(text) == method_id
        assert _bench_registry_entry(text) == expected
        assert (_REPO_ROOT / expected).is_file()


def test_aom_preprocessing_advertises_the_direct_operator_timing_benchmark():
    expected = "benchmarks/cross_binding/bench_aom_preprocess_timing.py"
    path = _CATALOG_METHODS / "aom_pop.aom_preprocessing.yaml"
    text = path.read_text(encoding="utf-8")
    assert _method_id(text) == "aom_pop.aom_preprocessing"
    assert _bench_registry_entry(text) == expected
    assert (_REPO_ROOT / expected).is_file()


def test_aom_ridge_superblock_advertises_its_timing_benchmark():
    expected = "benchmarks/cross_binding/bench_aom_ridge_superblock_timing.py"
    path = _CATALOG_METHODS / "aom_pop.ridge_superblock.yaml"
    text = path.read_text(encoding="utf-8")
    assert _method_id(text) == "aom_pop.ridge_superblock"
    assert _bench_registry_entry(text) == expected
    assert (_REPO_ROOT / expected).is_file()


def test_aom_ridge_active_superblock_advertises_its_timing_benchmark():
    expected = "benchmarks/cross_binding/bench_aom_ridge_active_superblock_timing.py"
    path = _CATALOG_METHODS / "aom_pop.ridge_active_superblock.yaml"
    text = path.read_text(encoding="utf-8")
    assert _method_id(text) == "aom_pop.ridge_active_superblock"
    assert _bench_registry_entry(text) == expected
    assert (_REPO_ROOT / expected).is_file()


def test_aom_ridge_mkl_superblock_advertises_its_timing_benchmark():
    expected = "benchmarks/cross_binding/bench_aom_ridge_mkl_superblock_timing.py"
    path = _CATALOG_METHODS / "aom_pop.ridge_mkl_superblock.yaml"
    text = path.read_text(encoding="utf-8")
    assert _method_id(text) == "aom_pop.ridge_mkl_superblock"
    assert _bench_registry_entry(text) == expected
    assert (_REPO_ROOT / expected).is_file()


def test_aom_pls_superblock_advertises_its_timing_benchmark():
    expected = "benchmarks/cross_binding/bench_aom_pls_superblock_timing.py"
    path = _CATALOG_METHODS / "aom_pop.aom_pls_superblock.yaml"
    text = path.read_text(encoding="utf-8")
    assert _method_id(text) == "aom_pop.aom_pls_superblock"
    assert _bench_registry_entry(text) == expected
    assert (_REPO_ROOT / expected).is_file()


def test_aom_ridge_pls_superblock_advertises_its_timing_benchmark():
    expected = "benchmarks/cross_binding/bench_aom_ridge_pls_superblock_timing.py"
    path = _CATALOG_METHODS / "aom_pop.aom_ridge_pls_superblock.yaml"
    text = path.read_text(encoding="utf-8")
    assert _method_id(text) == "aom_pop.aom_ridge_pls_superblock"
    assert _bench_registry_entry(text) == expected
    assert (_REPO_ROOT / expected).is_file()


def test_aom_ridge_global_advertises_its_timing_benchmark():
    expected = "benchmarks/cross_binding/bench_aom_ridge_global_timing.py"
    path = _CATALOG_METHODS / "aom_pop.ridge_global.yaml"
    text = path.read_text(encoding="utf-8")
    assert _method_id(text) == "aom_pop.ridge_global"
    assert _bench_registry_entry(text) == expected
    assert (_REPO_ROOT / expected).is_file()
