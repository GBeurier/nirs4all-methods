"""Facade-consistency guard for the n4m.aom / n4m.moment product surfaces.

Both facades are thin re-export layers over the single ``libn4m`` runtime: every
method they advertise in ``available_methods()`` must resolve as a real attribute
of the facade and be the exact ``n4m.python`` / ``n4m.sklearn`` object — never a
copy and never a name with no backing export. This caught the n4m.aom ``pop_pls``
gap, where the inventory named ``NativePOPPLSRegressor`` but the facade did not
re-export it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import n4m
import n4m.aom as aom
import n4m.moment as moment
from n4m import python as native
from n4m import sklearn as native_sklearn

_FACADES = {"n4m.aom": aom, "n4m.moment": moment}
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CATALOG_ROOT = _REPO_ROOT / "catalog" / "methods"

# Where each inventory ``kind`` is sourced from: sklearn estimators come from the
# scikit-learn layer, every other kind is an ABI-close function on ``n4m.python``.
_SOURCE_FOR_KIND = {"sklearn_estimator": native_sklearn}


def _rows(facade):
    return facade.available_methods()


def _catalog_path(catalog_id):
    return _CATALOG_ROOT / f"{catalog_id}.yaml"


def _yaml_scalar(line):
    stripped = line.strip()
    if stripped.startswith("- "):
        value = stripped[2:].strip()
    else:
        value = line.split(":", 1)[1].strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _catalog_method_id(text):
    for line in text.splitlines():
        if line.startswith("method_id:"):
            return _yaml_scalar(line)
    return None


def _catalog_python_binding(text):
    in_bindings = False
    in_python = False
    in_aliases = False
    module = None
    class_name = None
    aliases = []
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


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_inventory_entries_resolve_and_are_exported(label):
    facade = _FACADES[label]
    for row in _rows(facade):
        name, entry = row["name"], row["entry"]
        assert hasattr(facade, entry), (label, name, entry)
        assert entry in facade.__all__, (label, name, entry)


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_facade_entries_are_the_underlying_objects(label):
    facade = _FACADES[label]
    for row in _rows(facade):
        name, entry, kind = row["name"], row["entry"], row["kind"]
        source = _SOURCE_FOR_KIND.get(kind, native)
        assert getattr(facade, entry) is getattr(source, entry), (label, name, entry, kind)


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_inventory_catalog_and_docs_are_resolvable(label):
    facade = _FACADES[label]
    for row in _rows(facade):
        catalog_id = row.get("catalog_id")
        if catalog_id is None:
            assert row.get("non_catalog_reason"), (label, row["name"])
            continue

        catalog_path = _catalog_path(catalog_id)
        assert catalog_path.exists(), (label, row["name"], catalog_path)
        catalog_text = catalog_path.read_text(encoding="utf-8")
        assert _catalog_method_id(catalog_text) == catalog_id

        doc_path = row.get("doc_path")
        assert isinstance(doc_path, str) and doc_path, (label, row["name"])
        assert (_REPO_ROOT / doc_path).exists(), (label, row["name"], doc_path)
        assert row.get("catalog_role"), (label, row["name"], catalog_id)


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_inventory_catalog_binding_roles_are_coherent(label):
    facade = _FACADES[label]
    for row in _rows(facade):
        catalog_id = row.get("catalog_id")
        if catalog_id is None:
            continue

        catalog_text = _catalog_path(catalog_id).read_text(encoding="utf-8")
        binding = _catalog_python_binding(catalog_text)
        assert binding is not None, (label, row["name"], catalog_id)

        expected_class = row.get("wrapper_of", row["entry"])
        assert binding["module"] == "n4m.python", (
            label,
            row["name"],
            row["entry"],
            row.get("catalog_role"),
            binding,
        )
        assert binding["class"] == expected_class, (
            label,
            row["name"],
            row["entry"],
            row.get("catalog_role"),
            binding,
        )
        assert hasattr(native, binding["class"]), (label, row["name"], binding)
        for alias in binding["legacy_aliases"]:
            assert hasattr(native, alias), (label, row["name"], alias)
        if row.get("catalog_role") == "alias":
            assert row["entry"] in binding["legacy_aliases"], (label, row["name"], binding)


def test_pop_pls_wrapper_shared_across_surfaces():
    # The POP-PLS sklearn wrapper must be the same object on the aom facade, the
    # top-level n4m package, and the sklearn layer.
    assert aom.NativePOPPLSRegressor is n4m.NativePOPPLSRegressor
    assert aom.NativePOPPLSRegressor is native_sklearn.NativePOPPLSRegressor
