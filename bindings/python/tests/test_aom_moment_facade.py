"""Facade-consistency guard for the n4m.aom / n4m.moment product surfaces.

Both facades are thin re-export layers over the single ``libn4m`` runtime: every
method they advertise in ``available_methods()`` must resolve as a real attribute
of the facade and be the exact ``n4m.python`` / ``n4m.sklearn`` object — never a
copy and never a name with no backing export. This caught the n4m.aom ``pop_pls``
gap, where the inventory named ``NativePOPPLSRegressor`` but the facade did not
re-export it.
"""
from __future__ import annotations

from collections import defaultdict
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


def _rows_by_name(facade):
    return {row["name"]: row for row in _rows(facade)}


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


def _catalog_bench_registry_entry(text):
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


def _catalog_method_files():
    return sorted(_CATALOG_ROOT.glob("*.yaml"))


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_inventory_entries_resolve_and_are_exported(label):
    facade = _FACADES[label]
    for row in _rows(facade):
        name, entry = row["name"], row["entry"]
        assert hasattr(facade, entry), (label, name, entry)
        assert entry in facade.__all__, (label, name, entry)


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_inventory_declares_cpu_cuda_capability_flags(label):
    facade = _FACADES[label]
    for row in _rows(facade):
        name = row["name"]
        assert isinstance(row.get("cpu"), bool), (label, name, row.get("cpu"))
        assert isinstance(row.get("cuda"), bool), (label, name, row.get("cuda"))
        assert row["cpu"] or row["cuda"], (label, name)


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_facade_entries_are_the_underlying_objects(label):
    facade = _FACADES[label]
    for row in _rows(facade):
        name, entry, kind = row["name"], row["entry"], row["kind"]
        source = _SOURCE_FOR_KIND.get(kind, native)
        assert getattr(facade, entry) is getattr(source, entry), (label, name, entry, kind)


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_facade_entries_are_shared_with_top_level_package(label):
    facade = _FACADES[label]
    for row in _rows(facade):
        name, entry = row["name"], row["entry"]
        assert hasattr(n4m, entry), (label, name, entry)
        assert getattr(facade, entry) is getattr(n4m, entry), (label, name, entry)


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
def test_inventory_method_docs_are_linked_from_public_index(label):
    facade = _FACADES[label]
    index_text = (_REPO_ROOT / "docs" / "methods" / "index.md").read_text(
        encoding="utf-8"
    )
    for row in _rows(facade):
        doc_path = row.get("doc_path")
        if not isinstance(doc_path, str) or not doc_path.startswith("docs/methods/"):
            continue
        link = doc_path.removeprefix("docs/methods/")
        assert f"({link})" in index_text, (label, row["name"], link)


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_catalogued_inventory_methods_advertise_existing_timing_benchmarks(label):
    facade = _FACADES[label]
    for row in _rows(facade):
        catalog_id = row.get("catalog_id")
        if catalog_id is None:
            continue
        catalog_text = _catalog_path(catalog_id).read_text(encoding="utf-8")
        registry_entry = _catalog_bench_registry_entry(catalog_text)
        assert isinstance(registry_entry, str) and registry_entry, (
            label,
            row["name"],
            catalog_id,
        )
        assert registry_entry.startswith("benchmarks/cross_binding/"), (
            label,
            row["name"],
            catalog_id,
            registry_entry,
        )
        assert (_REPO_ROOT / registry_entry).is_file(), (
            label,
            row["name"],
            catalog_id,
            registry_entry,
        )


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


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_duplicate_catalog_ids_are_explicit_wrappers_or_aliases(label):
    facade = _FACADES[label]
    grouped = defaultdict(list)
    for row in _rows(facade):
        catalog_id = row.get("catalog_id")
        if catalog_id is not None:
            grouped[catalog_id].append(row)

    primary_roles = {"catalog_binding", "direct_native_binding"}
    secondary_roles = {
        "alias",
        "campaign_helper",
        "preset_function",
        "preset_sklearn_wrapper",
        "sklearn_wrapper",
    }
    for catalog_id, rows in grouped.items():
        if len(rows) <= 1:
            continue
        primaries = [row for row in rows if row.get("catalog_role") in primary_roles]
        assert len(primaries) <= 1, (label, catalog_id, rows)
        for row in rows:
            role = row.get("catalog_role")
            if role in primary_roles:
                continue
            assert role in secondary_roles, (label, catalog_id, row)
            assert row.get("wrapper_of"), (label, catalog_id, row)


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_inventory_wrapper_targets_are_exported_on_the_same_surface(label):
    facade = _FACADES[label]
    for row in _rows(facade):
        wrapper = row.get("wrapper_of")
        if not wrapper:
            continue
        assert hasattr(native, wrapper), (label, row["name"], wrapper)
        assert hasattr(n4m, wrapper), (label, row["name"], wrapper)
        assert hasattr(facade, wrapper), (label, row["name"], wrapper)
        assert wrapper in facade.__all__, (label, row["name"], wrapper)
        assert getattr(facade, wrapper) is getattr(native, wrapper), (
            label,
            row["name"],
            wrapper,
        )
        assert getattr(facade, wrapper) is getattr(n4m, wrapper), (
            label,
            row["name"],
            wrapper,
        )


def test_catalogued_native_aom_moment_bindings_are_exposed_on_a_facade():
    relevant_prefixes = (
        "aom_pop.",
        "utilities.",
        "models.ensembles.moment_stack",
        "models.pls.",
        "models.regularized.",
        "models.specialized.",
    )
    exposed_catalog_ids = {
        row["catalog_id"]
        for facade in _FACADES.values()
        for row in _rows(facade)
        if row.get("catalog_id")
    }
    missing = []
    for path in _catalog_method_files():
        text = path.read_text(encoding="utf-8")
        method_id = _catalog_method_id(text)
        if method_id is None or not method_id.startswith(relevant_prefixes):
            continue
        binding = _catalog_python_binding(text)
        if binding is None or binding["module"] != "n4m.python":
            continue
        if method_id not in exposed_catalog_ids:
            missing.append(method_id)

    assert missing == []


def test_aom_inventory_covers_reuse_presets_and_global_campaign_surfaces():
    rows = _rows_by_name(aom)
    required = {
        "screen_refit_campaign": "ultra_configurable_campaign",
        "moment_fast_screen_refit_campaign": "preconfigured_global_screen_refit",
        "staged_chain_campaign": "staged_screen_refit_orchestration",
        "staged_chain_campaign_estimator": "staged_screen_refit_estimator",
        "savgol_focus_regressor": "winning_preconfigured_screen_refit",
        "strict_family_lite_regressor": "cost_safe_family_audit",
        "aom_chain_fixed_fit": "winner_reuse",
        "fixed_candidate": "winner_reuse",
        "chain_sweep": "native_candidate_screen",
        "profile_sweep": "native_candidate_screen",
    }

    for name, role in required.items():
        row = rows[name]
        assert row["role"] == role
        assert row["cpu"] is True
        assert row["cuda"] is True
        assert row.get("doc_path"), (name, row)
        assert row.get("reuse"), (name, row)


def test_moment_inventory_covers_direct_heads_stack_and_aom_reuse_surfaces():
    rows = _rows_by_name(moment)
    required = {
        "moments",
        "sweep_run",
        "moment_sweep",
        "ridge",
        "ridge_regressor",
        "pls",
        "pls_regressor",
        "cppls",
        "cppls_regressor",
        "weighted_pls",
        "weighted_pls_regressor",
        "robust_pls",
        "robust_pls_regressor",
        "ridge_pls",
        "ridge_pls_regressor",
        "continuum_regression",
        "continuum_regression_regressor",
        "ecr",
        "ecr_regressor",
        "moment_stack",
        "moment_stack_regressor",
        "moment_mixed_screen_refit",
        "moment_pls_screen_refit",
        "moment_pls_exact_screen_refit",
        "moment_ridge_screen_refit",
        "aom_chain_fixed_fit",
        "fixed_candidate",
        "backend_recommendation",
    }

    missing = sorted(required - set(rows))
    assert missing == []
    for name in required:
        row = rows[name]
        assert row["cpu"] is True
        assert row["cuda"] is True
        assert row.get("reuse"), (name, row)


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_inventory_config_options_do_not_expose_dataset_identity_routing(label):
    facade = _FACADES[label]
    forbidden_exact = {
        "dataset",
        "dataset_name",
        "dataset_id",
        "source",
        "source_name",
        "source_id",
        "database",
        "database_name",
        "database_id",
        "metadata",
    }
    forbidden_fragments = (
        "dataset_",
        "_dataset",
        "source_",
        "_source",
        "database_",
        "_database",
        "route_by",
        "route_on",
    )
    for row in _rows(facade):
        options = tuple(row.get("config_options") or ())
        offenders = [
            option
            for option in options
            if option in forbidden_exact
            or any(fragment in option for fragment in forbidden_fragments)
        ]
        assert offenders == [], (label, row["name"], offenders)


@pytest.mark.parametrize("label", sorted(_FACADES))
def test_preset_sklearn_wrappers_are_bounded_reusable_presets(label):
    facade = _FACADES[label]
    fixed_by_preset = {"plan", "stages", "families", "templates"}
    presets = [
        row for row in _rows(facade)
        if row.get("catalog_role") == "preset_sklearn_wrapper"
    ]
    assert presets, label
    for row in presets:
        assert row["kind"] == "sklearn_estimator", (label, row["name"])
        assert row.get("wrapper_of"), (label, row["name"])
        assert isinstance(row.get("preset_plan"), str) and row["preset_plan"], (
            label,
            row["name"],
        )
        assert row["entry"] in facade.__all__, (label, row["name"], row["entry"])
        assert getattr(facade, row["entry"]) is getattr(native_sklearn, row["entry"]), (
            label,
            row["name"],
            row["entry"],
        )
        options = set(row.get("config_options") or ())
        assert options.isdisjoint(fixed_by_preset), (label, row["name"], options)


def test_pop_pls_wrapper_shared_across_surfaces():
    # The POP-PLS sklearn wrapper must be the same object on the aom facade, the
    # top-level n4m package, and the sklearn layer.
    assert aom.NativePOPPLSRegressor is n4m.NativePOPPLSRegressor
    assert aom.NativePOPPLSRegressor is native_sklearn.NativePOPPLSRegressor
