"""Regression tests for the evidence-bound V1 capability matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "docs" / "_extras" / "generate_v1_capability_matrix.py"
MATRIX_PATH = (
    ROOT / "docs" / "architecture" / "v1_estimator_stateful_preprocessor_matrix.json"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "v1_capability_matrix", GENERATOR_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_committed_matrix_is_exactly_regenerated_and_schema_valid() -> None:
    generator = _module()
    matrix = _matrix()
    generator.validate_matrix(matrix)
    assert matrix == generator.build_matrix()


def test_render_does_not_depend_on_head_revision() -> None:
    generator = _module()
    before = generator._serialized(generator.build_matrix())
    source = GENERATOR_PATH.read_text(encoding="utf-8")
    # A different HEAD can only affect rendering if it is queried by the
    # generator or declared in its evidence digest. Neither is permitted.
    assert "rev-parse" not in source
    assert "source_revision" not in source
    assert before == generator._serialized(generator.build_matrix())


def test_input_digest_covers_headers_catalog_abi_ffi_and_policy() -> None:
    generator = _module()
    paths = {
        record["path"]
        for record in generator.build_matrix()["generation"]["input_digest"]["files"]
    }
    assert generator._relative(generator.PYTHON_FFI) in paths
    assert generator._relative(generator.POLICY) in paths
    assert {
        generator._relative(path) for path in generator.ABI_SNAPSHOTS.values()
    } <= paths
    assert {
        generator._relative(path) for path in generator.HEADERS.rglob("*.h")
    } <= paths
    assert {
        generator._relative(path) for path in generator.CATALOG.glob("*.yaml")
    } <= paths


def test_catalog_model_universe_is_exact_and_has_no_unknown_native_symbol() -> None:
    generator = _module()
    catalog = generator._catalog_models()
    matrix_models = [row for row in _matrix()["rows"] if row["kind"] == "model"]
    assert {entry["method_id"] for entry in catalog} == {
        row["id"] for row in matrix_models
    }
    headers = generator._header_index(
        generator._header_symbols(generator.HEADERS.rglob("*.h"))
    )
    assert {
        s for entry in catalog for s in entry["abi_symbols"]
    } == generator._model_header_symbols(headers)
    rows = {row["id"]: row for row in matrix_models}
    for identifier in (
        "models.ensembles.bagging_pls",
        "models.ensembles.boosting_pls",
        "models.ensembles.random_subspace_pls",
        "models.transfer.di_pls",
        "models.transfer.ds",
        "models.transfer.pds",
    ):
        assert rows[identifier]["classification"] == "native"
    assert rows["models.ensembles.moment_stack"]["classification"] == "adapter"
    assert rows["models.ensembles.moment_stack"]["ffi_proof"]["declarations"].endswith(
        "_ffi_decls.py"
    )


def test_native_rows_require_all_three_abi_snapshots() -> None:
    generator = _module()
    snapshots = generator._snapshot_symbols()
    for row in _matrix()["rows"]:
        if row["classification"] == "native":
            assert set(row["abi_proof"]) == {"linux", "macos", "windows"}
            assert all(
                set(row["api_symbols"]) <= symbols for symbols in snapshots.values()
            )


def test_missing_symbol_fails_closed_for_native_and_adapter_proofs(
    tmp_path: Path,
) -> None:
    generator = _module()
    headers = generator._header_index(
        generator._header_symbols(generator.HEADERS.rglob("*.h"))
    )
    snapshots = generator._snapshot_symbols()
    snapshots["windows"] = set(snapshots["windows"])
    snapshots["windows"].remove("n4m_model_fit")
    with pytest.raises(generator.MatrixError, match="missing windows ABI"):
        generator._native_row(
            "mutated", "generic_model_api", ["n4m_model_fit"], headers, snapshots
        )

    ffi = generator.PYTHON_FFI.read_text(encoding="utf-8")
    missing_ffi = tmp_path / "_ffi_decls.py"
    missing_ffi.write_text(
        ffi.replace('"n4m_model_selection_sweep_run"', '"n4m_removed"', 1),
        encoding="utf-8",
    )
    original_ffi = generator.PYTHON_FFI
    generator.PYTHON_FFI = missing_ffi
    try:
        with pytest.raises(generator.MatrixError, match="FFI proof"):
            generator._moment_stack_adapter(generator._snapshot_symbols())
    finally:
        generator.PYTHON_FFI = original_ffi


def test_newly_discovered_unclassified_model_api_fails_closed() -> None:
    generator = _module()
    headers = generator._header_index(
        generator._header_symbols(generator.HEADERS.rglob("*.h"))
    )
    headers["n4m_estimators_newly_discovered_fit"] = [
        "cpp/include/n4m/estimators/regression.h"
    ]
    with pytest.raises(generator.MatrixError, match="model universe mismatch"):
        generator._model_rows(headers, generator._snapshot_symbols())


def test_stateful_rows_include_complete_public_lifecycle_symbols() -> None:
    rows = {row["id"]: row for row in _matrix()["rows"]}
    expected = {
        "domain_adaptation_epo": {
            "n4m_domain_adaptation_epo_transform_with_d",
            "n4m_domain_adaptation_epo_inverse_transform",
        },
        "transform_msc": {"n4m_transform_msc_inverse_transform"},
        "transform_baseline_center": {
            "n4m_transform_baseline_center_inverse_transform"
        },
        "transform_osc": {"n4m_transform_osc_inverse_transform"},
        "outlier_detection_high_leverage": {
            "n4m_outlier_detection_high_leverage_threshold"
        },
    }
    for identifier, symbols in expected.items():
        assert symbols <= set(rows[identifier]["api_symbols"])
    assert (
        rows["domain_adaptation_epo"]["lifecycle"]["transform_with_d"]
        == "n4m_domain_adaptation_epo_transform_with_d"
    )
    assert (
        rows["domain_adaptation_epo"]["lifecycle"]["inverse_transform"]
        == "n4m_domain_adaptation_epo_inverse_transform"
    )


def test_schema_rejects_duplicate_ids_and_unproved_native_rows() -> None:
    generator = _module()
    matrix = generator.build_matrix()
    matrix["rows"].append(dict(matrix["rows"][0]))
    with pytest.raises(generator.MatrixError, match="row IDs are not unique"):
        generator.validate_matrix(matrix)
    matrix = generator.build_matrix()
    native = next(row for row in matrix["rows"] if row["classification"] == "native")
    del native["abi_proof"]
    with pytest.raises(generator.MatrixError, match="schema validation"):
        generator.validate_matrix(matrix)
    matrix = generator.build_matrix()
    matrix["rows"][0]["invented_capability"] = True
    with pytest.raises(generator.MatrixError, match="schema validation"):
        generator.validate_matrix(matrix)
