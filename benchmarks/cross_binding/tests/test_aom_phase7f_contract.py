"""Phase 7f AOM robust-HPO integration contract.

This is a pre-implementation guard. It keeps the proposed AOM portfolio
integration aligned with the stop/go audit: no deployable route may depend on
dataset/source identity, and source-lookup rows must stay explicitly forbidden.
"""
from __future__ import annotations

import csv
import importlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "docs" / "architecture" / "aom_robust_hpo_integration_manifest.csv"
CATALOG_READY = REPO / "docs" / "architecture" / "aom_robust_hpo_catalog_ready_manifest.csv"
DASHBOARD = REPO / "docs" / "architecture" / "aom_robust_hpo_phase7f_dashboard.csv"
CATALOG = REPO / "catalog" / "methods.yaml"
PHASE_DOC = REPO / "roadmap" / "phase-7f-aom-robust-hpo-portfolio.md"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _rows() -> list[dict[str, str]]:
    return _csv_rows(MANIFEST)


def test_phase7f_manifest_is_present_and_parseable():
    rows = _rows()
    assert len(rows) >= 10
    assert {row["integration_target"] for row in rows} >= {
        "AOMStructuralPolicy",
        "AOMEndpointMarginStabilityGate",
        "AOMControlSelector",
        "source_route_or_dataset_lookup",
    }


def test_phase7f_dashboard_covers_pre_abi_python_references():
    catalog_rows = _csv_rows(CATALOG_READY)
    dashboard_rows = _csv_rows(DASHBOARD)
    dashboard_by_class = {row["python_class"]: row for row in dashboard_rows}
    python_refs = [
        row for row in catalog_rows
        if row["current_status"] == "python_reference"
    ]
    assert python_refs
    for row in python_refs:
        dashboard = dashboard_by_class.get(row["python_class"])
        assert dashboard is not None
        assert dashboard["method_id"] == row["method_id"]
        assert dashboard["catalog_status"] == "pre_abi_do_not_add_to_catalog"
        assert dashboard["phase_status"] == "python_reference_contract_tested"


def test_phase7f_python_references_are_exported():
    n4m = importlib.import_module("n4m")
    sklearn_surface = importlib.import_module("n4m.sklearn")
    for row in _csv_rows(CATALOG_READY):
        if row["current_status"] != "python_reference":
            continue
        class_name = row["python_class"]
        assert hasattr(n4m, class_name), class_name
        assert hasattr(sklearn_surface, class_name), class_name


def test_pre_abi_methods_are_not_in_canonical_catalog():
    text = CATALOG.read_text()
    assert "abi_symbols" in text
    for row in _csv_rows(CATALOG_READY):
        assert row["catalog_status"] == "pre_abi_do_not_add_to_catalog"
        assert row["method_id"] not in text


def test_source_lookup_is_never_integrable():
    rows = _rows()
    source_rows = [
        row for row in rows
        if row["integration_target"] == "source_route_or_dataset_lookup"
    ]
    assert len(source_rows) == 1
    source_row = source_rows[0]
    assert source_row["status"] == "do_not_integrate"
    assert source_row["no_name_selection_gate"] == "fail"


def test_integrable_rows_have_no_name_selection_gate():
    rows = _rows()
    integrable = [
        row for row in rows
        if row["status"] in {
            "integrate",
            "integrate_after_rename",
            "integrate_baseline",
            "integrate_guarded",
            "optional",
            "diagnostic_control",
        }
    ]
    assert integrable
    for row in integrable:
        gate = row["no_name_selection_gate"].lower()
        assert "dataset" not in gate or "must not change" in gate or "no dataset" in gate
        assert "source" not in gate or "no dataset/source" in gate or "source_free" in row["route_basis"]
        assert row["status"] != "do_not_integrate"


def test_phase_doc_declares_metadata_invariance_gate():
    text = PHASE_DOC.read_text()
    normalized = " ".join(text.split())
    assert "Deployable routing must not depend on dataset name" in text
    assert "Changing them must not change the selected route or predictions" in normalized
    assert "metadata-invariance tests for all routed selectors" in text
    assert "source-route lookup | do not integrate" in text
