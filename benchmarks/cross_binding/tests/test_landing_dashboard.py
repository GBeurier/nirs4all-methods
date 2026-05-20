from __future__ import annotations

import csv
import json
from pathlib import Path

from docs._extras import build_landing


def test_missing_reference_oracle_is_no_compare_not_divergent() -> None:
    row = {
        "algorithm": "pcr",
        "kind": "pls4all_core",
        "ok": "True",
        "reference_parity_ok": "False",
        "reference_parity_rmse_rel": "nan",
        "reference_parity_note": (
            "reference oracle missing; run canonical reference backend"
        ),
    }

    fields = build_landing.divergence_fields(row)

    assert build_landing.reference_parity_code(row) == "not_available"
    assert fields["divergence"] is None
    assert fields["divergence_status"] == "no_compare"


def test_missing_binding_gate_is_not_reported_as_drift() -> None:
    row = {
        "algorithm": "pcr",
        "kind": "pls4all_binding",
        "ok": "True",
        "binding_parity_ok": "",
        "binding_parity_max_diff": "",
    }

    assert build_landing.parity_code(row) == "not_available"
    assert build_landing.divergence_fields(row)["divergence_status"] == (
        "no_compare"
    )


def test_reference_release_policy_overrides_loose_method_tolerance() -> None:
    row = {
        "algorithm": "rosa",
        "kind": "pls4all_core",
        "ok": "True",
        "reference_parity_ok": "True",
        "reference_parity_rmse_abs": "0.1",
        "reference_parity_rmse_rel": "0.1",
        "reference_parity_tolerance": "1.0",
        "reference_kind": "external",
    }

    fields = build_landing.divergence_fields(row)

    assert build_landing.reference_parity_code(row) == "divergent"
    assert fields["divergence"] == 0.1
    assert fields["divergence_basis"] == "reference"
    assert fields["divergence_status"] == "divergent"


def test_reference_divergence_wins_over_zero_binding_diff() -> None:
    row = {
        "algorithm": "so_pls",
        "kind": "pls4all_binding",
        "ok": "True",
        "binding_parity_ok": "True",
        "binding_parity_max_diff": "0",
        "reference_parity_ok": "False",
        "reference_parity_rmse_abs": "1.0",
        "reference_parity_rmse_rel": "1.3739319604628795",
        "reference_parity_tolerance": "1.0",
        "reference_kind": "external",
    }

    fields = build_landing.divergence_fields(row)

    assert build_landing.reference_parity_code(row) == "divergent"
    assert fields["divergence"] == 1.3739319604628795
    assert fields["divergence_basis"] == "reference"
    assert fields["divergence_status"] == "divergent"


def test_ref_alias_gate_metadata_wins_over_legacy_missing_gate(
        tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    path = results / "full_matrix.csv"
    fieldnames = [
        "algorithm", "backend", "language", "tier", "kind", "n", "p",
        "threads", "libp4a_build", "seed_base", "prediction_seed", "ok",
        "reported_ms", "median_ms", "timing_statistic", "parity_ref",
        "parity_max_diff", "parity_ok", "parity_note",
        "binding_parity_ref", "binding_parity_max_diff",
        "binding_parity_ok", "binding_parity_note",
        "reference_parity_ref", "reference_parity_rmse_abs",
        "reference_parity_rmse_rel", "reference_parity_ok",
        "reference_parity_note", "reference_parity_tolerance",
        "reference_oracle_path", "reference_kind", "is_canonical_reference",
        "predictions_path", "versions_json", "reason",
    ]
    base = {
        "algorithm": "pcr",
        "language": "R",
        "tier": "external",
        "kind": "external",
        "n": "200",
        "p": "50",
        "threads": "1",
        "libp4a_build": "blas-omp",
        "seed_base": "123",
        "prediction_seed": "123",
        "ok": "True",
        "timing_statistic": "median",
        "reference_parity_ref": "ref_python_scikit_learn",
        "reference_parity_tolerance": "0.1",
        "reference_kind": "external",
        "is_canonical_reference": "False",
        "versions_json": json.dumps({"timing_schema": "adaptive-v1"}),
    }
    rows = [
        {
            **base,
            "backend": "r_pls",
            "reported_ms": "12",
            "median_ms": "12",
            "reference_parity_rmse_abs": "nan",
            "reference_parity_rmse_rel": "nan",
            "reference_parity_ok": "False",
            "reference_parity_note": (
                "reference oracle missing; run canonical reference backend"
            ),
        },
        {
            **base,
            "backend": "ref_r_pls",
            "reported_ms": "13",
            "median_ms": "13",
            "reference_parity_rmse_abs": "1e-14",
            "reference_parity_rmse_rel": "1e-14",
            "reference_parity_ok": "True",
            "reference_parity_note": "",
        },
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = build_landing.build_payload(results)["payload"]
    cell = payload["rows"][0]["cells"]["pls"]

    assert cell["ms"] == 12
    assert cell["reference_parity"] == "exact"
    assert cell["divergence"] == 1e-14
    assert cell["divergence_status"] == "exact"


def test_strict_gate_csv_replaces_stale_dashboard_refresh(
        tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    fieldnames = [
        "algorithm", "backend", "language", "tier", "kind", "n", "p",
        "threads", "libp4a_build", "seed_base", "prediction_seed", "ok",
        "reported_ms", "median_ms", "timing_statistic", "binding_parity_ok",
        "binding_parity_max_diff", "reference_parity_ref",
        "reference_parity_rmse_abs", "reference_parity_rmse_rel",
        "reference_parity_ok", "reference_parity_tolerance",
        "reference_kind", "is_canonical_reference", "versions_json",
    ]
    stale = {
        "algorithm": "pls_qda",
        "backend": "cpp",
        "language": "C++",
        "tier": "canonical",
        "kind": "pls4all_core",
        "n": "100",
        "p": "50",
        "threads": "1",
        "libp4a_build": "blas-omp",
        "seed_base": "123",
        "prediction_seed": "123",
        "ok": "True",
        "reported_ms": "1",
        "median_ms": "1",
        "timing_statistic": "median",
        "binding_parity_ok": "True",
        "binding_parity_max_diff": "0",
        "reference_parity_ref": "ref_python_scikit_learn",
        "reference_parity_rmse_abs": "3",
        "reference_parity_rmse_rel": "3",
        "reference_parity_ok": "False",
        "reference_parity_tolerance": "0.001",
        "reference_kind": "external",
        "is_canonical_reference": "False",
        "versions_json": json.dumps({"timing_schema": "adaptive-v1"}),
    }
    strict = {
        **stale,
        "n": "30",
        "p": "30",
        "reported_ms": "2",
        "median_ms": "2",
        "reference_parity_rmse_abs": "1e-14",
        "reference_parity_rmse_rel": "1e-14",
        "reference_parity_ok": "True",
    }
    for name, row in (
        ("dashboard_refresh_stale.csv", stale),
        ("parity_30x30_strict.csv", strict),
    ):
        with (results / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    payload = build_landing.build_payload(results)["payload"]

    assert [(r["algo"], r["n"], r["p"]) for r in payload["rows"]] == [
        ("pls_qda", 30, 30)
    ]
    cell = payload["rows"][0]["cells"]["pls4all.cpp.blas+omp"]
    assert cell["divergence"] == 1e-14
    assert cell["reference_parity"] == "exact"
