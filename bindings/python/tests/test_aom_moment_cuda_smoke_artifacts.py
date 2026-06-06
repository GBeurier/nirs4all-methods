"""Release-readiness checks for committed AOM/moment CUDA smoke artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH = _REPO_ROOT / "benchmarks" / "cross_binding"
_AOM_ARTIFACT_ABI = "1.22.0"


def _rows(name: str) -> list[dict[str, str]]:
    path = _BENCH / name
    assert path.exists(), path
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, path
    return rows


def _only_row(name: str) -> dict[str, str]:
    rows = _rows(name)
    assert len(rows) == 1, name
    return rows[0]


def _int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _bool(row: dict[str, str], key: str) -> bool:
    return row[key] == "True"


def _assert_cuda_build_path(path: str) -> None:
    assert "build/cuda-on" in path, path


def _assert_dev_release_build_path(path: str) -> None:
    assert "build/dev-release" in path, path


def _assert_current_aom_artifact_abi(row: dict[str, str]) -> None:
    assert row.get("abi") == _AOM_ARTIFACT_ABI


def _assert_direct_moment_heads_rows(
    rows: list[dict[str, str]],
    *,
    expect_cuda_pls: bool,
) -> None:
    expected_methods = {
        "ridge",
        "pls",
        "pcr",
        "cppls",
        "weighted_pls",
        "robust_pls",
        "ridge_pls",
        "continuum_regression",
        "ecr",
    }
    expected_backends = {"native_function", "sklearn_fit_predict"}
    required_route_fields = {
        "cuda_pls_parallel_folds",
        "cuda_pls_min_device_features",
        "cuda_pls_many_batched",
        "n_pls_moment_cv_fits",
        "n_pls_moment_host_cv_fits",
        "n_pls_moment_cuda_device_cv_fits",
        "n_pls_moment_cuda_parallel_fold_batches",
        "n_pls_moment_cuda_parallel_fold_jobs",
        "n_pls_moment_cuda_many_batched_batches",
        "n_pls_moment_cuda_many_batched_jobs",
        "n_pls_materialized_cv_fits",
        "n_pls_moment_final_fits",
        "n_pls_moment_host_final_fits",
        "n_pls_moment_cuda_device_final_fits",
        "n_pls_materialized_final_fits",
    }
    shapes_by_method: dict[str, set[tuple[int, int]]] = {}
    backends_by_cell: dict[tuple[str, tuple[int, int]], set[str]] = {}

    assert len(rows) == len(expected_methods) * 3 * len(expected_backends)
    assert {row["method"] for row in rows} == expected_methods
    assert required_route_fields.issubset(rows[0])

    for row in rows:
        method = row["method"]
        backend = row["backend"]
        shape = (_int(row, "n_samples"), _int(row, "n_features"))

        assert backend in expected_backends
        assert row["surface_status"] == "function_and_sklearn_replay"
        assert math.isfinite(_float(row, "rmse"))
        assert _float(row, "replay_max_abs_error") <= 1e-10
        if expect_cuda_pls:
            _assert_cuda_build_path(row["library_path"])
        else:
            assert "build/dev-release" in row["library_path"], row["library_path"]

        if method == "pls":
            assert _int(row, "n_pls_moment_cv_fits") > 0
            if expect_cuda_pls:
                assert _bool(row, "cuda_pls_parallel_folds") is True
                assert _int(row, "cuda_pls_min_device_features") == 1
                assert _bool(row, "cuda_pls_many_batched") is False
                assert _int(row, "n_pls_moment_host_cv_fits") == 0
                assert _int(row, "n_pls_moment_cuda_device_cv_fits") == _int(
                    row,
                    "n_pls_moment_cv_fits",
                )
                assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") > 0
            else:
                assert _bool(row, "cuda_pls_parallel_folds") is False
                assert row["cuda_pls_min_device_features"] == ""
                assert _bool(row, "cuda_pls_many_batched") is False
                assert _int(row, "n_pls_moment_host_cv_fits") == _int(
                    row,
                    "n_pls_moment_cv_fits",
                )
                assert _int(row, "n_pls_moment_cuda_device_cv_fits") == 0
                assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") == 0
            assert _int(row, "n_pls_moment_cuda_many_batched_batches") == 0
            assert _int(row, "n_pls_moment_cuda_many_batched_jobs") == 0

        shapes_by_method.setdefault(method, set()).add(shape)
        backends_by_cell.setdefault((method, shape), set()).add(backend)

    for method in expected_methods:
        assert len(shapes_by_method[method]) == 3
    assert len(backends_by_cell) == len(expected_methods) * 3
    for backends in backends_by_cell.values():
        assert backends == expected_backends


def _assert_pls_cross_validate_rows(
    rows: list[dict[str, str]],
    *,
    expect_cuda_pls: bool,
) -> None:
    expected_backends = {"pls_cross_validate", "pls_cross_validate_score_only"}
    required_fields = {
        "candidate_score_max_abs_delta",
        "oof_prediction_max_abs_delta",
        "prediction_max_abs_delta",
        "n_pls_moment_candidates",
        "n_pls_moment_cv_fits",
        "n_pls_moment_host_cv_fits",
        "n_pls_moment_cuda_device_cv_fits",
        "n_pls_moment_cuda_parallel_fold_batches",
        "n_pls_moment_cuda_parallel_fold_jobs",
        "n_pls_moment_cuda_many_batched_batches",
        "n_pls_moment_cuda_many_batched_jobs",
        "n_pls_moment_final_fits",
        "n_pls_moment_host_final_fits",
        "n_pls_moment_cuda_device_final_fits",
    }
    assert len(rows) == 6
    assert {row["backend"] for row in rows} == expected_backends
    assert required_fields.issubset(rows[0])
    assert {(_int(row, "n_samples"), _int(row, "n_features")) for row in rows} == {
        (48, 24),
        (96, 64),
        (160, 128),
    }

    backends_by_shape: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        backend = row["backend"]
        shape = (_int(row, "n_samples"), _int(row, "n_features"))
        score_only = backend.endswith("_score_only")
        assert _bool(row, "score_only") is score_only
        assert row["component_grid"] == "1|2|3"
        assert _int(row, "n_component_grid") == 3
        assert _int(row, "cv") == 4
        assert _int(row, "n_pls_moment_candidates") == 3
        assert _int(row, "n_pls_moment_cv_fits") == 4
        assert math.isfinite(_float(row, "selected_cv_rmse"))
        assert _float(row, "candidate_score_max_abs_delta") <= 1e-12
        assert _float(row, "oof_prediction_max_abs_delta") <= 1e-12
        assert _float(row, "prediction_max_abs_delta") <= 1e-12
        assert _int(row, "n_pls_moment_cuda_many_batched_batches") == 0
        assert _int(row, "n_pls_moment_cuda_many_batched_jobs") == 0

        if expect_cuda_pls:
            _assert_cuda_build_path(row["library_path"])
            assert _bool(row, "cuda_pls_parallel_folds") is True
            assert _int(row, "cuda_pls_min_device_features") == 1
            assert _bool(row, "cuda_pls_many_batched") is False
            assert _int(row, "n_pls_moment_host_cv_fits") == 0
            assert _int(row, "n_pls_moment_cuda_device_cv_fits") == 4
            assert _int(row, "n_pls_moment_cuda_parallel_fold_batches") == 1
            assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") == 4
            if score_only:
                assert _int(row, "n_pls_moment_final_fits") == 0
                assert _int(row, "n_pls_moment_cuda_device_final_fits") == 0
            else:
                assert _int(row, "n_pls_moment_final_fits") == 1
                assert _int(row, "n_pls_moment_host_final_fits") == 0
                assert _int(row, "n_pls_moment_cuda_device_final_fits") == 1
        else:
            assert "build/dev-release" in row["library_path"], row["library_path"]
            assert _bool(row, "cuda_pls_parallel_folds") is False
            assert row["cuda_pls_min_device_features"] == ""
            assert _bool(row, "cuda_pls_many_batched") is False
            assert _int(row, "n_pls_moment_host_cv_fits") == 4
            assert _int(row, "n_pls_moment_cuda_device_cv_fits") == 0
            assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") == 0
            if score_only:
                assert _int(row, "n_pls_moment_final_fits") == 0
                assert _int(row, "n_pls_moment_host_final_fits") == 0
            else:
                assert _int(row, "n_pls_moment_final_fits") == 1
                assert _int(row, "n_pls_moment_host_final_fits") == 1
                assert _int(row, "n_pls_moment_cuda_device_final_fits") == 0

        backends_by_shape.setdefault(shape, set()).add(backend)

    for backends in backends_by_shape.values():
        assert backends == expected_backends


def _loads_json_without_duplicate_keys(text: str):
    def no_duplicate_object(pairs):
        seen = {}
        for key, value in pairs:
            assert key not in seen, key
            seen[key] = value
        return seen

    return json.loads(text, object_pairs_hook=no_duplicate_object)


def test_cuda_facade_smoke_artifact_proves_facades_and_staged_cuda_route():
    path = _BENCH / "aom_moment_cuda_facade_smoke.json"
    data = _loads_json_without_duplicate_keys(path.read_text(encoding="utf-8"))

    assert data["report_schema"] == "n4m.aom_moment_cuda_facade_smoke.v1"
    assert data["cuda_available"] == 1
    _assert_cuda_build_path(data["library_path"])

    facades = data["facades"]
    assert facades["aom_staged_campaign_aliases_top_level"] is True
    assert facades["aom_staged_campaign_estimator_aliases_top_level"] is True
    assert facades["aom_savgol_focus_estimator_aliases_top_level"] is True
    assert facades["aom_strict_family_lite_estimator_aliases_top_level"] is True
    assert facades["aom_pls_exact_screen_refit_estimator_aliases_top_level"] is True
    assert facades["aom_pls_aliases_top_level"] is True
    assert facades["pop_pls_aliases_top_level"] is True
    assert facades["aom_diversity_aliases_top_level"] is True
    assert facades["aom_diversity_estimators_alias_top_level"] is True
    assert facades["n4m_moments_stays_callable"] is True

    for section in ("aom", "moment"):
        assert data[section]["n_pls_moment_cuda_device_cv_fits"] > 0
        assert data[section]["n_pls_moment_host_cv_fits"] == 0
    profile = data["aom_profile_sweep"]
    assert profile["n_pls_moment_cuda_device_cv_fits"] > 0
    assert profile["n_pls_moment_host_cv_fits"] == 0

    for section in ("aom_pls", "pop_pls"):
        assert data[section]["n_operators"] == 2
        assert data[section]["selected_n_components"] >= 1
        assert data[section]["prediction_replay_max_abs_error"] <= 1e-10

    staged = data["staged_estimator"]
    assert staged["screen_complete"] is True
    assert staged["selection_uses_test_set"] is False
    assert staged["split_head_scoring"] == "auto"
    assert staged["n_screen_split_head_chunks"] == 0
    assert staged["n_pls_moment_cuda_device_cv_fits"] > 0
    assert staged["n_pls_moment_host_cv_fits"] == 0
    assert staged["n_pls_moment_cuda_parallel_fold_jobs"] > 0

    pls_exact = data["pls_exact_screen_refit_estimator"]
    assert pls_exact["screen_complete"] is True
    assert pls_exact["selection_uses_test_set"] is False
    assert pls_exact["selected_head"] == "pls"
    assert pls_exact["selected_param"] >= 1
    assert pls_exact["preset"] == "moment_pls_exact_cv_screen_refit"
    assert pls_exact["pls_score_mode"] == "cv"
    assert pls_exact["refit_pls_score_mode"] == "cv"
    assert pls_exact["n_screen_candidates"] >= 2
    assert pls_exact["n_refit_candidates"] >= 1
    assert pls_exact["n_screen_pls_moment_cuda_device_cv_fits"] > 0
    assert pls_exact["n_screen_pls_moment_host_cv_fits"] == 0
    assert pls_exact["n_refit_pls_moment_cuda_device_cv_fits"] > 0
    assert pls_exact["n_refit_pls_moment_host_cv_fits"] == 0
    assert pls_exact["n_pls_gcv_proxy_fits"] == 0

    staged_mixed = data["staged_mixed_default_estimator"]
    assert staged_mixed["screen_complete"] is True
    assert staged_mixed["selection_uses_test_set"] is False
    assert staged_mixed["split_head_scoring"] == "auto"
    assert staged_mixed["selected_head"] in {"ridge", "pls"}
    assert staged_mixed["n_screen_split_head_chunks"] > 0
    assert staged_mixed["n_screen_chunk_score_calls"] > staged_mixed[
        "n_screen_split_head_chunks"
    ]
    assert staged_mixed["n_screen_pls_moment_cuda_device_cv_fits"] > 0
    assert staged_mixed["n_screen_pls_moment_host_cv_fits"] == 0
    assert staged_mixed["n_ridge_moment_cv_fits"] > 0
    assert staged_mixed["n_ridge_moment_score_batch_calls"] > 0
    assert staged_mixed["n_ridge_moment_score_batch_jobs"] > 0

    savgol = data["savgol_focus_estimator"]
    assert savgol["screen_complete"] is True
    assert savgol["selection_uses_test_set"] is False
    assert savgol["stage_names"] == [
        "compact",
        "savgol_smooth",
        "savgol_derivative",
        "savgol_combinations",
    ]
    assert savgol["selected_model_config"]["scale_x"] in {False, True}
    assert savgol["selected_scale_x"] in {False, True}
    assert savgol["n_screen_pls_moment_cuda_device_cv_fits"] > 0
    assert savgol["n_screen_pls_moment_host_cv_fits"] == 0
    assert savgol["n_pls_moment_cuda_device_cv_fits"] > 0
    assert savgol["n_pls_moment_host_cv_fits"] == 0
    assert savgol["n_pls_moment_cuda_parallel_fold_jobs"] > 0

    strict_lite = data["strict_family_lite_estimator"]
    assert strict_lite["screen_complete"] is True
    assert strict_lite["selection_uses_test_set"] is False
    assert strict_lite["stage_names"] == [
        "compact",
        "savgol_smooth",
        "savgol_derivative",
        "norris_williams",
        "finite_difference",
        "gaussian",
        "fck",
        "whittaker",
        "strict_combinations",
    ]
    assert strict_lite["selected_scale_x"] is False
    assert strict_lite["selected_model_config"] is None
    assert strict_lite["n_refit_candidates"] == 1
    assert strict_lite["n_screen_pls_moment_cuda_device_cv_fits"] > 0
    assert strict_lite["n_screen_pls_moment_host_cv_fits"] == 0
    assert strict_lite["n_pls_moment_cuda_device_cv_fits"] > 0
    assert strict_lite["n_pls_moment_host_cv_fits"] == 0
    assert strict_lite["n_pls_moment_cuda_parallel_fold_jobs"] > 0


def test_staged_campaign_cuda_smoke_artifact_proves_train_only_cuda_screen():
    row = _only_row("aom_staged_chain_campaign_timing_cuda_smoke.csv")

    assert row["backend"] == "native_aom_staged_chain_campaign"
    _assert_cuda_build_path(row["library_path"])
    _assert_current_aom_artifact_abi(row)
    assert _bool(row, "screen_complete") is True
    assert _int(row, "n_remaining_stage_chunks_total") == 0
    assert row["selection_metric"] == "refit_cv_rmse"
    assert _bool(row, "selection_uses_test_set") is False
    assert _int(row, "n_pls_moment_cuda_device_cv_fits") > 0
    assert _int(row, "n_pls_moment_host_cv_fits") == 0
    assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") > 0


def test_moment_stack_cuda_smoke_artifact_proves_pls_base_routes_on_device():
    row = _only_row("moment_stack_timing_cuda_smoke.csv")

    assert row["base_models"] == "pls"
    _assert_cuda_build_path(row["library_path"])
    assert _int(row, "n_base_oof_pls_moment_cuda_device_cv_fits") > 0
    assert _int(row, "n_base_oof_pls_moment_host_cv_fits") == 0
    assert _int(row, "n_base_oof_pls_moment_cuda_parallel_fold_jobs") > 0
    assert _int(row, "n_base_final_pls_moment_cuda_device_cv_fits") > 0
    assert _int(row, "n_base_final_pls_moment_host_cv_fits") == 0
    assert _int(row, "n_base_final_pls_moment_cuda_parallel_fold_jobs") > 0


def test_direct_moment_heads_cuda_smoke_artifact_covers_native_and_sklearn_replay():
    rows = _rows("direct_moment_heads_timing_cuda_smoke.csv")
    _assert_direct_moment_heads_rows(rows, expect_cuda_pls=True)


def test_direct_moment_heads_cpu_artifact_covers_current_schema_and_replay():
    rows = _rows("direct_moment_heads_timing.csv")
    _assert_direct_moment_heads_rows(rows, expect_cuda_pls=False)


def test_pls_cross_validate_cuda_smoke_artifact_routes_reference_abi_on_device():
    rows = _rows("pls_cross_validate_timing_cuda_smoke.csv")
    _assert_pls_cross_validate_rows(rows, expect_cuda_pls=True)


def test_pls_cross_validate_cpu_artifact_covers_reference_abi_schema():
    rows = _rows("pls_cross_validate_timing.csv")
    _assert_pls_cross_validate_rows(rows, expect_cuda_pls=False)


def test_moment_sweep_cuda_smoke_artifact_routes_exact_pls_cv_on_device():
    rows = _rows("moment_sweep_timing_cuda_smoke.csv")

    assert {row["backend"] for row in rows} == {
        "native_sweep_ridge",
        "native_sweep_ridge_score_only",
        "native_sweep_pls",
        "native_sweep_pls_score_only",
    }
    assert {(_int(row, "n_samples"), _int(row, "n_features")) for row in rows} == {
        (96, 48),
        (64, 64),
        (128, 128),
        (192, 256),
    }

    pls_cv_rows = [row for row in rows if _int(row, "n_pls_moment_cv_fits") > 0]
    assert pls_cv_rows
    ridge_moment_rows = [
        row for row in rows if _int(row, "n_ridge_moment_cv_fits") > 0
    ]
    assert ridge_moment_rows
    for row in rows:
        _assert_cuda_build_path(row["library_path"])
        assert _bool(row, "cuda_pls_parallel_folds") is True
        assert _int(row, "cuda_pls_min_device_features") == 1
        assert math.isfinite(_float(row, "selected_cv_rmse"))

    for row in pls_cv_rows:
        n_cv = _int(row, "n_pls_moment_cv_fits")
        assert _int(row, "n_pls_moment_host_cv_fits") == 0
        assert _int(row, "n_pls_moment_cuda_device_cv_fits") == n_cv
        assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") == n_cv

    for row in rows:
        n_final = _int(row, "n_pls_moment_final_fits")
        if n_final > 0:
            assert _int(row, "n_pls_moment_host_final_fits") == 0
            assert _int(row, "n_pls_moment_cuda_device_final_fits") == n_final

    for row in ridge_moment_rows:
        assert _int(row, "n_ridge_dual_materialized_cv_fits") == 0
        assert _int(row, "n_ridge_dual_cross_cv_fits") == 0

    tall_ridge_rows = {
        row["backend"]: row
        for row in ridge_moment_rows
        if (_int(row, "n_samples"), _int(row, "n_features")) == (96, 48)
    }
    full = tall_ridge_rows["native_sweep_ridge"]
    score_only = tall_ridge_rows["native_sweep_ridge_score_only"]
    assert _int(full, "n_ridge_moment_candidates") == 5
    assert _int(score_only, "n_ridge_moment_candidates") == 5
    assert _int(full, "n_ridge_moment_eigen_path_preparations") == 5
    assert _int(score_only, "n_ridge_moment_eigen_path_preparations") == 5
    assert _int(full, "n_ridge_moment_eigen_path_cv_fits") == 25
    assert _int(score_only, "n_ridge_moment_eigen_path_cv_fits") == 25
    assert _int(full, "n_ridge_moment_direct_cv_fits") == 0
    assert _int(score_only, "n_ridge_moment_direct_cv_fits") == 0
    assert _int(full, "n_ridge_moment_final_fits") == 1
    assert _int(score_only, "n_ridge_moment_final_fits") == 0
    assert _float(full, "selected_param") == _float(score_only, "selected_param")
    assert math.isclose(
        _float(full, "selected_cv_rmse"),
        _float(score_only, "selected_cv_rmse"),
        rel_tol=1e-10,
        abs_tol=1e-10,
    )


def test_moment_sweep_cuda_many_batched_smoke_artifact_routes_exact_pls_cv():
    rows = _rows("moment_sweep_timing_cuda_many_batched_smoke.csv")

    assert {row["backend"] for row in rows} == {
        "native_sweep_ridge",
        "native_sweep_ridge_score_only",
        "native_sweep_pls",
        "native_sweep_pls_score_only",
    }
    pls_cv_rows = [row for row in rows if _int(row, "n_pls_moment_cv_fits") > 0]
    assert pls_cv_rows

    for row in rows:
        _assert_cuda_build_path(row["library_path"])
        assert _bool(row, "cuda_pls_parallel_folds") is False
        assert _bool(row, "cuda_pls_many_batched") is True
        assert _int(row, "cuda_pls_min_device_features") == 1
        assert math.isfinite(_float(row, "selected_cv_rmse"))

    for row in pls_cv_rows:
        n_cv = _int(row, "n_pls_moment_cv_fits")
        assert _int(row, "n_pls_moment_host_cv_fits") == 0
        assert _int(row, "n_pls_moment_cuda_device_cv_fits") == n_cv
        assert _int(row, "n_pls_moment_cuda_parallel_fold_batches") == 0
        assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") == 0
        assert _int(row, "n_pls_moment_cuda_many_batched_batches") == 1
        assert _int(row, "n_pls_moment_cuda_many_batched_jobs") == n_cv

    for row in rows:
        n_final = _int(row, "n_pls_moment_final_fits")
        if n_final > 0:
            assert _int(row, "n_pls_moment_host_final_fits") == 0
            assert _int(row, "n_pls_moment_cuda_device_final_fits") == n_final


def test_aom_sweep_cuda_smoke_artifact_routes_exact_pls_cv_on_device():
    rows = _rows("aom_sweep_timing_cuda_smoke.csv")

    expected_backends = {
        "native_aom_sweep",
        "native_aom_chain_sweep",
        "native_aom_chain_sweep_whittaker",
        "native_aom_chain_sweep_fck",
        "native_aom_chain_sweep_gaussian",
        "native_aom_sweep_pls",
        "native_aom_chain_sweep_pls",
        "native_aom_chain_sweep_pls_gcv_proxy",
        "native_aom_chain_sweep_pls_exact_score_only",
        "native_aom_chain_screen_refit_pls_gcv_proxy",
        "native_aom_chain_fixed_fit_pls",
        "native_aom_chain_fixed_fit_ridge",
        "native_aom_sweep_ridge",
        "native_aom_chain_sweep_ridge",
        "native_aom_chain_sweep_ridge_exact_score_only",
    }
    assert {row["backend"] for row in rows} == expected_backends

    pls_cv_rows = [row for row in rows if _int(row, "n_pls_moment_cv_fits") > 0]
    assert pls_cv_rows
    assert any(row["profile"] == "custom3_fck" for row in rows)
    assert any(row["profile"] == "custom3_gaussian" for row in rows)
    assert any(row["profile"] == "custom2_whittaker" for row in rows)

    for row in rows:
        _assert_cuda_build_path(row["library_path"])
        assert _bool(row, "cuda_pls_parallel_folds") is True
        assert _int(row, "cuda_pls_min_device_features") == 1
        assert _int(row, "n_chains") > 0
        assert _int(row, "n_candidates") > 0
        if not row["backend"].startswith("native_aom_chain_fixed_fit"):
            assert math.isfinite(_float(row, "selected_cv_rmse"))

    for row in pls_cv_rows:
        n_cv = _int(row, "n_pls_moment_cv_fits")
        assert _int(row, "n_pls_moment_host_cv_fits") == 0
        assert _int(row, "n_pls_moment_cuda_device_cv_fits") == n_cv
        assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") == n_cv


def test_aom_selector_cuda_smoke_artifact_covers_function_and_sklearn_replay():
    _assert_aom_selector_rows(
        _rows("aom_selector_timing_cuda_smoke.csv"),
        expect_cuda_build=True,
    )


def test_aom_selector_cpu_artifact_covers_function_and_sklearn_replay():
    _assert_aom_selector_rows(
        _rows("aom_selector_timing.csv"),
        expect_cuda_build=False,
    )


def _assert_aom_selector_rows(
    rows: list[dict[str, str]],
    *,
    expect_cuda_build: bool,
) -> None:
    expected_methods = {"aom_pls", "pop_pls"}
    expected_surfaces = {"function", "sklearn"}
    shapes_by_method: dict[str, set[tuple[int, int]]] = {}
    surfaces_by_cell: dict[tuple[str, tuple[int, int]], set[str]] = {}

    assert len(rows) == len(expected_methods) * 3 * len(expected_surfaces)
    assert {row["method"] for row in rows} == expected_methods
    assert {row["surface"] for row in rows} == expected_surfaces

    for row in rows:
        method = row["method"]
        shape = (_int(row, "n_samples"), _int(row, "n_features"))
        if expect_cuda_build:
            _assert_cuda_build_path(row["library_path"])
        else:
            _assert_dev_release_build_path(row["library_path"])
        _assert_current_aom_artifact_abi(row)
        assert _int(row, "n_operators") > 0
        assert 1 <= _int(row, "selected_n_components") <= _int(
            row, "max_components"
        )
        assert math.isfinite(_float(row, "best_score"))
        assert _float(row, "replay_max_abs") <= 1e-10

        shapes_by_method.setdefault(method, set()).add(shape)
        surfaces_by_cell.setdefault((method, shape), set()).add(row["surface"])

    for method in expected_methods:
        assert len(shapes_by_method[method]) == 3
    assert len(surfaces_by_cell) == len(expected_methods) * 3
    for surfaces in surfaces_by_cell.values():
        assert surfaces == expected_surfaces


def test_aom_preprocess_cuda_smoke_artifact_covers_direct_linear_operators():
    _assert_aom_preprocess_rows(
        _rows("aom_preprocess_timing_cuda_smoke.csv"),
        expect_cuda_build=True,
    )


def test_aom_preprocess_cpu_artifact_covers_direct_linear_operators():
    _assert_aom_preprocess_rows(
        _rows("aom_preprocess_timing.csv"),
        expect_cuda_build=False,
    )


def _assert_aom_preprocess_rows(
    rows: list[dict[str, str]],
    *,
    expect_cuda_build: bool,
) -> None:
    expected_shapes = {(48, 24), (96, 64), (160, 128)}
    expected_operator_kinds = {
        "identity": 0,
        "detrend_poly_1": 7,
        "savgol_smooth_5_2": 8,
        "savgol_derivative_5_2_1": 9,
        "norris_williams_5_5_1": 10,
        "finite_difference_1": 15,
        "gaussian_1": 18,
        "whittaker_100": 16,
        "fck_1": 17,
    }

    assert len(rows) == len(expected_shapes) * 2 * len(expected_operator_kinds)
    assert {row["method"] for row in rows} == {"aom_preprocess"}
    assert {row["operator"] for row in rows} == set(expected_operator_kinds)
    assert {row["gating_mode"] for row in rows} == {"soft", "hard"}
    assert {(_int(row, "n_samples"), _int(row, "n_features")) for row in rows} == (
        expected_shapes
    )

    for row in rows:
        if expect_cuda_build:
            _assert_cuda_build_path(row["library_path"])
        else:
            _assert_dev_release_build_path(row["library_path"])
        _assert_current_aom_artifact_abi(row)
        operator = row["operator"]
        assert _int(row, "n_operators") == 1
        assert _int(row, "operator_kind") == expected_operator_kinds[operator]
        assert row["weight_shape"] == "1x1"
        assert abs(_float(row, "weight_sum") - 1.0) <= 1e-12
        assert _float(row, "replay_max_abs_error") <= 1e-12
        if operator == "identity":
            assert _float(row, "input_replay_max_abs_error") <= 1e-12
        assert math.isfinite(_float(row, "elapsed_ms_median"))


@pytest.mark.parametrize(
    ("csv_name", "native_backend", "wrapper_backend"),
    (
        (
            "aom_ridge_superblock_timing.csv",
            "native_aom_ridge_superblock",
            "native_aom_ridge_superblock_sklearn",
        ),
        (
            "aom_ridge_active_superblock_timing.csv",
            "native_aom_ridge_active_superblock",
            "native_aom_ridge_active_superblock_sklearn",
        ),
        (
            "aom_ridge_mkl_superblock_timing.csv",
            "native_aom_ridge_mkl_superblock",
            "native_aom_ridge_mkl_superblock_sklearn",
        ),
        (
            "aom_pls_superblock_timing.csv",
            "native_aom_pls_superblock",
            "native_aom_pls_superblock_sklearn",
        ),
        (
            "aom_ridge_pls_superblock_timing.csv",
            "native_aom_ridge_pls_superblock",
            "native_aom_ridge_pls_superblock_sklearn",
        ),
        (
            "aom_chain_ridge_pls_timing.csv",
            "native_aom_chain_ridge_pls",
            "native_aom_chain_ridge_pls_sklearn",
        ),
        (
            "aom_ridge_global_timing.csv",
            "native_aom_ridge_global",
            "native_aom_ridge_global_sklearn",
        ),
    ),
)
def test_diversity_cpu_artifacts_cover_native_and_sklearn_replay(
    csv_name: str,
    native_backend: str,
    wrapper_backend: str,
):
    rows = _rows(csv_name)
    backends = {row["backend"] for row in rows}
    assert native_backend in backends
    assert wrapper_backend in backends

    for row in rows:
        _assert_dev_release_build_path(row["library_path"])
        _assert_current_aom_artifact_abi(row)
        elapsed_key = (
            "elapsed_ms_median" if "elapsed_ms_median" in row else "elapsed_ms"
        )
        assert math.isfinite(_float(row, elapsed_key))
        if "ridge_backend" in row:
            assert row["ridge_backend"] in {"native", "native_aom_chain_sweep"}
        if "pls_backend" in row:
            assert row["pls_backend"] == "native"
            n_cv = _int(row, "n_pls_moment_cv_fits")
            assert n_cv > _int(row, "n_candidates")
            assert _int(row, "n_pls_moment_host_cv_fits") == n_cv
            assert _int(row, "n_pls_moment_cuda_device_cv_fits") == 0
            assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") == 0
            assert _int(row, "n_pls_moment_cuda_many_batched_batches") == 0
            assert _int(row, "n_pls_moment_cuda_many_batched_jobs") == 0
            assert _int(row, "n_pls_moment_final_fits") > 1
            assert _int(row, "n_pls_moment_host_final_fits") == _int(
                row,
                "n_pls_moment_final_fits",
            )
            assert _int(row, "n_pls_moment_cuda_device_final_fits") == 0
        if "ridge_pls_backend" in row:
            assert row["ridge_pls_backend"] == "native"
            assert _int(row, "n_ridge_pls_fit_calls") == (
                _int(row, "n_candidates") * _int(row, "cv") + 1
            )
    for row in rows:
        if row["backend"] == wrapper_backend:
            assert _float(row, "prediction_replay_max_abs_error") <= 1e-10


@pytest.mark.parametrize(
    ("csv_name", "native_backend", "wrapper_backend"),
    (
        (
            "aom_ridge_blender_timing_cuda_smoke.csv",
            "native_aom_ridge_blender",
            "native_aom_ridge_blender_sklearn",
        ),
        (
            "aom_operator_pls_stack_timing_cuda_smoke.csv",
            "native_aom_operator_pls_stack",
            "native_aom_operator_pls_stack_sklearn",
        ),
        (
            "aom_ridge_superblock_timing_cuda_smoke.csv",
            "native_aom_ridge_superblock",
            "native_aom_ridge_superblock_sklearn",
        ),
        (
            "aom_ridge_active_superblock_timing_cuda_smoke.csv",
            "native_aom_ridge_active_superblock",
            "native_aom_ridge_active_superblock_sklearn",
        ),
        (
            "aom_ridge_mkl_superblock_timing_cuda_smoke.csv",
            "native_aom_ridge_mkl_superblock",
            "native_aom_ridge_mkl_superblock_sklearn",
        ),
        (
            "aom_pls_superblock_timing_cuda_smoke.csv",
            "native_aom_pls_superblock",
            "native_aom_pls_superblock_sklearn",
        ),
        (
            "aom_ridge_pls_superblock_timing_cuda_smoke.csv",
            "native_aom_ridge_pls_superblock",
            "native_aom_ridge_pls_superblock_sklearn",
        ),
        (
            "aom_chain_ridge_pls_timing_cuda_smoke.csv",
            "native_aom_chain_ridge_pls",
            "native_aom_chain_ridge_pls_sklearn",
        ),
        (
            "aom_ridge_global_timing_cuda_smoke.csv",
            "native_aom_ridge_global",
            "native_aom_ridge_global_sklearn",
        ),
    ),
)
def test_diversity_cuda_smoke_artifacts_cover_native_and_sklearn_replay(
    csv_name: str,
    native_backend: str,
    wrapper_backend: str,
):
    rows = _rows(csv_name)
    backends = {row["backend"] for row in rows}
    assert native_backend in backends
    assert wrapper_backend in backends

    for row in rows:
        _assert_cuda_build_path(row["library_path"])
        _assert_current_aom_artifact_abi(row)
        if "ridge_backend" in row:
            assert row["ridge_backend"] in {"native", "native_aom_chain_sweep"}
        if "pls_backend" in row:
            assert row["pls_backend"] == "native"
            if csv_name == "aom_pls_superblock_timing_cuda_smoke.csv":
                n_cv = _int(row, "n_pls_moment_cv_fits")
                assert n_cv > _int(row, "n_candidates")
                assert _int(row, "n_pls_moment_host_cv_fits") == 0
                assert _int(row, "n_pls_moment_cuda_device_cv_fits") == n_cv
                assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") == n_cv
                assert _int(row, "n_pls_moment_final_fits") > 1
                assert _int(row, "n_pls_moment_host_final_fits") == 0
                assert _int(row, "n_pls_moment_cuda_device_final_fits") == _int(
                    row,
                    "n_pls_moment_final_fits",
                )
        if "ridge_pls_backend" in row:
            assert row["ridge_pls_backend"] == "native"
            if csv_name in {
                "aom_ridge_pls_superblock_timing_cuda_smoke.csv",
                "aom_chain_ridge_pls_timing_cuda_smoke.csv",
            }:
                assert _int(row, "n_ridge_pls_fit_calls") == (
                    _int(row, "n_candidates") * _int(row, "cv") + 1
                )
        if csv_name == "aom_ridge_blender_timing_cuda_smoke.csv":
            nc = _int(row, "n_candidates")
            cv = _int(row, "cv")
            assert _int(row, "n_ridge_blender_cv_fits") == nc * cv
            assert _int(row, "n_ridge_blender_final_fits") == nc
            assert _int(row, "n_ridge_blender_fit_calls") == nc * (cv + 1)
        if csv_name == "aom_operator_pls_stack_timing_cuda_smoke.csv":
            ns = _int(row, "n_specs")
            cv = _int(row, "cv")
            n_ops = _int(row, "n_operators")
            stack_fits = (ns + 1) * cv + 1
            assert _int(row, "n_pls_stack_cv_fits") == (ns + 1) * cv * n_ops
            assert _int(row, "n_pls_stack_final_fits") == n_ops
            assert _int(row, "n_ridge_stack_cv_fits") == (ns + 1) * cv
            assert _int(row, "n_ridge_stack_final_fits") == 1
            assert _int(row, "n_operator_pls_stack_fit_calls") == stack_fits
            assert _int(row, "n_operator_pls_stack_pls_fit_calls") == (
                stack_fits * n_ops
            )
            assert _int(row, "n_operator_pls_stack_ridge_fit_calls") == stack_fits
    for row in rows:
        if row["backend"] == wrapper_backend:
            assert _float(row, "prediction_replay_max_abs_error") <= 1e-10


def test_robust_hpo_cuda_smoke_artifact_covers_native_winner_path():
    _assert_robust_hpo_rows(
        _rows("aom_robust_hpo_timing_cuda_smoke.csv"),
        expect_cuda_build=True,
    )


def test_robust_hpo_cpu_artifact_covers_native_winner_path():
    _assert_robust_hpo_rows(
        _rows("aom_robust_hpo_timing.csv"),
        expect_cuda_build=False,
    )


def _assert_robust_hpo_rows(
    rows: list[dict[str, str]],
    *,
    expect_cuda_build: bool,
) -> None:
    assert {row["backend"] for row in rows} == {"native_abi", "native_sklearn"}
    assert {row["profile"] for row in rows} == {"compact"}
    seen_by_shape: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        if expect_cuda_build:
            _assert_cuda_build_path(row["library_path"])
        else:
            _assert_dev_release_build_path(row["library_path"])
        _assert_current_aom_artifact_abi(row)
        shape = (_int(row, "n_samples"), _int(row, "n_features"))
        seen_by_shape.setdefault(shape, set()).add(row["backend"])
        assert _int(row, "n_candidates") > 0
        assert _int(row, "selected_chain_id") >= 0
        assert _int(row, "selected_head_id") in {0, 1}
        assert math.isfinite(_float(row, "selected_cv_rmse"))
        if row["backend"] == "native_sklearn":
            assert _float(row, "prediction_replay_max_abs_error") <= 1e-10
    assert all(backends == {"native_abi", "native_sklearn"} for backends in seen_by_shape.values())


def test_aom_staged_chain_campaign_cpu_artifact_covers_host_route():
    row = _only_row("aom_staged_chain_campaign_timing.csv")

    _assert_dev_release_build_path(row["library_path"])
    _assert_current_aom_artifact_abi(row)
    assert row["backend"] == "native_aom_staged_chain_campaign"
    assert row["plan"] == "compact"
    assert row["heads"] == "pls"
    assert _bool(row, "selection_uses_test_set") is False
    assert _bool(row, "screen_complete") is True
    assert _int(row, "n_refit_candidates") >= 1
    assert _int(row, "n_pls_moment_cv_fits") > 0
    assert _int(row, "n_pls_moment_host_cv_fits") == _int(
        row,
        "n_pls_moment_cv_fits",
    )
    assert _int(row, "n_pls_moment_cuda_device_cv_fits") == 0
    assert _int(row, "n_pls_moment_cuda_parallel_fold_jobs") == 0
    assert _int(row, "n_pls_moment_cuda_many_batched_batches") == 0
    assert _int(row, "n_pls_moment_cuda_many_batched_jobs") == 0
    assert math.isfinite(_float(row, "best_refit_cv_rmse"))


@pytest.mark.parametrize(
    ("csv_name", "head", "min_refit_candidates"),
    (
        ("aom_screen_refit_scaling_cuda_smoke.csv", "pls", 1),
        ("aom_mixed_screen_refit_scaling_cuda_smoke.csv", "mixed", 1),
        ("aom_ridge_refit_scaling_cuda_smoke.csv", "ridge", 1),
    ),
)
def test_screen_refit_cuda_smoke_artifacts_cover_global_campaign_profiles(
    csv_name: str,
    head: str,
    min_refit_candidates: int,
):
    rows = _rows(csv_name)

    assert {row["backend"] for row in rows} == {"native_aom_screen_refit_scaling"}
    assert {row["head"] for row in rows} == {head}
    for row in rows:
        _assert_cuda_build_path(row["library_path"])
        assert _int(row, "n_screen_candidates") > 0
        assert _int(row, "n_screen_top_candidates") > 0
        assert _int(row, "n_refit_candidates") >= min_refit_candidates
        assert _int(row, "best_refit_cv_rank") >= 1
        assert math.isfinite(_float(row, "best_refit_cv_rmse"))
