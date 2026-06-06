"""Fast guards for the staged AOM benchmark helper scripts."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_moment_gpu_crossover_writes_markdown_summary(tmp_path):
    bench = _load_script(
        "bench_moment_gpu_crossover",
        "benchmarks/cross_binding/bench_moment_gpu_crossover.py",
    )
    rows = bench.add_speedups(
        [
            {
                "backend": "cpu",
                "cuda_pls_profile": "cpu_baseline",
                "head": "pls",
                "n_samples": 512,
                "n_features": 512,
                "elapsed_ms_median": 20.0,
                "error": "",
            },
            {
                "backend": "cuda",
                "cuda_pls_profile": "default",
                "head": "pls",
                "n_samples": 512,
                "n_features": 512,
                "elapsed_ms_median": 10.0,
                "error": "",
            },
            {
                "backend": "cuda",
                "cuda_pls_profile": "many_batched",
                "head": "pls",
                "n_samples": 512,
                "n_features": 512,
                "elapsed_ms_median": 12.5,
                "error": "",
            },
            {
                "backend": "cpu",
                "cuda_pls_profile": "cpu_baseline",
                "head": "ridge",
                "n_samples": 260,
                "n_features": 256,
                "elapsed_ms_median": 4.0,
                "error": "",
            },
            {
                "backend": "cuda",
                "cuda_pls_profile": "default",
                "head": "ridge",
                "n_samples": 260,
                "n_features": 256,
                "elapsed_ms_median": 4.2,
                "error": "",
            },
        ]
    )
    output = tmp_path / "moment_gpu_crossover.md"

    bench.write_markdown(output, rows)
    markdown = output.read_text(encoding="utf-8")

    assert "# Moment GPU Crossover" in markdown
    assert "512x512" in markdown
    assert "cuda:default" in markdown
    assert "2.00x" in markdown
    assert "0.80x" in markdown
    assert "| ridge | 260x256 |" in markdown
    assert "| pls | 512x512 |" in markdown
    assert "dataset identity" in markdown


def test_aom_preprocess_timing_script_writes_direct_operator_rows(tmp_path, monkeypatch):
    bench = _load_script(
        "bench_aom_preprocess_timing",
        "benchmarks/cross_binding/bench_aom_preprocess_timing.py",
    )
    output = tmp_path / "aom_preprocess.csv"
    calls: list[tuple[tuple[int, int], str, str]] = []

    def _operator_label(operator):
        if operator == "identity":
            return "identity"
        name, params = operator
        return "_".join([str(name), *(str(value) for value in params)])

    def fake_aom_preprocess(X, y, operators, gating_mode):
        operator = operators[0]
        label = _operator_label(operator)
        kind = {
            "identity": 0,
            "detrend_poly_1": 7,
            "savgol_smooth_5_2": 8,
            "savgol_derivative_5_2_1": 9,
            "norris_williams_5_5_1": 10,
            "finite_difference_1": 15,
            "gaussian_1.0": 18,
            "whittaker_100.0": 16,
            "fck_1.0": 17,
        }[label]
        transformed = np.array(X, copy=True)
        calls.append((X.shape, gating_mode, label))
        return {
            "transformed": transformed,
            "operator_outputs": transformed.reshape(1, -1),
            "weights": np.array([[1.0]], dtype=float),
            "operator_kinds": np.array([kind], dtype=int),
            "n_operators": 1.0,
            "mode": 1.0 if gating_mode == "soft" else 0.0,
        }

    monkeypatch.setattr(
        bench,
        "n4m",
        SimpleNamespace(
            aom_preprocess=fake_aom_preprocess,
            library_path=lambda: "build/cuda-on/cpp/src/libn4m.so",
            abi_version=lambda: (1, 21, 0),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_aom_preprocess_timing.py",
            "--output",
            str(output),
            "--repeats",
            "1",
        ],
    )

    assert bench.main() == 0
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    expected_operator_kinds = {
        "identity": "0",
        "detrend_poly_1": "7",
        "savgol_smooth_5_2": "8",
        "savgol_derivative_5_2_1": "9",
        "norris_williams_5_5_1": "10",
        "finite_difference_1": "15",
        "gaussian_1": "18",
        "whittaker_100": "16",
        "fck_1": "17",
    }

    assert len(rows) == 54
    assert {row["gating_mode"] for row in rows} == {"soft", "hard"}
    assert {row["operator"] for row in rows} == set(expected_operator_kinds)
    assert {
        row["operator_kind"] for row in rows
    } == set(expected_operator_kinds.values())
    for row in rows:
        assert row["operator_kind"] == expected_operator_kinds[row["operator"]]
    assert {row["weight_shape"] for row in rows} == {"1x1"}
    assert {row["weight_sum"] for row in rows} == {"1.0"}
    assert {row["replay_max_abs_error"] for row in rows} == {"0.0"}
    assert {row["library_path"] for row in rows} == {
        "build/cuda-on/cpp/src/libn4m.so"
    }
    assert {row["abi"] for row in rows} == {"1.21.0"}
    assert set(calls) == {
        (shape, mode, operator)
        for shape in ((48, 24), (96, 64), (160, 128))
        for mode in ("soft", "hard")
        for operator in {
            "identity",
            "detrend_poly_1",
            "savgol_smooth_5_2",
            "savgol_derivative_5_2_1",
            "norris_williams_5_5_1",
            "finite_difference_1",
            "gaussian_1.0",
            "whittaker_100.0",
            "fck_1.0",
        }
    }


def test_aom_ridge_superblock_timing_script_writes_native_and_wrapper_rows(tmp_path, monkeypatch):
    bench = _load_script(
        "bench_aom_ridge_superblock_timing",
        "benchmarks/cross_binding/bench_aom_ridge_superblock_timing.py",
    )
    output = tmp_path / "aom_ridge_superblock.csv"
    calls: list[tuple[str, tuple[int, int]]] = []

    def fake_result(X, y, operators, alphas, cv, fold_ids, block_scaling):
        calls.append(("function", X.shape))
        coef = np.zeros((X.shape[1], 1), dtype=float)
        intercept = np.array([[0.5]], dtype=float)
        pred = X @ coef + intercept
        return {
            "predictions": pred,
            "input_coefficients": coef,
            "intercept": intercept,
            "candidate_scores": np.array([[0.0, float(alphas[0]), 0.2]], dtype=float),
            "operator_kinds": np.array([0, 15, 8], dtype=int),
            "n_operators": 3.0,
            "n_features_superblock": float(3 * X.shape[1]),
            "n_candidates": 1.0,
            "selected_alpha": float(alphas[0]),
            "selected_cv_rmse": 0.2,
            "ridge_backend": "native",
        }

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            calls.append(("wrapper", X.shape))
            self.result_ = fake_result(
                X,
                y,
                operators=self.kwargs["operators"],
                alphas=self.kwargs["alphas"],
                cv=self.kwargs["cv"],
                fold_ids=self.kwargs["fold_ids"],
                block_scaling=self.kwargs["block_scaling"],
            )
            self.predictions_ = self.result_["predictions"]
            return self

        def predict(self, X):
            return self.result_["predictions"].ravel()

    monkeypatch.setattr(
        bench,
        "n4m",
        SimpleNamespace(
            aom_ridge_superblock=fake_result,
            library_path=lambda: "build/cuda-on/cpp/src/libn4m.so",
            abi_version=lambda: (1, 21, 0),
        ),
    )
    monkeypatch.setattr(bench, "NativeAOMRidgeSuperblockRegressor", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_aom_ridge_superblock_timing.py",
            "--output",
            str(output),
            "--repeats",
            "1",
            "--mode",
            "both",
        ],
    )

    assert bench.main() == 0
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert len(rows) == 6
    assert {row["backend"] for row in rows} == {
        "native_aom_ridge_superblock",
        "native_aom_ridge_superblock_sklearn",
    }
    assert {row["n_operators"] for row in rows} == {"3"}
    assert {row["operator_kinds"] for row in rows} == {"0 15 8"}
    assert {row["ridge_backend"] for row in rows} == {"native"}
    assert {row["prediction_replay_max_abs_error"] for row in rows} == {"0.0"}
    assert {row["library_path"] for row in rows} == {
        "build/cuda-on/cpp/src/libn4m.so"
    }
    assert {row["abi"] for row in rows} == {"1.21.0"}
    assert {name for name, _shape in calls} == {"function", "wrapper"}


def test_aom_ridge_active_superblock_timing_script_writes_native_and_wrapper_rows(tmp_path, monkeypatch):
    bench = _load_script(
        "bench_aom_ridge_active_superblock_timing",
        "benchmarks/cross_binding/bench_aom_ridge_active_superblock_timing.py",
    )
    output = tmp_path / "aom_ridge_active_superblock.csv"
    calls: list[tuple[str, tuple[int, int]]] = []

    def fake_result(X, y, operators, alphas, cv, fold_ids, active_top_m, block_scaling):
        calls.append(("function", X.shape))
        coef = np.zeros((X.shape[1], 1), dtype=float)
        intercept = np.array([[0.5]], dtype=float)
        pred = X @ coef + intercept
        return {
            "predictions": pred,
            "input_coefficients": coef,
            "intercept": intercept,
            "candidate_scores": np.array([[0.0, float(alphas[0]), 0.2]], dtype=float),
            "selected_operator_indices": np.array([0, 2], dtype=int),
            "selected_operator_kinds": np.array([0, 8], dtype=int),
            "n_operators": 4.0,
            "n_active_operators": 2.0,
            "n_active_pruned": 1.0,
            "n_features_superblock": float(2 * X.shape[1]),
            "n_candidates": 1.0,
            "active_score_method": "norm",
            "active_top_m": float(active_top_m),
            "selected_alpha": float(alphas[0]),
            "selected_cv_rmse": 0.2,
            "ridge_backend": "native",
        }

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            calls.append(("wrapper", X.shape))
            self.result_ = fake_result(
                X,
                y,
                operators=self.kwargs["operators"],
                alphas=self.kwargs["alphas"],
                cv=self.kwargs["cv"],
                fold_ids=self.kwargs["fold_ids"],
                active_top_m=self.kwargs["active_top_m"],
                block_scaling=self.kwargs["block_scaling"],
            )
            self.predictions_ = self.result_["predictions"]
            return self

        def predict(self, X):
            return self.result_["predictions"].ravel()

    monkeypatch.setattr(
        bench,
        "n4m",
        SimpleNamespace(
            aom_ridge_active_superblock=fake_result,
            library_path=lambda: "build/cuda-on/cpp/src/libn4m.so",
            abi_version=lambda: (1, 21, 0),
        ),
    )
    monkeypatch.setattr(bench, "NativeAOMRidgeActiveSuperblockRegressor", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_aom_ridge_active_superblock_timing.py",
            "--output",
            str(output),
            "--repeats",
            "1",
            "--mode",
            "both",
        ],
    )

    assert bench.main() == 0
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert len(rows) == 6
    assert {row["backend"] for row in rows} == {
        "native_aom_ridge_active_superblock",
        "native_aom_ridge_active_superblock_sklearn",
    }
    assert {row["n_operators"] for row in rows} == {"4"}
    assert {row["n_active_operators"] for row in rows} == {"2"}
    assert {row["selected_operator_indices"] for row in rows} == {"0 2"}
    assert {row["selected_operator_kinds"] for row in rows} == {"0 8"}
    assert {row["active_score_method"] for row in rows} == {"norm"}
    assert {row["ridge_backend"] for row in rows} == {"native"}
    assert {row["prediction_replay_max_abs_error"] for row in rows} == {"0.0"}
    assert {row["library_path"] for row in rows} == {
        "build/cuda-on/cpp/src/libn4m.so"
    }
    assert {row["abi"] for row in rows} == {"1.21.0"}
    assert {name for name, _shape in calls} == {"function", "wrapper"}


def test_aom_ridge_mkl_superblock_timing_script_writes_native_and_wrapper_rows(tmp_path, monkeypatch):
    bench = _load_script(
        "bench_aom_ridge_mkl_superblock_timing",
        "benchmarks/cross_binding/bench_aom_ridge_mkl_superblock_timing.py",
    )
    output = tmp_path / "aom_ridge_mkl_superblock.csv"
    calls: list[tuple[str, tuple[int, int]]] = []

    def fake_result(X, y, operators, alphas, cv, fold_ids, mkl_top_k, block_scaling):
        calls.append(("function", X.shape))
        coef = np.zeros((X.shape[1], 1), dtype=float)
        intercept = np.array([[0.5]], dtype=float)
        pred = X @ coef + intercept
        return {
            "predictions": pred,
            "input_coefficients": coef,
            "intercept": intercept,
            "candidate_scores": np.array([[0.0, float(alphas[0]), 0.2]], dtype=float),
            "operator_kinds": np.array([0, 15, 8, 9], dtype=int),
            "selected_operator_indices": np.array([0, 2], dtype=int),
            "mkl_weights": np.array([[0.6], [0.0], [0.4], [0.0]], dtype=float),
            "n_operators": 4.0,
            "n_mkl_active_operators": 2.0,
            "mkl_top_k": float(mkl_top_k),
            "n_features_superblock": float(4 * X.shape[1]),
            "n_candidates": 1.0,
            "selected_alpha": float(alphas[0]),
            "selected_cv_rmse": 0.2,
            "mkl_mode": "alignment",
            "ridge_backend": "native",
        }

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            calls.append(("wrapper", X.shape))
            self.result_ = fake_result(
                X,
                y,
                operators=self.kwargs["operators"],
                alphas=self.kwargs["alphas"],
                cv=self.kwargs["cv"],
                fold_ids=self.kwargs["fold_ids"],
                mkl_top_k=self.kwargs["mkl_top_k"],
                block_scaling=self.kwargs["block_scaling"],
            )
            self.predictions_ = self.result_["predictions"]
            return self

        def predict(self, X):
            return self.result_["predictions"].ravel()

    monkeypatch.setattr(
        bench,
        "n4m",
        SimpleNamespace(
            aom_ridge_mkl_superblock=fake_result,
            library_path=lambda: "build/cuda-on/cpp/src/libn4m.so",
            abi_version=lambda: (1, 21, 0),
        ),
    )
    monkeypatch.setattr(bench, "NativeAOMRidgeMKLSuperblockRegressor", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_aom_ridge_mkl_superblock_timing.py",
            "--output",
            str(output),
            "--repeats",
            "1",
            "--mode",
            "both",
        ],
    )

    assert bench.main() == 0
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert len(rows) == 6
    assert {row["backend"] for row in rows} == {
        "native_aom_ridge_mkl_superblock",
        "native_aom_ridge_mkl_superblock_sklearn",
    }
    assert {row["n_operators"] for row in rows} == {"4"}
    assert {row["operator_kinds"] for row in rows} == {"0 15 8 9"}
    assert {row["selected_operator_indices"] for row in rows} == {"0 2"}
    assert {row["n_mkl_active_operators"] for row in rows} == {"2"}
    assert {row["mkl_weight_sum"] for row in rows} == {"1.0"}
    assert {row["mkl_mode"] for row in rows} == {"alignment"}
    assert {row["ridge_backend"] for row in rows} == {"native"}
    assert {row["prediction_replay_max_abs_error"] for row in rows} == {"0.0"}
    assert {row["library_path"] for row in rows} == {
        "build/cuda-on/cpp/src/libn4m.so"
    }
    assert {row["abi"] for row in rows} == {"1.21.0"}
    assert {name for name, _shape in calls} == {"function", "wrapper"}


def test_aom_pls_superblock_timing_script_writes_native_and_wrapper_rows(tmp_path, monkeypatch):
    bench = _load_script(
        "bench_aom_pls_superblock_timing",
        "benchmarks/cross_binding/bench_aom_pls_superblock_timing.py",
    )
    output = tmp_path / "aom_pls_superblock.csv"
    calls: list[tuple[str, tuple[int, int]]] = []

    def fake_result(
        X,
        y,
        operators,
        pls_components,
        cv,
        fold_ids,
        block_scaling,
        cuda_pls_min_device_features,
        cuda_pls_parallel_folds,
    ):
        calls.append(("function", X.shape))
        coef = np.zeros((X.shape[1], 1), dtype=float)
        intercept = np.array([[0.5]], dtype=float)
        pred = X @ coef + intercept
        return {
            "predictions": pred,
            "input_coefficients": coef,
            "intercept": intercept,
            "candidate_scores": np.array([[0.0, float(pls_components[0]), 0.2]], dtype=float),
            "operator_kinds": np.array([0, 15, 8], dtype=int),
            "n_operators": 3.0,
            "n_features_superblock": float(3 * X.shape[1]),
            "n_candidates": 1.0,
            "n_components": float(pls_components[0]),
            "selected_cv_rmse": 0.2,
            "pls_backend": "native",
            "n_pls_moment_cv_fits": 1.0,
            "n_pls_moment_host_cv_fits": 0.0,
            "n_pls_moment_cuda_device_cv_fits": 1.0,
        }

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            calls.append(("wrapper", X.shape))
            self.result_ = fake_result(
                X,
                y,
                operators=self.kwargs["operators"],
                pls_components=self.kwargs["pls_components"],
                cv=self.kwargs["cv"],
                fold_ids=self.kwargs["fold_ids"],
                block_scaling=self.kwargs["block_scaling"],
                cuda_pls_min_device_features=self.kwargs["cuda_pls_min_device_features"],
                cuda_pls_parallel_folds=self.kwargs["cuda_pls_parallel_folds"],
            )
            self.predictions_ = self.result_["predictions"]
            return self

        def predict(self, X):
            return self.result_["predictions"].ravel()

    monkeypatch.setattr(
        bench,
        "n4m",
        SimpleNamespace(
            aom_pls_superblock=fake_result,
            library_path=lambda: "build/cuda-on/cpp/src/libn4m.so",
            abi_version=lambda: (1, 21, 0),
        ),
    )
    monkeypatch.setattr(bench, "NativeAOMPLSSuperblockRegressor", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_aom_pls_superblock_timing.py",
            "--output",
            str(output),
            "--repeats",
            "1",
            "--mode",
            "both",
            "--cuda-pls-min-device-features",
            "1",
        ],
    )

    assert bench.main() == 0
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert len(rows) == 6
    assert {row["backend"] for row in rows} == {
        "native_aom_pls_superblock",
        "native_aom_pls_superblock_sklearn",
    }
    assert {row["n_operators"] for row in rows} == {"3"}
    assert {row["operator_kinds"] for row in rows} == {"0 15 8"}
    assert {row["pls_backend"] for row in rows} == {"native"}
    assert {row["prediction_replay_max_abs_error"] for row in rows} == {"0.0"}
    assert {row["library_path"] for row in rows} == {
        "build/cuda-on/cpp/src/libn4m.so"
    }
    assert {row["abi"] for row in rows} == {"1.21.0"}
    assert {name for name, _shape in calls} == {"function", "wrapper"}


def test_aom_ridge_pls_superblock_timing_script_writes_native_and_wrapper_rows(
    tmp_path,
    monkeypatch,
):
    bench = _load_script(
        "bench_aom_ridge_pls_superblock_timing",
        "benchmarks/cross_binding/bench_aom_ridge_pls_superblock_timing.py",
    )
    output = tmp_path / "aom_ridge_pls_superblock.csv"
    calls: list[tuple[str, tuple[int, int]]] = []

    def fake_result(
        X,
        y,
        operators,
        pls_components,
        ridge_lambdas,
        cv,
        fold_ids,
        block_scaling,
    ):
        calls.append(("function", X.shape))
        coef = np.zeros((X.shape[1], 1), dtype=float)
        intercept = np.array([[0.5]], dtype=float)
        pred = X @ coef + intercept
        return {
            "predictions": pred,
            "input_coefficients": coef,
            "intercept": intercept,
            "candidate_scores": np.array(
                [[0.0, float(pls_components[0]), float(ridge_lambdas[0]), 0.2]],
                dtype=float,
            ),
            "operator_kinds": np.array([0, 15, 8], dtype=int),
            "n_operators": 3.0,
            "n_features_superblock": float(3 * X.shape[1]),
            "n_candidates": 1.0,
            "n_components": float(pls_components[0]),
            "ridge_lambda": float(ridge_lambdas[0]),
            "selected_cv_rmse": 0.2,
            "ridge_pls_backend": "native",
        }

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            calls.append(("wrapper", X.shape))
            self.result_ = fake_result(
                X,
                y,
                operators=self.kwargs["operators"],
                pls_components=self.kwargs["pls_components"],
                ridge_lambdas=self.kwargs["ridge_lambdas"],
                cv=self.kwargs["cv"],
                fold_ids=self.kwargs["fold_ids"],
                block_scaling=self.kwargs["block_scaling"],
            )
            self.predictions_ = self.result_["predictions"]
            return self

        def predict(self, X):
            return self.result_["predictions"].ravel()

    monkeypatch.setattr(
        bench,
        "n4m",
        SimpleNamespace(
            aom_ridge_pls_superblock=fake_result,
            library_path=lambda: "build/cuda-on/cpp/src/libn4m.so",
            abi_version=lambda: (1, 21, 0),
        ),
    )
    monkeypatch.setattr(bench, "NativeAOMRidgePLSSuperblockRegressor", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_aom_ridge_pls_superblock_timing.py",
            "--output",
            str(output),
            "--repeats",
            "1",
            "--mode",
            "both",
        ],
    )

    assert bench.main() == 0
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert len(rows) == 6
    assert {row["backend"] for row in rows} == {
        "native_aom_ridge_pls_superblock",
        "native_aom_ridge_pls_superblock_sklearn",
    }
    assert {row["n_operators"] for row in rows} == {"3"}
    assert {row["operator_kinds"] for row in rows} == {"0 15 8"}
    assert {row["ridge_pls_backend"] for row in rows} == {"native"}
    assert {row["prediction_replay_max_abs_error"] for row in rows} == {"0.0"}
    assert {row["library_path"] for row in rows} == {
        "build/cuda-on/cpp/src/libn4m.so"
    }
    assert {row["abi"] for row in rows} == {"1.21.0"}
    assert {name for name, _shape in calls} == {"function", "wrapper"}


def test_aom_ridge_global_timing_script_writes_native_and_wrapper_rows(tmp_path, monkeypatch):
    bench = _load_script(
        "bench_aom_ridge_global_timing",
        "benchmarks/cross_binding/bench_aom_ridge_global_timing.py",
    )
    output = tmp_path / "aom_ridge_global.csv"
    calls: list[tuple[str, tuple[int, int]]] = []

    def fake_result(X, y, operators, ridge_lambdas, cv, fold_ids, scale_x, moment_policy):
        calls.append(("function", X.shape))
        coef = np.zeros((X.shape[1], 1), dtype=float)
        intercept = np.array([[0.25]], dtype=float)
        pred = X @ coef + intercept
        return {
            "predictions": pred,
            "input_coefficients": coef,
            "intercept": intercept,
            "candidate_scores": np.array([[0.0, 0.0, 0.0, float(ridge_lambdas[0]), 0.3]], dtype=float),
            "n_operators": 3.0,
            "n_candidates": 1.0,
            "selected_operator_index": 0.0,
            "selected_operator_kind": 0.0,
            "selected_param": float(ridge_lambdas[0]),
            "selected_cv_rmse": 0.3,
            "ridge_backend": "native_aom_chain_sweep",
            "n_ridge_moment_cv_fits": 1.0,
            "n_ridge_dual_materialized_cv_fits": 0.0,
        }

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            calls.append(("wrapper", X.shape))
            self.result_ = fake_result(X, y, **self.kwargs)
            self.predictions_ = self.result_["predictions"]
            return self

        def predict(self, X):
            return self.result_["predictions"].ravel()

    monkeypatch.setattr(
        bench,
        "n4m",
        SimpleNamespace(
            aom_ridge_global=fake_result,
            library_path=lambda: "build/cuda-on/cpp/src/libn4m.so",
            abi_version=lambda: (1, 21, 0),
        ),
    )
    monkeypatch.setattr(bench, "NativeAOMRidgeGlobalRegressor", FakeModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_aom_ridge_global_timing.py",
            "--output",
            str(output),
            "--repeats",
            "1",
            "--mode",
            "both",
        ],
    )

    assert bench.main() == 0
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert len(rows) == 6
    assert {row["backend"] for row in rows} == {
        "native_aom_ridge_global",
        "native_aom_ridge_global_sklearn",
    }
    assert {row["n_operators"] for row in rows} == {"3"}
    assert {row["ridge_backend"] for row in rows} == {"native_aom_chain_sweep"}
    assert {row["prediction_replay_max_abs_error"] for row in rows} == {"0.0"}
    assert {row["library_path"] for row in rows} == {
        "build/cuda-on/cpp/src/libn4m.so"
    }
    assert {row["abi"] for row in rows} == {"1.21.0"}
    assert {name for name, _shape in calls} == {"function", "wrapper"}


def test_oracle_comparator_filters_and_keeps_oracles_separate(tmp_path):
    compare = _load_script(
        "compare_aom_staged_to_oracles",
        "benchmarks/cross_binding/compare_aom_staged_to_oracles.py",
    )

    aom_pls_csv = _write_csv(tmp_path / "aom_pls.csv", [
        {
            "database_name": "DB",
            "dataset": "one",
            "status": "ok",
            "model": "PLS-standard-numpy",
            "RMSEP": "0.10",
        },
        {
            "database_name": "DB",
            "dataset": "one",
            "status": "ok",
            "model": "AOM-compact-simpls",
            "RMSEP": "0.30",
        },
        {
            "database_name": "DB",
            "dataset": "one",
            "status": "ok",
            "model": "POP-wide-simpls",
            "RMSEP": "0.20",
        },
    ])
    aom_ridge_csv = _write_csv(tmp_path / "aom_ridge.csv", [
        {
            "dataset_group": "DB",
            "dataset": "one",
            "status": "ok",
            "variant": "Ridge-raw",
            "rmsep": "0.05",
        },
        {
            "dataset_group": "DB",
            "dataset": "one",
            "status": "ok",
            "variant": "AOMRidge-global-compact",
            "rmsep": "0.25",
        },
    ])
    tabpfn_csv = _write_csv(tmp_path / "tabpfn.csv", [
        {
            "database_name": "DB",
            "dataset": "one",
            "status": "ok",
            "ref_rmse_tabpfn_raw": "0.40",
            "ref_rmse_tabpfn_opt": "0.15",
        },
    ])
    target_csv = _write_csv(tmp_path / "target.csv", [
        {
            "database_name": "DB",
            "dataset": "one",
            "status": "ok",
            "canonical_name": "N4M-AOM-staged-chain-campaign",
            "rmsep": "0.22",
        },
    ])

    target = compare.best_by_dataset(
        [target_csv],
        source="n4m_staged",
        score_column="rmsep",
    )
    aom_pls = compare.best_by_dataset(
        [aom_pls_csv],
        source="aom_pls_oracle",
        score_column="RMSEP",
        model_contains=("AOM", "POP"),
    )
    aom_ridge = compare.best_by_dataset(
        [aom_ridge_csv],
        source="aom_ridge_oracle",
        score_column="rmsep",
        model_prefixes=("AOMRidge",),
    )
    tabpfn = compare.tabpfn_best_by_dataset([tabpfn_csv])

    assert target["DB/one"]["score"] == pytest.approx(0.22)
    assert aom_pls["DB/one"]["score"] == pytest.approx(0.20)
    assert aom_pls["DB/one"]["label"] == "POP-wide-simpls"
    assert aom_ridge["DB/one"]["score"] == pytest.approx(0.25)
    assert aom_ridge["DB/one"]["label"] == "AOMRidge-global-compact"
    assert tabpfn["DB/one"]["score"] == pytest.approx(0.15)
    assert tabpfn["DB/one"]["label"] == "TabPFN-opt"

    rows = compare.merge_sources(
        target,
        {
            "aom_pls_oracle": aom_pls,
            "aom_ridge_oracle": aom_ridge,
            "tabpfn_oracle": tabpfn,
        },
    )
    assert rows == [
        {
            "dataset_key": "DB/one",
            "target_score": pytest.approx(0.22),
            "target_label": "N4M-AOM-staged-chain-campaign",
            "aom_pls_oracle_score": pytest.approx(0.20),
            "aom_pls_oracle_label": "POP-wide-simpls",
            "target_delta_vs_aom_pls_oracle": pytest.approx(0.02),
            "target_ratio_vs_aom_pls_oracle": pytest.approx(1.1),
            "aom_ridge_oracle_score": pytest.approx(0.25),
            "aom_ridge_oracle_label": "AOMRidge-global-compact",
            "target_delta_vs_aom_ridge_oracle": pytest.approx(-0.03),
            "target_ratio_vs_aom_ridge_oracle": pytest.approx(0.88),
            "tabpfn_oracle_score": pytest.approx(0.15),
            "tabpfn_oracle_label": "TabPFN-opt",
            "target_delta_vs_tabpfn_oracle": pytest.approx(0.07),
            "target_ratio_vs_tabpfn_oracle": pytest.approx(1.4666666667),
            "winner": "tabpfn_oracle",
        }
    ]
    summary = "\n".join(
        compare.summarize(
            rows,
            ["aom_pls_oracle", "aom_ridge_oracle", "tabpfn_oracle"],
        )
    )
    assert "## Target Paired Rows" in summary
    assert (
        "| DB/one | 0.22 | 0.2 | 1.1 | 0.25 | 0.88 | 0.15 | 1.467 | tabpfn_oracle |"
        in summary
    )


def test_real_cohort_runner_reports_train_cv_selection_not_test_oracle():
    runner = _load_script(
        "run_aom_staged_real_cohort",
        "benchmarks/cross_binding/run_aom_staged_real_cohort.py",
    )
    selected = {"eval_rmse": 0.42, "head": "ridge", "param": 1.0}
    test_oracle = {"eval_rmse": 0.11, "head": "pls", "param": 2.0}
    report = {
        "audit": {
            "eval": {"best_cv": selected, "best_eval": test_oracle},
            "best_eval": test_oracle,
        }
    }

    assert runner.selected_eval_row(report) is selected
    assert runner.audit_oracle_row(report) is test_oracle


def test_real_cohort_runner_writes_route_counters_and_diagnostics(
    tmp_path, monkeypatch
):
    runner = _load_script(
        "run_aom_staged_real_cohort",
        "benchmarks/cross_binding/run_aom_staged_real_cohort.py",
    )

    X_train = np.ones((6, 4), dtype=np.float64)
    y_train = np.arange(6, dtype=np.float64)
    X_test = np.ones((3, 4), dtype=np.float64)
    y_test = np.arange(3, dtype=np.float64)

    def fake_load_split(data_root, database_name, dataset):
        return X_train, y_train, X_test, y_test, Path("/tmp/fake_split")

    selected = {
        "eval_rmse": 0.42,
        "eval_r2": 0.3,
        "head": "pls",
        "param": 1.0,
    }
    test_oracle = {"eval_rmse": 0.11, "head": "ridge", "param": 0.1}
    fake_report = {
        "audit": {
            "eval": {"best_cv": selected, "best_eval": test_oracle},
            "best_eval": test_oracle,
        },
        "rank_diagnostics": {"spearman_rank_correlation": 0.5},
        "impact": {
            "best_score": 0.4,
            "by_operator": [
                {
                    "group": "savgol_smooth",
                    "n_candidates": 2,
                    "best_score": 0.4,
                    "mean_score": 0.45,
                    "median_score": 0.45,
                    "best_rank": 1,
                    "mean_rank": 1.5,
                    "improvement_vs_identity": 0.1,
                }
            ],
            "by_stage_family": [],
            "by_stage_option": [
                {
                    "group": "sg_only/savgol_smooth",
                    "n_candidates": 2,
                    "best_score": 0.4,
                }
            ],
            "by_head_stage_option": [],
        },
        "best": {
            "head": "pls",
            "param": 1.0,
            "campaign_stage": "compact",
            "refit_cv_rmse": 0.4,
            "screen_cv_rmse": 0.45,
            "cv_rank": 1,
            "screen_rank": 2,
        },
        "screen_complete": True,
        "n_remaining_stage_chunks_total": 0,
        "n_screen_candidates_total": 9,
        "n_screen_split_head_chunks": 2,
        "n_screen_chunk_score_calls": 4,
        "n_ridge_moment_cv_fits": 24,
        "n_ridge_moment_eigen_path_preparations": 6,
        "n_ridge_moment_eigen_path_cv_fits": 18,
        "n_ridge_moment_direct_cv_fits": 6,
        "n_ridge_moment_score_batch_calls": 2,
        "n_ridge_moment_score_batch_jobs": 24,
        "n_refit_candidates": 3,
        "selection_uses_test_set": False,
        "plan": "compact",
        "n_screen_pls_moment_cv_fits": 12,
        "n_screen_pls_moment_host_cv_fits": 0,
        "n_screen_pls_moment_cuda_device_cv_fits": 12,
        "n_screen_pls_moment_cuda_parallel_fold_batches": 3,
        "n_screen_pls_moment_cuda_parallel_fold_jobs": 12,
        "n_screen_pls_moment_score_batch_calls": 3,
        "n_screen_pls_moment_score_batch_jobs": 12,
        "n_refit_pls_moment_cv_fits": 4,
        "n_refit_pls_moment_host_cv_fits": 0,
        "n_refit_pls_moment_cuda_device_cv_fits": 4,
        "n_refit_pls_moment_cuda_parallel_fold_batches": 1,
        "n_refit_pls_moment_cuda_parallel_fold_jobs": 4,
        "n_refit_pls_moment_score_batch_calls": 1,
        "n_refit_pls_moment_score_batch_jobs": 4,
        "model_config_summaries": [{"scale_x": True, "best_refit_cv_rmse": 0.4}],
        "selected_model_config": {"scale_x": True},
        "selected_model_config_id": 1,
        "route_summary": {"by_score_route": [{"score_route": "banded"}]},
    }
    runner.validate_ridge_moment_route_telemetry(fake_report)

    monkeypatch.setattr(runner, "load_split", fake_load_split)
    captured_campaign_kwargs = {}

    def fake_campaign(*args, **kwargs):
        captured_campaign_kwargs.update(kwargs)
        return fake_report

    monkeypatch.setattr(runner.n4m, "aom_staged_chain_campaign", fake_campaign)
    monkeypatch.setattr(runner.n4m, "library_path", lambda: "build/cuda-on/libn4m.so")
    monkeypatch.setattr(runner.n4m, "abi_version", lambda: (1, 21, 0))

    args = SimpleNamespace(
        data_root="/tmp",
        checkpoint_dir=None,
        no_resume=True,
        plan="compact",
        cv=3,
        ridge_lambdas="0.1",
        components="1",
        heads="pls",
        max_chains=4,
        chain_chunk_size=2,
        top_k=3,
        refit_top_k=2,
        refit_per_head_top_k=1,
        max_chunks_per_run=None,
        scale_x=False,
        moment_policy="auto",
        pls_score_mode="gcv_proxy",
        split_head_scoring="auto",
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
        cuda_pls_many_batched=False,
        backend_min_cuda_product=1,
        max_train_samples=None,
        max_features=None,
        max_train_feature_product=None,
        stages=[
            {
                "name": "sg_only",
                "profile": "compact",
                "families": ["savgol"],
                "max_chains": 2,
            }
        ],
        cohort_csv="cohort.csv",
        seed=0,
        diagnostics_dir=str(tmp_path / "diag"),
    )

    row = runner.run_one(args, {"database_name": "DB", "dataset": "one", "key": "DB/one"})

    assert row["status"] == "ok"
    assert row["n_screen_split_head_chunks"] == 2
    assert row["n_screen_chunk_score_calls"] == 4
    assert row["n_ridge_moment_cv_fits"] == 24
    assert row["n_ridge_moment_eigen_path_preparations"] == 6
    assert row["n_ridge_moment_eigen_path_cv_fits"] == 18
    assert row["n_ridge_moment_direct_cv_fits"] == 6
    assert row["n_ridge_moment_score_batch_calls"] == 2
    assert row["n_ridge_moment_score_batch_jobs"] == 24
    assert row["n_screen_pls_moment_cv_fits"] == 12
    assert row["n_screen_pls_moment_host_cv_fits"] == 0
    assert row["n_screen_pls_moment_cuda_device_cv_fits"] == 12
    assert row["n_screen_pls_moment_cuda_parallel_fold_jobs"] == 12
    assert row["n_screen_pls_moment_score_batch_calls"] == 3
    assert row["n_screen_pls_moment_score_batch_jobs"] == 12
    assert row["n_refit_pls_moment_cv_fits"] == 4
    assert row["n_refit_pls_moment_host_cv_fits"] == 0
    assert row["n_refit_pls_moment_cuda_device_cv_fits"] == 4
    assert row["n_refit_pls_moment_cuda_parallel_fold_jobs"] == 4
    assert row["n_refit_pls_moment_score_batch_calls"] == 1
    assert row["n_refit_pls_moment_score_batch_jobs"] == 4
    assert row["cuda_pls_parallel_folds"] is True
    assert row["cuda_pls_min_device_features"] == 1
    assert row["backend_min_cuda_product"] == 1
    assert row["plan"] == "compact"
    assert row["pls_score_mode"] == "gcv_proxy"
    assert row["split_head_scoring"] == "auto"
    assert row["scale_x"] is True
    assert row["stages_json"] == (
        '[{"families":["savgol"],"max_chains":2,'
        '"name":"sg_only","profile":"compact"}]'
    )
    assert captured_campaign_kwargs["stages"] == args.stages
    assert captured_campaign_kwargs["pls_score_mode"] == "gcv_proxy"
    assert captured_campaign_kwargs["split_head_scoring"] == "auto"

    diagnostics_path = (
        tmp_path / "diag" / "DB_one.diagnostics.json"
    )
    assert diagnostics_path.is_file()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["dataset_key"] == "DB/one"
    assert diagnostics["selection_uses_test_set"] is False
    assert diagnostics["selected_head"] == "pls"
    assert diagnostics["best"]["refit_cv_rmse"] == 0.4
    assert diagnostics["impact"]["by_operator"][0]["group"] == "savgol_smooth"
    assert diagnostics["rank_diagnostics"]["spearman_rank_correlation"] == 0.5
    assert diagnostics["selected_model_config"]["scale_x"] is True
    assert diagnostics["runner"]["pls_score_mode"] == "gcv_proxy"
    assert diagnostics["runner"]["split_head_scoring"] == "auto"
    assert diagnostics["counters"]["n_screen_split_head_chunks"] == 2
    assert diagnostics["counters"]["n_screen_chunk_score_calls"] == 4
    assert diagnostics["counters"]["n_ridge_moment_cv_fits"] == 24
    assert diagnostics["counters"]["n_ridge_moment_eigen_path_preparations"] == 6
    assert diagnostics["counters"]["n_ridge_moment_eigen_path_cv_fits"] == 18
    assert diagnostics["counters"]["n_ridge_moment_direct_cv_fits"] == 6
    assert diagnostics["counters"]["n_ridge_moment_score_batch_calls"] == 2
    assert diagnostics["counters"]["n_ridge_moment_score_batch_jobs"] == 24
    assert diagnostics["counters"]["n_screen_pls_moment_cuda_device_cv_fits"] == 12
    assert diagnostics["counters"]["n_screen_pls_moment_score_batch_calls"] == 3
    assert diagnostics["counters"]["n_screen_pls_moment_score_batch_jobs"] == 12
    assert diagnostics["counters"]["n_refit_pls_moment_score_batch_calls"] == 1
    assert diagnostics["counters"]["n_refit_pls_moment_score_batch_jobs"] == 4
    # audit payload persisted: offline test-set scoring never changes selection
    assert "audit" in diagnostics
    audit_diag = diagnostics["audit"]
    assert audit_diag["audit_only"] is True
    assert audit_diag["selected_cv"]["eval_rmse"] == pytest.approx(0.42)
    assert audit_diag["selected_cv"]["head"] == "pls"
    assert audit_diag["oracle"]["eval_rmse"] == pytest.approx(0.11)
    assert audit_diag["oracle"]["head"] == "ridge"
    assert diagnostics["selection_uses_test_set"] is False

    impact_rows = list(
        csv.DictReader((tmp_path / "diag" / "impact_groups.csv").open())
    )
    assert [row["group_kind"] for row in impact_rows] == [
        "by_operator",
        "by_stage_option",
    ]
    assert impact_rows[0]["dataset_key"] == "DB/one"
    assert impact_rows[0]["group"] == "savgol_smooth"
    assert impact_rows[0]["improvement_vs_identity"] == "0.1"
    assert impact_rows[0]["selected_model_config_id"] == "1"


def test_real_cohort_runner_rejects_inconsistent_ridge_route_telemetry():
    runner = _load_script(
        "run_aom_staged_real_cohort",
        "benchmarks/cross_binding/run_aom_staged_real_cohort.py",
    )

    with pytest.raises(ValueError, match="Ridge moment route telemetry"):
        runner.validate_ridge_moment_route_telemetry(
            {
                "n_ridge_moment_cv_fits": 24,
                "n_ridge_moment_eigen_path_preparations": 6,
                "n_ridge_moment_eigen_path_cv_fits": 18,
                "n_ridge_moment_direct_cv_fits": 7,
            }
        )

    with pytest.raises(ValueError, match="not an integer"):
        runner.validate_ridge_moment_route_telemetry(
            {
                "n_ridge_moment_cv_fits": 24,
                "n_ridge_moment_eigen_path_cv_fits": 18.5,
                "n_ridge_moment_direct_cv_fits": 6,
            }
        )

    with pytest.raises(ValueError, match="negative"):
        runner.validate_ridge_moment_route_telemetry(
            {
                "n_ridge_moment_cv_fits": 24,
                "n_ridge_moment_eigen_path_cv_fits": 25,
                "n_ridge_moment_direct_cv_fits": -1,
            }
        )


def test_real_cohort_runner_can_skip_by_dataset_properties(monkeypatch):
    runner = _load_script(
        "run_aom_staged_real_cohort",
        "benchmarks/cross_binding/run_aom_staged_real_cohort.py",
    )

    X_train = np.ones((10, 7), dtype=np.float64)
    y_train = np.arange(10, dtype=np.float64)
    X_test = np.ones((4, 7), dtype=np.float64)
    y_test = np.arange(4, dtype=np.float64)

    def fake_load_split(data_root, database_name, dataset):
        return X_train, y_train, X_test, y_test, Path("/tmp/fake_split")

    def fail_campaign(*args, **kwargs):
        raise AssertionError("campaign should not run for skipped property rows")

    monkeypatch.setattr(runner, "load_split", fake_load_split)
    monkeypatch.setattr(runner.n4m, "aom_staged_chain_campaign", fail_campaign)
    monkeypatch.setattr(runner.n4m, "library_path", lambda: "build/cuda-on/libn4m.so")
    monkeypatch.setattr(runner.n4m, "abi_version", lambda: (1, 21, 0))

    args = SimpleNamespace(
        data_root="/tmp",
        checkpoint_dir=None,
        no_resume=True,
        plan="compact_wide",
        cv=3,
        ridge_lambdas="0.1",
        components="1",
        heads="pls",
        max_chains=4,
        chain_chunk_size=2,
        top_k=3,
        refit_top_k=2,
        refit_per_head_top_k=1,
        max_chunks_per_run=None,
        scale_x=False,
        moment_policy="auto",
        pls_score_mode="cv",
        split_head_scoring="force",
        cuda_pls_parallel_folds=True,
        cuda_pls_min_device_features=1,
        cuda_pls_many_batched=False,
        backend_min_cuda_product=1,
        max_train_samples=20,
        max_features=5,
        max_train_feature_product=200,
        stages=None,
        cohort_csv="cohort.csv",
        seed=0,
    )

    row = runner.run_one(args, {"database_name": "DB", "dataset": "wide", "key": "DB/wide"})

    assert row["status"] == "skipped"
    assert row["error_message"] == "property_filter:n_features>5"
    assert row["n_train"] == 10
    assert row["n_test"] == 4
    assert row["n_features"] == 7
    assert row["selection_uses_test_set"] is False
    assert row["plan"] == "compact_wide"
    assert row["pls_score_mode"] == "cv"
    assert row["split_head_scoring"] == "force"
    assert row["cuda_pls_parallel_folds"] is True
    assert row["cuda_pls_min_device_features"] == 1


def test_real_cohort_runner_loads_custom_stages_json(tmp_path):
    runner = _load_script(
        "run_aom_staged_real_cohort",
        "benchmarks/cross_binding/run_aom_staged_real_cohort.py",
    )

    inline_args = SimpleNamespace(
        stages_json='["compact", {"name": "sg", "profile": "lab", "max_chains": 4}]',
        stages_json_file="",
    )
    assert runner.load_stages_config(inline_args) == [
        "compact",
        {"name": "sg", "profile": "lab", "max_chains": 4},
    ]

    path = tmp_path / "stages.json"
    path.write_text('[{"profile": "wide", "heads": ["ridge"]}]', encoding="utf-8")
    file_args = SimpleNamespace(stages_json="", stages_json_file=str(path))
    assert runner.load_stages_config(file_args) == [
        {"profile": "wide", "heads": ["ridge"]}
    ]

    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.load_stages_config(
            SimpleNamespace(stages_json='["compact"]', stages_json_file=str(path))
        )
    with pytest.raises(ValueError, match="non-empty list"):
        runner.load_stages_config(SimpleNamespace(stages_json="{}", stages_json_file=""))


def test_real_cohort_runner_rejects_incomplete_audit_reports():
    runner = _load_script(
        "run_aom_staged_real_cohort",
        "benchmarks/cross_binding/run_aom_staged_real_cohort.py",
    )

    with pytest.raises(ValueError, match="eval.best_cv"):
        runner.selected_eval_row({"audit": {"eval": {}}})
    with pytest.raises(ValueError, match="best_eval"):
        runner.audit_oracle_row({"audit": {"eval": {"best_cv": {}}}})


def test_real_cohort_runner_diagnostics_compact_audit_payload(tmp_path, monkeypatch):
    """Diagnostics JSON includes compact audit when report['audit'] is present.

    Verifies:
    - ``audit.audit_only`` is ``True`` (offline only)
    - ``audit.selected_cv`` preserves scalar fields, strips prediction arrays
    - ``audit.oracle`` is present
    - ``audit.n_candidates`` and ``audit_rank_diagnostics`` forwarded when present
    - ``selection_uses_test_set`` stays ``False``
    - No ``audit`` key when report has none
    """
    runner = _load_script(
        "run_aom_staged_real_cohort",
        "benchmarks/cross_binding/run_aom_staged_real_cohort.py",
    )

    # Rich fake candidate rows - include a prediction array that must be stripped
    selected_row = {
        "head": "ridge",
        "param": 10.0,
        "refit_cv_rmse": 0.30,
        "eval_rmse": 0.35,
        "eval_r2": 0.88,
        "cv_rank": 1,
        "eval_rank": 2,
        "rank_delta": 1,
        "eval_predictions": np.array([0.1, 0.2, 0.3]),  # must be stripped
    }
    oracle_row = {
        "head": "pls",
        "param": 2.0,
        "refit_cv_rmse": 0.32,
        "eval_rmse": 0.28,
        "eval_r2": 0.92,
        "cv_rank": 3,
        "eval_rank": 1,
        "rank_delta": -2,
    }
    audit_rank_diag = {"spearman_rank_correlation": 0.6, "n_candidates": 5}
    report_with_audit = {
        "audit": {
            "audit_only": True,
            "note": "Offline holdout audit.",
            "n_candidates": 5,
            "best_eval": oracle_row,
            "eval": {"best_cv": selected_row, "best_eval": oracle_row},
            "rank_diagnostics": audit_rank_diag,
        }
    }

    # With audit present
    payload_with = runner._compact_audit_payload(report_with_audit["audit"])
    assert payload_with["audit_only"] is True
    assert payload_with["n_candidates"] == 5
    assert payload_with["note"] == "Offline holdout audit."
    assert payload_with["selected_cv"]["eval_rmse"] == pytest.approx(0.35)
    assert payload_with["selected_cv"]["eval_rank"] == 2
    assert payload_with["selected_cv"]["cv_rank"] == 1
    # prediction array must be stripped
    assert "eval_predictions" not in payload_with["selected_cv"]
    assert payload_with["oracle"]["eval_rmse"] == pytest.approx(0.28)
    assert payload_with["oracle"]["cv_rank"] == 3
    assert payload_with["audit_rank_diagnostics"]["spearman_rank_correlation"] == pytest.approx(0.6)

    # Full diagnostics_payload includes audit and keeps selection_uses_test_set=False
    class _FakeArgs:
        cohort_csv = "cohort.csv"
        cv = 3
        max_chains = 4
        top_k = 3
        refit_top_k = 2
        refit_per_head_top_k = 1
        split_head_scoring = "auto"
        cuda_pls_parallel_folds = True
        cuda_pls_min_device_features = 1
        cuda_pls_many_batched = False
        backend_min_cuda_product = 1

    fake_row = {
        "dataset_key": "DB/x",
        "database_name": "DB",
        "dataset": "x",
        "selection_uses_test_set": False,
        "plan": "compact",
        "heads": "ridge,pls",
        "scale_x": True,
        "scale_x_values": "",
        "selected_model_config_id": 0,
        "selected_head": "ridge",
        "selected_param": 10.0,
    }
    full_report = {
        **report_with_audit,
        "best": {"head": "ridge", "param": 10.0, "refit_cv_rmse": 0.30, "cv_rank": 1},
        "impact": None,
        "rank_diagnostics": None,
        "model_config_summaries": None,
        "selected_model_config": None,
        "route_summary": None,
    }
    diag = runner.diagnostics_payload(report=full_report, row=fake_row, args=_FakeArgs())
    assert diag["selection_uses_test_set"] is False
    assert "audit" in diag
    assert diag["audit"]["audit_only"] is True
    assert diag["audit"]["selected_cv"]["eval_rmse"] == pytest.approx(0.35)
    assert "eval_predictions" not in diag["audit"]["selected_cv"]

    # Without audit: key must be absent
    diag_no_audit = runner.diagnostics_payload(
        report={**full_report, "audit": None},
        row=fake_row,
        args=_FakeArgs(),
    )
    assert "audit" not in diag_no_audit


def test_impact_group_summarizer_counts_dataset_winners(tmp_path, monkeypatch):
    summarize = _load_script(
        "summarize_aom_impact_groups",
        "benchmarks/cross_binding/summarize_aom_impact_groups.py",
    )

    impact_groups = _write_csv(tmp_path / "impact_groups.csv", [
        {
            "dataset_key": "DB/alpha",
            "database_name": "DB",
            "dataset": "alpha",
            "selected_head": "ridge",
            "plan": "compact",
            "heads": "ridge,pls",
            "group_kind": "by_operator",
            "group": "detrend_poly",
            "n_candidates": "4",
            "best_score": "0.8",
            "mean_score": "1.0",
            "median_score": "1.0",
            "best_rank": "1",
            "mean_rank": "2.0",
            "improvement_vs_identity": "0.2",
            "selected_model_config_id": "0",
            "scale_x": "True",
            "selection_uses_test_set": "False",
        },
        {
            "dataset_key": "DB/alpha",
            "database_name": "DB",
            "dataset": "alpha",
            "selected_head": "ridge",
            "plan": "compact",
            "heads": "ridge,pls",
            "group_kind": "by_operator",
            "group": "identity",
            "n_candidates": "2",
            "best_score": "1.0",
            "mean_score": "1.1",
            "median_score": "1.1",
            "best_rank": "2",
            "mean_rank": "3.0",
            "improvement_vs_identity": "0.0",
            "selected_model_config_id": "0",
            "scale_x": "True",
            "selection_uses_test_set": "False",
        },
        {
            "dataset_key": "DB/beta",
            "database_name": "DB",
            "dataset": "beta",
            "selected_head": "ridge",
            "plan": "compact",
            "heads": "ridge,pls",
            "group_kind": "by_operator",
            "group": "identity",
            "n_candidates": "2",
            "best_score": "0.5",
            "mean_score": "0.7",
            "median_score": "0.7",
            "best_rank": "1",
            "mean_rank": "2.0",
            "improvement_vs_identity": "0.0",
            "selected_model_config_id": "1",
            "scale_x": "False",
            "selection_uses_test_set": "False",
        },
        {
            "dataset_key": "DB/beta",
            "database_name": "DB",
            "dataset": "beta",
            "selected_head": "ridge",
            "plan": "compact",
            "heads": "ridge,pls",
            "group_kind": "by_stage_option",
            "group": "smooth|savgol_smooth(7,2)",
            "n_candidates": "1",
            "best_score": "0.6",
            "mean_score": "0.6",
            "median_score": "0.6",
            "best_rank": "1",
            "mean_rank": "1.0",
            "improvement_vs_identity": "0.1",
            "selected_model_config_id": "1",
            "scale_x": "False",
            "selection_uses_test_set": "False",
        },
    ])
    output = tmp_path / "impact_summary.csv"
    summary_output = tmp_path / "impact_summary.md"

    rows = summarize.summarize_impact_rows(summarize.read_csv(impact_groups))
    by_key = {(row["group_kind"], row["group"]): row for row in rows}
    assert by_key[("by_operator", "detrend_poly")]["dataset_wins"] == 1
    assert by_key[("by_operator", "identity")]["dataset_wins"] == 1
    assert by_key[("by_operator", "identity")]["n_datasets"] == 2
    assert by_key[
        ("by_stage_option", "smooth|savgol_smooth(7,2)")
    ]["dataset_wins"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_aom_impact_groups.py",
            "--impact-groups",
            str(impact_groups),
            "--output",
            str(output),
            "--summary-output",
            str(summary_output),
            "--top-n",
            "3",
        ],
    )
    assert summarize.main() == 0

    written = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert {
        (row["group_kind"], row["group"]) for row in written
    } >= {
        ("by_operator", "detrend_poly"),
        ("by_operator", "identity"),
        ("by_stage_option", "smooth|savgol_smooth(7,2)"),
    }
    markdown = summary_output.read_text(encoding="utf-8")
    assert "## by_operator" in markdown
    assert "detrend_poly" in markdown
    assert "smooth\\|savgol_smooth(7,2)" in markdown


def test_staged_variant_comparator_groups_configs_and_sums_routes(tmp_path):
    compare = _load_script(
        "compare_aom_staged_variants",
        "benchmarks/cross_binding/compare_aom_staged_variants.py",
    )

    path = _write_csv(tmp_path / "variants.csv", [
        {
            "database_name": "DB",
            "dataset": "alpha",
            "dataset_id": "secret-alpha",
            "source_name": "cohort-alpha",
            "status": "ok",
            "plan": "compact",
            "heads": "ridge,pls",
            "max_chains": "12",
            "top_k": "8",
            "refit_top_k": "4",
            "refit_per_head_top_k": "2",
            "moment_policy": "auto",
            "cuda_pls_parallel_folds": "True",
            "cuda_pls_min_device_features": "1",
            "backend_min_cuda_product": "1",
            "fit_time_s": "2.5",
            "rmsep": "1.0",
            "n_screen_pls_moment_cv_fits": "10",
            "n_screen_pls_moment_cuda_device_cv_fits": "10",
            "n_screen_pls_moment_score_batch_calls": "2",
            "n_screen_pls_moment_score_batch_jobs": "10",
            "n_refit_pls_moment_cv_fits": "4",
            "n_refit_pls_moment_cuda_device_cv_fits": "4",
            "n_refit_pls_moment_score_batch_calls": "1",
            "n_refit_pls_moment_score_batch_jobs": "4",
        },
        {
            "database_name": "DB",
            "dataset": "beta",
            "dataset_id": "secret-beta",
            "source_name": "cohort-beta",
            "status": "skipped",
            "plan": "compact",
            "heads": "ridge,pls",
            "max_chains": "12",
            "top_k": "8",
            "refit_top_k": "4",
            "refit_per_head_top_k": "2",
            "moment_policy": "auto",
            "cuda_pls_parallel_folds": "True",
            "cuda_pls_min_device_features": "1",
            "backend_min_cuda_product": "1",
            "fit_time_s": "",
            "rmsep": "",
            "n_screen_pls_moment_cv_fits": "",
            "n_screen_pls_moment_cuda_device_cv_fits": "",
            "n_screen_pls_moment_score_batch_calls": "",
            "n_screen_pls_moment_score_batch_jobs": "",
            "n_refit_pls_moment_cv_fits": "",
            "n_refit_pls_moment_cuda_device_cv_fits": "",
            "n_refit_pls_moment_score_batch_calls": "",
            "n_refit_pls_moment_score_batch_jobs": "",
        },
        {
            "database_name": "DB",
            "dataset": "gamma",
            "dataset_id": "secret-gamma",
            "source_name": "cohort-gamma",
            "status": "ok",
            "plan": "wide",
            "heads": "ridge",
            "max_chains": "24",
            "top_k": "16",
            "refit_top_k": "8",
            "refit_per_head_top_k": "3",
            "moment_policy": "auto",
            "cuda_pls_parallel_folds": "True",
            "cuda_pls_min_device_features": "1",
            "backend_min_cuda_product": "1",
            "fit_time_s": "3.5",
            "rmsep": "0.8",
            "n_screen_pls_moment_cv_fits": "5",
            "n_screen_pls_moment_cuda_device_cv_fits": "5",
            "n_screen_pls_moment_score_batch_calls": "1",
            "n_screen_pls_moment_score_batch_jobs": "5",
            "n_refit_pls_moment_cv_fits": "2",
            "n_refit_pls_moment_cuda_device_cv_fits": "2",
            "n_refit_pls_moment_score_batch_calls": "1",
            "n_refit_pls_moment_score_batch_jobs": "2",
        },
    ])

    records = compare.load_input_rows([f"trial={path}"])
    rows = compare.summarize_variant_groups(records, score_column="rmsep")
    by_plan = {}
    for row in rows:
        config = json.loads(row["config_key"])
        by_plan[config["plan"]] = row

    compact = by_plan["compact"]
    assert compact["n_rows"] == 2
    assert compact["n_ok"] == 1
    assert compact["n_skipped"] == 1
    assert compact["n_error"] == 0
    assert compact["median_score"] == pytest.approx(1.0)
    assert compact["total_n_screen_pls_moment_cv_fits"] == 10
    assert compact["total_n_screen_pls_moment_cuda_device_cv_fits"] == 10
    assert compact["total_n_screen_pls_moment_score_batch_calls"] == 2
    assert compact["total_n_screen_pls_moment_score_batch_jobs"] == 10
    assert compact["total_n_refit_pls_moment_cv_fits"] == 4
    assert compact["total_n_refit_pls_moment_cuda_device_cv_fits"] == 4
    assert compact["total_n_refit_pls_moment_score_batch_calls"] == 1
    assert compact["total_n_refit_pls_moment_score_batch_jobs"] == 4
    assert "alpha" not in compact["config_key"]
    assert "beta" not in compact["config_key"]
    assert "secret" not in compact["config_key"]
    assert "source_name" not in compact["config_key"]
    assert compare.config_uses_only_campaign_columns(compact["config_key"])


def test_staged_variant_comparator_pairs_against_baseline_label(tmp_path):
    compare = _load_script(
        "compare_aom_staged_variants",
        "benchmarks/cross_binding/compare_aom_staged_variants.py",
    )

    baseline = _write_csv(tmp_path / "baseline.csv", [
        {
            "database_name": "DB",
            "dataset": "one",
            "status": "ok",
            "plan": "compact",
            "heads": "ridge",
            "max_chains": "12",
            "rmsep": "1.0",
        },
        {
            "database_name": "DB",
            "dataset": "two",
            "status": "ok",
            "plan": "compact",
            "heads": "ridge",
            "max_chains": "12",
            "rmsep": "1.0",
        },
        {
            "database_name": "DB",
            "dataset": "three",
            "status": "ok",
            "plan": "compact",
            "heads": "ridge",
            "max_chains": "12",
            "rmsep": "1.0",
        },
    ])
    candidate = _write_csv(tmp_path / "candidate.csv", [
        {
            "database_name": "DB",
            "dataset": "one",
            "status": "ok",
            "plan": "wide",
            "heads": "ridge,pls",
            "max_chains": "24",
            "rmsep": "0.8",
        },
        {
            "database_name": "DB",
            "dataset": "two",
            "status": "ok",
            "plan": "wide",
            "heads": "ridge,pls",
            "max_chains": "24",
            "rmsep": "1.2",
        },
        {
            "database_name": "DB",
            "dataset": "three",
            "status": "ok",
            "plan": "wide",
            "heads": "ridge,pls",
            "max_chains": "24",
            "rmsep": "1.0",
        },
    ])

    records = compare.load_input_rows([f"base={baseline}", f"trial={candidate}"])
    baseline_scores = compare.baseline_from_label(
        records,
        label="base",
        score_column="rmsep",
    )
    rows = compare.summarize_variant_groups(
        records,
        score_column="rmsep",
        baseline=baseline_scores,
    )
    trial = next(row for row in rows if row["source_label"] == "trial")

    assert trial["n_paired_baseline"] == 3
    assert trial["wins_vs_baseline"] == 1
    assert trial["losses_vs_baseline"] == 1
    assert trial["ties_vs_baseline"] == 1
    assert trial["median_ratio_vs_baseline"] == pytest.approx(1.0)
    assert trial["mean_ratio_vs_baseline"] == pytest.approx(1.0)


def test_staged_variant_comparator_groups_custom_skips_with_custom_ok(tmp_path):
    compare = _load_script(
        "compare_aom_staged_variants",
        "benchmarks/cross_binding/compare_aom_staged_variants.py",
    )

    stages_json = '[{"name":"sg","profile":"lab","max_chains":8}]'
    path = _write_csv(tmp_path / "custom.csv", [
        {
            "database_name": "DB",
            "dataset": "ok-row",
            "status": "ok",
            "plan": "custom",
            "stages_json": stages_json,
            "heads": "ridge,pls",
            "max_chains": "8",
            "rmsep": "0.9",
        },
        {
            "database_name": "DB",
            "dataset": "skipped-row",
            "status": "skipped",
            "plan": "compact",
            "stages_json": stages_json,
            "heads": "ridge,pls",
            "max_chains": "8",
            "rmsep": "",
        },
    ])

    records = compare.load_input_rows([f"custom={path}"])
    rows = compare.summarize_variant_groups(records, score_column="rmsep")

    assert len(rows) == 1
    assert rows[0]["n_ok"] == 1
    assert rows[0]["n_skipped"] == 1
    assert json.loads(rows[0]["config_key"])["plan"] == "custom"


def test_staged_variant_comparator_distinguishes_pls_score_mode(tmp_path):
    compare = _load_script(
        "compare_aom_staged_variants",
        "benchmarks/cross_binding/compare_aom_staged_variants.py",
    )

    # Three rows identical in every campaign config column except
    # ``pls_score_mode``: two exact-CV (``cv``) rows on different datasets and
    # one proxy (``gcv_proxy``) row. Exact-vs-proxy must NOT collapse into one
    # variant, and dataset identity/source/name must never enter the config key.
    path = _write_csv(tmp_path / "score_modes.csv", [
        {
            "database_name": "DB",
            "dataset": "alpha",
            "dataset_id": "secret-alpha",
            "source_name": "cohort-alpha",
            "status": "ok",
            "plan": "compact",
            "heads": "ridge,pls",
            "max_chains": "12",
            "moment_policy": "auto",
            "pls_score_mode": "cv",
            "rmsep": "1.0",
        },
        {
            "database_name": "DB",
            "dataset": "beta",
            "dataset_id": "secret-beta",
            "source_name": "cohort-beta",
            "status": "ok",
            "plan": "compact",
            "heads": "ridge,pls",
            "max_chains": "12",
            "moment_policy": "auto",
            "pls_score_mode": "cv",
            "rmsep": "1.2",
        },
        {
            "database_name": "DB",
            "dataset": "gamma",
            "dataset_id": "secret-gamma",
            "source_name": "cohort-gamma",
            "status": "ok",
            "plan": "compact",
            "heads": "ridge,pls",
            "max_chains": "12",
            "moment_policy": "auto",
            "pls_score_mode": "gcv_proxy",
            "rmsep": "0.9",
        },
    ])

    records = compare.load_input_rows([f"trial={path}"])
    rows = compare.summarize_variant_groups(records, score_column="rmsep")

    by_mode = {}
    for row in rows:
        config = json.loads(row["config_key"])
        by_mode[config["pls_score_mode"]] = row

    # cv and gcv_proxy resolve to two distinct variants; same plan does not merge.
    assert set(by_mode) == {"cv", "gcv_proxy"}
    assert len(rows) == 2

    cv_row = by_mode["cv"]
    proxy_row = by_mode["gcv_proxy"]
    assert cv_row["variant_id"] != proxy_row["variant_id"]

    # Two different cv datasets collapse into one variant: grouping is config-only.
    assert cv_row["n_rows"] == 2
    assert cv_row["n_ok"] == 2
    assert proxy_row["n_rows"] == 1
    assert proxy_row["n_ok"] == 1

    # Labels disambiguate the score mode so summaries are not ambiguous.
    assert "pls_score_mode=cv" in cv_row["variant_label"]
    assert "pls_score_mode=gcv_proxy" in proxy_row["variant_label"]

    # No dataset-name/source routing: identity tokens never enter the config key.
    for row in rows:
        assert "alpha" not in row["config_key"]
        assert "beta" not in row["config_key"]
        assert "gamma" not in row["config_key"]
        assert "secret" not in row["config_key"]
        assert "source_name" not in row["config_key"]
        assert "cohort-" not in row["config_key"]
        assert compare.config_uses_only_campaign_columns(row["config_key"])


def test_rank_audit_summarizer_reads_diagnostics_and_writes_summary(tmp_path, monkeypatch):
    """summarize_aom_rank_audit reads diagnostics JSON files with audit payloads.

    Verifies:
    - Rows with ``audit`` section produce rank-comparison output rows.
    - ``selection_uses_test_set`` is always ``False`` in output.
    - ``test_rank_delta`` = ``selected_test_rank - 1``.
    - ``oracle_gap_ratio`` = ``(selected_test_rmse - oracle_test_rmse) / oracle_test_rmse``.
    - Files without ``audit`` are counted but not written as output rows.
    - CSV and Markdown outputs are written.
    - The script advertises itself as offline audit only.
    """
    summarize = _load_script(
        "summarize_aom_rank_audit",
        "benchmarks/cross_binding/summarize_aom_rank_audit.py",
    )

    diag_dir = tmp_path / "diag"
    diag_dir.mkdir()

    # File A: has full audit payload; CV winner slipped to test rank 3
    diag_a = {
        "dataset_key": "DB/alpha",
        "database_name": "DB",
        "dataset": "alpha",
        "selected_head": "ridge",
        "plan": "compact",
        "selection_uses_test_set": False,
        "best": {"refit_cv_rmse": 0.50, "cv_rank": 1},
        "audit": {
            "audit_only": True,
            "n_candidates": 6,
            "selected_cv": {
                "head": "ridge",
                "param": 0.1,
                "chain": [["savgol_smooth", [7, 2]]],
                "eval_rmse": 0.60,
                "eval_r2": 0.80,
                "cv_rank": 1,
                "eval_rank": 3,
                "rank_delta": 2,
            },
            "oracle": {
                "head": "pls",
                "param": 1.0,
                "chain": [["detrend_poly", [2]]],
                "eval_rmse": 0.45,
                "eval_r2": 0.90,
                "cv_rank": 4,
                "eval_rank": 1,
                "rank_delta": -3,
            },
            "audit_rank_diagnostics": {
                "spearman_rank_correlation": 0.7,
                "topk": [
                    {"k": 1, "eval_top_k_recall": 0.0},
                    {"k": 3, "eval_top_k_recall": 0.3333333333333333},
                    {"k": 5, "eval_top_k_recall": 0.6},
                ],
            },
        },
    }
    # File B: has audit with CV winner = oracle (test_rank_delta=0)
    diag_b = {
        "dataset_key": "DB/beta",
        "database_name": "DB",
        "dataset": "beta",
        "selected_head": "pls",
        "plan": "compact",
        "selection_uses_test_set": False,
        "best": {"refit_cv_rmse": 0.20, "cv_rank": 1},
        "audit": {
            "audit_only": True,
            "n_candidates": 4,
            "selected_cv": {
                "head": "pls",
                "param": 2.0,
                "chain": [["identity", []]],
                "eval_rmse": 0.22,
                "cv_rank": 1,
                "eval_rank": 1,
                "rank_delta": 0,
            },
            "oracle": {
                "head": "pls",
                "param": 2.0,
                "chain": [["identity", []]],
                "eval_rmse": 0.22,
                "cv_rank": 1,
                "eval_rank": 1,
                "rank_delta": 0,
            },
        },
    }
    # File C: no audit section (pre-feature diagnostics)
    diag_c = {
        "dataset_key": "DB/gamma",
        "database_name": "DB",
        "dataset": "gamma",
        "selected_head": "ridge",
        "plan": "compact",
        "selection_uses_test_set": False,
        "best": {"refit_cv_rmse": 0.40, "cv_rank": 1},
    }
    for name, diag in [("DB_alpha.diagnostics.json", diag_a),
                        ("DB_beta.diagnostics.json", diag_b),
                        ("DB_gamma.diagnostics.json", diag_c)]:
        (diag_dir / name).write_text(json.dumps(diag), encoding="utf-8")

    output = tmp_path / "rank_audit.csv"
    summary_md = tmp_path / "rank_audit.md"

    monkeypatch.setattr(
        sys, "argv",
        [
            "summarize_aom_rank_audit.py",
            "--diagnostics-dir", str(diag_dir),
            "--output", str(output),
            "--summary-output", str(summary_md),
        ],
    )
    assert summarize.main() == 0

    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    # only 2 rows with audit (gamma has none)
    assert len(rows) == 2
    by_key = {row["dataset_key"]: row for row in rows}

    # alpha: CV winner slipped to test rank 3
    a = by_key["DB/alpha"]
    assert a["selection_uses_test_set"] == "False"
    assert int(a["selected_test_rank"]) == 3
    assert int(a["test_rank_delta"]) == 2
    assert a["selected_head"] == "ridge"
    assert float(a["selected_param"]) == pytest.approx(0.1)
    assert a["selected_chain"] == "savgol_smooth(7,2)"
    assert a["oracle_head"] == "pls"
    assert float(a["oracle_param"]) == pytest.approx(1.0)
    assert a["oracle_chain"] == "detrend_poly(2)"
    assert float(a["selected_cv_rmse"]) == pytest.approx(0.50)
    assert float(a["selected_test_rmse"]) == pytest.approx(0.60)
    assert float(a["oracle_test_rmse"]) == pytest.approx(0.45)
    # oracle_gap_ratio = (0.60 - 0.45) / 0.45
    assert float(a["oracle_gap_ratio"]) == pytest.approx((0.60 - 0.45) / 0.45)
    assert int(a["n_audit_candidates"]) == 6
    assert float(a["audit_spearman"]) == pytest.approx(0.7)
    assert float(a["audit_top1_recall"]) == pytest.approx(0.0)
    assert float(a["audit_top3_recall"]) == pytest.approx(1 / 3)
    assert float(a["audit_top5_recall"]) == pytest.approx(0.6)

    # beta: CV winner is oracle (rank delta = 0)
    b = by_key["DB/beta"]
    assert b["selection_uses_test_set"] == "False"
    assert int(b["selected_test_rank"]) == 1
    assert int(b["test_rank_delta"]) == 0
    assert float(b["oracle_gap_ratio"]) == pytest.approx(0.0)
    assert b["selected_chain"] == "identity"
    assert b["oracle_chain"] == "identity"

    # Markdown mentions offline audit
    md = summary_md.read_text(encoding="utf-8")
    assert "offline" in md.lower()
    assert "selection_uses_test_set" in md
    assert "Rows with audit data: **2** of 3 total." in md
    assert "1 file(s) lacked an `audit` section" in md
    assert "savgol_smooth(7,2)" in md
    assert "detrend_poly(2)" in md
    assert "DB/alpha" in md or "alpha" in md
    # Mismatch aggregate section (post-hoc audit only; diag_a has delta>0, diag_b does not)
    assert "## Mismatch patterns" in md
    assert "ridge -> pls" in md
    assert "savgol_smooth(7,2) -> detrend_poly(2)" in md
