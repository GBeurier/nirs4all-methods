# SPDX-License-Identifier: CECILL-2.1
"""Sklearn regressors backed by native moment/AOM sweep MethodResults."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import python as _native
from ._compat import BaseEstimator


def _as_X(X) -> np.ndarray:
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    return np.ascontiguousarray(X_arr)


def _as_y(y, n_samples: int) -> tuple[np.ndarray, bool]:
    y_arr = np.asarray(y, dtype=np.float64)
    y_was_1d = y_arr.ndim == 1
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    if y_arr.ndim != 2:
        raise ValueError("y must be 1D or 2D")
    if y_arr.shape[0] != int(n_samples):
        raise ValueError("X and y have incompatible lengths")
    return np.ascontiguousarray(y_arr), y_was_1d


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    residual = yt - yp
    centered = yt - float(np.mean(yt))
    denom = float(centered @ centered)
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - (residual @ residual) / denom)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(residual * residual)))


def _normalize_aom_head(head) -> str:
    if head in (0, "0", "ridge"):
        return "ridge"
    if head in (1, "1", "pls"):
        return "pls"
    raise ValueError("head must be 'ridge' or 'pls'")


def _aom_head_id_name(head_id) -> str:
    try:
        return _normalize_aom_head(int(head_id))
    except (TypeError, ValueError):
        return f"head_{head_id}"


def _aom_profile_name(profile) -> str:
    if profile in (0, "0", "compact"):
        return "compact"
    if profile in (1, "1", "wide", "lab", "moment_lab", "large"):
        return "wide"
    return str(profile)


def _aom_profile_expected_bank_size(profile) -> int | None:
    name = _aom_profile_name(profile)
    if name == "compact":
        return 12
    if name == "wide":
        return 31
    return None


def _candidate_chain(candidate: dict):
    if "chain" not in candidate:
        raise ValueError("candidate must contain a decoded 'chain'")
    chain = candidate["chain"]
    if isinstance(chain, tuple):
        chain = list(chain)
    if not chain:
        raise ValueError("candidate chain must not be empty")
    return chain


def _candidate_head(candidate: dict) -> str:
    if "head" in candidate:
        return _normalize_aom_head(candidate["head"])
    if "head_id" in candidate:
        return _normalize_aom_head(candidate["head_id"])
    raise ValueError("candidate must contain 'head' or 'head_id'")


class _NativeLinearResultRegressor(BaseEstimator):
    """Base class for native sweep results that export linear input-space state."""

    _result_coef_key = "coefficients"

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        raise NotImplementedError

    def fit(self, X, y):
        X_arr = _as_X(X)
        y_arr, y_was_1d = _as_y(y, X_arr.shape[0])
        result = self._fit_native_result(X_arr, y_arr)
        coef = np.asarray(result[self._result_coef_key], dtype=np.float64)
        intercept = np.asarray(result["intercept"], dtype=np.float64).reshape(-1)
        if coef.ndim != 2 or coef.shape[0] != X_arr.shape[1]:
            raise ValueError("native result coefficients do not match X")
        if intercept.size != coef.shape[1]:
            raise ValueError("native result intercept does not match coefficients")

        self.result_ = result
        self._coef_matrix_ = np.ascontiguousarray(coef)
        self.intercept_ = float(intercept[0]) if intercept.size == 1 else intercept.copy()
        self.coef_ = (
            self._coef_matrix_[:, 0].copy()
            if self._coef_matrix_.shape[1] == 1
            else self._coef_matrix_.T.copy()
        )
        self.predictions_ = np.asarray(result["predictions"], dtype=np.float64).copy()
        self.oof_predictions_ = np.asarray(result["oof_predictions"], dtype=np.float64).copy()
        self.candidate_scores_ = np.asarray(result["candidate_scores"], dtype=np.float64).copy()
        self.selected_candidate_id_ = int(result["selected_candidate_id"])
        self.selected_head_id_ = int(result["selected_head_id"])
        self.selected_param_ = float(result["selected_param"])
        self.selected_cv_rmse_ = float(result["selected_cv_rmse"])
        self.n_features_in_ = int(X_arr.shape[1])
        self.n_targets_ = int(self._coef_matrix_.shape[1])
        self._y_was_1d_ = bool(y_was_1d)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "_coef_matrix_"):
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        X_arr = _as_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        preds = X_arr @ self._coef_matrix_ + np.asarray(self.intercept_).reshape(1, -1)
        if self._y_was_1d_ and preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def score(self, X, y) -> float:
        return _r2_score(y, self.predict(X))

    def get_diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "result_"):
            raise RuntimeError(f"{type(self).__name__}.get_diagnostics called before fit")
        return {
            "selected_candidate_id": self.selected_candidate_id_,
            "selected_head_id": self.selected_head_id_,
            "selected_param": self.selected_param_,
            "selected_cv_rmse": self.selected_cv_rmse_,
            "n_candidates": int(self.result_.get("n_candidates", 0.0)),
            "n_ridge_moment_candidates": int(
                self.result_.get("n_ridge_moment_candidates", 0.0)
            ),
            "n_ridge_dual_materialized_candidates": int(
                self.result_.get("n_ridge_dual_materialized_candidates", 0.0)
            ),
            "n_ridge_moment_cv_fits": int(
                self.result_.get("n_ridge_moment_cv_fits", 0.0)
            ),
            "n_ridge_dual_materialized_cv_fits": int(
                self.result_.get("n_ridge_dual_materialized_cv_fits", 0.0)
            ),
            "n_ridge_dual_cross_cv_fits": int(
                self.result_.get("n_ridge_dual_cross_cv_fits", 0.0)
            ),
            "n_ridge_moment_score_batch_calls": int(
                self.result_.get("n_ridge_moment_score_batch_calls", 0.0)
            ),
            "n_ridge_moment_score_batch_jobs": int(
                self.result_.get("n_ridge_moment_score_batch_jobs", 0.0)
            ),
            "n_ridge_moment_final_fits": int(
                self.result_.get("n_ridge_moment_final_fits", 0.0)
            ),
            "n_ridge_dual_materialized_final_fits": int(
                self.result_.get("n_ridge_dual_materialized_final_fits", 0.0)
            ),
            "n_pls_moment_cv_fits": int(self.result_.get("n_pls_moment_cv_fits", 0.0)),
            "n_pls_moment_host_cv_fits": int(self.result_.get("n_pls_moment_host_cv_fits", 0.0)),
            "n_pls_moment_cuda_device_cv_fits": int(self.result_.get("n_pls_moment_cuda_device_cv_fits", 0.0)),
            "n_pls_moment_cuda_parallel_fold_batches": int(self.result_.get("n_pls_moment_cuda_parallel_fold_batches", 0.0)),
            "n_pls_moment_cuda_parallel_fold_jobs": int(self.result_.get("n_pls_moment_cuda_parallel_fold_jobs", 0.0)),
            "n_pls_moment_score_batch_calls": int(self.result_.get("n_pls_moment_score_batch_calls", 0.0)),
            "n_pls_moment_score_batch_jobs": int(self.result_.get("n_pls_moment_score_batch_jobs", 0.0)),
            "n_pls_materialized_cv_fits": int(self.result_.get("n_pls_materialized_cv_fits", 0.0)),
            "n_pls_gcv_proxy_candidates": int(self.result_.get("n_pls_gcv_proxy_candidates", 0.0)),
            "n_pls_gcv_proxy_fits": int(self.result_.get("n_pls_gcv_proxy_fits", 0.0)),
            "n_pls_gcv_proxy_batch_calls": int(self.result_.get("n_pls_gcv_proxy_batch_calls", 0.0)),
            "n_pls_gcv_proxy_batch_jobs": int(self.result_.get("n_pls_gcv_proxy_batch_jobs", 0.0)),
            "n_pls_moment_final_fits": int(self.result_.get("n_pls_moment_final_fits", 0.0)),
            "n_pls_moment_host_final_fits": int(self.result_.get("n_pls_moment_host_final_fits", 0.0)),
            "n_pls_moment_cuda_device_final_fits": int(self.result_.get("n_pls_moment_cuda_device_final_fits", 0.0)),
            "n_pls_materialized_final_fits": int(self.result_.get("n_pls_materialized_final_fits", 0.0)),
            "cv": int(self.result_.get("cv", 0.0)),
            "n_features": int(self.result_.get("n_features", self.n_features_in_)),
            "n_targets": int(self.result_.get("n_targets", self.n_targets_)),
            "cuda_pls_min_device_features": getattr(
                self, "cuda_pls_min_device_features", None
            ),
            "cuda_pls_many_batched": getattr(
                self, "cuda_pls_many_batched", None
            ),
        }


class _NativeDirectMomentRegressor(BaseEstimator):
    """Base class for direct native moment/linear heads with reusable coefficients."""

    _method_name = "native_direct"
    _result_coef_key = "coefficients"
    _diagnostic_scalar_keys: tuple[str, ...] = ()

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        raise NotImplementedError

    def _config_diagnostics(self) -> dict[str, object]:
        return {}

    @staticmethod
    def _result_vector(result: dict, key: str, expected_size: int) -> np.ndarray:
        vec = np.asarray(result[key], dtype=np.float64).reshape(-1)
        if vec.size != int(expected_size):
            raise ValueError(f"native result {key} does not match coefficients")
        return vec

    @staticmethod
    def _scalar_value(value, *, integer: bool = False):
        arr = np.asarray(value)
        if arr.size != 1:
            return np.asarray(value).copy()
        scalar = float(arr.reshape(-1)[0])
        return int(scalar) if integer else scalar

    def fit(self, X, y):
        X_arr = _as_X(X)
        y_arr, y_was_1d = _as_y(y, X_arr.shape[0])
        result = self._fit_native_result(X_arr, y_arr)
        coef = np.asarray(result[self._result_coef_key], dtype=np.float64)
        if coef.ndim == 1:
            coef = coef.reshape(-1, 1)
        if coef.ndim != 2 or coef.shape[0] != X_arr.shape[1]:
            raise ValueError("native result coefficients do not match X")

        if "x_mean" in result and "y_mean" in result:
            x_mean = self._result_vector(result, "x_mean", X_arr.shape[1])
            y_mean = self._result_vector(result, "y_mean", coef.shape[1])
            folded_intercept = y_mean - x_mean @ coef
        else:
            x_mean = np.zeros(X_arr.shape[1], dtype=np.float64)
            y_mean = np.zeros(coef.shape[1], dtype=np.float64)
            folded_intercept = None

        if "intercept" in result:
            intercept = np.asarray(result["intercept"], dtype=np.float64).reshape(-1)
            if intercept.size != coef.shape[1]:
                raise ValueError("native result intercept does not match coefficients")
        elif folded_intercept is not None:
            intercept = np.asarray(folded_intercept, dtype=np.float64).reshape(-1)
        else:
            raise ValueError("native result must expose an intercept or x/y means")

        self.result_ = result
        self._coef_matrix_ = np.ascontiguousarray(coef)
        self.input_coefficients_ = self._coef_matrix_.copy()
        self.intercept_ = float(intercept[0]) if intercept.size == 1 else intercept.copy()
        self.coef_ = (
            self._coef_matrix_[:, 0].copy()
            if self._coef_matrix_.shape[1] == 1
            else self._coef_matrix_.T.copy()
        )
        self.predictions_ = np.asarray(result["predictions"], dtype=np.float64).copy()
        self.rmse_ = float(result.get("rmse", np.nan))
        self.x_mean_ = x_mean.copy()
        self.y_mean_ = y_mean.copy()
        if "x_scale" in result:
            self.x_scale_ = np.asarray(result["x_scale"], dtype=np.float64).reshape(-1).copy()
        if "y_scale" in result:
            self.y_scale_ = np.asarray(result["y_scale"], dtype=np.float64).reshape(-1).copy()
        self.n_features_in_ = int(X_arr.shape[1])
        self.n_targets_ = int(self._coef_matrix_.shape[1])
        self._y_was_1d_ = bool(y_was_1d)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "_coef_matrix_"):
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        X_arr = _as_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        preds = X_arr @ self._coef_matrix_ + np.asarray(self.intercept_).reshape(1, -1)
        if self._y_was_1d_ and preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def score(self, X, y) -> float:
        return _r2_score(y, self.predict(X))

    def get_diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "result_"):
            raise RuntimeError(f"{type(self).__name__}.get_diagnostics called before fit")
        report: dict[str, object] = {
            "method": self._method_name,
            "rmse": self.rmse_,
            "n_samples": int(self.result_.get("n_samples", self.predictions_.shape[0])),
            "n_features": int(self.result_.get("n_features", self.n_features_in_)),
            "n_targets": int(self.result_.get("n_targets", self.n_targets_)),
        }
        report.update(self._config_diagnostics())
        for key in self._diagnostic_scalar_keys:
            if key in self.result_:
                report[key] = self._scalar_value(
                    self.result_[key],
                    integer=key in {"n_samples", "n_features", "n_targets", "n_components"},
                )
        return report


class NativeRidgeRegressor(_NativeDirectMomentRegressor):
    """Sklearn-style wrapper for the native closed-form Ridge head."""

    _method_name = "ridge"
    _diagnostic_scalar_keys = ("lambda",)

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        return _native.ridge(
            X,
            y,
            alpha=self.alpha,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
        )

    def _config_diagnostics(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "center_x": self.center_x,
            "scale_x": self.scale_x,
            "center_y": self.center_y,
        }


class NativePCRRegressor(_NativeDirectMomentRegressor):
    """Sklearn-style wrapper for native Principal Components Regression."""

    _method_name = "pcr"
    _diagnostic_scalar_keys = ("n_components",)

    def __init__(
        self,
        *,
        n_components: int = 2,
        center_x: bool | None = True,
        scale_x: bool | None = True,
        center_y: bool | None = True,
        scale_y: bool | None = False,
    ) -> None:
        self.n_components = int(n_components)
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        return _native.pcr(
            X,
            y,
            n_components=self.n_components,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
        )

    def _config_diagnostics(self) -> dict[str, object]:
        return {
            "n_components": self.n_components,
            "center_x": self.center_x,
            "scale_x": self.scale_x,
            "center_y": self.center_y,
            "scale_y": self.scale_y,
        }


class NativeCPPLSRegressor(_NativeDirectMomentRegressor):
    """Sklearn-style wrapper for native Powered PLS."""

    _method_name = "cppls"
    _diagnostic_scalar_keys = ("gamma",)

    def __init__(self, *, gamma: float = 0.5, n_components: int = 2) -> None:
        self.gamma = float(gamma)
        self.n_components = int(n_components)

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        return _native.cppls(
            X,
            y,
            gamma=self.gamma,
            n_components=self.n_components,
        )

    def _config_diagnostics(self) -> dict[str, object]:
        return {
            "gamma": self.gamma,
            "n_components": self.n_components,
        }


class NativeContinuumRegressionRegressor(_NativeDirectMomentRegressor):
    """Sklearn-style wrapper for native continuum regression."""

    _method_name = "continuum_regression"
    _diagnostic_scalar_keys = ("tau",)

    def __init__(self, *, tau: float = 0.5, n_components: int = 2) -> None:
        self.tau = float(tau)
        self.n_components = int(n_components)

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        return _native.continuum_regression(
            X,
            y,
            tau=self.tau,
            n_components=self.n_components,
        )

    def _config_diagnostics(self) -> dict[str, object]:
        return {
            "tau": self.tau,
            "n_components": self.n_components,
        }


class NativeECRRegressor(_NativeDirectMomentRegressor):
    """Sklearn-style wrapper for native Elastic Component Regression."""

    _method_name = "ecr"
    _diagnostic_scalar_keys = ("alpha", "n_components")

    def __init__(self, *, alpha: float = 0.5, n_components: int = 2) -> None:
        self.alpha = float(alpha)
        self.n_components = int(n_components)

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        return _native.ecr(
            X,
            y,
            alpha=self.alpha,
            n_components=self.n_components,
        )

    def _config_diagnostics(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "n_components": self.n_components,
        }


class NativeMomentSweepRegressor(_NativeLinearResultRegressor):
    """Native `sweep_run` Ridge/PLS screening estimator.

    The fit uses the C/CUDA libn4m sweep path and stores the selected
    input-space linear model for normal sklearn-style `predict(X_new)`.
    """

    _result_coef_key = "coefficients"

    def __init__(
        self,
        *,
        cv: int = 5,
        fold_ids=None,
        ridge_lambdas: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
        pls_components: Sequence[int] | None = None,
        heads: Sequence[str] = ("ridge",),
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
        cuda_pls_parallel_folds: bool | None = None,
        cuda_pls_min_device_features: int | None = None,
        cuda_pls_many_batched: bool | None = None,
    ) -> None:
        self.cv = int(cv)
        self.fold_ids = fold_ids
        self.ridge_lambdas = tuple(float(v) for v in ridge_lambdas)
        self.pls_components = None if pls_components is None else tuple(int(v) for v in pls_components)
        self.heads = tuple(heads)
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.cuda_pls_parallel_folds = cuda_pls_parallel_folds
        self.cuda_pls_min_device_features = cuda_pls_min_device_features
        self.cuda_pls_many_batched = cuda_pls_many_batched

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        return _native.sweep_run(
            X,
            y,
            cv=self.cv,
            fold_ids=self.fold_ids,
            ridge_lambdas=self.ridge_lambdas,
            pls_components=self.pls_components,
            heads=self.heads,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
            cuda_pls_parallel_folds=self.cuda_pls_parallel_folds,
            cuda_pls_min_device_features=self.cuda_pls_min_device_features,
            cuda_pls_many_batched=self.cuda_pls_many_batched,
        )


class NativeMomentStackRegressor(BaseEstimator):
    """OOF linear stack over native moment-compatible base regressors.

    The heavy base predictions are produced by native Ridge/PLS/PCR/continuum/
    ECR/CPPLS paths. The meta-model is a small train-only Ridge solve over OOF
    predictions, so fitting never selects from test data or dataset identity.
    """

    _ALIASES = {
        "ridge": "ridge",
        "pls": "pls",
        "pcr": "pcr",
        "continuum": "continuum",
        "continuum_regression": "continuum",
        "ecr": "ecr",
        "cppls": "cppls",
    }

    def __init__(
        self,
        *,
        base_models: Sequence[str] = (
            "ridge",
            "pls",
            "pcr",
            "continuum",
            "ecr",
            "cppls",
        ),
        cv: int = 5,
        fold_ids=None,
        inner_cv: int = 3,
        meta_alpha: float = 1e-6,
        ridge_alpha: float = 0.1,
        n_components: int = 3,
        pls_components: Sequence[int] | None = None,
        cppls_gamma: float = 0.5,
        continuum_tau: float = 0.5,
        ecr_alpha: float = 0.5,
        center_x: bool | None = None,
        scale_x: bool | None = False,
        center_y: bool | None = None,
        scale_y: bool | None = False,
        cuda_pls_parallel_folds: bool | None = None,
        cuda_pls_min_device_features: int | None = None,
        cuda_pls_many_batched: bool | None = None,
    ) -> None:
        self.base_models = tuple(base_models)
        self.cv = int(cv)
        self.fold_ids = fold_ids
        self.inner_cv = int(inner_cv)
        self.meta_alpha = float(meta_alpha)
        self.ridge_alpha = float(ridge_alpha)
        self.n_components = int(n_components)
        self.pls_components = (
            None if pls_components is None else tuple(int(v) for v in pls_components)
        )
        self.cppls_gamma = float(cppls_gamma)
        self.continuum_tau = float(continuum_tau)
        self.ecr_alpha = float(ecr_alpha)
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.cuda_pls_parallel_folds = cuda_pls_parallel_folds
        self.cuda_pls_min_device_features = cuda_pls_min_device_features
        self.cuda_pls_many_batched = cuda_pls_many_batched

    @classmethod
    def _normalize_base_models(cls, base_models: Sequence[str]) -> tuple[str, ...]:
        normalized = []
        seen: set[str] = set()
        for name in base_models:
            key = str(name).strip().lower()
            if key not in cls._ALIASES:
                allowed = ", ".join(sorted(cls._ALIASES))
                raise ValueError(f"unknown base model {name!r}; expected one of {allowed}")
            canonical = cls._ALIASES[key]
            if canonical in seen:
                continue
            seen.add(canonical)
            normalized.append(canonical)
        if not normalized:
            raise ValueError("base_models must contain at least one model")
        return tuple(normalized)

    @staticmethod
    def _fold_ids(n_samples: int, cv: int, fold_ids) -> np.ndarray:
        if fold_ids is not None:
            folds = np.asarray(fold_ids, dtype=np.int32).reshape(-1)
            if folds.size != int(n_samples):
                raise ValueError("fold_ids length must match X.shape[0]")
            unique = np.unique(folds)
            if unique.size < 2:
                raise ValueError("fold_ids must define at least two folds")
            return np.ascontiguousarray(folds)
        if cv < 2:
            raise ValueError("cv must be at least 2")
        if cv > n_samples:
            raise ValueError("cv must be <= n_samples")
        return np.arange(n_samples, dtype=np.int32) % int(cv)

    @staticmethod
    def _fit_meta(Z: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        z_mean = np.mean(Z, axis=0)
        y_mean = np.mean(y, axis=0)
        Zc = Z - z_mean
        yc = y - y_mean
        penalty = max(float(alpha), 0.0)
        lhs = Zc.T @ Zc
        if penalty > 0.0:
            lhs = lhs + penalty * np.eye(lhs.shape[0], dtype=np.float64)
        rhs = Zc.T @ yc
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        intercept = y_mean - z_mean @ coef
        return np.ascontiguousarray(coef), np.asarray(intercept, dtype=np.float64).reshape(-1)

    def _pls_component_grid(self, n_features: int, n_train: int) -> tuple[int, ...]:
        if self.pls_components is not None:
            grid = tuple(int(v) for v in self.pls_components)
        else:
            max_comp = max(1, min(self.n_components, n_features, n_train - 1))
            grid = tuple(range(1, max_comp + 1))
        if not grid:
            raise ValueError("pls_components must not be empty")
        return grid

    def _base_estimator(self, name: str, n_features: int, n_train: int):
        n_components = max(1, min(self.n_components, n_features, n_train - 1))
        if name == "ridge":
            return NativeRidgeRegressor(
                alpha=self.ridge_alpha,
                center_x=self.center_x,
                scale_x=self.scale_x,
                center_y=self.center_y,
            )
        if name == "pls":
            return NativeMomentSweepRegressor(
                cv=max(2, min(self.inner_cv, n_train)),
                ridge_lambdas=(),
                pls_components=self._pls_component_grid(n_features, n_train),
                heads=("pls",),
                center_x=self.center_x,
                scale_x=self.scale_x,
                center_y=self.center_y,
                scale_y=self.scale_y,
                cuda_pls_parallel_folds=self.cuda_pls_parallel_folds,
                cuda_pls_min_device_features=self.cuda_pls_min_device_features,
                cuda_pls_many_batched=self.cuda_pls_many_batched,
            )
        if name == "pcr":
            return NativePCRRegressor(
                n_components=n_components,
                center_x=self.center_x,
                scale_x=self.scale_x,
                center_y=self.center_y,
                scale_y=self.scale_y,
            )
        if name == "continuum":
            return NativeContinuumRegressionRegressor(
                tau=self.continuum_tau,
                n_components=n_components,
            )
        if name == "ecr":
            return NativeECRRegressor(alpha=self.ecr_alpha, n_components=n_components)
        if name == "cppls":
            return NativeCPPLSRegressor(gamma=self.cppls_gamma, n_components=n_components)
        raise AssertionError(name)

    @staticmethod
    def _predict_2d(model, X: np.ndarray) -> np.ndarray:
        pred = np.asarray(model.predict(X), dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if pred.ndim != 2:
            raise ValueError("base model prediction must be 1D or 2D")
        return np.ascontiguousarray(pred)

    def fit(self, X, y):
        X_arr = _as_X(X)
        y_arr, y_was_1d = _as_y(y, X_arr.shape[0])
        base_names = self._normalize_base_models(self.base_models)
        folds = self._fold_ids(X_arr.shape[0], self.cv, self.fold_ids)
        unique_folds = np.unique(folds)
        min_train = X_arr.shape[0] - max(int(np.sum(folds == f)) for f in unique_folds)
        if min_train < 2:
            raise ValueError("each fold must leave at least two training samples")

        oof_blocks = [
            np.full((X_arr.shape[0], y_arr.shape[1]), np.nan, dtype=np.float64)
            for _ in base_names
        ]
        fold_scores = np.empty((len(unique_folds), len(base_names)), dtype=np.float64)
        for fold_ix, fold in enumerate(unique_folds):
            valid = folds == fold
            train = ~valid
            X_train = X_arr[train]
            y_train = y_arr[train]
            X_valid = X_arr[valid]
            y_valid = y_arr[valid]
            for base_ix, name in enumerate(base_names):
                model = self._base_estimator(name, X_train.shape[1], X_train.shape[0])
                model.fit(X_train, y_train)
                pred = self._predict_2d(model, X_valid)
                if pred.shape != y_valid.shape:
                    raise ValueError(f"{name} prediction shape does not match y")
                oof_blocks[base_ix][valid] = pred
                fold_scores[fold_ix, base_ix] = _rmse(y_valid, pred)

        if any(np.isnan(block).any() for block in oof_blocks):
            raise ValueError("OOF splitter did not cover every sample")

        Z = np.column_stack(oof_blocks)
        meta_coef, meta_intercept = self._fit_meta(Z, y_arr, self.meta_alpha)
        oof_pred = Z @ meta_coef + meta_intercept.reshape(1, -1)

        final_models = []
        final_blocks = []
        for name in base_names:
            model = self._base_estimator(name, X_arr.shape[1], X_arr.shape[0])
            model.fit(X_arr, y_arr)
            final_models.append(model)
            final_blocks.append(self._predict_2d(model, X_arr))
        Z_train = np.column_stack(final_blocks)
        train_pred = Z_train @ meta_coef + meta_intercept.reshape(1, -1)

        self.base_model_names_ = base_names
        self.fold_ids_ = folds.copy()
        self.base_oof_predictions_ = Z.copy()
        self.oof_predictions_ = oof_pred.copy()
        self.predictions_ = train_pred.copy()
        self.base_oof_rmse_ = {
            name: _rmse(y_arr, block) for name, block in zip(base_names, oof_blocks)
        }
        self.fold_rmse_ = fold_scores.copy()
        self.oof_rmse_ = _rmse(y_arr, oof_pred)
        self.rmse_ = _rmse(y_arr, train_pred)
        self.meta_coefficients_ = meta_coef.copy()
        self.intercept_ = (
            float(meta_intercept[0]) if meta_intercept.size == 1 else meta_intercept.copy()
        )
        self.base_models_ = final_models
        self.n_features_in_ = int(X_arr.shape[1])
        self.n_targets_ = int(y_arr.shape[1])
        self._y_was_1d_ = bool(y_was_1d)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "base_models_"):
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        X_arr = _as_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        blocks = [self._predict_2d(model, X_arr) for model in self.base_models_]
        Z = np.column_stack(blocks)
        pred = Z @ self.meta_coefficients_ + np.asarray(self.intercept_).reshape(1, -1)
        if self._y_was_1d_ and pred.shape[1] == 1:
            return pred.ravel()
        return pred

    def score(self, X, y) -> float:
        return _r2_score(y, self.predict(X))

    def get_diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "base_models_"):
            raise RuntimeError(f"{type(self).__name__}.get_diagnostics called before fit")
        return {
            "method": "moment_stack",
            "base_models": self.base_model_names_,
            "cv": int(np.unique(self.fold_ids_).size),
            "inner_cv": self.inner_cv,
            "meta_alpha": self.meta_alpha,
            "ridge_alpha": self.ridge_alpha,
            "n_components": self.n_components,
            "pls_components": (
                None if self.pls_components is None else tuple(self.pls_components)
            ),
            "cppls_gamma": self.cppls_gamma,
            "continuum_tau": self.continuum_tau,
            "ecr_alpha": self.ecr_alpha,
            "oof_rmse": self.oof_rmse_,
            "rmse": self.rmse_,
            "base_oof_rmse": dict(self.base_oof_rmse_),
            "n_samples": int(self.oof_predictions_.shape[0]),
            "n_features": self.n_features_in_,
            "n_targets": self.n_targets_,
        }


class NativeAOMSweepRegressor(_NativeLinearResultRegressor):
    """Native strict-linear AOM profile sweep estimator."""

    _result_coef_key = "input_coefficients"

    def __init__(
        self,
        *,
        profile: str | int = "compact",
        cv: int = 5,
        fold_ids=None,
        ridge_lambdas: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
        pls_components: Sequence[int] | None = None,
        heads: Sequence[str] = ("ridge",),
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
        moment_policy: str | int = "auto",
        cuda_pls_parallel_folds: bool | None = None,
        cuda_pls_min_device_features: int | None = None,
        cuda_pls_many_batched: bool | None = None,
    ) -> None:
        self.profile = profile
        self.cv = int(cv)
        self.fold_ids = fold_ids
        self.ridge_lambdas = tuple(float(v) for v in ridge_lambdas)
        self.pls_components = None if pls_components is None else tuple(int(v) for v in pls_components)
        self.heads = tuple(heads)
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.moment_policy = moment_policy
        self.cuda_pls_parallel_folds = cuda_pls_parallel_folds
        self.cuda_pls_min_device_features = cuda_pls_min_device_features
        self.cuda_pls_many_batched = cuda_pls_many_batched

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        return _native.aom_sweep_run(
            X,
            y,
            profile=self.profile,
            cv=self.cv,
            fold_ids=self.fold_ids,
            ridge_lambdas=self.ridge_lambdas,
            pls_components=self.pls_components,
            heads=self.heads,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
            moment_policy=self.moment_policy,
            cuda_pls_parallel_folds=self.cuda_pls_parallel_folds,
            cuda_pls_min_device_features=self.cuda_pls_min_device_features,
            cuda_pls_many_batched=self.cuda_pls_many_batched,
        )

    def get_diagnostics(self) -> dict[str, object]:
        report = super().get_diagnostics()
        profile_name = _aom_profile_name(self.result_["profile"])
        report.update({
            "selected_chain_id": int(self.result_["selected_chain_id"]),
            "selected_sweep_candidate_id": int(self.result_["selected_sweep_candidate_id"]),
            "n_chains": int(self.result_["n_chains"]),
            "profile": int(self.result_["profile"]),
            "profile_name": profile_name,
            "expected_bank_size": _aom_profile_expected_bank_size(profile_name),
            "n_operator_moment_candidates": int(self.result_["n_operator_moment_candidates"]),
            "n_ridge_operator_moment_candidates": int(self.result_["n_ridge_operator_moment_candidates"]),
            "n_pls_operator_moment_candidates": int(self.result_["n_pls_operator_moment_candidates"]),
            "n_materialized_candidates": int(self.result_["n_materialized_candidates"]),
            "n_ridge_materialized_candidates": int(self.result_["n_ridge_materialized_candidates"]),
            "n_pls_materialized_candidates": int(self.result_["n_pls_materialized_candidates"]),
            "n_moment_prefix_cache_hits": int(self.result_["n_moment_prefix_cache_hits"]),
            "n_moment_prefix_cache_misses": int(self.result_["n_moment_prefix_cache_misses"]),
            "n_ridge_moment_cv_fits": int(self.result_.get("n_ridge_moment_cv_fits", 0.0)),
            "n_ridge_dual_materialized_cv_fits": int(self.result_.get("n_ridge_dual_materialized_cv_fits", 0.0)),
            "n_ridge_dual_cross_cv_fits": int(self.result_.get("n_ridge_dual_cross_cv_fits", 0.0)),
            "n_ridge_moment_score_batch_calls": int(self.result_.get("n_ridge_moment_score_batch_calls", 0.0)),
            "n_ridge_moment_score_batch_jobs": int(self.result_.get("n_ridge_moment_score_batch_jobs", 0.0)),
            "n_ridge_moment_final_fits": int(self.result_.get("n_ridge_moment_final_fits", 0.0)),
            "n_ridge_dual_materialized_final_fits": int(self.result_.get("n_ridge_dual_materialized_final_fits", 0.0)),
            "n_pls_moment_cv_fits": int(self.result_["n_pls_moment_cv_fits"]),
            "n_pls_moment_host_cv_fits": int(self.result_.get("n_pls_moment_host_cv_fits", 0.0)),
            "n_pls_moment_cuda_device_cv_fits": int(self.result_.get("n_pls_moment_cuda_device_cv_fits", 0.0)),
            "n_pls_moment_cuda_parallel_fold_batches": int(self.result_.get("n_pls_moment_cuda_parallel_fold_batches", 0.0)),
            "n_pls_moment_cuda_parallel_fold_jobs": int(self.result_.get("n_pls_moment_cuda_parallel_fold_jobs", 0.0)),
            "n_pls_moment_score_batch_calls": int(self.result_.get("n_pls_moment_score_batch_calls", 0.0)),
            "n_pls_moment_score_batch_jobs": int(self.result_.get("n_pls_moment_score_batch_jobs", 0.0)),
            "n_pls_materialized_cv_fits": int(self.result_["n_pls_materialized_cv_fits"]),
            "n_pls_gcv_proxy_candidates": int(self.result_.get("n_pls_gcv_proxy_candidates", 0.0)),
            "n_pls_gcv_proxy_fits": int(self.result_.get("n_pls_gcv_proxy_fits", 0.0)),
            "n_pls_gcv_proxy_batch_calls": int(self.result_.get("n_pls_gcv_proxy_batch_calls", 0.0)),
            "n_pls_gcv_proxy_batch_jobs": int(self.result_.get("n_pls_gcv_proxy_batch_jobs", 0.0)),
            "n_pls_moment_final_fits": int(self.result_["n_pls_moment_final_fits"]),
            "n_pls_moment_host_final_fits": int(self.result_.get("n_pls_moment_host_final_fits", 0.0)),
            "n_pls_moment_cuda_device_final_fits": int(self.result_.get("n_pls_moment_cuda_device_final_fits", 0.0)),
            "n_pls_materialized_final_fits": int(self.result_["n_pls_materialized_final_fits"]),
            "aom_pls_score_mode": int(self.result_.get("aom_pls_score_mode", 0.0)),
            "moment_policy": self.moment_policy,
            "cuda_pls_min_device_features": self.cuda_pls_min_device_features,
            "cuda_pls_many_batched": self.cuda_pls_many_batched,
        })
        return report


class NativeAOMChainSweepRegressor(NativeAOMSweepRegressor):
    """Native AOM sweep over caller-provided strict-linear chain descriptors."""

    def __init__(
        self,
        chains=None,
        *,
        cv: int = 5,
        fold_ids=None,
        ridge_lambdas: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
        pls_components: Sequence[int] | None = None,
        heads: Sequence[str] = ("ridge",),
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
        moment_policy: str | int = "auto",
        cuda_pls_parallel_folds: bool | None = None,
        cuda_pls_min_device_features: int | None = None,
        cuda_pls_many_batched: bool | None = None,
    ) -> None:
        self.chains = chains
        super().__init__(
            profile=-1,
            cv=cv,
            fold_ids=fold_ids,
            ridge_lambdas=ridge_lambdas,
            pls_components=pls_components,
            heads=heads,
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
            moment_policy=moment_policy,
            cuda_pls_parallel_folds=cuda_pls_parallel_folds,
            cuda_pls_min_device_features=cuda_pls_min_device_features,
            cuda_pls_many_batched=cuda_pls_many_batched,
        )

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        if self.chains is None:
            raise ValueError("chains must be provided")
        return _native.aom_chain_sweep_run(
            X,
            y,
            self.chains,
            cv=self.cv,
            fold_ids=self.fold_ids,
            ridge_lambdas=self.ridge_lambdas,
            pls_components=self.pls_components,
            heads=self.heads,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
            moment_policy=self.moment_policy,
            cuda_pls_parallel_folds=self.cuda_pls_parallel_folds,
            cuda_pls_min_device_features=self.cuda_pls_min_device_features,
            cuda_pls_many_batched=self.cuda_pls_many_batched,
        )


class NativeAOMFixedCandidateRegressor(_NativeLinearResultRegressor):
    """Reusable native model for one decoded AOM strict-linear candidate."""

    _result_coef_key = "input_coefficients"

    def __init__(
        self,
        chain,
        *,
        head: str | int = "ridge",
        param: float = 0.1,
        cv: int = 5,
        fold_ids=None,
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
        moment_policy: str | int = "auto",
        cuda_pls_min_device_features: int | None = None,
        fit_mode: str = "cv",
        precomputed_cv_rmse: float | None = None,
    ) -> None:
        self.chain = chain
        self.head = _normalize_aom_head(head)
        self.param = float(param)
        self.cv = int(cv)
        self.fold_ids = fold_ids
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.moment_policy = moment_policy
        self.cuda_pls_min_device_features = cuda_pls_min_device_features
        self.fit_mode = str(fit_mode)
        self.precomputed_cv_rmse = (
            None if precomputed_cv_rmse is None else float(precomputed_cv_rmse)
        )

    @classmethod
    def from_candidate(cls, candidate: dict, **kwargs):
        """Build an estimator from a row returned by `aom_candidate_table` or a campaign."""
        return cls(
            _candidate_chain(candidate),
            head=_candidate_head(candidate),
            param=float(candidate["param"]),
            **kwargs,
        )

    @classmethod
    def from_campaign(cls, report: dict, *, head=None, rank: int = 0, **kwargs):
        """Build an estimator from a campaign report winner.

        With ``head=None`` this selects from the global ``top_candidates`` list.
        With ``head="ridge"`` or ``head="pls"`` it selects from
        ``top_candidates_by_head`` when available, falling back to
        ``best_by_head`` for ``rank=0``. ``rank`` is zero-based.
        """
        if not isinstance(report, dict):
            raise ValueError("report must be a campaign dictionary")
        rank_index = int(rank)
        if rank_index < 0:
            raise ValueError("rank must be >= 0")

        if head is None:
            rows = list(report.get("top_candidates", ()))
            if not rows:
                best = report.get("best")
                rows = [] if best is None else [best]
        else:
            head_name = _normalize_aom_head(head)
            by_head = report.get("top_candidates_by_head", {})
            rows = list(by_head.get(head_name, ())) if isinstance(by_head, dict) else []
            if not rows and rank_index == 0:
                best_by_head = report.get("best_by_head", {})
                if isinstance(best_by_head, dict) and head_name in best_by_head:
                    rows = [best_by_head[head_name]]
            if not rows:
                raise ValueError(f"campaign report has no candidates for head {head_name!r}")

        if rank_index >= len(rows):
            raise ValueError("rank is outside the available campaign candidates")
        return cls.from_candidate(rows[rank_index], **kwargs)

    @classmethod
    def from_refit_report(cls, report: dict, *, rank: int = 0, **kwargs):
        """Build an estimator from an exact-CV refit report row.

        ``aom_refit_candidates`` rows can be sorted by different analysis keys,
        so this constructor always ranks rows by ``refit_cv_rmse``. ``rank`` is
        zero-based in that exact-CV ordering.
        """
        if not isinstance(report, dict):
            raise ValueError("report must be an AOM refit report dictionary")
        rank_index = int(rank)
        if rank_index < 0:
            raise ValueError("rank must be >= 0")

        rows = list(report.get("rows", ()))
        if not rows and rank_index == 0:
            best = report.get("best_cv")
            rows = [] if best is None else [best]
        if not rows:
            raise ValueError("refit report has no candidate rows")

        def refit_key(row):
            return (
                float(row.get("refit_cv_rmse", np.inf)),
                int(row.get("source_index", 0)),
            )

        rows = sorted(rows, key=refit_key)
        if rank_index >= len(rows):
            raise ValueError("rank is outside the available refit candidates")
        return cls.from_candidate(rows[rank_index], **kwargs)

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        mode = str(self.fit_mode).lower()
        if mode in {"final_only", "fit_only", "fixed_fit"}:
            result = _native.aom_chain_fixed_fit_run(
                X,
                y,
                self.chain,
                head=self.head,
                param=self.param,
                center_x=self.center_x,
                scale_x=self.scale_x,
                center_y=self.center_y,
                scale_y=self.scale_y,
                moment_policy=self.moment_policy,
                cuda_pls_min_device_features=self.cuda_pls_min_device_features,
            )
            if self.precomputed_cv_rmse is not None:
                result = dict(result)
                result["selected_cv_rmse"] = float(self.precomputed_cv_rmse)
                scores = np.asarray(result["candidate_scores"], dtype=np.float64).copy()
                if scores.ndim == 2 and scores.size:
                    scores[0, scores.shape[1] - 1] = float(self.precomputed_cv_rmse)
                    result["candidate_scores"] = scores
            return result
        if mode not in {"cv", "full_cv"}:
            raise ValueError("fit_mode must be 'cv' or 'final_only'")
        if self.head == "ridge":
            ridge_lambdas = (self.param,)
            pls_components = ()
        else:
            component = int(round(self.param))
            if abs(float(component) - self.param) > 1e-9 or component < 1:
                raise ValueError("PLS candidate param must be a positive integer component count")
            ridge_lambdas = ()
            pls_components = (component,)
        return _native.aom_chain_sweep_run(
            X,
            y,
            [self.chain],
            cv=self.cv,
            fold_ids=self.fold_ids,
            ridge_lambdas=ridge_lambdas,
            pls_components=pls_components,
            heads=(self.head,),
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
            moment_policy=self.moment_policy,
            cuda_pls_min_device_features=self.cuda_pls_min_device_features,
        )

    def get_diagnostics(self) -> dict[str, object]:
        report = super().get_diagnostics()
        report.update({
            "chain": self.chain,
            "head": self.head,
            "param": self.param,
            "selected_chain_id": int(self.result_["selected_chain_id"]),
            "selected_sweep_candidate_id": int(self.result_["selected_sweep_candidate_id"]),
            "n_chains": int(self.result_["n_chains"]),
            "n_operator_moment_candidates": int(self.result_["n_operator_moment_candidates"]),
            "n_ridge_operator_moment_candidates": int(self.result_["n_ridge_operator_moment_candidates"]),
            "n_pls_operator_moment_candidates": int(self.result_["n_pls_operator_moment_candidates"]),
            "n_materialized_candidates": int(self.result_["n_materialized_candidates"]),
            "n_ridge_materialized_candidates": int(self.result_["n_ridge_materialized_candidates"]),
            "n_pls_materialized_candidates": int(self.result_["n_pls_materialized_candidates"]),
            "n_pls_gcv_proxy_candidates": int(self.result_.get("n_pls_gcv_proxy_candidates", 0.0)),
            "n_pls_gcv_proxy_fits": int(self.result_.get("n_pls_gcv_proxy_fits", 0.0)),
            "aom_pls_score_mode": int(self.result_.get("aom_pls_score_mode", 0.0)),
            "moment_policy": self.moment_policy,
            "cuda_pls_min_device_features": self.cuda_pls_min_device_features,
            "fit_mode": self.fit_mode,
            "precomputed_cv_rmse": self.precomputed_cv_rmse,
        })
        return report


class NativeAOMScreenRefitRegressor(BaseEstimator):
    """Two-pass AOM preprocessing screen with exact-CV candidate reuse.

    The estimator runs ``aom_chain_screen_refit_campaign`` during ``fit`` and
    then fits one verified candidate through ``NativeAOMFixedCandidateRegressor``.
    This keeps the broad screen and exact refit reports inspectable while
    exposing a normal sklearn-style reusable linear model for ``predict``.

    ``split_head_scoring`` defaults to ``"auto"`` so the default mixed
    ``("ridge", "pls")`` screen scores each chunk as separate Ridge-only and
    PLS-only native score-only calls. That merge is score-preserving (the
    retained ``(chain_id, head, param)`` candidate scores are identical to a
    single mixed call) but lets each head hit its batched native fast path
    (Ridge moment score batch + PLS exact/GCV-proxy batch), which a single
    mixed call cannot use. For a single-head screen the policy is inert and
    ``n_split_head_chunks`` stays ``0``. The lower-level campaign helpers
    (``aom_chain_score_campaign`` / ``aom_chain_screen_refit_campaign``) keep
    the historical ``"off"`` default.
    """

    def __init__(
        self,
        chains=None,
        *,
        profile: str = "lab",
        families: dict | None = None,
        templates: Sequence[Sequence[str]] | None = None,
        max_chains: int | None = None,
        chain_chunk_size: int = 4096,
        top_k: int = 50,
        refit_top_k: int | None = None,
        refit_per_head_top_k: int | None = None,
        final_rank: int = 0,
        cv: int = 5,
        fold_ids=None,
        ridge_lambdas: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
        pls_components: Sequence[int] | None = None,
        heads: Sequence[str] = ("ridge", "pls"),
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
        moment_policy: str | int = "auto",
        refit_moment_policy: str | int | None = None,
        final_moment_policy: str | int | None = None,
        pls_score_mode: str | int = "cv",
        chain_ordering: str = "input",
        split_head_scoring: str = "auto",
        checkpoint_path=None,
        resume: bool = True,
        max_chunks_per_run: int | None = None,
        backend_cuda_available: bool | None = None,
        backend_min_cuda_product: int | None = None,
        cuda_pls_parallel_folds: bool | None = None,
        cuda_pls_min_device_features: int | None = None,
        cuda_pls_many_batched: bool | None = None,
        refit_sort_by: str | None = "refit_cv_rmse",
        refit_execution: str = "auto",
        refit_auto_max_extra_fraction: float = 1.0,
    ) -> None:
        self.chains = chains
        self.profile = profile
        self.families = families
        self.templates = templates
        self.max_chains = None if max_chains is None else int(max_chains)
        self.chain_chunk_size = int(chain_chunk_size)
        self.top_k = int(top_k)
        self.refit_top_k = None if refit_top_k is None else int(refit_top_k)
        self.refit_per_head_top_k = (
            None if refit_per_head_top_k is None else int(refit_per_head_top_k)
        )
        self.final_rank = int(final_rank)
        self.cv = int(cv)
        self.fold_ids = fold_ids
        self.ridge_lambdas = tuple(float(v) for v in ridge_lambdas)
        self.pls_components = (
            None if pls_components is None
            else tuple(int(v) for v in pls_components)
        )
        self.heads = tuple(heads)
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y
        self.moment_policy = moment_policy
        self.refit_moment_policy = refit_moment_policy
        self.final_moment_policy = final_moment_policy
        self.pls_score_mode = pls_score_mode
        self.chain_ordering = chain_ordering
        self.split_head_scoring = split_head_scoring
        self.checkpoint_path = checkpoint_path
        self.resume = bool(resume)
        self.max_chunks_per_run = (
            None if max_chunks_per_run is None else int(max_chunks_per_run)
        )
        self.backend_cuda_available = backend_cuda_available
        self.backend_min_cuda_product = (
            None if backend_min_cuda_product is None
            else int(backend_min_cuda_product)
        )
        self.cuda_pls_parallel_folds = cuda_pls_parallel_folds
        self.cuda_pls_min_device_features = cuda_pls_min_device_features
        self.cuda_pls_many_batched = cuda_pls_many_batched
        self.refit_sort_by = refit_sort_by
        self.refit_execution = refit_execution
        self.refit_auto_max_extra_fraction = float(refit_auto_max_extra_fraction)

    def fit(self, X, y):
        X_arr = _as_X(X)
        y_arr, _ = _as_y(y, X_arr.shape[0])
        report = _native.aom_chain_screen_refit_campaign(
            X_arr,
            y_arr,
            chains=self.chains,
            profile=self.profile,
            families=self.families,
            templates=self.templates,
            max_chains=self.max_chains,
            chain_chunk_size=self.chain_chunk_size,
            top_k=self.top_k,
            refit_top_k=self.refit_top_k,
            refit_per_head_top_k=self.refit_per_head_top_k,
            cv=self.cv,
            fold_ids=self.fold_ids,
            ridge_lambdas=self.ridge_lambdas,
            pls_components=self.pls_components,
            heads=self.heads,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
            moment_policy=self.moment_policy,
            refit_moment_policy=self.refit_moment_policy,
            pls_score_mode=self.pls_score_mode,
            chain_ordering=self.chain_ordering,
            split_head_scoring=self.split_head_scoring,
            checkpoint_path=self.checkpoint_path,
            resume=self.resume,
            max_chunks_per_run=self.max_chunks_per_run,
            backend_cuda_available=self.backend_cuda_available,
            backend_min_cuda_product=self.backend_min_cuda_product,
            cuda_pls_parallel_folds=self.cuda_pls_parallel_folds,
            cuda_pls_min_device_features=self.cuda_pls_min_device_features,
            cuda_pls_many_batched=self.cuda_pls_many_batched,
            refit_sort_by=self.refit_sort_by,
            refit_execution=self.refit_execution,
            refit_auto_max_extra_fraction=self.refit_auto_max_extra_fraction,
        )

        ordered_rows = sorted(
            list(report["rows"]),
            key=lambda row: (
                float(row.get("refit_cv_rmse", np.inf)),
                int(row.get("source_index", 0)),
            ),
        )
        selected_row = ordered_rows[self.final_rank]
        final_policy = self.final_moment_policy
        if final_policy is None:
            final_policy = (
                self.refit_moment_policy
                if self.refit_moment_policy is not None
                else self.moment_policy
            )
        selected_model = NativeAOMFixedCandidateRegressor.from_candidate(
            selected_row,
            cv=self.cv,
            fold_ids=self.fold_ids,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
            moment_policy=final_policy,
            cuda_pls_min_device_features=self.cuda_pls_min_device_features,
            fit_mode="final_only",
            precomputed_cv_rmse=float(selected_row["refit_cv_rmse"]),
        ).fit(X_arr, y)

        self.campaign_report_ = report
        self.screen_report_ = report["screen"]
        self.refit_report_ = report["refit"]
        self.selected_refit_row_ = selected_row
        self.selected_model_ = selected_model
        self.result_ = selected_model.result_
        self._coef_matrix_ = selected_model._coef_matrix_.copy()
        self.coef_ = np.asarray(selected_model.coef_).copy()
        self.intercept_ = (
            float(selected_model.intercept_)
            if np.asarray(selected_model.intercept_).ndim == 0
            else np.asarray(selected_model.intercept_).copy()
        )
        self.predictions_ = selected_model.predictions_.copy()
        self.oof_predictions_ = selected_model.oof_predictions_.copy()
        self.candidate_scores_ = selected_model.candidate_scores_.copy()
        self.selected_candidate_id_ = selected_model.selected_candidate_id_
        self.selected_head_id_ = selected_model.selected_head_id_
        self.selected_param_ = selected_model.selected_param_
        self.selected_cv_rmse_ = float(selected_row["refit_cv_rmse"])
        self.selected_chain_ = selected_model.chain
        self.selected_head_ = selected_model.head
        self.n_features_in_ = selected_model.n_features_in_
        self.n_targets_ = selected_model.n_targets_
        self._y_was_1d_ = selected_model._y_was_1d_
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "selected_model_"):
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        return self.selected_model_.predict(X)

    def score(self, X, y) -> float:
        return _r2_score(y, self.predict(X))

    def get_diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "campaign_report_"):
            raise RuntimeError(
                f"{type(self).__name__}.get_diagnostics called before fit"
            )
        final = self.selected_model_.get_diagnostics()
        return {
            "screen_complete": bool(self.campaign_report_["screen_complete"]),
            "n_screen_candidates": int(self.campaign_report_["n_screen_candidates"]),
            "n_screen_top_candidates": int(
                self.campaign_report_["n_screen_top_candidates"]
            ),
            "n_refit_candidates": int(self.campaign_report_["n_refit_candidates"]),
            "refit_per_head_top_k": self.campaign_report_.get(
                "refit_per_head_top_k"
            ),
            "n_refit_global_candidates": int(
                self.campaign_report_.get("n_refit_global_candidates", 0)
            ),
            "n_refit_per_head_candidates": int(
                self.campaign_report_.get("n_refit_per_head_candidates", 0)
            ),
            "n_refit_per_head_extra_candidates": int(
                self.campaign_report_.get("n_refit_per_head_extra_candidates", 0)
            ),
            "n_refit_groups": int(self.refit_report_.get("n_refit_groups", 0)),
            "n_refit_scored_candidates": int(
                self.refit_report_.get("n_refit_scored_candidates", 0)
            ),
            "n_refit_extra_scored_candidates": int(
                self.refit_report_.get("n_refit_extra_scored_candidates", 0)
            ),
            "n_chains": int(self.screen_report_["n_chains"]),
            "n_candidates": int(self.screen_report_["n_candidates"]),
            "n_operator_moment_candidates": int(
                self.screen_report_["n_operator_moment_candidates"]
            ),
            "n_materialized_candidates": int(
                self.screen_report_["n_materialized_candidates"]
            ),
            "n_ridge_moment_cv_fits": int(
                self.screen_report_.get("n_ridge_moment_cv_fits", 0)
            ),
            "n_ridge_moment_score_batch_calls": int(
                self.screen_report_.get("n_ridge_moment_score_batch_calls", 0)
            ),
            "n_ridge_moment_score_batch_jobs": int(
                self.screen_report_.get("n_ridge_moment_score_batch_jobs", 0)
            ),
            "n_pls_gcv_proxy_candidates": int(
                self.screen_report_.get("n_pls_gcv_proxy_candidates", 0)
            ),
            "n_pls_gcv_proxy_fits": int(
                self.screen_report_.get("n_pls_gcv_proxy_fits", 0)
            ),
            "n_pls_moment_score_batch_calls": int(
                self.screen_report_.get("n_pls_moment_score_batch_calls", 0)
            ),
            "n_pls_moment_score_batch_jobs": int(
                self.screen_report_.get("n_pls_moment_score_batch_jobs", 0)
            ),
            "n_pls_gcv_proxy_batch_calls": int(
                self.screen_report_.get("n_pls_gcv_proxy_batch_calls", 0)
            ),
            "n_pls_gcv_proxy_batch_jobs": int(
                self.screen_report_.get("n_pls_gcv_proxy_batch_jobs", 0)
            ),
            "n_refit_ridge_moment_cv_fits": int(
                self.refit_report_.get("n_ridge_moment_cv_fits", 0)
            ),
            "n_refit_ridge_dual_materialized_cv_fits": int(
                self.refit_report_.get("n_ridge_dual_materialized_cv_fits", 0)
            ),
            "n_refit_ridge_dual_cross_cv_fits": int(
                self.refit_report_.get("n_ridge_dual_cross_cv_fits", 0)
            ),
            "n_refit_ridge_moment_score_batch_calls": int(
                self.refit_report_.get("n_ridge_moment_score_batch_calls", 0)
            ),
            "n_refit_ridge_moment_score_batch_jobs": int(
                self.refit_report_.get("n_ridge_moment_score_batch_jobs", 0)
            ),
            "n_refit_pls_moment_cv_fits": int(
                self.refit_report_.get("n_pls_moment_cv_fits", 0)
            ),
            "n_refit_pls_moment_host_cv_fits": int(
                self.refit_report_.get("n_pls_moment_host_cv_fits", 0)
            ),
            "n_refit_pls_moment_cuda_device_cv_fits": int(
                self.refit_report_.get("n_pls_moment_cuda_device_cv_fits", 0)
            ),
            "n_refit_pls_moment_cuda_parallel_fold_batches": int(
                self.refit_report_.get("n_pls_moment_cuda_parallel_fold_batches", 0)
            ),
            "n_refit_pls_moment_cuda_parallel_fold_jobs": int(
                self.refit_report_.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
            ),
            "n_refit_pls_moment_score_batch_calls": int(
                self.refit_report_.get("n_pls_moment_score_batch_calls", 0)
            ),
            "n_refit_pls_moment_score_batch_jobs": int(
                self.refit_report_.get("n_pls_moment_score_batch_jobs", 0)
            ),
            "n_refit_pls_materialized_cv_fits": int(
                self.refit_report_.get("n_pls_materialized_cv_fits", 0)
            ),
            "pls_score_mode": self.campaign_report_["pls_score_mode"],
            "refit_pls_score_mode": self.campaign_report_["refit_pls_score_mode"],
            "refit_execution": self.campaign_report_["refit_execution"],
            "refit_execution_requested": self.campaign_report_.get(
                "refit_execution_requested", self.refit_execution
            ),
            "refit_execution_auto_reason": self.campaign_report_.get(
                "refit_execution_auto_reason"
            ),
            "refit_auto_max_extra_fraction": float(
                self.campaign_report_.get(
                    "refit_auto_max_extra_fraction",
                    self.refit_auto_max_extra_fraction,
                )
            ),
            "chain_ordering": self.campaign_report_.get("chain_ordering", "input"),
            "split_head_scoring": self.campaign_report_.get(
                "split_head_scoring", "off"
            ),
            "cuda_pls_parallel_folds": self.campaign_report_.get(
                "cuda_pls_parallel_folds"
            ),
            "cuda_pls_min_device_features": self.campaign_report_.get(
                "cuda_pls_min_device_features"
            ),
            "cuda_pls_many_batched": self.campaign_report_.get(
                "cuda_pls_many_batched"
            ),
            "n_split_head_chunks": int(
                self.campaign_report_.get("n_split_head_chunks", 0)
            ),
            "n_chunk_score_calls": int(
                self.campaign_report_.get("n_chunk_score_calls", 0)
            ),
            "moment_backend_recommendations": self.campaign_report_.get(
                "moment_backend_recommendations", {}
            ),
            "backend_min_cuda_product": self.campaign_report_.get(
                "backend_min_cuda_product"
            ),
            "selected_chain": self.selected_chain_,
            "selected_head": self.selected_head_,
            "selected_param": self.selected_param_,
            "selected_cv_rmse": self.selected_cv_rmse_,
            "final_selected_cv_rmse": float(final["selected_cv_rmse"]),
            "final_n_candidates": int(final.get("n_candidates", 0)),
            "final_n_operator_moment_candidates": int(
                final.get("n_operator_moment_candidates", 0)
            ),
            "final_n_materialized_candidates": int(
                final.get("n_materialized_candidates", 0)
            ),
            "final_n_pls_moment_cv_fits": int(
                final.get("n_pls_moment_cv_fits", 0)
            ),
            "final_n_pls_moment_host_cv_fits": int(
                final.get("n_pls_moment_host_cv_fits", 0)
            ),
            "final_n_pls_moment_cuda_device_cv_fits": int(
                final.get("n_pls_moment_cuda_device_cv_fits", 0)
            ),
            "final_n_pls_moment_cuda_parallel_fold_batches": int(
                final.get("n_pls_moment_cuda_parallel_fold_batches", 0)
            ),
            "final_n_pls_moment_cuda_parallel_fold_jobs": int(
                final.get("n_pls_moment_cuda_parallel_fold_jobs", 0)
            ),
            "final_n_pls_materialized_cv_fits": int(
                final.get("n_pls_materialized_cv_fits", 0)
            ),
            "final_n_pls_gcv_proxy_fits": int(
                final.get("n_pls_gcv_proxy_fits", 0)
            ),
            "final_n_pls_moment_final_fits": int(
                final.get("n_pls_moment_final_fits", 0)
            ),
            "final_n_pls_moment_host_final_fits": int(
                final.get("n_pls_moment_host_final_fits", 0)
            ),
            "final_n_pls_moment_cuda_device_final_fits": int(
                final.get("n_pls_moment_cuda_device_final_fits", 0)
            ),
            "final_n_pls_materialized_final_fits": int(
                final.get("n_pls_materialized_final_fits", 0)
            ),
            "moment_policy": self.moment_policy,
            "refit_moment_policy": self.campaign_report_["refit_moment_policy"],
        }


class NativeAOMMomentScreenRefitRegressor(NativeAOMScreenRefitRegressor):
    """Preset mixed Ridge/PLS AOM moment screen with exact-CV refit.

    The first pass screens Ridge candidates by exact CV and PLS candidates by
    the explicit GCV proxy, then exact-CV refits the retained union of global
    top rows and per-head top rows. This keeps broad mixed campaigns cheap
    while making the second-pass candidate pool less sensitive to one head
    dominating the global screen list.
    """

    def __init__(
        self,
        chains=None,
        *,
        profile: str = "lab",
        families: dict | None = None,
        templates: Sequence[Sequence[str]] | None = None,
        max_chains: int | None = None,
        chain_chunk_size: int = 4096,
        top_k: int = 50,
        refit_top_k: int | None = None,
        refit_per_head_top_k: int | None = 10,
        final_rank: int = 0,
        cv: int = 5,
        fold_ids=None,
        ridge_lambdas: Sequence[float] = (
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0,
            10.0,
            100.0,
        ),
        pls_components: Sequence[int] | None = (1, 2, 3, 4, 6, 8),
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
        moment_policy: str | int = "force_moments",
        refit_moment_policy: str | int | None = None,
        final_moment_policy: str | int | None = None,
        chain_ordering: str = "prefix",
        split_head_scoring: str = "auto",
        checkpoint_path=None,
        resume: bool = True,
        max_chunks_per_run: int | None = None,
        backend_cuda_available: bool | None = None,
        backend_min_cuda_product: int | None = None,
        cuda_pls_parallel_folds: bool | None = None,
        cuda_pls_min_device_features: int | None = None,
        cuda_pls_many_batched: bool | None = None,
        refit_sort_by: str | None = "refit_cv_rmse",
        refit_execution: str = "auto",
        refit_auto_max_extra_fraction: float = 1.0,
    ) -> None:
        super().__init__(
            chains=chains,
            profile=profile,
            families=families,
            templates=templates,
            max_chains=max_chains,
            chain_chunk_size=chain_chunk_size,
            top_k=top_k,
            refit_top_k=refit_top_k,
            refit_per_head_top_k=refit_per_head_top_k,
            final_rank=final_rank,
            cv=cv,
            fold_ids=fold_ids,
            ridge_lambdas=ridge_lambdas,
            pls_components=pls_components,
            heads=("ridge", "pls"),
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
            moment_policy=moment_policy,
            refit_moment_policy=refit_moment_policy,
            final_moment_policy=final_moment_policy,
            pls_score_mode="gcv_proxy",
            chain_ordering=chain_ordering,
            split_head_scoring=split_head_scoring,
            checkpoint_path=checkpoint_path,
            resume=resume,
            max_chunks_per_run=max_chunks_per_run,
            backend_cuda_available=backend_cuda_available,
            backend_min_cuda_product=backend_min_cuda_product,
            cuda_pls_parallel_folds=cuda_pls_parallel_folds,
            cuda_pls_min_device_features=cuda_pls_min_device_features,
            cuda_pls_many_batched=cuda_pls_many_batched,
            refit_sort_by=refit_sort_by,
            refit_execution=refit_execution,
            refit_auto_max_extra_fraction=refit_auto_max_extra_fraction,
        )

    def get_diagnostics(self) -> dict[str, object]:
        report = super().get_diagnostics()
        report["preset"] = "moment_mixed_gcv_proxy_screen_refit"
        return report


class NativeAOMMomentPLSScreenRefitRegressor(NativeAOMScreenRefitRegressor):
    """Preset PLS AOM moment screen: GCV proxy screen, exact-CV refit.

    This is the end-user PLS preset for broad strict-linear preprocessing
    selection. It keeps the full `NativeAOMScreenRefitRegressor` workflow but
    fixes the model head to PLS, uses the explicit moment-only GCV proxy for
    the first pass, then verifies retained rows by exact CV before the final
    reusable fit.
    """

    def __init__(
        self,
        chains=None,
        *,
        profile: str = "lab",
        families: dict | None = None,
        templates: Sequence[Sequence[str]] | None = None,
        max_chains: int | None = None,
        chain_chunk_size: int = 4096,
        top_k: int = 50,
        refit_top_k: int | None = None,
        final_rank: int = 0,
        cv: int = 5,
        fold_ids=None,
        pls_components: Sequence[int] | None = (1, 2, 3, 4, 6, 8),
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
        moment_policy: str | int = "force_moments",
        refit_moment_policy: str | int | None = None,
        final_moment_policy: str | int | None = None,
        chain_ordering: str = "prefix",
        checkpoint_path=None,
        resume: bool = True,
        max_chunks_per_run: int | None = None,
        backend_cuda_available: bool | None = None,
        backend_min_cuda_product: int | None = None,
        cuda_pls_parallel_folds: bool | None = None,
        cuda_pls_min_device_features: int | None = None,
        cuda_pls_many_batched: bool | None = None,
        refit_sort_by: str | None = "refit_cv_rmse",
        refit_execution: str = "auto",
        refit_auto_max_extra_fraction: float = 1.0,
    ) -> None:
        super().__init__(
            chains=chains,
            profile=profile,
            families=families,
            templates=templates,
            max_chains=max_chains,
            chain_chunk_size=chain_chunk_size,
            top_k=top_k,
            refit_top_k=refit_top_k,
            final_rank=final_rank,
            cv=cv,
            fold_ids=fold_ids,
            ridge_lambdas=(),
            pls_components=pls_components,
            heads=("pls",),
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
            moment_policy=moment_policy,
            refit_moment_policy=refit_moment_policy,
            final_moment_policy=final_moment_policy,
            pls_score_mode="gcv_proxy",
            chain_ordering=chain_ordering,
            checkpoint_path=checkpoint_path,
            resume=resume,
            max_chunks_per_run=max_chunks_per_run,
            backend_cuda_available=backend_cuda_available,
            backend_min_cuda_product=backend_min_cuda_product,
            cuda_pls_parallel_folds=cuda_pls_parallel_folds,
            cuda_pls_min_device_features=cuda_pls_min_device_features,
            cuda_pls_many_batched=cuda_pls_many_batched,
            refit_sort_by=refit_sort_by,
            refit_execution=refit_execution,
            refit_auto_max_extra_fraction=refit_auto_max_extra_fraction,
        )

    def get_diagnostics(self) -> dict[str, object]:
        report = super().get_diagnostics()
        report["preset"] = "moment_pls_gcv_proxy_screen_refit"
        return report


class NativeAOMMomentRidgeScreenRefitRegressor(NativeAOMScreenRefitRegressor):
    """Preset Ridge AOM moment screen with exact-CV refit.

    This is the end-user Ridge preset for broad strict-linear preprocessing
    selection. It fixes the model head to Ridge, ranks chains by exact native
    CV, exact-refits the retained rows, then builds a reusable final-only
    selected candidate.
    """

    def __init__(
        self,
        chains=None,
        *,
        profile: str = "lab",
        families: dict | None = None,
        templates: Sequence[Sequence[str]] | None = None,
        max_chains: int | None = None,
        chain_chunk_size: int = 4096,
        top_k: int = 50,
        refit_top_k: int | None = None,
        final_rank: int = 0,
        cv: int = 5,
        fold_ids=None,
        ridge_lambdas: Sequence[float] = (
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0,
            10.0,
            100.0,
        ),
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
        moment_policy: str | int = "force_moments",
        refit_moment_policy: str | int | None = None,
        final_moment_policy: str | int | None = None,
        chain_ordering: str = "prefix",
        checkpoint_path=None,
        resume: bool = True,
        max_chunks_per_run: int | None = None,
        backend_cuda_available: bool | None = None,
        backend_min_cuda_product: int | None = None,
        cuda_pls_parallel_folds: bool | None = None,
        cuda_pls_min_device_features: int | None = None,
        cuda_pls_many_batched: bool | None = None,
        refit_sort_by: str | None = "refit_cv_rmse",
        refit_execution: str = "auto",
        refit_auto_max_extra_fraction: float = 1.0,
    ) -> None:
        super().__init__(
            chains=chains,
            profile=profile,
            families=families,
            templates=templates,
            max_chains=max_chains,
            chain_chunk_size=chain_chunk_size,
            top_k=top_k,
            refit_top_k=refit_top_k,
            final_rank=final_rank,
            cv=cv,
            fold_ids=fold_ids,
            ridge_lambdas=ridge_lambdas,
            pls_components=(),
            heads=("ridge",),
            center_x=center_x,
            scale_x=scale_x,
            center_y=center_y,
            scale_y=scale_y,
            moment_policy=moment_policy,
            refit_moment_policy=refit_moment_policy,
            final_moment_policy=final_moment_policy,
            pls_score_mode="cv",
            chain_ordering=chain_ordering,
            checkpoint_path=checkpoint_path,
            resume=resume,
            max_chunks_per_run=max_chunks_per_run,
            backend_cuda_available=backend_cuda_available,
            backend_min_cuda_product=backend_min_cuda_product,
            cuda_pls_parallel_folds=cuda_pls_parallel_folds,
            cuda_pls_min_device_features=cuda_pls_min_device_features,
            cuda_pls_many_batched=cuda_pls_many_batched,
            refit_sort_by=refit_sort_by,
            refit_execution=refit_execution,
            refit_auto_max_extra_fraction=refit_auto_max_extra_fraction,
        )

    def get_diagnostics(self) -> dict[str, object]:
        report = super().get_diagnostics()
        report["preset"] = "moment_ridge_exact_screen_refit"
        return report


class NativeAOMRobustHPORegressor(BaseEstimator):
    """Native strict-linear AOM robust-HPO estimator with reusable coefficients."""

    def __init__(
        self,
        *,
        profile: str | int = "compact",
        cv: int = 5,
        heads: Sequence[str] = ("ridge", "pls"),
    ) -> None:
        self.profile = profile
        self.cv = int(cv)
        self.heads = tuple(heads)

    def fit(self, X, y):
        X_arr = _as_X(X)
        y_arr, y_was_1d = _as_y(y, X_arr.shape[0])
        if y_arr.shape[1] != 1:
            raise ValueError("NativeAOMRobustHPORegressor supports one target")
        result = _native.aom_robust_hpo(
            X_arr,
            y_arr,
            profile=self.profile,
            cv=self.cv,
            heads=self.heads,
        )
        coef = np.asarray(result["input_coefficients"], dtype=np.float64)
        intercept = np.asarray(result["intercept"], dtype=np.float64).reshape(-1)
        if coef.ndim != 2 or coef.shape != (X_arr.shape[1], 1):
            raise ValueError("native AOM robust-HPO coefficients do not match X")
        if intercept.size != 1:
            raise ValueError("native AOM robust-HPO intercept is invalid")

        self.result_ = result
        self._coef_matrix_ = np.ascontiguousarray(coef)
        self.intercept_ = float(intercept[0])
        self.coef_ = self._coef_matrix_[:, 0].copy()
        self.predictions_ = np.asarray(result["predictions"], dtype=np.float64).copy()
        self.candidate_scores_ = np.asarray(result["candidate_scores"], dtype=np.float64).copy()
        self.selected_chain_id_ = int(result["selected_chain_id"])
        self.selected_head_id_ = int(result["selected_head_id"])
        self.selected_param_ = float(result["selected_param"])
        self.selected_cv_rmse_ = float(result["selected_cv_rmse"])
        self.n_features_in_ = int(X_arr.shape[1])
        self.n_targets_ = 1
        self._y_was_1d_ = bool(y_was_1d)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "_coef_matrix_"):
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        X_arr = _as_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        preds = X_arr @ self._coef_matrix_ + self.intercept_
        if self._y_was_1d_:
            return preds.ravel()
        return preds

    def score(self, X, y) -> float:
        return _r2_score(y, self.predict(X))

    def get_diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "result_"):
            raise RuntimeError(f"{type(self).__name__}.get_diagnostics called before fit")
        profile_name = _aom_profile_name(self.result_["profile"])
        return {
            "selected_chain_id": self.selected_chain_id_,
            "selected_head_id": self.selected_head_id_,
            "selected_head": _aom_head_id_name(self.selected_head_id_),
            "selected_param": self.selected_param_,
            "selected_cv_rmse": self.selected_cv_rmse_,
            "n_chains": int(self.result_["n_chains"]),
            "n_candidates": int(self.result_["n_candidates"]),
            "profile": int(self.result_["profile"]),
            "profile_name": profile_name,
            "expected_bank_size": _aom_profile_expected_bank_size(profile_name),
            "cv": int(self.result_["cv"]),
            "n_features": int(self.result_["n_features"]),
            "n_features_transformed": int(self.result_["n_features_transformed"]),
            "n_targets": int(self.result_["n_targets"]),
        }


class NativeAOMRidgeBlenderRegressor(BaseEstimator):
    """Native strict-linear AOM Ridge simplex blender with reusable coefficients."""

    def __init__(
        self,
        *,
        profile: str | int = "compact",
        cv: int = 5,
        fold_ids=None,
        ridge_lambdas: Sequence[float] = (1e-4, 1e-2, 1.0, 100.0),
        regularizer: float = 0.01,
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
    ) -> None:
        self.profile = profile
        self.cv = int(cv)
        self.fold_ids = fold_ids
        self.ridge_lambdas = tuple(float(v) for v in ridge_lambdas)
        self.regularizer = float(regularizer)
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y

    def fit(self, X, y):
        X_arr = _as_X(X)
        y_arr, y_was_1d = _as_y(y, X_arr.shape[0])
        result = _native.aom_ridge_blender(
            X_arr,
            y_arr,
            profile=self.profile,
            cv=self.cv,
            fold_ids=self.fold_ids,
            ridge_lambdas=self.ridge_lambdas,
            regularizer=self.regularizer,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
        )
        coef = np.asarray(result["input_coefficients"], dtype=np.float64)
        intercept = np.asarray(result["intercept"], dtype=np.float64).reshape(-1)
        if coef.ndim != 2 or coef.shape[0] != X_arr.shape[1]:
            raise ValueError("native AOM Ridge blender coefficients do not match X")
        if intercept.size != coef.shape[1]:
            raise ValueError("native AOM Ridge blender intercept does not match coefficients")

        self.result_ = result
        self._coef_matrix_ = np.ascontiguousarray(coef)
        self.intercept_ = float(intercept[0]) if intercept.size == 1 else intercept.copy()
        self.coef_ = (
            self._coef_matrix_[:, 0].copy()
            if self._coef_matrix_.shape[1] == 1
            else self._coef_matrix_.T.copy()
        )
        self.predictions_ = np.asarray(result["predictions"], dtype=np.float64).copy()
        self.oof_predictions_ = np.asarray(result["oof_predictions"], dtype=np.float64).copy()
        self.candidate_scores_ = np.asarray(result["candidate_scores"], dtype=np.float64).copy()
        self.weights_ = np.asarray(result["weights"], dtype=np.float64).reshape(-1).copy()
        self.selected_candidate_id_ = int(result["selected_candidate_id"])
        self.selected_chain_id_ = int(result["selected_chain_id"])
        self.selected_param_ = float(result["selected_param"])
        self.selected_cv_rmse_ = float(result["selected_cv_rmse"])
        self.blend_oof_rmse_ = float(result["blend_oof_rmse"])
        self.n_features_in_ = int(X_arr.shape[1])
        self.n_targets_ = int(coef.shape[1])
        self._y_was_1d_ = bool(y_was_1d)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "_coef_matrix_"):
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        X_arr = _as_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        preds = X_arr @ self._coef_matrix_ + np.asarray(self.intercept_).reshape(1, -1)
        if self._y_was_1d_ and preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def score(self, X, y) -> float:
        return _r2_score(y, self.predict(X))

    def get_diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "result_"):
            raise RuntimeError(f"{type(self).__name__}.get_diagnostics called before fit")
        profile_name = _aom_profile_name(self.result_["profile"])
        return {
            "selected_candidate_id": self.selected_candidate_id_,
            "selected_chain_id": self.selected_chain_id_,
            "selected_param": self.selected_param_,
            "selected_cv_rmse": self.selected_cv_rmse_,
            "blend_oof_rmse": self.blend_oof_rmse_,
            "regularizer": float(self.result_["regularizer"]),
            "n_candidates": int(self.result_["n_candidates"]),
            "n_chains": int(self.result_["n_chains"]),
            "profile": int(self.result_["profile"]),
            "profile_name": profile_name,
            "expected_bank_size": _aom_profile_expected_bank_size(profile_name),
            "cv": int(self.result_["cv"]),
            "n_features": int(self.result_["n_features"]),
            "n_targets": int(self.result_["n_targets"]),
        }


class NativeAOMOperatorPLSStackRegressor(BaseEstimator):
    """Native strict-linear AOM operator PLS stack with reusable coefficients."""

    def __init__(
        self,
        *,
        profile: str | int = "compact",
        cv: int = 5,
        fold_ids=None,
        components: Sequence[int] = (2, 4, 8),
        alphas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        std_penalty: float = 0.0,
        gap_penalty: float = 0.0,
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
    ) -> None:
        self.profile = profile
        self.cv = int(cv)
        self.fold_ids = fold_ids
        self.components = tuple(int(v) for v in components)
        self.alphas = tuple(float(v) for v in alphas)
        self.std_penalty = float(std_penalty)
        self.gap_penalty = float(gap_penalty)
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y

    def fit(self, X, y):
        X_arr = _as_X(X)
        y_arr, y_was_1d = _as_y(y, X_arr.shape[0])
        if y_arr.shape[1] != 1:
            raise ValueError("NativeAOMOperatorPLSStackRegressor supports one target")
        result = _native.aom_operator_pls_stack(
            X_arr,
            y_arr,
            profile=self.profile,
            cv=self.cv,
            fold_ids=self.fold_ids,
            components=self.components,
            alphas=self.alphas,
            std_penalty=self.std_penalty,
            gap_penalty=self.gap_penalty,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
        )
        coef = np.asarray(result["input_coefficients"], dtype=np.float64)
        intercept = np.asarray(result["input_intercept"], dtype=np.float64).reshape(-1)
        if coef.ndim != 2 or coef.shape != (X_arr.shape[1], 1):
            raise ValueError("native AOM operator PLS stack coefficients do not match X")
        if intercept.size != 1:
            raise ValueError("native AOM operator PLS stack input intercept is invalid")

        self.result_ = result
        self._coef_matrix_ = np.ascontiguousarray(coef)
        self.intercept_ = float(intercept[0])
        self.coef_ = self._coef_matrix_[:, 0].copy()
        self.predictions_ = np.asarray(result["predictions"], dtype=np.float64).copy()
        self.oof_predictions_ = np.asarray(result["oof_predictions"], dtype=np.float64).copy()
        self.stack_features_ = np.asarray(result["stack_features"], dtype=np.float64).copy()
        self.stack_coefficients_ = np.asarray(result["coefficients"], dtype=np.float64).copy()
        self.stack_intercept_ = np.asarray(result["intercept"], dtype=np.float64).copy()
        self.candidate_scores_ = np.asarray(result["candidate_scores"], dtype=np.float64).copy()
        self.selected_spec_id_ = int(result["selected_spec_id"])
        self.selected_components_ = int(result["selected_components"])
        self.selected_alpha_ = float(result["selected_alpha"])
        self.selected_oof_rmse_ = float(result["selected_oof_rmse"])
        self.selected_train_rmse_ = float(result["selected_train_rmse"])
        self.selected_criterion_ = float(result["selected_criterion"])
        self.n_features_in_ = int(X_arr.shape[1])
        self.n_targets_ = 1
        self._y_was_1d_ = bool(y_was_1d)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "_coef_matrix_"):
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        X_arr = _as_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        preds = X_arr @ self._coef_matrix_ + self.intercept_
        if self._y_was_1d_:
            return preds.ravel()
        return preds

    def score(self, X, y) -> float:
        return _r2_score(y, self.predict(X))

    def get_diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "result_"):
            raise RuntimeError(f"{type(self).__name__}.get_diagnostics called before fit")
        profile_name = _aom_profile_name(self.result_["profile"])
        return {
            "selected_spec_id": self.selected_spec_id_,
            "selected_components": self.selected_components_,
            "selected_alpha": self.selected_alpha_,
            "selected_oof_rmse": self.selected_oof_rmse_,
            "selected_train_rmse": self.selected_train_rmse_,
            "selected_criterion": self.selected_criterion_,
            "std_penalty": float(self.result_["std_penalty"]),
            "gap_penalty": float(self.result_["gap_penalty"]),
            "n_operator_features": int(self.result_["n_operator_features"]),
            "n_specs": int(self.result_["n_specs"]),
            "n_operators": int(self.result_["n_operators"]),
            "profile": int(self.result_["profile"]),
            "profile_name": profile_name,
            "expected_bank_size": _aom_profile_expected_bank_size(profile_name),
            "cv": int(self.result_["cv"]),
            "n_features": int(self.result_["n_features"]),
            "n_targets": int(self.result_["n_targets"]),
        }


class _NativeAOMSelectorRegressor(BaseEstimator):
    """Base class for native AOM/POP selector results with input coefficients."""

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        raise NotImplementedError

    def fit(self, X, y):
        X_arr = _as_X(X)
        y_arr, y_was_1d = _as_y(y, X_arr.shape[0])
        result = self._fit_native_result(X_arr, y_arr)
        coef = np.asarray(result["input_coefficients"], dtype=np.float64)
        intercept = np.asarray(result["intercept"], dtype=np.float64).reshape(-1)
        if coef.ndim != 2 or coef.shape[0] != X_arr.shape[1]:
            raise ValueError("native AOM selector coefficients do not match X")
        if intercept.size != coef.shape[1]:
            raise ValueError("native AOM selector intercept does not match coefficients")

        self.result_ = result
        self._coef_matrix_ = np.ascontiguousarray(coef)
        self.intercept_ = float(intercept[0]) if intercept.size == 1 else intercept.copy()
        self.coef_ = (
            self._coef_matrix_[:, 0].copy()
            if self._coef_matrix_.shape[1] == 1
            else self._coef_matrix_.T.copy()
        )
        self.predictions_ = np.asarray(result["predictions"], dtype=np.float64).copy()
        self.best_score_ = float(result["best_score"])
        self.selected_n_components_ = int(result["selected_n_components"])
        self.operator_kinds_ = np.asarray(result["operator_kinds"], dtype=np.int32).copy()
        self.n_features_in_ = int(X_arr.shape[1])
        self.n_targets_ = int(self._coef_matrix_.shape[1])
        self._y_was_1d_ = bool(y_was_1d)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "_coef_matrix_"):
            raise RuntimeError(f"{type(self).__name__}.predict called before fit")
        X_arr = _as_X(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        preds = X_arr @ self._coef_matrix_ + np.asarray(self.intercept_).reshape(1, -1)
        if self._y_was_1d_ and preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def score(self, X, y) -> float:
        return _r2_score(y, self.predict(X))

    def get_diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "result_"):
            raise RuntimeError(f"{type(self).__name__}.get_diagnostics called before fit")
        return {
            "best_score": self.best_score_,
            "selected_n_components": self.selected_n_components_,
            "n_operators": int(self.result_["n_operators"]),
            "max_components": int(self.result_["max_components"]),
            "cv": int(self.result_["cv"]),
            "n_features": int(self.result_["n_features"]),
            "n_targets": int(self.result_["n_targets"]),
        }


class NativeAOMPLSRegressor(_NativeAOMSelectorRegressor):
    """Native global AOM-PLS selector with reusable input-space coefficients."""

    def __init__(
        self,
        *,
        max_components: int = 3,
        operators=None,
        cv: int = 3,
        fold_ids=None,
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
    ) -> None:
        self.max_components = int(max_components)
        self.operators = operators
        self.cv = int(cv)
        self.fold_ids = fold_ids
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        return _native.aom_pls(
            X,
            y,
            max_components=self.max_components,
            operators=self.operators,
            cv=self.cv,
            fold_ids=self.fold_ids,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
        )

    def get_diagnostics(self) -> dict[str, object]:
        report = super().get_diagnostics()
        report["selected_operator_index"] = int(self.result_["selected_operator_index"])
        return report


class NativePOPPLSRegressor(_NativeAOMSelectorRegressor):
    """Native POP-PLS per-component selector with reusable coefficients."""

    def __init__(
        self,
        *,
        max_components: int = 3,
        operators=None,
        cv: int = 3,
        fold_ids=None,
        center_x: bool | None = None,
        scale_x: bool | None = None,
        center_y: bool | None = None,
        scale_y: bool | None = None,
    ) -> None:
        self.max_components = int(max_components)
        self.operators = operators
        self.cv = int(cv)
        self.fold_ids = fold_ids
        self.center_x = center_x
        self.scale_x = scale_x
        self.center_y = center_y
        self.scale_y = scale_y

    def _fit_native_result(self, X: np.ndarray, y: np.ndarray) -> dict:
        return _native.pop_pls(
            X,
            y,
            max_components=self.max_components,
            operators=self.operators,
            cv=self.cv,
            fold_ids=self.fold_ids,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=self.center_y,
            scale_y=self.scale_y,
        )

    def get_diagnostics(self) -> dict[str, object]:
        report = super().get_diagnostics()
        report["selected_operator_indices"] = np.asarray(
            self.result_["selected_operator_indices"], dtype=np.int32).copy()
        return report


__all__ = [
    "NativeAOMChainSweepRegressor",
    "NativeAOMFixedCandidateRegressor",
    "NativeAOMMomentScreenRefitRegressor",
    "NativeAOMMomentPLSScreenRefitRegressor",
    "NativeAOMMomentRidgeScreenRefitRegressor",
    "NativeAOMOperatorPLSStackRegressor",
    "NativeAOMPLSRegressor",
    "NativeAOMRidgeBlenderRegressor",
    "NativeAOMRobustHPORegressor",
    "NativeAOMScreenRefitRegressor",
    "NativeAOMSweepRegressor",
    "NativeContinuumRegressionRegressor",
    "NativeCPPLSRegressor",
    "NativeECRRegressor",
    "NativeMomentStackRegressor",
    "NativeMomentSweepRegressor",
    "NativePCRRegressor",
    "NativePOPPLSRegressor",
    "NativeRidgeRegressor",
]
