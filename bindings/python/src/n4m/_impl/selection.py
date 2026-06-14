# SPDX-License-Identifier: CECILL-2.1
"""Feature-selection wrappers over the tier-1 ``n4m_feature_selection_*`` ABI.

Public role names live in ``n4m.feature_selection.{wrapper,ranking}``; these
implementation classes are imported and re-exported there under their canonical
names (``CARS``, ``UVE``, ``VariableSelect``, ...). They are scikit-learn
``SelectorMixin`` estimators backed entirely by the libn4m C ABI through
``n4m._impl.native``.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from . import native as _native

try:  # scikit-learn is an optional runtime dependency.
    from sklearn.base import BaseEstimator
    from sklearn.feature_selection import SelectorMixin
    from sklearn.utils.validation import check_is_fitted
except Exception:  # pragma: no cover - keep the binding dependency-light
    from .compat import BaseEstimator

    class SelectorMixin:  # minimal fallback
        def get_support(self, indices: bool = False):
            mask = self._get_support_mask()
            return np.where(mask)[0] if indices else mask

        def transform(self, X):
            X = np.asarray(X, dtype=np.float64)
            return X[:, self._get_support_mask()]

    def check_is_fitted(estimator) -> None:
        if not hasattr(estimator, "support_"):
            raise RuntimeError(f"{type(estimator).__name__} is not fitted")


def _check_X_y(estimator, X, y):
    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X_arr.ndim != 2:
        raise ValueError("X must be 2-D")
    y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    estimator.n_features_in_ = int(X_arr.shape[1])
    return X_arr, y_arr


class _BaseSelector(SelectorMixin, BaseEstimator):
    """Common machinery for every libn4m SelectorMixin wrapper."""

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self)
        return self.support_

    def _commit_fit(self, *, indices: np.ndarray, scores: np.ndarray | None = None) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[indices] = True
        self.support_ = mask
        self.selected_indices_ = indices.copy()
        if scores is None:
            self.scores_ = np.empty(self.n_features_in_, dtype=np.float64)
        else:
            self.scores_ = np.asarray(scores, dtype=np.float64).ravel().copy()


# ----------------------------------------------------------------------
# Scoring-then-top-k ranking selector (variable_select_rank).
# ----------------------------------------------------------------------

class _Ranker(_BaseSelector):
    _rank_method: int  # 0=VIP, 1=|coef|, 2=SR

    def __init__(self, top_k: int, *, n_components: int = 2) -> None:
        self.top_k = top_k
        self.n_components = n_components

    def fit(self, X: Any, y: Any):
        X_arr, y_arr = _check_X_y(self, X, y)
        top_k = int(self.top_k)
        if top_k < 1 or top_k > self.n_features_in_:
            raise ValueError(
                f"top_k must be in [1, n_features_in_={self.n_features_in_}]; got {self.top_k!r}"
            )
        indices, scores = _native.variable_select_rank(
            X_arr, y_arr, method=self._rank_method, top_k=top_k, n_components=int(self.n_components)
        )
        self._commit_fit(indices=indices, scores=scores)
        return self


class VariableSelect(_Ranker):
    """VIP-ranked top-k variable selection (default rank by VIP)."""

    def __init__(self, top_k: int, *, n_components: int = 2, rank_method: int = 0) -> None:
        super().__init__(top_k, n_components=n_components)
        self.rank_method = rank_method

    @property
    def _rank_method(self) -> int:  # type: ignore[override]
        return int(self.rank_method)


# ----------------------------------------------------------------------
# Plan-driven wrapper selectors.
# ----------------------------------------------------------------------

class SPA(_BaseSelector):
    """Successive Projections Algorithm selector (Araujo 2001)."""

    def __init__(self, top_k: int, *, n_components: int = 2) -> None:
        self.top_k = top_k
        self.n_components = n_components

    def fit(self, X, y):
        X_arr, y_arr = _check_X_y(self, X, y)
        import ctypes
        fields = _native._run_select(
            "n4m_feature_selection_spa_select", X_arr, y_arr,
            ctypes.c_int32(int(self.top_k)),
            n_components=int(self.n_components), plan_folds=None,
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("coefficient_scores"))
        return self


class Stability(_BaseSelector):
    """MCUVE-style stability selector via Monte-Carlo subsampling."""

    def __init__(self, top_k: int, *, n_components: int = 2,
                 n_iterations: int = 50, seed: int = 0, n_folds: int = 3) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.seed = seed
        self.n_folds = n_folds

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_stability_select", X_arr, y_arr,
            ctypes.c_int32(int(self.top_k)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds), plan_seed=int(self.seed),
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("stability_scores"))
        return self


class UVE(_BaseSelector):
    """Uninformative Variable Elimination (Centner 1996)."""

    def __init__(self, *, n_components: int = 2, noise_features: int = 50,
                 noise_seed: int = 0, min_features: int | None = None) -> None:
        self.n_components = n_components
        self.noise_features = noise_features
        self.noise_seed = noise_seed
        self.min_features = min_features

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        min_features = int(self.min_features if self.min_features is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_uve_select", X_arr, y_arr,
            ctypes.c_int32(int(self.noise_features)),
            ctypes.c_uint64(int(self.noise_seed)),
            n_components=int(self.n_components),
            plan_folds=3, plan_seed=int(self.noise_seed),
        )
        indices = _native._selected_indices(fields)
        scores = fields.get("real_stability_scores")
        if indices.size < min_features:
            if scores is None or np.asarray(scores).size != self.n_features_in_:
                raise RuntimeError(
                    "UVE selected fewer than min_features and real_stability_scores "
                    "are unavailable for a deterministic fallback; set min_features=0."
                )
            score_arr = np.where(np.isfinite(scores), scores, -np.inf).ravel()
            order = sorted(range(self.n_features_in_), key=lambda i: (-float(score_arr[i]), int(i)))
            existing = {int(i) for i in indices}
            needed = min_features - indices.size
            additions = [i for i in order if i not in existing][:needed]
            indices = np.concatenate([indices, np.asarray(additions, dtype=np.int64)])
            warnings.warn(
                "UVE selected fewer than min_features; added the highest "
                "real_stability_scores deterministically.", UserWarning, stacklevel=2,
            )
        self._commit_fit(indices=indices, scores=scores)
        return self


class CARS(_BaseSelector):
    """Competitive Adaptive Reweighted Sampling (Li 2009)."""

    def __init__(self, *, n_components: int = 2, n_iterations: int = 50,
                 min_features: int | None = None, n_folds: int = 3, seed: int = 0) -> None:
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.min_features = min_features
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        min_features = int(self.min_features if self.min_features is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_cars_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_iterations)),
            ctypes.c_int32(min_features),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds), plan_seed=int(self.seed),
        )
        self._commit_fit(indices=_native._selected_indices(fields), scores=None)
        return self


class RandomFrog(_BaseSelector):
    """Random Frog feature selection (Li 2012)."""

    def __init__(self, top_k: int, *, n_components: int = 2, n_iterations: int = 100,
                 initial_size: int = 20, min_size: int | None = None,
                 max_size: int | None = None, n_folds: int = 3, seed: int = 0) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.initial_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        max_size = int(self.max_size if self.max_size is not None else self.n_features_in_)
        min_size = int(self.min_size if self.min_size is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_random_frog_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_iterations)),
            ctypes.c_int32(int(self.initial_size)),
            ctypes.c_int32(min_size),
            ctypes.c_int32(max_size),
            ctypes.c_int32(int(self.top_k)),
            ctypes.c_uint64(int(self.seed)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds), plan_seed=int(self.seed),
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("global_scores"))
        return self


class SCARS(_BaseSelector):
    """Stability-CARS hybrid (Zheng 2014)."""

    def __init__(self, *, n_components: int = 2, n_iterations: int = 50,
                 min_features: int | None = None, sample_fraction: float = 0.8,
                 n_folds: int = 3, seed: int = 0) -> None:
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.min_features = min_features
        self.sample_fraction = sample_fraction
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        min_features = int(self.min_features if self.min_features is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_scars_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_iterations)),
            ctypes.c_int32(min_features),
            ctypes.c_double(float(self.sample_fraction)),
            ctypes.c_uint64(int(self.seed)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds), plan_seed=int(self.seed),
        )
        scores = fields.get("coefficient_scores")
        if scores is not None and scores.ndim == 2 and scores.shape[0]:
            scores = scores[-1].ravel()
        self._commit_fit(indices=_native._selected_indices(fields), scores=scores)
        return self


class GA(_BaseSelector):
    """Genetic Algorithm feature selection."""

    def __init__(self, *, n_components: int = 2, n_generations: int = 30,
                 population_size: int = 40, min_features: int | None = None,
                 max_features: int | None = None, mutation_rate: float = 0.05,
                 n_folds: int = 3, seed: int = 0) -> None:
        self.n_components = n_components
        self.n_generations = n_generations
        self.population_size = population_size
        self.min_features = min_features
        self.max_features = max_features
        self.mutation_rate = mutation_rate
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        max_features = int(self.max_features if self.max_features is not None else self.n_features_in_)
        min_features = int(self.min_features if self.min_features is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_ga_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_generations)),
            ctypes.c_int32(int(self.population_size)),
            ctypes.c_int32(min_features),
            ctypes.c_int32(max_features),
            ctypes.c_double(float(self.mutation_rate)),
            ctypes.c_uint64(int(self.seed)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds), plan_seed=int(self.seed),
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("global_scores"))
        return self


class PSO(_BaseSelector):
    """Binary Particle Swarm Optimization selector."""

    def __init__(self, *, n_components: int = 2, n_swarm: int = 30, n_iterations: int = 50,
                 w: float = 0.729, c1: float = 1.494, c2: float = 1.494, v_max: float = 4.0,
                 n_folds: int = 3, seed: int = 0) -> None:
        self.n_components = n_components
        self.n_swarm = n_swarm
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_pso_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_swarm)),
            ctypes.c_int32(int(self.n_iterations)),
            ctypes.c_double(float(self.w)),
            ctypes.c_double(float(self.c1)),
            ctypes.c_double(float(self.c2)),
            ctypes.c_double(float(self.v_max)),
            ctypes.c_uint64(int(self.seed)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds), plan_seed=int(self.seed),
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("inclusion_frequencies"))
        return self


class VISSA(_BaseSelector):
    """Variable Iterative Subspace Shrinkage Approach (Deng 2014)."""

    def __init__(self, *, n_components: int = 2, n_iterations: int = 10, n_submodels: int = 60,
                 ratio_kept: float = 0.1, threshold: float = 0.5, floor_probability: float = 0.05,
                 n_folds: int = 3, seed: int = 0) -> None:
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.n_submodels = n_submodels
        self.ratio_kept = ratio_kept
        self.threshold = threshold
        self.floor_probability = floor_probability
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_vissa_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_iterations)),
            ctypes.c_int32(int(self.n_submodels)),
            ctypes.c_double(float(self.ratio_kept)),
            ctypes.c_double(float(self.threshold)),
            ctypes.c_double(float(self.floor_probability)),
            ctypes.c_uint64(int(self.seed)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds), plan_seed=int(self.seed),
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("final_probabilities"))
        return self


class Shaving(_BaseSelector):
    """Iterative SR-shaving variable elimination."""

    def __init__(self, *, n_components: int = 2, n_steps: int = 10,
                 min_features: int | None = None, shave_fraction: float = 0.2,
                 n_folds: int = 3) -> None:
        self.n_components = n_components
        self.n_steps = n_steps
        self.min_features = min_features
        self.shave_fraction = shave_fraction
        self.n_folds = n_folds

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        min_features = int(self.min_features if self.min_features is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_shaving_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_steps)),
            ctypes.c_int32(min_features),
            ctypes.c_double(float(self.shave_fraction)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds),
        )
        self._commit_fit(indices=_native._selected_indices(fields), scores=None)
        return self


class BVE(_BaseSelector):
    """Backward Variable Elimination with VIP filter."""

    def __init__(self, *, n_components: int = 2, n_steps: int = 10,
                 min_features: int | None = None, n_folds: int = 3) -> None:
        self.n_components = n_components
        self.n_steps = n_steps
        self.min_features = min_features
        self.n_folds = n_folds

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        min_features = int(self.min_features if self.min_features is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_bve_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_steps)),
            ctypes.c_int32(min_features),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds),
        )
        self._commit_fit(indices=_native._selected_indices(fields), scores=None)
        return self


class REP(_BaseSelector):
    """REP-PLS — repeated VIP-thresholded variable selection."""

    def __init__(self, *, n_components: int = 2, n_steps: int = 10,
                 min_features: int | None = None, remove_count: int = 1, n_folds: int = 3) -> None:
        self.n_components = n_components
        self.n_steps = n_steps
        self.min_features = min_features
        self.remove_count = remove_count
        self.n_folds = n_folds

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        min_features = int(self.min_features if self.min_features is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_rep_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_steps)),
            ctypes.c_int32(min_features),
            ctypes.c_int32(int(self.remove_count)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds),
        )
        self._commit_fit(indices=_native._selected_indices(fields), scores=None)
        return self


class IPW(_BaseSelector):
    """Iterative Predictor Weighting PLS selector."""

    def __init__(self, top_k: int, *, n_components: int = 2, n_iterations: int = 20,
                 damping: float = 0.5, weight_floor: float = 1e-6,
                 n_folds: int = 3, seed: int = 0) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.damping = damping
        self.weight_floor = weight_floor
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_ipw_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_iterations)),
            ctypes.c_int32(int(self.top_k)),
            ctypes.c_double(float(self.damping)),
            ctypes.c_double(float(self.weight_floor)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds), plan_seed=int(self.seed),
        )
        scores = fields.get("score_path")
        if scores is not None and scores.ndim == 2 and scores.shape[0]:
            scores = scores[-1].ravel()
        self._commit_fit(indices=_native._selected_indices(fields), scores=scores)
        return self


class ST(_BaseSelector):
    """ST-PLS — soft-thresholded sparse PLS selector."""

    def __init__(self, thresholds, *, n_components: int = 2, min_selected: int | None = None) -> None:
        self.thresholds = thresholds
        self.n_components = n_components
        self.min_selected = min_selected

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        thr = np.ascontiguousarray(self.thresholds, dtype=np.float64).reshape(-1)
        min_selected = int(self.min_selected if self.min_selected is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_st_select", X_arr, y_arr,
            thr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(int(thr.size)),
            ctypes.c_int32(min_selected),
            n_components=int(self.n_components),
            plan_folds=3,
        )
        self._commit_fit(indices=_native._selected_indices(fields), scores=None)
        return self


# ----------------------------------------------------------------------
# Interval-family selectors.
# ----------------------------------------------------------------------

class BiPLS(_BaseSelector):
    """biPLS — backward interval elimination (Nørgaard 2000)."""

    def __init__(self, *, n_components: int = 2, interval_width: int = 10,
                 min_intervals: int = 2, n_folds: int = 3) -> None:
        self.n_components = n_components
        self.interval_width = interval_width
        self.min_intervals = min_intervals
        self.n_folds = n_folds

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_bipls_select", X_arr, y_arr,
            ctypes.c_int32(int(self.interval_width)),
            ctypes.c_int32(int(self.min_intervals)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds),
        )
        self._commit_fit(indices=_native._selected_indices(fields), scores=None)
        return self


class SiPLS(_BaseSelector):
    """siPLS — synergistic interval combinations (Nørgaard 2000)."""

    def __init__(self, *, n_components: int = 2, interval_width: int = 10,
                 combination_size: int = 2, n_folds: int = 3) -> None:
        self.n_components = n_components
        self.interval_width = interval_width
        self.combination_size = combination_size
        self.n_folds = n_folds

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_sipls_select", X_arr, y_arr,
            ctypes.c_int32(int(self.interval_width)),
            ctypes.c_int32(int(self.combination_size)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds),
        )
        self._commit_fit(indices=_native._selected_indices(fields), scores=None)
        return self


# ----------------------------------------------------------------------
# T² / WVC / threshold-based selectors.
# ----------------------------------------------------------------------

class T2(_BaseSelector):
    """T²-PLS loading-weight selection (plsVarSel::T2_pls style)."""

    def __init__(self, alpha_thresholds, *, n_components: int = 2, min_selected: int | None = None) -> None:
        self.alpha_thresholds = alpha_thresholds
        self.n_components = n_components
        self.min_selected = min_selected

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        alpha = np.ascontiguousarray(self.alpha_thresholds, dtype=np.float64).reshape(-1)
        min_selected = int(self.min_selected if self.min_selected is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_t2_select", X_arr, y_arr,
            alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(int(alpha.size)),
            ctypes.c_int32(min_selected),
            n_components=int(self.n_components),
            plan_folds=3,
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("t2_scores"))
        return self


class WVC(_BaseSelector):
    """WVC-PLS — weighted variable contribution top-k selector (no plan)."""

    def __init__(self, top_k: int, *, n_components: int = 2, normalize: bool = True) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.normalize = normalize

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_wvc_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_components)),
            ctypes.c_int32(int(self.top_k)),
            ctypes.c_int32(1 if self.normalize else 0),
            n_components=int(self.n_components),
            plan_folds=None, with_config=False,
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("final_scores"))
        return self


class WVCThreshold(_BaseSelector):
    """Threshold-/factor-based WVC-PLS selector (no plan)."""

    def __init__(self, *, n_components: int = 2, normalize: bool = True,
                 score_threshold: float = 0.0, threshold_factor: float = 1.0,
                 min_selected: int | None = None) -> None:
        self.n_components = n_components
        self.normalize = normalize
        self.score_threshold = score_threshold
        self.threshold_factor = threshold_factor
        self.min_selected = min_selected

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        min_selected = int(self.min_selected if self.min_selected is not None else self.n_components)
        fields = _native._run_select(
            "n4m_feature_selection_wvc_threshold_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_components)),
            ctypes.c_int32(1 if self.normalize else 0),
            ctypes.c_double(float(self.score_threshold)),
            ctypes.c_double(float(self.threshold_factor)),
            ctypes.c_int32(min_selected),
            n_components=int(self.n_components),
            plan_folds=None, with_config=False,
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("final_scores"))
        return self


class EMCUVE(_BaseSelector):
    """Ensemble Monte-Carlo UVE selector."""

    def __init__(self, *, n_components: int = 2, noise_features: int = 50, noise_seed: int = 0,
                 n_ensembles: int = 10, vote_threshold: float = 0.5, n_folds: int = 3) -> None:
        self.n_components = n_components
        self.noise_features = noise_features
        self.noise_seed = noise_seed
        self.n_ensembles = n_ensembles
        self.vote_threshold = vote_threshold
        self.n_folds = n_folds

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_emcuve_select", X_arr, y_arr,
            ctypes.c_int32(int(self.noise_features)),
            ctypes.c_uint64(int(self.noise_seed)),
            ctypes.c_int32(int(self.n_ensembles)),
            ctypes.c_double(float(self.vote_threshold)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds), plan_seed=int(self.noise_seed),
        )
        self._commit_fit(indices=_native._selected_indices(fields), scores=None)
        return self


class Randomization(_BaseSelector):
    """Randomization-test PLS selector (Y-permutation p-values, no plan)."""

    def __init__(self, *, n_components: int = 2, n_permutations: int = 200,
                 randomization_seed: int = 0, alpha: float = 0.05) -> None:
        self.n_components = n_components
        self.n_permutations = n_permutations
        self.randomization_seed = randomization_seed
        self.alpha = alpha

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_randomization_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_permutations)),
            ctypes.c_uint64(int(self.randomization_seed)),
            ctypes.c_double(float(self.alpha)),
            n_components=int(self.n_components),
            plan_folds=None,
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("p_values"))
        return self


class IRIV(_BaseSelector):
    """IRIV — Iteratively Retains Informative Variables (Yun 2014)."""

    def __init__(self, *, n_components: int = 2, max_rounds: int = 5,
                 n_folds: int = 5, seed: int = 0) -> None:
        self.n_components = n_components
        self.max_rounds = max_rounds
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_iriv_select", X_arr, y_arr,
            ctypes.c_int32(int(self.max_rounds)),
            ctypes.c_uint64(int(self.seed)),
            n_components=int(self.n_components),
            plan_folds=int(self.n_folds), plan_seed=int(self.seed),
        )
        self._commit_fit(indices=_native._selected_indices(fields), scores=None)
        return self


class IRF(_BaseSelector):
    """IRF — Interval Random Frog (Yun 2013)."""

    def __init__(self, top_k: int, *, n_components: int = 2, n_iterations: int = 100,
                 window_size: int = 5, initial_intervals: int = 5, seed: int = 0) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.window_size = window_size
        self.initial_intervals = initial_intervals
        self.seed = seed

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_irf_select", X_arr, y_arr,
            ctypes.c_int32(int(self.n_iterations)),
            ctypes.c_int32(int(self.window_size)),
            ctypes.c_int32(int(self.initial_intervals)),
            ctypes.c_int32(int(self.top_k)),
            ctypes.c_uint64(int(self.seed)),
            n_components=int(self.n_components),
            plan_folds=3, plan_seed=int(self.seed),
        )
        self._commit_fit(indices=_native._selected_indices(fields), scores=None)
        probability = fields.get("probability")
        if probability is not None:
            self.interval_probability_ = probability
        return self


class VIPSPA(_BaseSelector):
    """VIP_SPA — VIP-mask + SPA greedy (no plan)."""

    def __init__(self, top_k: int, *, n_components: int = 2, vip_threshold: float = 0.3) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.vip_threshold = vip_threshold

    def fit(self, X, y):
        import ctypes
        X_arr, y_arr = _check_X_y(self, X, y)
        fields = _native._run_select(
            "n4m_feature_selection_vip_spa_select", X_arr, y_arr,
            ctypes.c_double(float(self.vip_threshold)),
            ctypes.c_int32(int(self.top_k)),
            n_components=int(self.n_components),
            plan_folds=None,
        )
        self._commit_fit(indices=_native._selected_indices(fields),
                         scores=fields.get("vip_scores"))
        return self


__all__ = [
    "VariableSelect",
    "SPA", "Stability", "UVE", "CARS", "RandomFrog", "SCARS", "GA", "PSO",
    "VISSA", "Shaving", "BVE", "REP", "IPW", "ST", "BiPLS", "SiPLS", "T2",
    "WVC", "WVCThreshold", "EMCUVE", "Randomization", "IRIV", "IRF", "VIPSPA",
]
