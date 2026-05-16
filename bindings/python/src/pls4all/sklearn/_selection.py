"""sklearn-compatible variable-selection wrappers.

Selectors are MethodResult-based on the C side: the C ABI returns a
scoring + selection report, not a re-fittable ``p4a_model_t``. Persistence
therefore uses plain NumPy state (mask + scores) rather than the `.n4a`
bundle path used by Model-based regressors.

Hybrid contract for every class:
    * ``support_`` (bool[n_features_in_]) — sklearn SelectorMixin mask
    * ``selected_indices_`` (int64[n_selected]) — selection-order indices
    * ``scores_`` (float64[n_features_in_]) — per-feature score from the
      underlying selector (VIP, coef, RC, RMSE drop, etc.). May be empty
      for selectors that don't expose a per-feature score.

Each class hand-rolls its own ``fit(X, y)`` so the signature, docstring
and parameter semantics stay readable; common machinery lives in
:class:`_BasePls4allSelector`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from .. import _methods
from .._aom import ValidationPlan
from .._context import Context
from .._model import Model
from .._types import Algorithm, Deflation
from ._base import _check_X_y_p4a, _config_from_params, _resolve_solver


def _default_plan(n_samples: int, *, n_folds: int = 3,
                    seed: int = 0) -> ValidationPlan:
    """Deterministic k-fold ValidationPlan used by every selector that
    needs cross-validated scoring. Mirrors the pattern in
    ``benchmarks/parity_timing/registry.py::_build_default_plan``."""
    plan = ValidationPlan()
    plan.n_samples = int(n_samples)
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    fold_size = max(1, n_samples // n_folds)
    for f in range(n_folds):
        start = f * fold_size
        end = (f + 1) * fold_size if f < n_folds - 1 else n_samples
        test = idx[start:end]
        train = np.setdiff1d(idx, test, assume_unique=False)
        plan.add_fold([int(x) for x in train], [int(x) for x in test])
    return plan


class _BasePls4allSelector(SelectorMixin, BaseEstimator):
    """Common machinery for every pls4all SelectorMixin wrapper.

    Subclasses override ``_run`` to call the tier-1 selector function
    and return a tuple ``(selected_indices, scores_or_None)``. The base
    handles validation, fitted state and the SelectorMixin contract.
    """

    # Subclass-tunable defaults — most selectors fit a PLS regression on
    # (X, y) under the hood and don't care about the solver / centering
    # knobs; expose them anyway so power users can tune.

    def _resolve_solver_(self):
        return _resolve_solver(getattr(self, "solver", "nipals"))

    def _make_config(self, *, store_scores: bool = False):
        return _config_from_params(
            n_components=int(getattr(self, "n_components", 2)),
            algorithm=Algorithm.PLS_REGRESSION,
            solver=self._resolve_solver_(),
            deflation=Deflation.REGRESSION,
            center_x=bool(getattr(self, "center_x", True)),
            scale_x=bool(getattr(self, "scale_x", True)),
            center_y=True,
            scale_y=False,
            tol=float(getattr(self, "tol", 1e-6)),
            max_iter=int(getattr(self, "max_iter", 500)),
            store_scores=store_scores,
        )

    # SelectorMixin hook.
    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self)
        return self.support_

    def _commit_fit(self, *, indices: np.ndarray,
                     scores: np.ndarray | None = None) -> None:
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
# Scoring-then-top-k selectors (variable_select_rank)
# ----------------------------------------------------------------------

class _RankerSelector(_BasePls4allSelector):
    """Common machinery for VIP / |coef| / SR top-k selectors."""

    _rank_method: int  # 0=VIP, 1=|coef|, 2=SR (see variable_selection.cpp)

    def fit(self, X: Any, y: Any):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        top_k = int(self.top_k)
        if top_k < 1 or top_k > self.n_features_in_:
            raise ValueError(
                f"top_k must be in [1, n_features_in_={self.n_features_in_}]; "
                f"got {self.top_k!r}"
            )
        ctx = Context()
        cfg = self._make_config(store_scores=True)
        try:
            model = Model.fit(ctx, cfg, X_arr, y_arr)
        finally:
            cfg.close()
        try:
            res = _methods.variable_select_rank(
                ctx, model, X_arr, method=self._rank_method, top_k=top_k)
            scores = res.matrix("scores").ravel()
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            model.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class VIPSelector(_RankerSelector):
    """Variable Importance in Projection top-k selector (Favilla 2013).

    Parameters
    ----------
    top_k : int
        Number of features to keep.
    n_components, solver, center_x, scale_x, tol, max_iter
        Underlying PLS hyperparameters used for VIP scoring.

    Notes
    -----
    Exposes ``vip_scores_`` as an alias for the generic ``scores_``
    attribute, for callers used to the chemometrics naming convention.
    """
    _rank_method = 0

    def __init__(self, top_k: int, *, n_components: int = 2,
                  solver: str = "nipals",
                  center_x: bool = True, scale_x: bool = True,
                  tol: float = 1e-6, max_iter: int = 500) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.solver = solver
        self.center_x = center_x
        self.scale_x = scale_x
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        super().fit(X, y)
        # Domain alias for chemometricians.
        self.vip_scores_ = self.scores_
        return self


class CoefficientSelector(_RankerSelector):
    """|coef| top-k selector. Ranks features by the magnitude of their
    PLS regression coefficient on Y."""
    _rank_method = 1

    def __init__(self, top_k: int, *, n_components: int = 2,
                  solver: str = "nipals",
                  center_x: bool = True, scale_x: bool = True,
                  tol: float = 1e-6, max_iter: int = 500) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.solver = solver
        self.center_x = center_x
        self.scale_x = scale_x
        self.tol = tol
        self.max_iter = max_iter


class SelectivityRatioSelector(_RankerSelector):
    """Selectivity Ratio top-k selector (Rajalahti 2009)."""
    _rank_method = 2

    def __init__(self, top_k: int, *, n_components: int = 2,
                  solver: str = "nipals",
                  center_x: bool = True, scale_x: bool = True,
                  tol: float = 1e-6, max_iter: int = 500) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.solver = solver
        self.center_x = center_x
        self.scale_x = scale_x
        self.tol = tol
        self.max_iter = max_iter


# ----------------------------------------------------------------------
# Plan-based selectors (need a ValidationPlan)
# ----------------------------------------------------------------------

def _open_plan(n_samples: int, n_folds: int, seed: int) -> ValidationPlan:
    return _default_plan(n_samples, n_folds=n_folds, seed=seed)


class SPASelector(_BasePls4allSelector):
    """Successive Projections Algorithm selector (Araujo 2001)."""

    def __init__(self, top_k: int, *, n_components: int = 2) -> None:
        self.top_k = top_k
        self.n_components = n_components

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        try:
            res = _methods.spa_select(ctx, cfg, X_arr, y_arr,
                                        top_k=int(self.top_k))
            scores = res.matrix("coefficient_scores").ravel()
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class StabilitySelector(_BasePls4allSelector):
    """MCUVE-style stability selector via Monte-Carlo subsampling."""

    def __init__(self, top_k: int, *, n_components: int = 2,
                  n_iterations: int = 50, seed: int = 0,
                  n_folds: int = 3) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.seed = seed
        self.n_folds = n_folds

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), int(self.seed))
        try:
            res = _methods.stability_select(ctx, cfg, X_arr, y_arr, plan,
                                              top_k=int(self.top_k))
            scores = res.matrix("stability_scores").ravel()
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class UVESelector(_BasePls4allSelector):
    """Uninformative Variable Elimination (Centner 1996) with artificial
    noise variables for the threshold."""

    def __init__(self, *, n_components: int = 2,
                  noise_features: int = 50, noise_seed: int = 0) -> None:
        self.n_components = n_components
        self.noise_features = noise_features
        self.noise_seed = noise_seed

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], 3, int(self.noise_seed))
        try:
            res = _methods.uve_select(
                ctx, cfg, X_arr, y_arr, plan,
                noise_features=int(self.noise_features),
                noise_seed=int(self.noise_seed))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
            try:
                scores = res.matrix("real_stability_scores").ravel()
            except Exception:
                scores = None
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class CARSSelector(_BasePls4allSelector):
    """Competitive Adaptive Reweighted Sampling (Li 2009)."""

    def __init__(self, *, n_components: int = 2,
                  n_iterations: int = 50, min_features: int | None = None,
                  n_folds: int = 3, seed: int = 0) -> None:
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.min_features = min_features
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), int(self.seed))
        # C-side requires min_features >= n_components; default to that.
        min_features = int(self.min_features if self.min_features is not None
                            else self.n_components)
        try:
            res = _methods.cars_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_iterations=int(self.n_iterations),
                min_features=min_features)
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=None)
        return self


class RandomFrogSelector(_BasePls4allSelector):
    """Random Frog feature selection (Li 2012)."""

    def __init__(self, top_k: int, *, n_components: int = 2,
                  n_iterations: int = 100, initial_size: int = 20,
                  min_size: int | None = None, max_size: int | None = None,
                  n_folds: int = 3, seed: int = 0) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.initial_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), int(self.seed))
        max_size = int(self.max_size if self.max_size is not None
                        else self.n_features_in_)
        min_size = int(self.min_size if self.min_size is not None
                        else self.n_components)
        try:
            res = _methods.random_frog_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_iterations=int(self.n_iterations),
                initial_size=int(self.initial_size),
                min_size=min_size,
                max_size=max_size,
                top_k=int(self.top_k),
                seed=int(self.seed))
            scores = res.matrix("global_scores").ravel()
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class SCARSSelector(_BasePls4allSelector):
    """Stability-CARS hybrid (Zheng 2014)."""

    def __init__(self, *, n_components: int = 2,
                  n_iterations: int = 50, min_features: int | None = None,
                  sample_fraction: float = 0.8,
                  n_folds: int = 3, seed: int = 0) -> None:
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.min_features = min_features
        self.sample_fraction = sample_fraction
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), int(self.seed))
        min_features = int(self.min_features if self.min_features is not None
                            else self.n_components)
        try:
            res = _methods.scars_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_iterations=int(self.n_iterations),
                min_features=min_features,
                sample_fraction=float(self.sample_fraction),
                seed=int(self.seed))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
            try:
                scores = res.matrix("coefficient_scores")[-1].ravel()
            except Exception:
                scores = None
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class GASelector(_BasePls4allSelector):
    """Genetic Algorithm feature selection."""

    def __init__(self, *, n_components: int = 2,
                  n_generations: int = 30, population_size: int = 40,
                  min_features: int | None = None,
                  max_features: int | None = None,
                  mutation_rate: float = 0.05,
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
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), int(self.seed))
        max_features = int(self.max_features if self.max_features is not None
                            else self.n_features_in_)
        min_features = int(self.min_features if self.min_features is not None
                            else self.n_components)
        try:
            res = _methods.ga_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_generations=int(self.n_generations),
                population_size=int(self.population_size),
                min_features=min_features,
                max_features=max_features,
                mutation_rate=float(self.mutation_rate),
                seed=int(self.seed))
            scores = res.matrix("global_scores").ravel()
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class PSOSelector(_BasePls4allSelector):
    """Binary Particle Swarm Optimization selector."""

    def __init__(self, *, n_components: int = 2,
                  n_swarm: int = 30, n_iterations: int = 50,
                  w: float = 0.729, c1: float = 1.494, c2: float = 1.494,
                  v_max: float = 4.0,
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
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), int(self.seed))
        try:
            res = _methods.pso_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_swarm=int(self.n_swarm),
                n_iterations=int(self.n_iterations),
                w=float(self.w), c1=float(self.c1), c2=float(self.c2),
                v_max=float(self.v_max), seed=int(self.seed))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
            try:
                scores = res.matrix("inclusion_frequencies").ravel()
            except Exception:
                scores = None
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class VISSASelector(_BasePls4allSelector):
    """Variable Iterative Subspace Shrinkage Approach (Deng 2014)."""

    def __init__(self, *, n_components: int = 2,
                  n_iterations: int = 10, n_submodels: int = 60,
                  ratio_kept: float = 0.1,
                  threshold: float = 0.5,
                  floor_probability: float = 0.05,
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
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), int(self.seed))
        try:
            res = _methods.vissa_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_iterations=int(self.n_iterations),
                n_submodels=int(self.n_submodels),
                ratio_kept=float(self.ratio_kept),
                threshold=float(self.threshold),
                floor_probability=float(self.floor_probability),
                seed=int(self.seed))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
            try:
                scores = res.matrix("final_probabilities").ravel()
            except Exception:
                scores = None
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class ShavingSelector(_BasePls4allSelector):
    """Iterative SR-shaving variable elimination."""

    def __init__(self, *, n_components: int = 2, n_steps: int = 10,
                  min_features: int | None = None,
                  shave_fraction: float = 0.2,
                  n_folds: int = 3) -> None:
        self.n_components = n_components
        self.n_steps = n_steps
        self.min_features = min_features
        self.shave_fraction = shave_fraction
        self.n_folds = n_folds

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), 0)
        min_features = int(self.min_features if self.min_features is not None
                            else self.n_components)
        try:
            res = _methods.shaving_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_steps=int(self.n_steps),
                min_features=min_features,
                shave_fraction=float(self.shave_fraction))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=None)
        return self


class BVESelector(_BasePls4allSelector):
    """Backward Variable Elimination with VIP filter."""

    def __init__(self, *, n_components: int = 2,
                  n_steps: int = 10, min_features: int | None = None,
                  n_folds: int = 3) -> None:
        self.n_components = n_components
        self.n_steps = n_steps
        self.min_features = min_features
        self.n_folds = n_folds

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), 0)
        min_features = int(self.min_features if self.min_features is not None
                            else self.n_components)
        try:
            res = _methods.bve_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_steps=int(self.n_steps),
                min_features=min_features)
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=None)
        return self


class REPSelector(_BasePls4allSelector):
    """REP-PLS — repeated VIP-thresholded variable selection."""

    def __init__(self, *, n_components: int = 2,
                  n_steps: int = 10, min_features: int | None = None,
                  remove_count: int = 1, n_folds: int = 3) -> None:
        self.n_components = n_components
        self.n_steps = n_steps
        self.min_features = min_features
        self.remove_count = remove_count
        self.n_folds = n_folds

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), 0)
        min_features = int(self.min_features if self.min_features is not None
                            else self.n_components)
        try:
            res = _methods.rep_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_steps=int(self.n_steps),
                min_features=min_features,
                remove_count=int(self.remove_count))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=None)
        return self


class IPWSelector(_BasePls4allSelector):
    """Iterative Predictor Weighting PLS selector."""

    def __init__(self, top_k: int, *, n_components: int = 2,
                  n_iterations: int = 20, damping: float = 0.5,
                  weight_floor: float = 1e-6,
                  n_folds: int = 3, seed: int = 0) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.damping = damping
        self.weight_floor = weight_floor
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), int(self.seed))
        try:
            res = _methods.ipw_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_iterations=int(self.n_iterations),
                top_k=int(self.top_k),
                damping=float(self.damping),
                weight_floor=float(self.weight_floor))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
            try:
                # score_path is per-iteration; take the last (final scores)
                scores = res.matrix("score_path")[-1].ravel()
            except Exception:
                scores = None
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class STSelector(_BasePls4allSelector):
    """ST-PLS — soft-thresholded sparse PLS selector.

    ``thresholds`` is a sequence of soft-threshold values in ``[0, 1]``;
    the routine sweeps them and picks the most aggressive one that still
    retains ``>= min_selected`` features.
    """

    def __init__(self, thresholds, *, n_components: int = 2,
                  min_selected: int | None = None) -> None:
        self.thresholds = thresholds
        self.n_components = n_components
        self.min_selected = min_selected

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], 3, 0)
        thresholds = np.asarray(self.thresholds, dtype=np.float64).ravel()
        min_selected = int(self.min_selected if self.min_selected is not None
                            else self.n_components)
        try:
            res = _methods.st_select(
                ctx, cfg, X_arr, y_arr, plan,
                thresholds=thresholds,
                min_selected=min_selected)
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=None)
        return self


# ----------------------------------------------------------------------
# Interval-family selectors
# ----------------------------------------------------------------------

class IntervalSelector(_BasePls4allSelector):
    """Forward interval PLS (iPLS, Nørgaard 2000)."""

    def __init__(self, *, n_components: int = 2,
                  interval_width: int = 10, step: int = 5,
                  n_folds: int = 3) -> None:
        self.n_components = n_components
        self.interval_width = interval_width
        self.step = step
        self.n_folds = n_folds

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), 0)
        try:
            res = _methods.interval_select(
                ctx, cfg, X_arr, y_arr, plan,
                interval_width=int(self.interval_width),
                step=int(self.step))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=None)
        return self


class BiPLSSelector(_BasePls4allSelector):
    """biPLS — backward interval elimination (Nørgaard 2000)."""

    def __init__(self, *, n_components: int = 2,
                  interval_width: int = 10, min_intervals: int = 2,
                  n_folds: int = 3) -> None:
        self.n_components = n_components
        self.interval_width = interval_width
        self.min_intervals = min_intervals
        self.n_folds = n_folds

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), 0)
        try:
            res = _methods.bipls_select(
                ctx, cfg, X_arr, y_arr, plan,
                interval_width=int(self.interval_width),
                min_intervals=int(self.min_intervals))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=None)
        return self


class SiPLSSelector(_BasePls4allSelector):
    """siPLS — synergistic interval combinations (Nørgaard 2000)."""

    def __init__(self, *, n_components: int = 2,
                  interval_width: int = 10, combination_size: int = 2,
                  n_folds: int = 3) -> None:
        self.n_components = n_components
        self.interval_width = interval_width
        self.combination_size = combination_size
        self.n_folds = n_folds

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), 0)
        try:
            res = _methods.sipls_select(
                ctx, cfg, X_arr, y_arr, plan,
                interval_width=int(self.interval_width),
                combination_size=int(self.combination_size))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=None)
        return self


# ----------------------------------------------------------------------
# T² / WVC / threshold-based selectors
# ----------------------------------------------------------------------

class T2Selector(_BasePls4allSelector):
    """T²-PLS loading-weight selection (plsVarSel::T2_pls style)."""

    def __init__(self, alpha_thresholds, *, n_components: int = 2,
                  min_selected: int | None = None) -> None:
        self.alpha_thresholds = alpha_thresholds
        self.n_components = n_components
        self.min_selected = min_selected

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], 3, 0)
        alpha = np.asarray(self.alpha_thresholds, dtype=np.float64).ravel()
        min_selected = int(self.min_selected if self.min_selected is not None
                            else self.n_components)
        try:
            res = _methods.t2_select(
                ctx, cfg, X_arr, y_arr, plan,
                alpha_thresholds=alpha,
                min_selected=min_selected)
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
            try:
                scores = res.matrix("t2_scores").ravel()
            except Exception:
                scores = None
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


class WVCSelector(_BasePls4allSelector):
    """WVC-PLS — weighted variable contribution top-k selector."""

    def __init__(self, top_k: int, *, n_components: int = 2,
                  normalize: bool = True) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.normalize = normalize

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        res = _methods.wvc_select(
            ctx, X_arr, y_arr,
            n_components=int(self.n_components),
            top_k=int(self.top_k),
            normalize=bool(self.normalize))
        scores = res.matrix("final_scores").ravel()
        indices = np.asarray(res.vector_int64("selected_indices"),
                              dtype=np.int64)
        self._commit_fit(indices=indices, scores=scores)
        return self


class WVCThresholdSelector(_BasePls4allSelector):
    """Threshold-/factor-based WVC-PLS selector."""

    def __init__(self, *, n_components: int = 2, normalize: bool = True,
                  score_threshold: float = 0.0,
                  threshold_factor: float = 1.0,
                  min_selected: int | None = None) -> None:
        self.n_components = n_components
        self.normalize = normalize
        self.score_threshold = score_threshold
        self.threshold_factor = threshold_factor
        self.min_selected = min_selected

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        min_selected = int(self.min_selected if self.min_selected is not None
                            else self.n_components)
        res = _methods.wvc_threshold_select(
            ctx, X_arr, y_arr,
            n_components=int(self.n_components),
            normalize=bool(self.normalize),
            score_threshold=float(self.score_threshold),
            threshold_factor=float(self.threshold_factor),
            min_selected=min_selected)
        scores = res.matrix("final_scores").ravel()
        indices = np.asarray(res.vector_int64("selected_indices"),
                              dtype=np.int64)
        self._commit_fit(indices=indices, scores=scores)
        return self


class EMCUVESelector(_BasePls4allSelector):
    """Ensemble Monte-Carlo UVE selector."""

    def __init__(self, *, n_components: int = 2,
                  noise_features: int = 50, noise_seed: int = 0,
                  n_ensembles: int = 10, vote_threshold: float = 0.5,
                  n_folds: int = 3) -> None:
        self.n_components = n_components
        self.noise_features = noise_features
        self.noise_seed = noise_seed
        self.n_ensembles = n_ensembles
        self.vote_threshold = vote_threshold
        self.n_folds = n_folds

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), int(self.noise_seed))
        try:
            res = _methods.emcuve_select(
                ctx, cfg, X_arr, y_arr, plan,
                noise_features=int(self.noise_features),
                noise_seed=int(self.noise_seed),
                n_ensembles=int(self.n_ensembles),
                vote_threshold=float(self.vote_threshold))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=None)
        return self


class RandomizationSelector(_BasePls4allSelector):
    """Randomization-test PLS selector (Y-permutation p-values)."""

    def __init__(self, *, n_components: int = 2,
                  n_permutations: int = 200,
                  randomization_seed: int = 0,
                  alpha: float = 0.05) -> None:
        self.n_components = n_components
        self.n_permutations = n_permutations
        self.randomization_seed = randomization_seed
        self.alpha = alpha

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        try:
            res = _methods.randomization_select(
                ctx, cfg, X_arr, y_arr,
                n_permutations=int(self.n_permutations),
                randomization_seed=int(self.randomization_seed),
                alpha=float(self.alpha))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
            try:
                scores = res.matrix("p_values").ravel()
            except Exception:
                scores = None
        finally:
            cfg.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


# ----------------------------------------------------------------------
# Phase 50+ selectors (IRIV, IRF, VIPSPA)
# ----------------------------------------------------------------------

class IRIVSelector(_BasePls4allSelector):
    """IRIV — Iteratively Retains Informative Variables (Yun 2014)."""

    def __init__(self, *, n_components: int = 2,
                  max_rounds: int = 5, n_folds: int = 5,
                  seed: int = 0) -> None:
        self.n_components = n_components
        self.max_rounds = max_rounds
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], int(self.n_folds), int(self.seed))
        try:
            res = _methods.iriv_select(
                ctx, cfg, X_arr, y_arr, plan,
                max_rounds=int(self.max_rounds),
                seed=int(self.seed))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
            plan.close()
        self._commit_fit(indices=indices, scores=None)
        return self


class IRFSelector(_BasePls4allSelector):
    """IRF — Interval Random Frog (Yun 2013)."""

    def __init__(self, top_k: int, *, n_components: int = 2,
                  n_iterations: int = 100, window_size: int = 5,
                  initial_intervals: int = 5,
                  seed: int = 0) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.window_size = window_size
        self.initial_intervals = initial_intervals
        self.seed = seed

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        plan = _open_plan(X_arr.shape[0], 3, int(self.seed))
        try:
            res = _methods.irf_select(
                ctx, cfg, X_arr, y_arr, plan,
                n_iterations=int(self.n_iterations),
                window_size=int(self.window_size),
                initial_intervals=int(self.initial_intervals),
                top_k=int(self.top_k),
                seed=int(self.seed))
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
            try:
                scores = res.matrix("probability").ravel()
            except Exception:
                scores = None
        finally:
            cfg.close()
            plan.close()
        # IRF scores are per-interval; expand to per-feature is non-trivial,
        # so we leave scores_ as-is (NaN-fill).
        self._commit_fit(indices=indices, scores=None)
        if scores is not None:
            self.interval_probability_ = scores
        return self


class VIPSPASelector(_BasePls4allSelector):
    """VIP_SPA — VIP-mask + SPA greedy (Phase 53)."""

    def __init__(self, top_k: int, *, n_components: int = 2,
                  vip_threshold: float = 0.3) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.vip_threshold = vip_threshold

    def fit(self, X, y):
        X_arr, y_arr, _ = _check_X_y_p4a(self, X, y)
        ctx = Context()
        cfg = self._make_config()
        try:
            res = _methods.vip_spa_select(
                ctx, cfg, X_arr, y_arr,
                vip_threshold=float(self.vip_threshold),
                top_k=int(self.top_k))
            scores = res.matrix("vip_scores").ravel()
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            cfg.close()
        self._commit_fit(indices=indices, scores=scores)
        return self


__all__ = [
    "VIPSelector",
    "CoefficientSelector",
    "SelectivityRatioSelector",
    "SPASelector",
    "StabilitySelector",
    "UVESelector",
    "CARSSelector",
    "RandomFrogSelector",
    "SCARSSelector",
    "GASelector",
    "PSOSelector",
    "VISSASelector",
    "ShavingSelector",
    "BVESelector",
    "REPSelector",
    "IPWSelector",
    "STSelector",
    "IntervalSelector",
    "BiPLSSelector",
    "SiPLSSelector",
    "T2Selector",
    "WVCSelector",
    "WVCThresholdSelector",
    "EMCUVESelector",
    "RandomizationSelector",
    "IRIVSelector",
    "IRFSelector",
    "VIPSPASelector",
]
