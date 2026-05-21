"""Internal helpers shared by every pls4all.sklearn wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils.validation import check_array, check_X_y

from .._config import Config
from .._context import Context
from .._model import Model, ModelArrayKind
from .._types import Algorithm, Deflation, Solver


_SOLVER_ALIASES: dict[str, Solver] = {
    "nipals": Solver.NIPALS,
    "simpls": Solver.SIMPLS,
    "orthogonal-scores": Solver.ORTHOGONAL_SCORES,
    "kernel": Solver.KERNEL_ALGORITHM,
    "wide-kernel": Solver.WIDE_KERNEL,
    "svd": Solver.SVD,
    "power": Solver.POWER,
    "randomized-svd": Solver.RANDOMIZED_SVD,
}


def _resolve_solver(value: str | Solver) -> Solver:
    if isinstance(value, Solver):
        return value
    key = str(value).strip().lower().replace("_", "-")
    try:
        return _SOLVER_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown solver {value!r}; expected one of "
            f"{sorted(_SOLVER_ALIASES)}"
        ) from exc


def _as_2d_float64(arr: Any, *, name: str) -> np.ndarray:
    """Coerce ``arr`` to a C-contiguous float64 2-D ndarray.

    Used at predict / transform time on the X side, where sklearn-style
    validation (NaN guard, dtype check, feature-count check) has already
    been done by ``_check_X_y_p4a`` or ``_check_X_p4a``. Reshape 1-D to
    column vector; reject anything else.
    """
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    elif out.ndim != 2:
        raise ValueError(
            f"{name} must be 1-D or 2-D; got ndim={out.ndim}"
        )
    return out


def _check_X_y_p4a(estimator: Any, X: Any, y: Any
                    ) -> tuple[np.ndarray, np.ndarray, int]:
    """sklearn-style X/y validation for fit().

    Returns ``(X_2d_f64, y_2d_f64, y_ndim_original)``. ``y_ndim_original``
    captures whether the caller passed 1-D or 2-D so ``predict()`` can
    return the matching shape. Casts to float64, enforces C-contiguity,
    rejects NaN/Inf, sets ``estimator.n_features_in_`` and (when present)
    ``estimator.feature_names_in_``.
    """
    y_ndim = int(np.asarray(y).ndim) if y is not None else 0
    X_chk, y_chk = check_X_y(
        X, y,
        dtype=np.float64,
        ensure_2d=True,
        allow_nd=False,
        ensure_all_finite=True,
        multi_output=True,
        y_numeric=False,  # we accept int class labels too
    )
    X_arr = np.ascontiguousarray(X_chk, dtype=np.float64)
    y_arr = np.ascontiguousarray(y_chk, dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    estimator.n_features_in_ = int(X_arr.shape[1])
    if hasattr(X, "columns"):
        estimator.feature_names_in_ = np.asarray(X.columns, dtype=object)
    return X_arr, y_arr, y_ndim


def _check_X_p4a(estimator: Any, X: Any) -> np.ndarray:
    """sklearn-style X validation for predict() / transform(). Enforces
    feature count against ``estimator.n_features_in_``."""
    X_chk = check_array(
        X,
        dtype=np.float64,
        ensure_2d=True,
        allow_nd=False,
        ensure_all_finite=True,
    )
    X_arr = np.ascontiguousarray(X_chk, dtype=np.float64)
    expected = getattr(estimator, "n_features_in_", None)
    if expected is not None and X_arr.shape[1] != expected:
        raise ValueError(
            f"X has {X_arr.shape[1]} features, but "
            f"{type(estimator).__name__} was fitted with {expected}"
        )
    return X_arr


def _validate_X_y_no_mutate(X: Any, y: Any
                              ) -> tuple[np.ndarray, np.ndarray, int]:
    """sklearn-style X/y validation that does NOT mutate the caller's
    fitted state. Returns ``(X_2d_f64, y_2d_f64, y_ndim_original)``.

    Use this in ``fit()`` paths that want to keep the previous fitted
    state intact when the *new* call validates badly — opposite of
    :func:`_check_X_y_p4a` which immediately sets ``n_features_in_``.
    """
    y_ndim = int(np.asarray(y).ndim) if y is not None else 0
    X_chk, y_chk = check_X_y(
        X, y,
        dtype=np.float64,
        ensure_2d=True,
        allow_nd=False,
        ensure_all_finite=True,
        multi_output=True,
        y_numeric=False,
    )
    X_arr = np.ascontiguousarray(X_chk, dtype=np.float64)
    y_arr = np.ascontiguousarray(y_chk, dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    return X_arr, y_arr, y_ndim


def _config_from_params(*,
                          n_components: int,
                          algorithm: Algorithm,
                          solver: Solver,
                          deflation: Deflation = Deflation.REGRESSION,
                          center_x: bool = True,
                          scale_x: bool = True,
                          center_y: bool = True,
                          scale_y: bool = False,
                          tol: float = 1e-6,
                          max_iter: int = 500,
                          store_scores: bool = False) -> Config:
    cfg = Config()
    cfg.algorithm = algorithm
    cfg.solver = solver
    cfg.deflation = deflation
    cfg.n_components = int(n_components)
    cfg.center_x = bool(center_x)
    cfg.scale_x = bool(scale_x)
    cfg.center_y = bool(center_y)
    cfg.scale_y = bool(scale_y)
    cfg.tol = float(tol)
    cfg.max_iter = int(max_iter)
    cfg.store_scores = bool(store_scores)
    return cfg


class _Pls4allModelEstimator:
    """Persistence + handle-caching mixin for Model-based sklearn wrappers.

    Fitted state lives in two places:

    * ``self._bundle_`` — the C ABI `.n4a` byte buffer, the only state
      that survives ``pickle.dumps`` / ``copy.deepcopy``.
    * ``self._model_handle_`` / ``self._model_ctx_`` — lazily rebuilt
      C handles for ``predict`` / ``transform`` hot paths.

    Both handles are dropped from ``__getstate__`` because they are
    bound to a specific in-process libn4m.so and cannot be serialized.

    **set_params on a fitted estimator**: this mixin deliberately does
    NOT invalidate the cached bundle when ``set_params`` is called on a
    fitted instance — same convention as every other sklearn estimator
    (``LogisticRegression(C=1).fit(X, y).set_params(C=10).predict(X)``
    returns predictions from the ``C=1`` fit until ``fit`` is called
    again). Call ``fit`` again after ``set_params`` to apply
    hyperparameter changes. ``clone()`` is unaffected because it builds
    a fresh instance via ``__init__``, leaving fitted state behind.
    """

    _bundle_: bytes | None = None

    def _store_fitted_model(self, ctx: Context, model: Model) -> None:
        """Persist the fitted model bundle and cache the active handle."""
        self._bundle_ = model.to_bytes()
        self._model_ctx_ = ctx
        self._model_handle_ = model

    def _ensure_model_handle(self) -> tuple[Context, Model]:
        if not getattr(self, "_bundle_", None):
            raise RuntimeError(
                f"{type(self).__name__}: estimator is not fitted yet"
            )
        cached_ctx = getattr(self, "_model_ctx_", None)
        cached_handle = getattr(self, "_model_handle_", None)
        if cached_ctx is not None and cached_handle is not None:
            return cached_ctx, cached_handle
        ctx = Context()
        model = Model.from_bytes(ctx, self._bundle_)
        self._model_ctx_ = ctx
        self._model_handle_ = model
        return ctx, model

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        # Drop the live C handles — only the byte bundle survives pickling.
        state.pop("_model_handle_", None)
        state.pop("_model_ctx_", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Lazy: handles get rebuilt on the next predict/transform call.

    def _drop_handle(self) -> None:
        # Useful when set_params changes a hyperparameter and we want to
        # invalidate the cached handle so refit happens on the next fit.
        if getattr(self, "_model_handle_", None) is not None:
            try:
                self._model_handle_.close()
            except Exception:
                pass
        self._model_handle_ = None
        self._model_ctx_ = None
