"""sklearn-compatible variable-selection wrappers.

Selectors are MethodResult-based on the C side: the C ABI returns a
scoring + selection report, not a re-fittable `p4a_model_t`. Persistence
therefore uses plain NumPy state (mask + scores) rather than the `.n4a`
bundle path used by Model-based regressors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from .._context import Context
from .._methods import variable_select_rank
from .._model import Model
from .._types import Algorithm, Deflation
from ._base import _check_X_y_p4a, _config_from_params, _resolve_solver


class VIPSelector(SelectorMixin, BaseEstimator):
    """Variable Importance in Projection (Favilla 2013) top-k selector.

    Fits a PLS regression on (X, y), computes the VIP score per feature
    from the fitted model, and retains the `top_k` highest-scoring
    features. Compatible with `Pipeline`, `ColumnTransformer`, and
    `feature_selection.SelectFromModel`-style introspection.

    Hybrid contract: implements the sklearn `SelectorMixin` (so
    `get_support()` and `transform()` work out of the box) AND exposes
    the raw scoring artefacts as attributes for analysis:

    * `support_` : bool ndarray of shape (n_features_in_,)
    * `vip_scores_` : float64 ndarray of shape (n_features_in_,)
    * `selected_indices_` : int64 ndarray of selected feature indices
      (in selection order, not numerical order)

    Parameters
    ----------
    top_k : int
        Number of features to keep.
    n_components : int, default=2
        Components used to fit the underlying PLS for VIP scoring.
    solver : str, default='nipals'
    center_x, scale_x : bool, default=True
    tol : float, default=1e-6
    max_iter : int, default=500
    """

    def __init__(self,
                  top_k: int,
                  *,
                  n_components: int = 2,
                  solver: str = "nipals",
                  center_x: bool = True,
                  scale_x: bool = True,
                  tol: float = 1e-6,
                  max_iter: int = 500) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.solver = solver
        self.center_x = center_x
        self.scale_x = scale_x
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X: Any, y: Any) -> "VIPSelector":
        X_arr, y_arr, _y_ndim = _check_X_y_p4a(self, X, y)
        # Validate top_k now that we know n_features_in_.
        top_k = int(self.top_k)
        if top_k < 1 or top_k > self.n_features_in_:
            raise ValueError(
                f"top_k must be in [1, n_features_in_={self.n_features_in_}]; "
                f"got top_k={self.top_k!r}"
            )
        ctx = Context()
        cfg = _config_from_params(
            n_components=self.n_components,
            algorithm=Algorithm.PLS_REGRESSION,
            solver=_resolve_solver(self.solver),
            deflation=Deflation.REGRESSION,
            center_x=self.center_x,
            scale_x=self.scale_x,
            center_y=True,
            scale_y=False,
            tol=self.tol,
            max_iter=self.max_iter,
            store_scores=True,  # variable_select_rank uses VIP -> needs scores
        )
        try:
            model = Model.fit(ctx, cfg, X_arr, y_arr)
        finally:
            cfg.close()
        try:
            res = variable_select_rank(
                ctx, model, X_arr,
                method=0,  # 0 = VIP scoring (see variable_selection.cpp)
                top_k=top_k,
            )
            # Persist as pure NumPy state — survives pickle without C handles.
            scores = res.matrix("scores").ravel()
            indices = np.asarray(res.vector_int64("selected_indices"),
                                  dtype=np.int64)
        finally:
            model.close()
        # n_features_in_ was set by _check_X_y_p4a.
        self.vip_scores_ = scores.astype(np.float64, copy=True)
        self.selected_indices_ = indices.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[indices] = True
        self.support_ = mask
        return self

    # SelectorMixin hook — must return a bool[n_features_in_] mask.
    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self)
        return self.support_
