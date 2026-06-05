# SPDX-License-Identifier: CECILL-2.1
"""AOM robust-HPO portfolio reference objects.

The Phase 7f integration starts with source-free routing primitives. Dataset
metadata may be reported for audit, but it must never affect deployable route
selection.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from ._compat import BaseEstimator


@dataclass(frozen=True)
class AOMDatasetMetadata:
    """Audit-only dataset metadata.

    These fields are intentionally not used by :class:`AOMStructuralPolicy`.
    They are accepted so callers can keep audit context without creating a
    source/dataset-name route.
    """

    dataset_id: str | None = None
    database_name: str | None = None


EstimatorFactory = Callable[[], object]


@dataclass(frozen=True)
class EndpointStabilityDecision:
    """Serializable train-only endpoint decision."""

    chosen_endpoint: str
    cv_best_endpoint: str
    oof_best_endpoint: str
    accepted_endpoint: bool
    cv_margin_ratio: float
    oof_margin_ratio: float


@dataclass(frozen=True)
class AOMCandidateSpec:
    """One chain plus one prediction head configuration."""

    chain_name: str
    head: str
    param: float

    @property
    def name(self) -> str:
        if self.head == "ridge":
            return f"{self.chain_name}__ridge_a{self.param:g}"
        return f"{self.chain_name}__{self.head}_{self.param:g}"


@dataclass
class _FittedAOMCandidate:
    spec: AOMCandidateSpec
    chain: object
    head: object


class _IdentityTransform(BaseEstimator):
    def fit(self, X, y=None) -> "_IdentityTransform":
        X_arr = _as_feature_matrix(X)
        self.n_features_in_ = int(X_arr.shape[1])
        return self

    def transform(self, X) -> np.ndarray:
        return _as_feature_matrix(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _as_target_array(y) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    if y_arr.ndim not in (1, 2):
        raise ValueError("y must be 1D or 2D")
    return y_arr


def _match_prediction_shape(pred, reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    arr = np.asarray(pred, dtype=float)
    if ref.ndim == 1:
        arr = arr.reshape(-1)
        if arr.shape[0] != ref.shape[0]:
            raise ValueError("prediction length does not match validation fold")
        return arr
    if arr.ndim == 1:
        arr = arr.reshape(ref.shape)
    else:
        arr = arr.reshape(ref.shape)
    return arr


def _as_feature_matrix(X) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.ndim != 2:
        raise ValueError("transformed X must be a 2D array")
    return X_arr


def _make_estimator(estimator_or_factory: object | EstimatorFactory) -> object:
    if callable(estimator_or_factory) and not hasattr(estimator_or_factory, "fit"):
        return estimator_or_factory()
    return deepcopy(estimator_or_factory)


def _fit_transform_chain(chain: object, X: np.ndarray, y: np.ndarray) -> tuple[object, np.ndarray]:
    fitted = deepcopy(chain)
    if hasattr(fitted, "fit_transform"):
        try:
            return fitted, _as_feature_matrix(fitted.fit_transform(X, y=y))
        except TypeError:
            try:
                return fitted, _as_feature_matrix(fitted.fit_transform(X, y))
            except TypeError:
                return fitted, _as_feature_matrix(fitted.fit_transform(X))
    if hasattr(fitted, "fit"):
        try:
            fitted.fit(X, y)
        except TypeError:
            fitted.fit(X)
    if not hasattr(fitted, "transform"):
        raise TypeError("chain must expose transform or fit_transform")
    return fitted, _as_feature_matrix(fitted.transform(X))


def _transform_chain(chain: object, X: np.ndarray) -> np.ndarray:
    if not hasattr(chain, "transform"):
        raise TypeError("chain must expose transform")
    return _as_feature_matrix(chain.transform(X))


def _fold_indices(
    n_samples: int,
    *,
    cv: int,
    shuffle: bool,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(int(cv), int(n_samples))
    if n_splits < 2:
        raise ValueError("at least two CV folds are required")
    indices = np.arange(n_samples)
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    folds = [np.sort(fold) for fold in np.array_split(indices, n_splits) if fold.size]
    out = []
    for val_idx in folds:
        mask = np.ones(n_samples, dtype=bool)
        mask[val_idx] = False
        out.append((np.nonzero(mask)[0], val_idx))
    return out


def _fit_predict_oof(
    estimator_like: object | EstimatorFactory,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: int,
    shuffle: bool,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    pred = np.empty(X.shape[0], dtype=float)
    losses: list[float] = []
    for train_idx, val_idx in _fold_indices(
        X.shape[0],
        cv=cv,
        shuffle=shuffle,
        seed=seed,
    ):
        estimator = _make_estimator(estimator_like)
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError("endpoint estimator must expose fit and predict")
        estimator.fit(X[train_idx], y[train_idx])
        fold_pred = np.asarray(estimator.predict(X[val_idx]), dtype=float).ravel()
        pred[val_idx] = fold_pred
        losses.append(_rmse(y[val_idx], fold_pred))
    return pred, losses


class _RidgeHead:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeHead":
        X = _as_feature_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible lengths")
        x_mean = X.mean(axis=0)
        y_mean = float(y.mean())
        Xc = X - x_mean
        yc = y - y_mean
        gram = Xc.T @ Xc
        gram.flat[:: gram.shape[0] + 1] += max(self.alpha, 0.0)
        try:
            coef = np.linalg.solve(gram, Xc.T @ yc)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(gram) @ (Xc.T @ yc)
        self.coef_ = np.asarray(coef, dtype=float).ravel()
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _as_feature_matrix(X)
        return X @ self.coef_ + self.intercept_


class _PLS1Head:
    """Dependency-light single-target PLS head for fold-local screening."""

    def __init__(self, n_components: int) -> None:
        self.n_components = int(n_components)
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_PLS1Head":
        X = _as_feature_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible lengths")
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = float(y.mean())
        Xr = X - self.x_mean_
        yr = y - self.y_mean_
        max_components = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        weights = []
        loadings = []
        y_loadings = []
        for _component in range(max_components):
            w = Xr.T @ yr
            w_norm = float(np.linalg.norm(w))
            if w_norm <= 1e-12:
                break
            w = w / w_norm
            t = Xr @ w
            denom = float(t @ t)
            if denom <= 1e-12:
                break
            p = (Xr.T @ t) / denom
            q = float((yr @ t) / denom)
            Xr = Xr - np.outer(t, p)
            yr = yr - q * t
            weights.append(w)
            loadings.append(p)
            y_loadings.append(q)
        if not weights:
            self.coef_ = np.zeros(X.shape[1], dtype=float)
            self.intercept_ = self.y_mean_
            self.n_components_ = 0
            return self
        W = np.column_stack(weights)
        P = np.column_stack(loadings)
        q = np.asarray(y_loadings, dtype=float)
        rotations = W @ np.linalg.pinv(P.T @ W)
        self.coef_ = np.asarray(rotations @ q, dtype=float).ravel()
        self.intercept_ = float(self.y_mean_ - self.x_mean_ @ self.coef_)
        self.n_components_ = int(len(weights))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _as_feature_matrix(X)
        return X @ self.coef_ + self.intercept_


class AOMPreprocessingChain(BaseEstimator):
    """Fold-safe sequential spectral preprocessing chain.

    The chain is intentionally small: each step must expose either
    ``fit_transform`` or ``fit`` + ``transform``. During :meth:`fit`, every
    step is deep-copied before being fitted, so the same chain object can be
    reused safely inside CV selectors.
    """

    def __init__(self, steps: Sequence[tuple[str, object]] | None = None) -> None:
        self.steps = tuple(steps or ())

    def fit(self, X, y=None) -> "AOMPreprocessingChain":
        Xt = _as_feature_matrix(X)
        fitted_steps = []
        for name, step in self.steps:
            if not name:
                raise ValueError("chain step names must be non-empty")
            fitted = deepcopy(step)
            if hasattr(fitted, "fit_transform"):
                try:
                    Xt = _as_feature_matrix(fitted.fit_transform(Xt, y=y))
                except TypeError:
                    try:
                        Xt = _as_feature_matrix(fitted.fit_transform(Xt, y))
                    except TypeError:
                        Xt = _as_feature_matrix(fitted.fit_transform(Xt))
            else:
                if hasattr(fitted, "fit"):
                    try:
                        fitted.fit(Xt, y)
                    except TypeError:
                        fitted.fit(Xt)
                if not hasattr(fitted, "transform"):
                    raise TypeError("chain steps must expose transform or fit_transform")
                Xt = _as_feature_matrix(fitted.transform(Xt))
            fitted_steps.append((name, fitted))
        self.steps_ = tuple(fitted_steps)
        self.n_features_in_ = int(_as_feature_matrix(X).shape[1])
        self.n_features_out_ = int(Xt.shape[1])
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "steps_"):
            raise RuntimeError("AOMPreprocessingChain.transform called before fit")
        Xt = _as_feature_matrix(X)
        for _name, step in self.steps_:
            Xt = _as_feature_matrix(step.transform(Xt))
        return Xt

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


def _chain(*steps: tuple[str, object]) -> AOMPreprocessingChain:
    return AOMPreprocessingChain(steps)


def build_aom_control_chain_bank(
    profile: str = "compact",
    *,
    include_identity: bool = True,
    extra_chains: Mapping[str, object] | None = None,
) -> dict[str, AOMPreprocessingChain | object]:
    """Build the source-free preprocessing chain bank used by AOM robust HPO.

    Profiles:

    - ``compact``: fast end-user bank with the recurring robust controls;
    - ``robust``: broader default with extra SavGol, derivative, and scatter
      variants;
    - ``wide`` / ``lab``: larger lab-style bank for reproducing the diversity
      of the archived campaigns without source/dataset routing.
    """

    from .baseline import AsLS, Detrend
    from .preprocessing import (
        AreaNormalization,
        BaselineCenter,
        Derivate,
        EMSC,
        Gaussian,
        MSC,
        NorrisWilliams,
        RNV,
        SNV,
        SavitzkyGolay,
    )

    normalized = profile.lower().replace("-", "_")
    aliases = {
        "end_user": "compact",
        "default": "robust",
        "large": "wide",
        "moment_lab": "wide",
        "lab": "wide",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"compact", "robust", "wide"}:
        raise ValueError("profile must be compact, robust, wide, lab, or moment_lab")

    bank: dict[str, AOMPreprocessingChain | object] = {}
    if include_identity:
        bank["raw"] = _chain(("identity", _IdentityTransform()))

    bank.update({
        "row_center": _chain(("row_center", BaselineCenter())),
        "snv": _chain(("snv", SNV())),
        "msc": _chain(("msc", MSC())),
        "emsc2": _chain(("emsc2", EMSC(degree=2))),
        "detrend1": _chain(("detrend1", Detrend(polyorder=1))),
        "detrend2": _chain(("detrend2", Detrend(polyorder=2))),
        "savgol_w5_p2_d0": _chain(("savgol_w5_p2_d0", SavitzkyGolay(5, 2, 0))),
        "savgol_w7_p2_d1": _chain(("savgol_w7_p2_d1", SavitzkyGolay(7, 2, 1))),
        "savgol_w11_p2_d2": _chain(("savgol_w11_p2_d2", SavitzkyGolay(11, 2, 2))),
        "nw_s5_g5_d1": _chain(("nw_s5_g5_d1", NorrisWilliams(5, 5, 1))),
        "finite_diff1": _chain(("finite_diff1", Derivate(order=1))),
    })

    if normalized in {"robust", "wide"}:
        bank.update({
            "rnv": _chain(("rnv", RNV())),
            "area_sum": _chain(("area_sum", AreaNormalization("sum"))),
            "emsc1": _chain(("emsc1", EMSC(degree=1))),
            "emsc3": _chain(("emsc3", EMSC(degree=3))),
            "gaussian_s1": _chain(("gaussian_s1", Gaussian(sigma=1.0))),
            "gaussian_s2": _chain(("gaussian_s2", Gaussian(sigma=2.0))),
            "savgol_w9_p2_d0": _chain(("savgol_w9_p2_d0", SavitzkyGolay(9, 2, 0))),
            "savgol_w9_p3_d1": _chain(("savgol_w9_p3_d1", SavitzkyGolay(9, 3, 1))),
            "savgol_w15_p3_d0": _chain(("savgol_w15_p3_d0", SavitzkyGolay(15, 3, 0))),
            "savgol_w15_p3_d1": _chain(("savgol_w15_p3_d1", SavitzkyGolay(15, 3, 1))),
            "nw_s7_g7_d1": _chain(("nw_s7_g7_d1", NorrisWilliams(7, 7, 1))),
            "snv_savgol_w7_d1": _chain(
                ("snv", SNV()),
                ("savgol_w7_p2_d1", SavitzkyGolay(7, 2, 1)),
            ),
            "detrend1_snv": _chain(("detrend1", Detrend(1)), ("snv", SNV())),
            "msc_savgol_w7_d1": _chain(
                ("msc", MSC()),
                ("savgol_w7_p2_d1", SavitzkyGolay(7, 2, 1)),
            ),
        })

    if normalized == "wide":
        bank.update({
            "asls_l1e5": _chain(("asls_l1e5", AsLS(lam=1e5))),
            "asls_l1e6": _chain(("asls_l1e6", AsLS(lam=1e6))),
            "finite_diff2": _chain(("finite_diff2", Derivate(order=2))),
            "savgol_w21_p3_d0": _chain(("savgol_w21_p3_d0", SavitzkyGolay(21, 3, 0))),
            "savgol_w21_p3_d1": _chain(("savgol_w21_p3_d1", SavitzkyGolay(21, 3, 1))),
            "savgol_w21_p3_d2": _chain(("savgol_w21_p3_d2", SavitzkyGolay(21, 3, 2))),
            "savgol_w31_p3_d0": _chain(("savgol_w31_p3_d0", SavitzkyGolay(31, 3, 0))),
            "savgol_w31_p3_d1": _chain(("savgol_w31_p3_d1", SavitzkyGolay(31, 3, 1))),
            "nw_s11_g11_d1": _chain(("nw_s11_g11_d1", NorrisWilliams(11, 11, 1))),
            "snv_detrend1": _chain(("snv", SNV()), ("detrend1", Detrend(1))),
            "snv_savgol_w11_d2": _chain(
                ("snv", SNV()),
                ("savgol_w11_p2_d2", SavitzkyGolay(11, 2, 2)),
            ),
            "emsc2_savgol_w7_d1": _chain(
                ("emsc2", EMSC(2)),
                ("savgol_w7_p2_d1", SavitzkyGolay(7, 2, 1)),
            ),
            "detrend1_nw_s5_g5_d1": _chain(
                ("detrend1", Detrend(1)),
                ("nw_s5_g5_d1", NorrisWilliams(5, 5, 1)),
            ),
        })

    if extra_chains:
        overlap = set(bank).intersection(extra_chains)
        if overlap:
            raise ValueError(f"extra_chains duplicate built-in names: {sorted(overlap)}")
        bank.update(extra_chains)
    return bank


class AOMStructuralPolicy(BaseEstimator):
    """Source-free structural router for AOM portfolio estimators.

    Routing is based on ``X.shape[1]`` and explicit constructor flags only:

    - ``p <= 700``: ``p700_truebank_endpoint_portfolio`` by default;
    - ``701 <= p <= 1200``: ``mid_p_margin_stability_gate``;
    - ``p > 1200``: ``pgt1200_control_selector`` by default, or
      ``pgt1200_kernel_rff_admission`` when enabled.

    ``metadata`` passed to :meth:`fit` is audit-only. Changing dataset/source
    identifiers must not change the selected route or predictions.
    """

    def __init__(
        self,
        route_estimators: Mapping[str, object | EstimatorFactory],
        *,
        enable_pgt1200_admissions: bool = False,
        enable_p700_protocol_unified: bool = False,
        enable_p700_block_local_admission: bool = False,
        p700_protocol_min_features: int | None = None,
    ) -> None:
        if not route_estimators:
            raise ValueError("route_estimators must not be empty")
        self.route_estimators = dict(route_estimators)
        self.enable_pgt1200_admissions = bool(enable_pgt1200_admissions)
        self.enable_p700_protocol_unified = bool(enable_p700_protocol_unified)
        self.enable_p700_block_local_admission = bool(enable_p700_block_local_admission)
        self.p700_protocol_min_features = p700_protocol_min_features

    def fit(
        self,
        X,
        y,
        *,
        metadata: AOMDatasetMetadata | Mapping[str, str | None] | None = None,
    ) -> "AOMStructuralPolicy":
        X_arr, y_arr = self._validate_xy(X, y)
        meta = self._coerce_metadata(metadata)
        route = self.choose_route(n_features=X_arr.shape[1])
        estimator = self._make_estimator(route)
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError(f"estimator for route {route!r} must expose fit and predict")
        estimator.fit(X_arr, y_arr)
        self.route_ = route
        self.estimator_ = estimator
        self.routing_report_ = {
            "route": route,
            "n_features": int(X_arr.shape[1]),
            "dataset_id": meta.dataset_id,
            "database_name": meta.database_name,
            "metadata_used_for_routing": False,
            "enable_pgt1200_admissions": self.enable_pgt1200_admissions,
            "enable_p700_protocol_unified": self.enable_p700_protocol_unified,
            "enable_p700_block_local_admission": self.enable_p700_block_local_admission,
            "p700_protocol_min_features": self.p700_protocol_min_features,
        }
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("AOMStructuralPolicy.predict called before fit")
        X_arr = self._as_2d_float(X)
        return np.asarray(self.estimator_.predict(X_arr), dtype=float).ravel()

    def choose_route(self, *, n_features: int) -> str:
        """Return the deployable route for a feature count."""
        n_features = int(n_features)
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if n_features <= 700:
            if self.enable_p700_block_local_admission:
                return "p700_block_local_admission"
            protocol_allowed = (
                self.p700_protocol_min_features is None
                or n_features >= int(self.p700_protocol_min_features)
            )
            if self.enable_p700_protocol_unified and protocol_allowed:
                return "p700_protocol_unified_selector"
            return "p700_truebank_endpoint_portfolio"
        if n_features <= 1200:
            return "mid_p_margin_stability_gate"
        if self.enable_pgt1200_admissions:
            return "pgt1200_kernel_rff_admission"
        return "pgt1200_control_selector"

    def _make_estimator(self, route: str) -> object:
        if route not in self.route_estimators:
            raise ValueError(f"missing estimator for selected route: {route}")
        return _make_estimator(self.route_estimators[route])

    @staticmethod
    def _coerce_metadata(
        metadata: AOMDatasetMetadata | Mapping[str, str | None] | None,
    ) -> AOMDatasetMetadata:
        if metadata is None:
            return AOMDatasetMetadata()
        if isinstance(metadata, AOMDatasetMetadata):
            return metadata
        return AOMDatasetMetadata(
            dataset_id=metadata.get("dataset_id"),
            database_name=metadata.get("database_name"),
        )

    @classmethod
    def _validate_xy(cls, X, y) -> tuple[np.ndarray, np.ndarray]:
        X_arr = cls._as_2d_float(X)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        if y_arr.ndim != 1:
            raise ValueError("y must be 1D or single-column 2D")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y have incompatible lengths")
        if X_arr.shape[0] < 2:
            raise ValueError("at least two samples are required")
        return X_arr, y_arr

    @staticmethod
    def _as_2d_float(X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        return X_arr


class AOMStructuralPolicyWithPgt1200Admissions(AOMStructuralPolicy):
    """Structural policy with the train-only high-p admission route enabled."""

    def __init__(self, route_estimators: Mapping[str, object | EstimatorFactory]) -> None:
        super().__init__(route_estimators, enable_pgt1200_admissions=True)


class AOMStructuralPolicyWithP700ProtocolUnified(AOMStructuralPolicy):
    """Structural policy with the source-free p<=700 protocol route enabled."""

    def __init__(
        self,
        route_estimators: Mapping[str, object | EstimatorFactory],
        *,
        p700_protocol_min_features: int | None = None,
    ) -> None:
        super().__init__(
            route_estimators,
            enable_pgt1200_admissions=True,
            enable_p700_protocol_unified=True,
            p700_protocol_min_features=p700_protocol_min_features,
        )


class AOMStructuralPolicyWithP700BlockLocalAdmission(AOMStructuralPolicy):
    """Structural policy with a source-free p<=700 block-local admission route."""

    def __init__(self, route_estimators: Mapping[str, object | EstimatorFactory]) -> None:
        super().__init__(
            route_estimators,
            enable_pgt1200_admissions=True,
            enable_p700_block_local_admission=True,
        )


class AOMControlSelector(BaseEstimator):
    """Train-CV selector over AOM-style chains and Ridge heads.

    This first Phase 7f reference implementation uses only the arrays passed to
    :meth:`fit`, fold-local chain fitting, and a closed-form Ridge head.
    Dataset/source identifiers are not part of the API.
    """

    def __init__(
        self,
        chain_bank: Mapping[str, object],
        *,
        chains: Sequence[str] | None = None,
        alphas: Sequence[float] = (1e-4, 1e-2, 1.0, 100.0),
        pls_components: Sequence[int] = (2, 4, 8, 12),
        heads: Sequence[str] = ("ridge",),
        cv: int = 5,
        repeats: int = 1,
        cv_mode: str = "contiguous",
        criterion_aggregation: str = "mean",
        std_penalty: float = 0.0,
        ensemble_top_k: int = 1,
        ensemble_aggregation: str = "mean",
        random_state: int = 2026,
    ) -> None:
        if not chain_bank:
            raise ValueError("chain_bank must not be empty")
        if cv < 2:
            raise ValueError("cv must be >= 2")
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        if ensemble_top_k < 1:
            raise ValueError("ensemble_top_k must be >= 1")
        unknown_heads = set(heads) - {"ridge", "pls"}
        if unknown_heads:
            raise ValueError(f"unknown heads: {sorted(unknown_heads)}")
        if cv_mode not in {"contiguous", "shuffled", "both"}:
            raise ValueError("cv_mode must be contiguous, shuffled, or both")
        if criterion_aggregation not in {"mean", "median", "q75", "q90", "max"}:
            raise ValueError("unsupported criterion_aggregation")
        if ensemble_aggregation not in {"mean", "median", "rank_mean"}:
            raise ValueError("unsupported ensemble_aggregation")
        self.chain_bank = dict(chain_bank)
        self.chains = tuple(chains) if chains is not None else None
        self.alphas = tuple(float(alpha) for alpha in alphas)
        self.pls_components = tuple(int(n) for n in pls_components)
        if any(n < 1 for n in self.pls_components):
            raise ValueError("pls_components must all be >= 1")
        self.heads = tuple(heads)
        self.cv = int(cv)
        self.repeats = int(repeats)
        self.cv_mode = cv_mode
        self.criterion_aggregation = criterion_aggregation
        self.std_penalty = float(std_penalty)
        self.ensemble_top_k = int(ensemble_top_k)
        self.ensemble_aggregation = ensemble_aggregation
        self.random_state = int(random_state)

    def fit(self, X, y) -> "AOMControlSelector":
        X_arr, y_arr = AOMStructuralPolicy._validate_xy(X, y)
        chain_bank = self._selected_chain_bank()
        specs = self._candidate_specs(chain_bank)
        selection = self._select(X_arr, y_arr, chain_bank, specs)
        self._fitted_candidates_ = [
            self._fit_candidate(X_arr, y_arr, chain_bank, spec)
            for spec in selection["ensemble_specs"]
        ]
        self.selection_report_ = {
            "selected": selection["spec"].name,
            "ensemble": [spec.name for spec in selection["ensemble_specs"]],
            "ranked": selection["ranked_names"],
            "criteria": selection["criteria"],
            "mean_scores": selection["mean_scores"],
            "std_scores": selection["std_scores"],
            "cv": self.cv,
            "repeats": self.repeats,
            "cv_mode": self.cv_mode,
            "criterion_aggregation": self.criterion_aggregation,
            "std_penalty": self.std_penalty,
            "heads": self.heads,
            "alphas": self.alphas,
            "pls_components": self.pls_components,
            "ensemble_top_k": self.ensemble_top_k,
            "ensemble_aggregation": self.ensemble_aggregation,
            "metadata_used_for_routing": False,
        }
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "_fitted_candidates_"):
            raise RuntimeError("AOMControlSelector.predict called before fit")
        X_arr = AOMStructuralPolicy._as_2d_float(X)
        preds = []
        for fitted in self._fitted_candidates_:
            Xt = _transform_chain(fitted.chain, X_arr)
            preds.append(np.asarray(fitted.head.predict(Xt), dtype=float).ravel())
        stacked = np.vstack(preds)
        if self.ensemble_aggregation == "mean":
            return np.mean(stacked, axis=0)
        if self.ensemble_aggregation == "median":
            return np.median(stacked, axis=0)
        weights = 1.0 / np.arange(1, stacked.shape[0] + 1, dtype=float)
        weights = weights / weights.sum()
        return np.average(stacked, axis=0, weights=weights)

    def _selected_chain_bank(self) -> dict[str, object]:
        if self.chains is None:
            return dict(self.chain_bank)
        missing = [name for name in self.chains if name not in self.chain_bank]
        if missing:
            raise ValueError(f"unknown chains: {missing}")
        return {name: self.chain_bank[name] for name in self.chains}

    def _candidate_specs(self, chain_bank: Mapping[str, object]) -> list[AOMCandidateSpec]:
        specs = []
        for chain_name in chain_bank:
            if "ridge" in self.heads:
                specs.extend(AOMCandidateSpec(chain_name, "ridge", alpha) for alpha in self.alphas)
            if "pls" in self.heads:
                specs.extend(
                    AOMCandidateSpec(chain_name, "pls", float(n_components))
                    for n_components in self.pls_components
                )
        if not specs:
            raise ValueError("candidate pool is empty")
        return specs

    def _splitters(self, n_samples: int) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        out = []
        if self.cv_mode in {"contiguous", "both"}:
            out.append(self._fold_indices(n_samples, shuffle=False, seed=self.random_state))
        if self.cv_mode in {"shuffled", "both"}:
            for repeat_id in range(self.repeats):
                out.append(
                    self._fold_indices(
                        n_samples,
                        shuffle=True,
                        seed=self.random_state + repeat_id,
                    )
                )
        return out

    def _fold_indices(
        self,
        n_samples: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        n_splits = min(self.cv, int(n_samples))
        if n_splits < 2:
            raise ValueError("at least two CV folds are required")
        indices = np.arange(n_samples)
        if shuffle:
            indices = np.random.default_rng(seed).permutation(indices)
        folds = [np.sort(fold) for fold in np.array_split(indices, n_splits) if fold.size]
        out = []
        for val_idx in folds:
            mask = np.ones(n_samples, dtype=bool)
            mask[val_idx] = False
            out.append((np.nonzero(mask)[0], val_idx))
        return out

    def _select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        chain_bank: Mapping[str, object],
        specs: list[AOMCandidateSpec],
    ) -> dict[str, object]:
        scores = {spec.name: [] for spec in specs}
        spec_by_name = {spec.name: spec for spec in specs}
        for folds in self._splitters(X.shape[0]):
            for train_idx, val_idx in folds:
                transformed = {}
                for chain_name, chain in chain_bank.items():
                    fitted_chain, X_train_t = _fit_transform_chain(chain, X[train_idx], y[train_idx])
                    X_val_t = _transform_chain(fitted_chain, X[val_idx])
                    transformed[chain_name] = (X_train_t, X_val_t)
                for spec in specs:
                    X_train_t, X_val_t = transformed[spec.chain_name]
                    try:
                        head = self._fit_head(spec, X_train_t, y[train_idx])
                        scores[spec.name].append(_rmse(y[val_idx], head.predict(X_val_t)))
                    except Exception:
                        scores[spec.name].append(float("inf"))
        mean_scores = {name: float(np.mean(values)) for name, values in scores.items()}
        std_scores = {
            name: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            for name, values in scores.items()
        }
        aggregate_scores = {name: self._aggregate(values) for name, values in scores.items()}
        criteria = {
            name: aggregate_scores[name] + self.std_penalty * std_scores[name]
            for name in scores
        }
        ranked_names = [name for name, _value in sorted(criteria.items(), key=lambda item: item[1])]
        top_names = ranked_names[: max(1, min(self.ensemble_top_k, len(ranked_names)))]
        return {
            "spec": spec_by_name[top_names[0]],
            "ensemble_specs": [spec_by_name[name] for name in top_names],
            "criteria": criteria,
            "mean_scores": mean_scores,
            "std_scores": std_scores,
            "ranked_names": ranked_names,
        }

    def _aggregate(self, values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if self.criterion_aggregation == "mean":
            return float(np.mean(arr))
        if self.criterion_aggregation == "median":
            return float(np.median(arr))
        if self.criterion_aggregation == "q75":
            return float(np.quantile(arr, 0.75))
        if self.criterion_aggregation == "q90":
            return float(np.quantile(arr, 0.90))
        return float(np.max(arr))

    def _fit_candidate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        chain_bank: Mapping[str, object],
        spec: AOMCandidateSpec,
    ) -> _FittedAOMCandidate:
        fitted_chain, Xt = _fit_transform_chain(chain_bank[spec.chain_name], X, y)
        head = self._fit_head(spec, Xt, y)
        return _FittedAOMCandidate(spec=spec, chain=fitted_chain, head=head)

    @staticmethod
    def _fit_head(spec: AOMCandidateSpec, X: np.ndarray, y: np.ndarray) -> object:
        if spec.head == "ridge":
            return _RidgeHead(alpha=float(spec.param)).fit(X, y)
        if spec.head == "pls":
            return _PLS1Head(n_components=int(spec.param)).fit(X, y)
        raise ValueError(f"unsupported head: {spec.head}")


class AOMRobustHPORegressor(BaseEstimator):
    """Configurable end-user AOM robust-HPO regressor.

    This estimator builds a source-free spectral preprocessing bank, evaluates
    Ridge and/or PLS heads with fold-local CV, then refits the selected
    candidate(s) on the full training data. It is the product-facing wrapper
    around :class:`AOMControlSelector`.
    """

    def __init__(
        self,
        *,
        profile: str = "robust",
        chain_bank: Mapping[str, object] | None = None,
        extra_chains: Mapping[str, object] | None = None,
        chains: Sequence[str] | None = None,
        heads: Sequence[str] = ("ridge", "pls"),
        alphas: Sequence[float] = (1e-4, 1e-2, 1.0, 100.0),
        pls_components: Sequence[int] = (2, 4, 8, 12),
        cv: int = 5,
        repeats: int = 1,
        cv_mode: str = "contiguous",
        criterion_aggregation: str = "mean",
        std_penalty: float = 0.0,
        ensemble_top_k: int = 1,
        ensemble_aggregation: str = "mean",
        random_state: int = 2026,
    ) -> None:
        self.profile = str(profile)
        self.chain_bank = dict(chain_bank) if chain_bank is not None else None
        self.extra_chains = dict(extra_chains) if extra_chains is not None else None
        self.chains = tuple(chains) if chains is not None else None
        self.heads = tuple(heads)
        self.alphas = tuple(float(alpha) for alpha in alphas)
        self.pls_components = tuple(int(n) for n in pls_components)
        self.cv = int(cv)
        self.repeats = int(repeats)
        self.cv_mode = cv_mode
        self.criterion_aggregation = criterion_aggregation
        self.std_penalty = float(std_penalty)
        self.ensemble_top_k = int(ensemble_top_k)
        self.ensemble_aggregation = ensemble_aggregation
        self.random_state = int(random_state)

    def fit(
        self,
        X,
        y,
        *,
        metadata: AOMDatasetMetadata | Mapping[str, str | None] | None = None,
    ) -> "AOMRobustHPORegressor":
        X_arr, y_arr = AOMStructuralPolicy._validate_xy(X, y)
        meta = AOMStructuralPolicy._coerce_metadata(metadata)
        chain_bank = (
            dict(self.chain_bank)
            if self.chain_bank is not None
            else build_aom_control_chain_bank(
                self.profile,
                extra_chains=self.extra_chains,
            )
        )
        selector = AOMControlSelector(
            chain_bank,
            chains=self.chains,
            alphas=self.alphas,
            pls_components=self.pls_components,
            heads=self.heads,
            cv=self.cv,
            repeats=self.repeats,
            cv_mode=self.cv_mode,
            criterion_aggregation=self.criterion_aggregation,
            std_penalty=self.std_penalty,
            ensemble_top_k=self.ensemble_top_k,
            ensemble_aggregation=self.ensemble_aggregation,
            random_state=self.random_state,
        ).fit(X_arr, y_arr)
        self.selector_ = selector
        self.chain_bank_ = chain_bank
        self.chain_names_ = tuple(chain_bank)
        self.n_features_in_ = int(X_arr.shape[1])
        self.selection_report_ = dict(selector.selection_report_)
        self.selection_report_.update({
            "profile": self.profile,
            "n_chains": len(chain_bank),
            "chain_names": list(self.chain_names_),
            "dataset_id": meta.dataset_id,
            "database_name": meta.database_name,
            "metadata_used_for_routing": False,
        })
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "selector_"):
            raise RuntimeError("AOMRobustHPORegressor.predict called before fit")
        return self.selector_.predict(X)


class AOMRobustHPOCompact(AOMRobustHPORegressor):
    """Fast end-user preset over the compact AOM control bank."""

    def __init__(
        self,
        *,
        cv: int = 5,
        random_state: int = 2026,
    ) -> None:
        super().__init__(
            profile="compact",
            heads=("ridge", "pls"),
            alphas=(1e-4, 1e-2, 1.0, 100.0),
            pls_components=(2, 4, 8),
            cv=cv,
            cv_mode="contiguous",
            ensemble_top_k=1,
            random_state=random_state,
        )


class AOMRobustHPOWide(AOMRobustHPORegressor):
    """Larger lab-style preset for reproducing broad preprocessing diversity."""

    def __init__(
        self,
        *,
        cv: int = 5,
        repeats: int = 1,
        ensemble_top_k: int = 3,
        random_state: int = 2026,
    ) -> None:
        super().__init__(
            profile="wide",
            heads=("ridge", "pls"),
            alphas=(1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0),
            pls_components=(2, 4, 8, 12, 16),
            cv=cv,
            repeats=repeats,
            cv_mode="both",
            criterion_aggregation="mean",
            std_penalty=0.05,
            ensemble_top_k=ensemble_top_k,
            ensemble_aggregation="rank_mean",
            random_state=random_state,
        )


class _ChainRidgeCandidate(BaseEstimator):
    """One fold-local preprocessing chain followed by a dependency-light Ridge head."""

    def __init__(self, chain: object, alpha: float) -> None:
        self.chain = chain
        self.alpha = float(alpha)

    def fit(self, X, y) -> "_ChainRidgeCandidate":
        X_arr = _as_feature_matrix(X)
        y_arr = _as_target_array(y)
        if y_arr.ndim != 1:
            raise ValueError("default AOMRidgeBlender Ridge candidates require 1D y")
        chain, Xt = _fit_transform_chain(self.chain, X_arr, y_arr)
        head = _RidgeHead(self.alpha).fit(Xt, y_arr)
        self.chain_ = chain
        self.head_ = head
        self.n_features_in_ = int(X_arr.shape[1])
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "head_"):
            raise RuntimeError("_ChainRidgeCandidate.predict called before fit")
        Xt = _transform_chain(self.chain_, _as_feature_matrix(X))
        return np.asarray(self.head_.predict(Xt), dtype=float).ravel()


def _project_simplex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("simplex projection needs at least one value")
    sorted_values = np.sort(values)[::-1]
    cssv = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, values.size + 1, dtype=float)
    active = sorted_values - cssv / indices > 0.0
    if not np.any(active):
        return np.full(values.size, 1.0 / values.size, dtype=float)
    rho = int(np.nonzero(active)[0][-1])
    theta = float(cssv[rho] / (rho + 1.0))
    projected = np.maximum(values - theta, 0.0)
    total = float(projected.sum())
    if total <= 0.0:
        return np.full(values.size, 1.0 / values.size, dtype=float)
    return projected / total


def _solve_simplex_qp(Z: np.ndarray, y: np.ndarray, regularizer: float) -> np.ndarray:
    """Solve the regularized non-negative simplex QP used by AOMRidgeBlender."""

    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D prediction matrix")
    if Z.shape[0] != y.shape[0]:
        raise ValueError("Z and y have incompatible lengths")
    k = int(Z.shape[1])
    if k < 1:
        raise ValueError("Z must have at least one candidate column")
    uniform = np.full(k, 1.0 / k, dtype=float)
    lam = float(max(regularizer, 0.0))
    ZtZ = Z.T @ Z
    Zty = Z.T @ y

    try:  # Prefer the exact SLSQP path when scipy is installed.
        from scipy.optimize import minimize  # type: ignore

        def objective(w: np.ndarray) -> float:
            residual = Z @ w - y
            data_term = 0.5 * float(residual @ residual)
            reg_term = 0.5 * lam * float(np.sum((w - uniform) ** 2))
            return data_term + reg_term

        def gradient(w: np.ndarray) -> np.ndarray:
            return ZtZ @ w - Zty + lam * (w - uniform)

        constraints = ({
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones_like(w),
        },)
        result = minimize(
            objective,
            x0=uniform,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10, "disp": False},
        )
        if result.success and np.all(np.isfinite(result.x)):
            return _project_simplex(np.asarray(result.x, dtype=float))
    except Exception:
        pass

    # Dependency-light fallback: projected gradient on the simplex.
    lipschitz = float(np.linalg.norm(ZtZ, ord=2) + lam)
    step = 1.0 / max(lipschitz, 1e-12)
    weights = uniform.copy()
    for _iteration in range(2000):
        grad = ZtZ @ weights - Zty + lam * (weights - uniform)
        updated = _project_simplex(weights - step * grad)
        if float(np.linalg.norm(updated - weights, ord=1)) <= 1e-12:
            weights = updated
            break
        weights = updated
    return weights


class AOMRidgeBlender(BaseEstimator):
    """Fold-safe simplex blender for AOM-Ridge preprocessing candidates.

    Each candidate is evaluated with outer-CV out-of-fold predictions. The OOF
    columns are then blended with non-negative weights that sum to one, using
    a small optional shrinkage toward the uniform average. Dataset metadata is
    audit-only and never participates in candidate construction or weighting.
    """

    def __init__(
        self,
        candidates: (
            Mapping[str, object | EstimatorFactory]
            | Sequence[tuple[str, object | EstimatorFactory]]
            | Sequence[object | EstimatorFactory]
            | None
        ) = None,
        *,
        profile: str = "compact",
        chain_bank: Mapping[str, object] | None = None,
        extra_chains: Mapping[str, object] | None = None,
        chains: Sequence[str] | None = None,
        alphas: Sequence[float] = (1e-4, 1e-2, 1.0, 100.0),
        cv: int = 5,
        shuffle: bool = True,
        regularizer: float = 0.01,
        random_state: int = 2026,
        n_jobs: int = 1,
        drop_failed_candidates: bool = True,
    ) -> None:
        if cv < 2:
            raise ValueError("cv must be >= 2")
        self.candidates = candidates
        self.profile = str(profile)
        self.chain_bank = dict(chain_bank) if chain_bank is not None else None
        self.extra_chains = dict(extra_chains) if extra_chains is not None else None
        self.chains = tuple(chains) if chains is not None else None
        self.alphas = tuple(float(alpha) for alpha in alphas)
        if not self.alphas:
            raise ValueError("alphas must not be empty")
        self.cv = int(cv)
        self.shuffle = bool(shuffle)
        self.regularizer = float(regularizer)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.drop_failed_candidates = bool(drop_failed_candidates)

    def fit(
        self,
        X,
        y,
        *,
        metadata: AOMDatasetMetadata | Mapping[str, str | None] | None = None,
    ) -> "AOMRidgeBlender":
        X_arr = AOMStructuralPolicy._as_2d_float(X)
        y_arr = _as_target_array(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y have incompatible lengths")
        if X_arr.shape[0] < 2:
            raise ValueError("at least two samples are required")
        meta = AOMStructuralPolicy._coerce_metadata(metadata)
        labels, candidate_likes, default_chain_names = self._resolved_candidates()
        folds = _fold_indices(
            X_arr.shape[0],
            cv=self.cv,
            shuffle=self.shuffle,
            seed=self.random_state,
        )
        oof_results = self._candidate_oof_predictions(
            labels,
            candidate_likes,
            X_arr,
            y_arr,
            folds,
        )
        successful = [(label, candidate, oof) for label, candidate, oof, _err in oof_results if oof is not None]
        failed = [(label, str(err)) for label, _candidate, oof, err in oof_results if oof is None]
        if failed and not self.drop_failed_candidates:
            label, err = failed[0]
            raise RuntimeError(f"AOMRidgeBlender candidate {label!r} failed: {err}")
        if not successful:
            raise ValueError("all AOMRidgeBlender candidates failed during OOF scoring")

        labels = [label for label, _candidate, _oof in successful]
        candidate_likes = [candidate for _label, candidate, _oof in successful]
        oof_list = [np.asarray(oof, dtype=float) for _label, _candidate, oof in successful]
        if y_arr.ndim == 1:
            Z = np.column_stack(oof_list)
            Z_flat = Z
            y_flat = y_arr.reshape(-1)
        else:
            Z = np.stack(oof_list, axis=-1)
            Z_flat = Z.reshape(-1, Z.shape[-1])
            y_flat = y_arr.reshape(-1)
        weights = _solve_simplex_qp(Z_flat, y_flat, self.regularizer)
        cv_scores = {
            label: _rmse(y_flat, Z_flat[:, idx])
            for idx, label in enumerate(labels)
        }

        estimators = []
        for label, candidate_like in zip(labels, candidate_likes):
            estimator = _make_estimator(candidate_like)
            self._check_estimator(estimator, label)
            estimator.fit(X_arr, y_arr)
            estimators.append(estimator)

        best_idx = int(np.argmax(weights))
        self.candidate_labels_ = tuple(labels)
        self.estimators_ = tuple(estimators)
        self.weights_ = np.asarray(weights, dtype=float)
        self.oof_predictions_ = Z
        self.cv_scores_ = cv_scores
        self.failed_candidates_ = tuple(failed)
        self.selected_variant_index_ = best_idx
        self.selected_variant_label_ = labels[best_idx]
        self.n_features_in_ = int(X_arr.shape[1])
        self.n_targets_ = 1 if y_arr.ndim == 1 else int(y_arr.shape[1])
        self.blend_report_ = {
            "model": "AOMRidgeBlender",
            "selected_variant_label": self.selected_variant_label_,
            "selected_variant_index": self.selected_variant_index_,
            "candidate_labels": list(self.candidate_labels_),
            "weights": [float(weight) for weight in self.weights_],
            "cv_scores": self.cv_scores_,
            "failed_candidates": [
                {"label": label, "error": error}
                for label, error in self.failed_candidates_
            ],
            "cv": self.cv,
            "shuffle": self.shuffle,
            "regularizer": self.regularizer,
            "random_state": self.random_state,
            "profile": self.profile,
            "chain_names": list(default_chain_names),
            "alphas": list(self.alphas),
            "dataset_id": meta.dataset_id,
            "database_name": meta.database_name,
            "metadata_used_for_routing": False,
        }
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("AOMRidgeBlender.predict called before fit")
        X_arr = AOMStructuralPolicy._as_2d_float(X)
        preds = [
            _match_prediction_shape(estimator.predict(X_arr), self._prediction_reference(X_arr.shape[0]))
            for estimator in self.estimators_
        ]
        if self.n_targets_ == 1:
            stack = np.column_stack([pred.reshape(-1) for pred in preds])
            return stack @ self.weights_
        stack = np.stack(preds, axis=-1)
        return stack @ self.weights_

    def score(self, X, y) -> float:
        y_true = _as_target_array(y)
        y_pred = _match_prediction_shape(self.predict(X), y_true)
        residual = y_true.reshape(-1) - y_pred.reshape(-1)
        centered = y_true.reshape(-1) - float(np.mean(y_true))
        denom = float(centered @ centered)
        if denom <= 1e-12:
            return 0.0
        return float(1.0 - (residual @ residual) / denom)

    def get_diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "blend_report_"):
            raise RuntimeError("AOMRidgeBlender.get_diagnostics called before fit")
        ranking = sorted(
            zip(self.candidate_labels_, self.weights_, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        report = dict(self.blend_report_)
        report["weight_ranking"] = [
            {"label": str(label), "weight": float(weight)}
            for label, weight in ranking
        ]
        return report

    def _prediction_reference(self, n_samples: int) -> np.ndarray:
        if int(self.n_targets_) == 1:
            return np.empty(int(n_samples), dtype=float)
        return np.empty((int(n_samples), int(self.n_targets_)), dtype=float)

    def _resolved_candidates(self) -> tuple[list[str], list[object | EstimatorFactory], tuple[str, ...]]:
        if self.candidates is not None:
            labels, candidates = self._normalize_explicit_candidates()
            return labels, candidates, ()
        chain_bank = (
            dict(self.chain_bank)
            if self.chain_bank is not None
            else build_aom_control_chain_bank(
                self.profile,
                extra_chains=self.extra_chains,
            )
        )
        if self.chains is not None:
            missing = [name for name in self.chains if name not in chain_bank]
            if missing:
                raise ValueError(f"unknown chains: {missing}")
            chain_bank = {name: chain_bank[name] for name in self.chains}
        labels: list[str] = []
        candidates: list[object] = []
        for chain_name, chain in chain_bank.items():
            for alpha in self.alphas:
                labels.append(f"{chain_name}__ridge_a{alpha:g}")
                candidates.append(_ChainRidgeCandidate(chain, alpha))
        if not candidates:
            raise ValueError("AOMRidgeBlender candidate pool is empty")
        return labels, candidates, tuple(chain_bank)

    def _normalize_explicit_candidates(self) -> tuple[list[str], list[object | EstimatorFactory]]:
        candidates = self.candidates
        if isinstance(candidates, Mapping):
            items = list(candidates.items())
        else:
            items = []
            for idx, item in enumerate(candidates or ()):
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], str)
                ):
                    items.append((item[0], item[1]))
                else:
                    items.append((f"candidate_{idx}", item))
        if not items:
            raise ValueError("candidates must not be empty")
        labels = [str(label) for label, _candidate in items]
        if len(set(labels)) != len(labels):
            raise ValueError("candidate labels must be unique")
        return labels, [candidate for _label, candidate in items]

    def _candidate_oof_predictions(
        self,
        labels: Sequence[str],
        candidate_likes: Sequence[object | EstimatorFactory],
        X: np.ndarray,
        y: np.ndarray,
        folds: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> list[tuple[str, object | EstimatorFactory, np.ndarray | None, Exception | None]]:
        if abs(int(self.n_jobs)) == 1 or len(candidate_likes) <= 1:
            return [
                self._candidate_oof_prediction(label, candidate, X, y, folds)
                for label, candidate in zip(labels, candidate_likes)
            ]
        try:
            from joblib import Parallel, delayed  # type: ignore

            return Parallel(n_jobs=int(self.n_jobs), backend="loky")(
                delayed(self._candidate_oof_prediction)(label, candidate, X, y, folds)
                for label, candidate in zip(labels, candidate_likes)
            )
        except Exception:
            return [
                self._candidate_oof_prediction(label, candidate, X, y, folds)
                for label, candidate in zip(labels, candidate_likes)
            ]

    def _candidate_oof_prediction(
        self,
        label: str,
        candidate_like: object | EstimatorFactory,
        X: np.ndarray,
        y: np.ndarray,
        folds: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[str, object | EstimatorFactory, np.ndarray | None, Exception | None]:
        oof = np.full(y.shape, np.nan, dtype=float)
        try:
            for train_idx, val_idx in folds:
                estimator = _make_estimator(candidate_like)
                self._check_estimator(estimator, label)
                estimator.fit(X[train_idx], y[train_idx])
                pred = estimator.predict(X[val_idx])
                oof[val_idx] = _match_prediction_shape(pred, y[val_idx])
            if np.isnan(oof).any():
                raise ValueError("OOF splitter did not cover every sample")
            return label, candidate_like, oof, None
        except Exception as exc:
            return label, candidate_like, None, exc

    @staticmethod
    def _check_estimator(estimator: object, label: str) -> None:
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError(f"candidate {label!r} must expose fit and predict")


@dataclass(frozen=True)
class AOMOperatorPLSSpec:
    """One operator-PLS score stack configuration."""

    n_components: int
    alpha: float

    @property
    def name(self) -> str:
        return f"op_pls_c{self.n_components}_ridge_a{self.alpha:g}"


class _ColumnStandardizer:
    def fit(self, X: np.ndarray) -> "_ColumnStandardizer":
        X = _as_feature_matrix(X)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("_ColumnStandardizer.transform called before fit")
        X = _as_feature_matrix(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class _PLS1ScoreProjector:
    def __init__(self, n_components: int) -> None:
        self.n_components = int(n_components)
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_PLS1ScoreProjector":
        X = _as_feature_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible lengths")
        self.x_mean_ = X.mean(axis=0)
        y_mean = float(y.mean())
        Xr = X - self.x_mean_
        yr = y - y_mean
        max_components = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        weights = []
        loadings = []
        for _component in range(max_components):
            w = Xr.T @ yr
            w_norm = float(np.linalg.norm(w))
            if w_norm <= 1e-12:
                break
            w = w / w_norm
            t = Xr @ w
            denom = float(t @ t)
            if denom <= 1e-12:
                break
            p = (Xr.T @ t) / denom
            q = float((yr @ t) / denom)
            Xr = Xr - np.outer(t, p)
            yr = yr - q * t
            weights.append(w)
            loadings.append(p)
        if not weights:
            self.rotations_ = np.zeros((X.shape[1], 1), dtype=float)
            self.n_components_ = 1
            return self
        W = np.column_stack(weights)
        P = np.column_stack(loadings)
        self.rotations_ = np.asarray(W @ np.linalg.pinv(P.T @ W), dtype=float)
        self.n_components_ = int(self.rotations_.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "rotations_"):
            raise RuntimeError("_PLS1ScoreProjector.transform called before fit")
        X = _as_feature_matrix(X)
        return (X - self.x_mean_) @ self.rotations_


@dataclass
class _FittedOperatorPLSStack:
    spec: AOMOperatorPLSSpec
    matrices: tuple[np.ndarray, ...]
    operator_names: tuple[str, ...]
    scalers: tuple[_ColumnStandardizer, ...]
    projectors: tuple[_PLS1ScoreProjector, ...]
    ridge: _RidgeHead

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _as_feature_matrix(X)
        features = []
        for matrix, scaler, projector in zip(
            self.matrices,
            self.scalers,
            self.projectors,
            strict=True,
        ):
            X_op = X @ matrix
            X_scaled = scaler.transform(X_op)
            features.append(projector.transform(X_scaled))
        return np.hstack(features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ridge.predict(self.transform(X))


def _operator_matrix_from_value(p: int, value: object, label: str) -> np.ndarray:
    if callable(value) and not hasattr(value, "fit") and not hasattr(value, "transform"):
        value = value(int(p))
    if isinstance(value, np.ndarray):
        raw = np.asarray(value, dtype=float)
    elif hasattr(value, "fit_transform") or hasattr(value, "transform"):
        raw = _matrix_from_transformer(int(p), value)
    else:
        raw = np.asarray(value, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"operator {label!r} must resolve to a 2D matrix")
    if raw.shape[0] == int(p):
        matrix = raw
    elif raw.shape[1] == int(p):
        matrix = raw.T
    else:
        raise ValueError(
            f"operator {label!r} matrix must have one dimension equal to n_features"
        )
    if matrix.shape[1] < 1:
        raise ValueError(f"operator {label!r} has no output columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"operator {label!r} matrix contains NaN or Inf")
    return np.ascontiguousarray(matrix, dtype=float)


def _matrix_from_transformer(p: int, transformer_like: object) -> np.ndarray:
    identity = np.eye(int(p), dtype=float)
    transformer = deepcopy(transformer_like)
    if hasattr(transformer, "fit_transform"):
        try:
            return _as_feature_matrix(transformer.fit_transform(identity))
        except TypeError:
            return _as_feature_matrix(transformer.fit_transform(identity, None))
    if hasattr(transformer, "fit"):
        try:
            transformer.fit(identity)
        except TypeError:
            transformer.fit(identity, None)
    if not hasattr(transformer, "transform"):
        raise TypeError("operator transformer must expose transform or fit_transform")
    return _as_feature_matrix(transformer.transform(identity))


def _default_operator_pls_bank(p: int) -> dict[str, np.ndarray]:
    from .baseline import Detrend
    from .preprocessing import Derivate, Gaussian, NorrisWilliams, SavitzkyGolay

    candidates: list[tuple[str, object]] = [
        ("raw", _IdentityTransform()),
        ("detrend1", Detrend(polyorder=1)),
        ("detrend2", Detrend(polyorder=2)),
        ("finite_diff1", Derivate(order=1)),
        ("finite_diff2", Derivate(order=2)),
        ("gaussian_s1", Gaussian(sigma=1.0)),
        ("gaussian_s2", Gaussian(sigma=2.0)),
        ("savgol_w5_p2_d0", SavitzkyGolay(5, 2, 0)),
        ("savgol_w7_p2_d1", SavitzkyGolay(7, 2, 1)),
        ("savgol_w11_p2_d2", SavitzkyGolay(11, 2, 2)),
        ("nw_s5_g5_d1", NorrisWilliams(5, 5, 1)),
    ]
    out: dict[str, np.ndarray] = {}
    for name, operator in candidates:
        try:
            out[name] = _operator_matrix_from_value(int(p), operator, name)
        except Exception:
            continue
    if not out:
        out["raw"] = np.eye(int(p), dtype=float)
    return out


class AOMOperatorPLSStack(BaseEstimator):
    """Guarded per-operator PLS score stack with a Ridge head.

    This reference estimator reproduces the deployable-compatible
    ``operator_pls_stack`` experiment: build fixed strict-linear operator views,
    fit a PLS1 score projector per view, concatenate the scores, and fit a
    Ridge head. Selection uses train-only CV. If ``baseline_estimator`` is
    supplied, the operator stack is admitted only when its OOF RMSE improves
    the baseline by ``min_relative_oof_gain``.
    """

    def __init__(
        self,
        operator_bank: Mapping[str, object] | None = None,
        *,
        components: Sequence[int] = (2, 4, 8),
        alphas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
        cv: int = 5,
        cv_mode: str = "shuffled",
        std_penalty: float = 0.0,
        gap_penalty: float = 0.0,
        baseline_estimator: object | EstimatorFactory | None = None,
        min_relative_oof_gain: float = 0.0,
        random_state: int = 2026,
        drop_failed_specs: bool = True,
    ) -> None:
        if cv < 2:
            raise ValueError("cv must be >= 2")
        if cv_mode not in {"contiguous", "shuffled", "both"}:
            raise ValueError("cv_mode must be contiguous, shuffled, or both")
        self.operator_bank = dict(operator_bank) if operator_bank is not None else None
        self.components = tuple(int(component) for component in components)
        if not self.components or any(component < 1 for component in self.components):
            raise ValueError("components must contain positive integers")
        self.alphas = tuple(float(alpha) for alpha in alphas)
        if not self.alphas:
            raise ValueError("alphas must not be empty")
        self.cv = int(cv)
        self.cv_mode = cv_mode
        self.std_penalty = float(std_penalty)
        self.gap_penalty = float(gap_penalty)
        self.baseline_estimator = baseline_estimator
        self.min_relative_oof_gain = float(min_relative_oof_gain)
        self.random_state = int(random_state)
        self.drop_failed_specs = bool(drop_failed_specs)

    def fit(
        self,
        X,
        y,
        *,
        metadata: AOMDatasetMetadata | Mapping[str, str | None] | None = None,
    ) -> "AOMOperatorPLSStack":
        X_arr, y_arr = AOMStructuralPolicy._validate_xy(X, y)
        meta = AOMStructuralPolicy._coerce_metadata(metadata)
        matrices, operator_names = self._resolved_operator_matrices(X_arr.shape[1])
        specs = [
            AOMOperatorPLSSpec(component, alpha)
            for component in self.components
            for alpha in self.alphas
        ]
        selected_spec, score_map = self._select_spec(
            X_arr,
            y_arr,
            matrices,
            operator_names,
            specs,
        )
        operator_stack = self._fit_stack(X_arr, y_arr, matrices, operator_names, selected_spec)
        selected_score = score_map[selected_spec.name]
        accepted = True
        baseline_rmse = float("nan")
        relative_gain = float("nan")
        baseline = None
        if self.baseline_estimator is not None:
            baseline_rmse = self._score_baseline(X_arr, y_arr)
            relative_gain = (
                baseline_rmse - float(selected_score["mean_rmse"])
            ) / max(baseline_rmse, 1e-12)
            accepted = bool(relative_gain >= self.min_relative_oof_gain)
            baseline = _make_estimator(self.baseline_estimator)
            if not hasattr(baseline, "fit") or not hasattr(baseline, "predict"):
                raise TypeError("baseline_estimator must expose fit and predict")
            baseline.fit(X_arr, y_arr)

        self.operator_stack_ = operator_stack
        self.baseline_estimator_ = baseline
        self.accepted_operator_stack_ = bool(accepted)
        self.selected_spec_ = selected_spec
        self.selected_variant_label_ = selected_spec.name
        self.operator_names_ = tuple(operator_names)
        self.operator_matrices_ = tuple(np.asarray(matrix, dtype=float) for matrix in matrices)
        self.cv_scores_ = score_map
        self.n_features_in_ = int(X_arr.shape[1])
        self.n_operator_features_ = int(operator_stack.transform(X_arr[:1]).shape[1])
        self.stack_report_ = {
            "model": "AOMOperatorPLSStack",
            "selected_spec": selected_spec.name,
            "selected_components": int(selected_spec.n_components),
            "selected_alpha": float(selected_spec.alpha),
            "accepted_operator_stack": self.accepted_operator_stack_,
            "baseline_oof_rmse": float(baseline_rmse),
            "operator_oof_rmse": float(selected_score["mean_rmse"]),
            "relative_oof_gain": float(relative_gain),
            "min_relative_oof_gain": self.min_relative_oof_gain,
            "cv_scores": score_map,
            "operator_names": list(self.operator_names_),
            "n_operator_features": self.n_operator_features_,
            "cv": self.cv,
            "cv_mode": self.cv_mode,
            "std_penalty": self.std_penalty,
            "gap_penalty": self.gap_penalty,
            "random_state": self.random_state,
            "dataset_id": meta.dataset_id,
            "database_name": meta.database_name,
            "metadata_used_for_routing": False,
        }
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "operator_stack_"):
            raise RuntimeError("AOMOperatorPLSStack.predict called before fit")
        X_arr = AOMStructuralPolicy._as_2d_float(X)
        if not self.accepted_operator_stack_ and self.baseline_estimator_ is not None:
            return np.asarray(self.baseline_estimator_.predict(X_arr), dtype=float).ravel()
        return np.asarray(self.operator_stack_.predict(X_arr), dtype=float).ravel()

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "operator_stack_"):
            raise RuntimeError("AOMOperatorPLSStack.transform called before fit")
        return self.operator_stack_.transform(AOMStructuralPolicy._as_2d_float(X))

    def get_diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "stack_report_"):
            raise RuntimeError("AOMOperatorPLSStack.get_diagnostics called before fit")
        return dict(self.stack_report_)

    def _resolved_operator_matrices(self, n_features: int) -> tuple[list[np.ndarray], list[str]]:
        bank = (
            _default_operator_pls_bank(int(n_features))
            if self.operator_bank is None
            else dict(self.operator_bank)
        )
        if not bank:
            raise ValueError("operator_bank must not be empty")
        matrices = []
        names = []
        for name, value in bank.items():
            matrix = _operator_matrix_from_value(int(n_features), value, str(name))
            matrices.append(matrix)
            names.append(str(name))
        return matrices, names

    def _splitters(self, n_samples: int) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        out = []
        if self.cv_mode in {"contiguous", "both"}:
            out.append(
                _fold_indices(
                    n_samples,
                    cv=self.cv,
                    shuffle=False,
                    seed=self.random_state,
                )
            )
        if self.cv_mode in {"shuffled", "both"}:
            out.append(
                _fold_indices(
                    n_samples,
                    cv=self.cv,
                    shuffle=True,
                    seed=self.random_state,
                )
            )
        return out

    def _select_spec(
        self,
        X: np.ndarray,
        y: np.ndarray,
        matrices: Sequence[np.ndarray],
        operator_names: Sequence[str],
        specs: Sequence[AOMOperatorPLSSpec],
    ) -> tuple[AOMOperatorPLSSpec, dict[str, dict[str, float | list[float]]]]:
        score_map: dict[str, dict[str, float | list[float]]] = {}
        best_spec = None
        best_criterion = float("inf")
        failures: dict[str, str] = {}
        for spec in specs:
            try:
                scores = self._score_spec(X, y, matrices, operator_names, spec)
            except Exception as exc:
                if not self.drop_failed_specs:
                    raise
                failures[spec.name] = str(exc)
                continue
            score_map[spec.name] = scores
            criterion = float(scores["criterion"])
            if criterion < best_criterion:
                best_criterion = criterion
                best_spec = spec
        if best_spec is None:
            raise ValueError(f"all operator PLS specs failed: {failures}")
        if failures:
            for name, error in failures.items():
                score_map[name] = {
                    "mean_rmse": float("inf"),
                    "std_rmse": float("inf"),
                    "mean_train_rmse": float("inf"),
                    "gap": float("inf"),
                    "criterion": float("inf"),
                    "fold_scores": [],
                    "error": error,
                }
        return best_spec, score_map

    def _score_spec(
        self,
        X: np.ndarray,
        y: np.ndarray,
        matrices: Sequence[np.ndarray],
        operator_names: Sequence[str],
        spec: AOMOperatorPLSSpec,
    ) -> dict[str, float | list[float]]:
        fold_scores: list[float] = []
        train_scores: list[float] = []
        for folds in self._splitters(X.shape[0]):
            for train_idx, val_idx in folds:
                fitted = self._fit_stack(
                    X[train_idx],
                    y[train_idx],
                    matrices,
                    operator_names,
                    spec,
                )
                pred_train = fitted.predict(X[train_idx])
                pred_val = fitted.predict(X[val_idx])
                train_scores.append(_rmse(y[train_idx], pred_train))
                fold_scores.append(_rmse(y[val_idx], pred_val))
        mean_rmse = float(np.mean(fold_scores))
        std_rmse = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
        mean_train_rmse = float(np.mean(train_scores))
        gap = max(0.0, mean_rmse - mean_train_rmse)
        criterion = mean_rmse + self.std_penalty * std_rmse + self.gap_penalty * gap
        return {
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "mean_train_rmse": mean_train_rmse,
            "gap": float(gap),
            "criterion": float(criterion),
            "fold_scores": [float(value) for value in fold_scores],
        }

    def _fit_stack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        matrices: Sequence[np.ndarray],
        operator_names: Sequence[str],
        spec: AOMOperatorPLSSpec,
    ) -> _FittedOperatorPLSStack:
        scalers = []
        projectors = []
        features = []
        for matrix in matrices:
            X_op = X @ matrix
            scaler = _ColumnStandardizer()
            X_scaled = scaler.fit_transform(X_op)
            n_components = max(
                1,
                min(int(spec.n_components), X_scaled.shape[0] - 1, X_scaled.shape[1]),
            )
            projector = _PLS1ScoreProjector(n_components).fit(X_scaled, y)
            scores = projector.transform(X_scaled)
            scalers.append(scaler)
            projectors.append(projector)
            features.append(scores)
        X_features = np.hstack(features)
        ridge = _RidgeHead(alpha=float(spec.alpha)).fit(X_features, y)
        return _FittedOperatorPLSStack(
            spec=spec,
            matrices=tuple(np.asarray(matrix, dtype=float) for matrix in matrices),
            operator_names=tuple(operator_names),
            scalers=tuple(scalers),
            projectors=tuple(projectors),
            ridge=ridge,
        )

    def _score_baseline(self, X: np.ndarray, y: np.ndarray) -> float:
        losses = []
        for folds in self._splitters(X.shape[0]):
            pred = np.empty(X.shape[0], dtype=float)
            for train_idx, val_idx in folds:
                estimator = _make_estimator(self.baseline_estimator)
                if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
                    raise TypeError("baseline_estimator must expose fit and predict")
                estimator.fit(X[train_idx], y[train_idx])
                pred[val_idx] = np.asarray(estimator.predict(X[val_idx]), dtype=float).ravel()
            losses.append(_rmse(y, pred))
        return float(np.mean(losses))


class AOMTrueBankEndpointPortfolio(BaseEstimator):
    """Guarded train-CV selector over true-bank endpoint estimators.

    This reference object receives already-built endpoint estimators. It does
    not inspect dataset/source metadata, and it does not perform a broad moment
    grind. The selected endpoint is the lowest train-CV criterion
    ``mean_rmse + std_penalty * std_rmse``.
    """

    def __init__(
        self,
        endpoint_estimators: Mapping[str, object | EstimatorFactory],
        *,
        cv: int = 5,
        cv_mode: str = "contiguous",
        std_penalty: float = 1.0,
        random_state: int = 2026,
    ) -> None:
        if not endpoint_estimators:
            raise ValueError("endpoint_estimators must not be empty")
        if cv < 2:
            raise ValueError("cv must be >= 2")
        if cv_mode not in {"contiguous", "shuffled", "both"}:
            raise ValueError("cv_mode must be contiguous, shuffled, or both")
        self.endpoint_estimators = dict(endpoint_estimators)
        self.cv = int(cv)
        self.cv_mode = cv_mode
        self.std_penalty = float(std_penalty)
        self.random_state = int(random_state)

    def fit(
        self,
        X,
        y,
        *,
        metadata: AOMDatasetMetadata | Mapping[str, str | None] | None = None,
    ) -> "AOMTrueBankEndpointPortfolio":
        X_arr, y_arr = AOMStructuralPolicy._validate_xy(X, y)
        meta = AOMStructuralPolicy._coerce_metadata(metadata)
        scores = self._score_endpoints(X_arr, y_arr)
        criteria = {
            name: values["mean_rmse"] + self.std_penalty * values["std_rmse"]
            for name, values in scores.items()
        }
        ranked = [name for name, _value in sorted(criteria.items(), key=lambda item: item[1])]
        selected = ranked[0]
        estimator = _make_estimator(self.endpoint_estimators[selected])
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError("selected endpoint estimator must expose fit and predict")
        estimator.fit(X_arr, y_arr)
        self.selected_endpoint_ = selected
        self.estimator_ = estimator
        self.portfolio_report_ = {
            "selected_endpoint": selected,
            "ranked_endpoints": ranked,
            "criteria": criteria,
            "cv_scores": scores,
            "cv": self.cv,
            "cv_mode": self.cv_mode,
            "std_penalty": self.std_penalty,
            "random_state": self.random_state,
            "dataset_id": meta.dataset_id,
            "database_name": meta.database_name,
            "metadata_used_for_routing": False,
        }
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("AOMTrueBankEndpointPortfolio.predict called before fit")
        X_arr = AOMStructuralPolicy._as_2d_float(X)
        return np.asarray(self.estimator_.predict(X_arr), dtype=float).ravel()

    def _splitters(self, n_samples: int) -> list[tuple[bool, int]]:
        out: list[tuple[bool, int]] = []
        if self.cv_mode in {"contiguous", "both"}:
            out.append((False, self.random_state))
        if self.cv_mode in {"shuffled", "both"}:
            out.append((True, self.random_state))
        return out

    def _score_endpoints(self, X: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float]]:
        out = {}
        for name, estimator_like in self.endpoint_estimators.items():
            losses: list[float] = []
            for shuffle, seed in self._splitters(X.shape[0]):
                try:
                    _pred, fold_losses = _fit_predict_oof(
                        estimator_like,
                        X,
                        y,
                        cv=self.cv,
                        shuffle=shuffle,
                        seed=seed,
                    )
                    losses.extend(fold_losses)
                except Exception:
                    losses.append(float("inf"))
            arr = np.asarray(losses, dtype=float)
            out[name] = {
                "mean_rmse": float(np.mean(arr)),
                "std_rmse": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "n_folds": int(arr.size),
            }
        return out


class AOMMidPEndpointStack(BaseEstimator):
    """Train-OOF pairwise endpoint stack for the mid-p AOM portfolio.

    The implementation is intentionally dependency-light. It evaluates
    endpoint estimators with fold-local OOF predictions, then chooses either the
    best single endpoint or the best two-endpoint convex blend. Metadata is
    audit-only and cannot affect the selected members or weights.
    """

    def __init__(
        self,
        endpoint_estimators: Mapping[str, object | EstimatorFactory],
        *,
        cv: int = 5,
        max_pair_oof_ratio: float = 1.1,
        random_state: int = 2026,
    ) -> None:
        if not endpoint_estimators:
            raise ValueError("endpoint_estimators must not be empty")
        if cv < 2:
            raise ValueError("cv must be >= 2")
        if max_pair_oof_ratio < 1.0:
            raise ValueError("max_pair_oof_ratio must be >= 1")
        self.endpoint_estimators = dict(endpoint_estimators)
        self.cv = int(cv)
        self.max_pair_oof_ratio = float(max_pair_oof_ratio)
        self.random_state = int(random_state)

    def fit(
        self,
        X,
        y,
        *,
        metadata: AOMDatasetMetadata | Mapping[str, str | None] | None = None,
    ) -> "AOMMidPEndpointStack":
        X_arr, y_arr = AOMStructuralPolicy._validate_xy(X, y)
        meta = AOMStructuralPolicy._coerce_metadata(metadata)
        endpoint_oof, endpoint_scores = self._endpoint_oof_predictions(X_arr, y_arr)
        selected = self._select_blend(endpoint_oof, endpoint_scores, y_arr)
        self.selected_members_ = tuple(selected["members"])
        self.weights_ = tuple(float(w) for w in selected["weights"])
        self.estimators_ = {}
        for name in self.selected_members_:
            estimator = _make_estimator(self.endpoint_estimators[name])
            if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
                raise TypeError("selected endpoint estimator must expose fit and predict")
            estimator.fit(X_arr, y_arr)
            self.estimators_[name] = estimator
        self.stack_report_ = {
            "selected_members": list(self.selected_members_),
            "weights": list(self.weights_),
            "oof_rmse": float(selected["oof_rmse"]),
            "endpoint_scores": endpoint_scores,
            "cv": self.cv,
            "max_pair_oof_ratio": self.max_pair_oof_ratio,
            "random_state": self.random_state,
            "dataset_id": meta.dataset_id,
            "database_name": meta.database_name,
            "metadata_used_for_routing": False,
        }
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("AOMMidPEndpointStack.predict called before fit")
        X_arr = AOMStructuralPolicy._as_2d_float(X)
        preds = []
        for name in self.selected_members_:
            preds.append(np.asarray(self.estimators_[name].predict(X_arr), dtype=float).ravel())
        stacked = np.vstack(preds)
        return np.average(stacked, axis=0, weights=np.asarray(self.weights_, dtype=float))

    def _endpoint_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        oofs: dict[str, np.ndarray] = {}
        scores: dict[str, float] = {}
        for name, estimator_like in self.endpoint_estimators.items():
            try:
                pred, _losses = _fit_predict_oof(
                    estimator_like,
                    X,
                    y,
                    cv=self.cv,
                    shuffle=True,
                    seed=self.random_state,
                )
                score = _rmse(y, pred)
            except Exception:
                pred = np.full(X.shape[0], np.nan, dtype=float)
                score = float("inf")
            oofs[name] = pred
            scores[name] = float(score)
        return oofs, scores

    def _select_blend(
        self,
        endpoint_oof: Mapping[str, np.ndarray],
        endpoint_scores: Mapping[str, float],
        y: np.ndarray,
    ) -> dict[str, object]:
        finite = [(name, score) for name, score in endpoint_scores.items() if np.isfinite(score)]
        if not finite:
            raise ValueError("all endpoint estimators failed during OOF scoring")
        ranked = sorted(finite, key=lambda item: item[1])
        best_name, best_score = ranked[0]
        best = {
            "members": (best_name,),
            "weights": (1.0,),
            "oof_rmse": float(best_score),
        }
        eligible = [
            name
            for name, score in ranked
            if score <= best_score * self.max_pair_oof_ratio + 1e-12
        ]
        for i, left in enumerate(eligible):
            for right in eligible[i + 1 :]:
                p_left = np.asarray(endpoint_oof[left], dtype=float)
                p_right = np.asarray(endpoint_oof[right], dtype=float)
                weights = self._two_member_convex_weights(p_left, p_right, y)
                pred = weights[0] * p_left + weights[1] * p_right
                rmse = _rmse(y, pred)
                if rmse < float(best["oof_rmse"]):
                    best = {
                        "members": (left, right),
                        "weights": weights,
                        "oof_rmse": float(rmse),
                    }
        return best

    @staticmethod
    def _two_member_convex_weights(
        left: np.ndarray,
        right: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float]:
        direction = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        target = np.asarray(y, dtype=float).ravel() - np.asarray(right, dtype=float)
        denom = float(direction @ direction)
        if denom <= 1e-12:
            return (0.5, 0.5)
        w_left = float(np.clip((target @ direction) / denom, 0.0, 1.0))
        return (w_left, 1.0 - w_left)


class AOMFallbackBlendGate(BaseEstimator):
    """Conservative OOF gate for an optional candidate endpoint.

    The base estimator is always available. The candidate estimator is admitted
    only when a train-only OOF convex blend improves the base by at least
    ``min_relative_oof_gain``. If the gate is not accepted, predictions are the
    fitted base estimator only.
    """

    def __init__(
        self,
        base_estimator: object | EstimatorFactory,
        candidate_estimator: object | EstimatorFactory,
        *,
        cv: int = 5,
        blend_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        min_relative_oof_gain: float = 0.05,
        random_state: int = 2026,
    ) -> None:
        if cv < 2:
            raise ValueError("cv must be >= 2")
        weights = tuple(float(weight) for weight in blend_weights)
        if not weights:
            raise ValueError("blend_weights must not be empty")
        if any(weight < 0.0 or weight > 1.0 for weight in weights):
            raise ValueError("blend_weights must be in [0, 1]")
        self.base_estimator = base_estimator
        self.candidate_estimator = candidate_estimator
        self.cv = int(cv)
        self.blend_weights = weights
        self.min_relative_oof_gain = float(min_relative_oof_gain)
        self.random_state = int(random_state)

    def fit(
        self,
        X,
        y,
        *,
        metadata: AOMDatasetMetadata | Mapping[str, str | None] | None = None,
    ) -> "AOMFallbackBlendGate":
        X_arr, y_arr = AOMStructuralPolicy._validate_xy(X, y)
        meta = AOMStructuralPolicy._coerce_metadata(metadata)
        base_oof, _base_losses = _fit_predict_oof(
            self.base_estimator,
            X_arr,
            y_arr,
            cv=self.cv,
            shuffle=True,
            seed=self.random_state,
        )
        candidate_oof, _candidate_losses = _fit_predict_oof(
            self.candidate_estimator,
            X_arr,
            y_arr,
            cv=self.cv,
            shuffle=True,
            seed=self.random_state,
        )
        base_rmse = _rmse(y_arr, base_oof)
        candidate_rmse = _rmse(y_arr, candidate_oof)
        best_weight, best_rmse = self._select_blend_weight(
            y_arr,
            base_oof,
            candidate_oof,
        )
        relative_gain = (base_rmse - best_rmse) / max(base_rmse, 1e-12)
        accepted = best_weight > 0.0 and relative_gain >= self.min_relative_oof_gain

        base = _make_estimator(self.base_estimator)
        candidate = _make_estimator(self.candidate_estimator)
        for estimator in (base, candidate):
            if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
                raise TypeError("base and candidate estimators must expose fit and predict")
        base.fit(X_arr, y_arr)
        candidate.fit(X_arr, y_arr)
        self.base_estimator_ = base
        self.candidate_estimator_ = candidate
        self.accepted_candidate_ = bool(accepted)
        self.candidate_weight_ = float(best_weight if accepted else 0.0)
        self.blend_report_ = {
            "accepted_candidate": self.accepted_candidate_,
            "candidate_weight": self.candidate_weight_,
            "base_weight": float(1.0 - self.candidate_weight_),
            "base_oof_rmse": float(base_rmse),
            "candidate_oof_rmse": float(candidate_rmse),
            "blend_oof_rmse": float(best_rmse),
            "relative_oof_gain": float(relative_gain),
            "min_relative_oof_gain": self.min_relative_oof_gain,
            "cv": self.cv,
            "blend_weights": list(self.blend_weights),
            "random_state": self.random_state,
            "dataset_id": meta.dataset_id,
            "database_name": meta.database_name,
            "metadata_used_for_routing": False,
        }
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "base_estimator_"):
            raise RuntimeError("AOMFallbackBlendGate.predict called before fit")
        X_arr = AOMStructuralPolicy._as_2d_float(X)
        base_pred = np.asarray(self.base_estimator_.predict(X_arr), dtype=float).ravel()
        if self.candidate_weight_ <= 0.0:
            return base_pred
        candidate_pred = np.asarray(
            self.candidate_estimator_.predict(X_arr),
            dtype=float,
        ).ravel()
        return (1.0 - self.candidate_weight_) * base_pred + self.candidate_weight_ * candidate_pred

    def _select_blend_weight(
        self,
        y: np.ndarray,
        base_oof: np.ndarray,
        candidate_oof: np.ndarray,
    ) -> tuple[float, float]:
        best_weight = 0.0
        best_rmse = float("inf")
        for weight in self.blend_weights:
            pred = (1.0 - weight) * base_oof + weight * candidate_oof
            rmse = _rmse(y, pred)
            if rmse < best_rmse:
                best_weight = float(weight)
                best_rmse = float(rmse)
        return best_weight, best_rmse


class AOMEndpointMarginStabilityGate(BaseEstimator):
    """Train-only CV/OOF agreement gate for endpoint candidates.

    The gate scores endpoint candidates with two train-only splitters:
    contiguous folds and shuffled folds. A candidate endpoint is admitted only
    when both splitters select the same best endpoint and the best-vs-second
    margins pass the configured thresholds. Otherwise the fitted ``stack``
    estimator is used as fallback.
    """

    def __init__(
        self,
        stack_estimator: object | EstimatorFactory,
        endpoint_estimators: Mapping[str, object | EstimatorFactory],
        *,
        cv: int = 5,
        margin_cv_threshold: float = 1.005,
        margin_oof_threshold: float = 1.0,
        random_state: int = 2026,
    ) -> None:
        if not endpoint_estimators:
            raise ValueError("endpoint_estimators must not be empty")
        if cv < 2:
            raise ValueError("cv must be >= 2")
        self.stack_estimator = stack_estimator
        self.endpoint_estimators = dict(endpoint_estimators)
        self.cv = int(cv)
        self.margin_cv_threshold = float(margin_cv_threshold)
        self.margin_oof_threshold = float(margin_oof_threshold)
        self.random_state = int(random_state)

    def fit(self, X, y) -> "AOMEndpointMarginStabilityGate":
        X_arr, y_arr = AOMStructuralPolicy._validate_xy(X, y)
        cv_scores = self._score_endpoints(X_arr, y_arr, shuffle=False)
        oof_scores = self._score_endpoints(X_arr, y_arr, shuffle=True)
        decision = self._decide(cv_scores, oof_scores)
        if decision.chosen_endpoint == "stack":
            estimator = _make_estimator(self.stack_estimator)
        else:
            estimator = _make_estimator(self.endpoint_estimators[decision.chosen_endpoint])
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError("selected estimator must expose fit and predict")
        estimator.fit(X_arr, y_arr)
        self.estimator_ = estimator
        self.decision_ = decision
        self.stability_report_ = {
            "chosen_endpoint": decision.chosen_endpoint,
            "cv_best_endpoint": decision.cv_best_endpoint,
            "oof_best_endpoint": decision.oof_best_endpoint,
            "accepted_endpoint": decision.accepted_endpoint,
            "cv_margin_ratio": decision.cv_margin_ratio,
            "oof_margin_ratio": decision.oof_margin_ratio,
            "cv_scores": cv_scores,
            "oof_scores": oof_scores,
            "cv": self.cv,
            "margin_cv_threshold": self.margin_cv_threshold,
            "margin_oof_threshold": self.margin_oof_threshold,
            "random_state": self.random_state,
            "metadata_used_for_routing": False,
        }
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("AOMEndpointMarginStabilityGate.predict called before fit")
        X_arr = AOMStructuralPolicy._as_2d_float(X)
        return np.asarray(self.estimator_.predict(X_arr), dtype=float).ravel()

    def _score_endpoints(self, X: np.ndarray, y: np.ndarray, *, shuffle: bool) -> dict[str, float]:
        scores = {name: [] for name in self.endpoint_estimators}
        for train_idx, val_idx in self._fold_indices(X.shape[0], shuffle=shuffle):
            for name, estimator_like in self.endpoint_estimators.items():
                estimator = _make_estimator(estimator_like)
                try:
                    estimator.fit(X[train_idx], y[train_idx])
                    pred = estimator.predict(X[val_idx])
                    scores[name].append(_rmse(y[val_idx], pred))
                except Exception:
                    scores[name].append(float("inf"))
        return {name: float(np.mean(values)) for name, values in scores.items()}

    def _fold_indices(self, n_samples: int, *, shuffle: bool) -> list[tuple[np.ndarray, np.ndarray]]:
        n_splits = min(self.cv, int(n_samples))
        if n_splits < 2:
            raise ValueError("at least two CV folds are required")
        indices = np.arange(n_samples)
        if shuffle:
            indices = np.random.default_rng(self.random_state).permutation(indices)
        folds = [np.sort(fold) for fold in np.array_split(indices, n_splits) if fold.size]
        out = []
        for val_idx in folds:
            mask = np.ones(n_samples, dtype=bool)
            mask[val_idx] = False
            out.append((np.nonzero(mask)[0], val_idx))
        return out

    def _decide(
        self,
        cv_scores: Mapping[str, float],
        oof_scores: Mapping[str, float],
    ) -> EndpointStabilityDecision:
        cv_best, cv_margin = self._best_and_margin(cv_scores)
        oof_best, oof_margin = self._best_and_margin(oof_scores)
        accepted = (
            cv_best == oof_best
            and cv_margin >= self.margin_cv_threshold
            and oof_margin >= self.margin_oof_threshold
        )
        return EndpointStabilityDecision(
            chosen_endpoint=cv_best if accepted else "stack",
            cv_best_endpoint=cv_best,
            oof_best_endpoint=oof_best,
            accepted_endpoint=accepted,
            cv_margin_ratio=cv_margin,
            oof_margin_ratio=oof_margin,
        )

    @staticmethod
    def _best_and_margin(scores: Mapping[str, float]) -> tuple[str, float]:
        ranked = sorted(scores.items(), key=lambda item: item[1])
        finite = [(name, score) for name, score in ranked if np.isfinite(score)]
        if not finite:
            return ranked[0][0], 0.0
        best_name, best_score = finite[0]
        if len(finite) < 2:
            return best_name, 0.0
        second_score = finite[1][1]
        return best_name, float(second_score / max(best_score, 1e-12))


__all__ = [
    "AOMCandidateSpec",
    "AOMControlSelector",
    "AOMDatasetMetadata",
    "AOMEndpointMarginStabilityGate",
    "AOMFallbackBlendGate",
    "AOMMidPEndpointStack",
    "AOMOperatorPLSStack",
    "AOMOperatorPLSSpec",
    "AOMPreprocessingChain",
    "AOMRidgeBlender",
    "AOMRobustHPOCompact",
    "AOMRobustHPORegressor",
    "AOMRobustHPOWide",
    "AOMStructuralPolicy",
    "AOMStructuralPolicyWithP700BlockLocalAdmission",
    "AOMStructuralPolicyWithP700ProtocolUnified",
    "AOMStructuralPolicyWithPgt1200Admissions",
    "AOMTrueBankEndpointPortfolio",
    "EndpointStabilityDecision",
    "build_aom_control_chain_bank",
]
