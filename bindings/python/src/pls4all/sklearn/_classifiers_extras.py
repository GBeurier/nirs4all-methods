"""Additional sklearn-compatible classifiers / survival regressors.

These wrap the C *_fit functions that don't fit the basic
PLSDAClassifier mold:

* :class:`SparsePLSDAClassifier` — sparse PLS-DA with int label input
* :class:`PLSLDAClassifier` — in-sample LDA on PLS scores
* :class:`PLSLogisticClassifier` — softmax-IRLS on PLS scores
* :class:`PLSQDAClassifier` — quadratic-discriminant on PLS scores
* :class:`PLSCoxRegressor` — Cox proportional-hazards on PLS scores

For the in-sample-only variants (PLS-LDA, PLS-QDA, PLS-Cox), see the
contract notes in :mod:`pls4all.sklearn._in_sample`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, check_X_y

from .. import _methods
from .._config import Config
from .._context import Context
from .._types import Algorithm, Deflation, Solver
from ._base import _check_X_p4a, _validate_X_y_no_mutate


def _basic_cfg(n_components: int) -> Config:
    cfg = Config()
    cfg.algorithm = Algorithm.PLS_REGRESSION
    cfg.solver = Solver.SIMPLS
    cfg.deflation = Deflation.REGRESSION
    cfg.n_components = int(n_components)
    cfg.center_x = True
    cfg.scale_x = False
    cfg.center_y = True
    cfg.scale_y = False
    return cfg


class SparsePLSDAClassifier(BaseEstimator, ClassifierMixin):
    """Sparse PLS-DA classifier.

    Takes integer class labels (0, 1, …, n_classes-1) and a sparsity
    regularizer. Predict-on-new-X uses the C result's `coefficients`
    matrix and the in-sample `class_priors` decoding the soft scores.
    """

    def __init__(self, n_components: int = 2,
                  *, sparsity_lambda: float = 0.05) -> None:
        self.n_components = n_components
        self.sparsity_lambda = sparsity_lambda

    def fit(self, X, y):
        X_arr, y_arr, _ = _validate_X_y_no_mutate(X, y)
        encoder = LabelEncoder()
        y_int = encoder.fit_transform(np.asarray(y).ravel())
        classes = np.asarray(encoder.classes_)
        if classes.size < 2:
            raise ValueError(
                f"SparsePLSDAClassifier needs >= 2 classes, got {classes.size}")
        ctx = Context()
        cfg = _basic_cfg(self.n_components)
        cfg.sparsity_lambda = float(self.sparsity_lambda)
        try:
            res = _methods.sparse_pls_da_fit(
                ctx, cfg, X_arr, y_labels=y_int.astype(np.int32))
            coef = np.asarray(res.matrix("coefficients"), dtype=np.float64)
            x_mean = np.asarray(res.matrix("x_mean"), dtype=np.float64).ravel()
            y_mean = np.asarray(res.matrix("y_mean"), dtype=np.float64).ravel()
        finally:
            cfg.close()
        self._label_encoder_ = encoder
        self.classes_ = classes
        self.n_features_in_ = int(X_arr.shape[1])
        self.coef_ = coef.T
        self.x_mean_ = x_mean
        self.y_mean_ = y_mean
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X_arr = _check_X_p4a(self, X)
        return (X_arr - self.x_mean_) @ self.coef_.T + self.y_mean_

    def predict(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1 or scores.shape[1] == 1:
            int_pred = (scores.ravel() >= 0.5).astype(np.int64)
        else:
            int_pred = np.argmax(scores, axis=1)
        return self._label_encoder_.inverse_transform(int_pred)


class PLSLogisticClassifier(BaseEstimator, ClassifierMixin):
    """PLS-Logistic: PLS scores fed into multinomial softmax IRLS.

    **In-sample only**: the C kernel fits multinomial logistic on the
    PLS scores (not the raw features), exports
    ``(n_classes-1, n_components)`` coefficients with class 0 as the
    implicit baseline, and returns the in-sample ``decision_scores`` /
    ``probabilities`` directly. To predict on new X we would need both
    the PLS transform (currently not exported) AND the baseline-zero
    decoding convention. Until then this wrapper:

    * fits via ``pls_logistic_fit`` and stores the in-sample
      ``decision_scores`` / ``probabilities``;
    * ``predict(X)`` and ``predict_proba(X)`` return the stored
      values IFF ``X`` matches the training array content;
    * raises :class:`NotImplementedError` otherwise.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def fit(self, X, y):
        X_arr, _, _ = _validate_X_y_no_mutate(X, y)
        y_arr = np.asarray(y).ravel()
        encoder = LabelEncoder()
        y_int = encoder.fit_transform(y_arr)
        classes = np.asarray(encoder.classes_)
        if classes.size < 2:
            raise ValueError(
                f"PLSLogisticClassifier needs >= 2 classes, got {classes.size}")
        ctx = Context()
        cfg = _basic_cfg(self.n_components)
        try:
            res = _methods.pls_logistic_fit(
                ctx, cfg, X_arr,
                y_labels=y_int.astype(np.int32),
                n_classes=int(classes.size))
            scores = np.asarray(
                res.matrix("decision_scores"), dtype=np.float64)
            try:
                probs = np.asarray(
                    res.matrix("probabilities"), dtype=np.float64)
            except Exception:
                probs = None
        finally:
            cfg.close()
        self._label_encoder_ = encoder
        self.classes_ = classes
        self.n_features_in_ = int(X_arr.shape[1])
        self._decision_scores_train = scores if scores.ndim == 2 \
            else scores.reshape(-1, 1)
        self._proba_train = probs
        self._X_train_ = X_arr.copy()
        return self

    def _check_X_match(self, X):
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        if (X_arr.shape != self._X_train_.shape
                or not np.array_equal(X_arr, self._X_train_)):
            raise NotImplementedError(
                "PLSLogisticClassifier is in-sample only: the C ABI does "
                "not export the PLS-score-space transform needed for "
                "predict-on-new-X. Pass the same X used in fit() to "
                "retrieve the stored decision scores.")

    def decision_function(self, X):
        check_is_fitted(self)
        self._check_X_match(X)
        return self._decision_scores_train

    def predict_proba(self, X):
        check_is_fitted(self)
        self._check_X_match(X)
        if self._proba_train is not None:
            return self._proba_train
        scores = self._decision_scores_train
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1 or scores.shape[1] == 1:
            int_pred = (scores.ravel() >= 0.5).astype(np.int64)
        else:
            int_pred = np.argmax(scores, axis=1)
        return self._label_encoder_.inverse_transform(int_pred)


class _InSampleClassifierBase(BaseEstimator, ClassifierMixin):
    """Base for in-sample-only classifiers (PLSLDA, PLSQDA, PLSCox)."""

    _X_train_ref: int = 0
    _decision_scores_train: np.ndarray | None = None

    def _fit_classifier(self, ctx, X, y_int, n_classes):
        raise NotImplementedError

    def fit(self, X, y):
        X_arr, _, _ = _validate_X_y_no_mutate(X, y)
        encoder = LabelEncoder()
        y_int = encoder.fit_transform(np.asarray(y).ravel())
        classes = np.asarray(encoder.classes_)
        ctx = Context()
        cfg = _basic_cfg(self.n_components)
        try:
            res = self._fit_classifier(ctx, X_arr, y_int, classes.size, cfg)
            scores = np.asarray(
                res.matrix("decision_scores"), dtype=np.float64)
        finally:
            cfg.close()
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        self._label_encoder_ = encoder
        self.classes_ = classes
        self.n_features_in_ = int(X_arr.shape[1])
        self._decision_scores_train = scores
        self._X_train_ref = id(X)
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        if id(X) != self._X_train_ref:
            raise NotImplementedError(
                f"{type(self).__name__} is in-sample only — the C ABI "
                f"does not export coefficients for predict-on-new-X. "
                f"Pass the same X used in fit() to retrieve the stored "
                f"decision scores.")
        return self._decision_scores_train

    def predict(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1 or scores.shape[1] == 1:
            int_pred = (scores.ravel() >= 0.5).astype(np.int64)
        else:
            int_pred = np.argmax(scores, axis=1)
        return self._label_encoder_.inverse_transform(int_pred)


class PLSLDAClassifier(_InSampleClassifierBase):
    """PLS-LDA on PLS scores (in-sample only)."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def _fit_classifier(self, ctx, X, y_int, n_classes, cfg):
        return _methods.pls_lda_fit(
            ctx, cfg, X, y_labels=y_int.astype(np.int32),
            n_classes=int(n_classes))


class PLSQDAClassifier(_InSampleClassifierBase):
    """PLS-QDA on PLS scores (in-sample only).

    QDA uses class_means + class_covariances which we keep around so a
    future predict-on-new-X path can compute the Mahalanobis-style
    decision boundary externally.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def _fit_classifier(self, ctx, X, y_int, n_classes, cfg):
        res = _methods.pls_qda_fit(
            ctx, cfg, X, y_labels=y_int.astype(np.int32))
        # QDA result returns "predictions" instead of "decision_scores";
        # repackage so the in-sample base class can read it uniformly.
        preds = np.asarray(res.matrix("predictions"), dtype=np.float64)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        class _Shim:
            def matrix(self, key):
                if key == "decision_scores":
                    return preds
                return res.matrix(key)

        return _Shim()


class PLSCoxRegressor(BaseEstimator, RegressorMixin):
    """PLS + Cox proportional-hazards regression on PLS scores.

    Survival-regression rather than classifier; predicts the linear
    predictor (relative risk) for the training set via the C result's
    `predictions` and on new X via the stored coefficients.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def fit(self, X, survival_times, event_indicators):
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        times = np.ascontiguousarray(
            survival_times, dtype=np.float64).ravel()
        events = np.ascontiguousarray(
            event_indicators, dtype=np.int32).ravel()
        ctx = Context()
        cfg = _basic_cfg(self.n_components)
        try:
            res = _methods.pls_cox_fit(
                ctx, cfg, X_arr,
                survival_times=times,
                event_indicators=events)
            coef = np.asarray(
                res.matrix("coefficients"), dtype=np.float64).ravel()
            x_mean = np.asarray(
                res.matrix("x_mean"), dtype=np.float64).ravel()
        finally:
            cfg.close()
        self.coef_ = coef
        self.x_mean_ = x_mean
        self.n_features_in_ = int(X_arr.shape[1])
        return self

    def predict(self, X):
        check_is_fitted(self)
        X_arr = _check_X_p4a(self, X)
        return (X_arr - self.x_mean_) @ self.coef_


__all__ = [
    "SparsePLSDAClassifier",
    "PLSLogisticClassifier",
    "PLSLDAClassifier",
    "PLSQDAClassifier",
    "PLSCoxRegressor",
]
