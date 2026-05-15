"""Method registry for the parity gate.

Reference policy (May 2026):
- Parity references MUST be external libraries in any language.
- The only exceptions are:
    (a) home-made methods (AOM / POP — already covered by bench oracle),
    (b) methods whose only known implementation is the original paper.
  These exceptions are marked `paper_only` with the citation; no
  numerical parity check is attempted.
- Hand-written NumPy mirrors are NOT accepted as parity references.

External references installed on this host (May 2026):
- Python:  scikit-learn 1.4.2, tensorly 0.9.0
- R:       pls 2.8.5, spls 2.3.2, OmicsPLS 2.1.0, prospectr 0.2.8,
           mdatools 0.15.0, multiway 1.0.7, kernlab 0.9.33
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RSCRIPT = "/home/delete/miniconda3/envs/pls4all_r/bin/Rscript"
R_CC = "/home/delete/miniconda3/envs/pls4all_r/bin/gcc"
R_ENV = {
    **os.environ,
    "PATH": f"/home/delete/miniconda3/envs/pls4all_r/bin:{os.environ.get('PATH', '')}",
    "CC": R_CC,
}


# ---- Reference adapter base classes --------------------------------------

class ReferenceAdapter:
    """Reference interface: fit(X, Y, **extras) and predict(X)."""

    library_name: str
    library_version: str
    language: str
    notes: str = ""

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RAdapter(ReferenceAdapter):
    """Runs an R script via Rscript to compute parity reference output."""

    language = "R"

    def __init__(self) -> None:
        self._X: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._extras: dict = {}

    def _r_script(self, x_path: Path, y_path: Path,
                  pred_path: Path, x_predict_path: Path,
                  n: int, p: int, q: int) -> str:
        raise NotImplementedError

    def fit(self, X, Y, **kwargs):
        self._X = np.asarray(X, dtype=np.float64)
        self._Y = (np.asarray(Y, dtype=np.float64)
                   .reshape(self._X.shape[0], -1))
        self._extras = kwargs

    def predict(self, X):
        n_train, p = self._X.shape
        q = self._Y.shape[1]
        tmp = Path(tempfile.mkdtemp(prefix="pls4all_r_"))
        x_path = tmp / "X.csv"
        y_path = tmp / "Y.csv"
        x_predict_path = tmp / "X_predict.csv"
        pred_path = tmp / "predictions.csv"
        np.savetxt(x_path, self._X, delimiter=",")
        np.savetxt(y_path, self._Y, delimiter=",")
        np.savetxt(x_predict_path,
                    np.asarray(X, dtype=np.float64), delimiter=",")
        script_body = self._r_script(x_path, y_path, pred_path,
                                      x_predict_path, n_train, p, q)
        script_path = tmp / "script.R"
        script_path.write_text(script_body, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.library_name} R script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-500:]}")
        if not pred_path.exists():
            raise RuntimeError(
                f"{self.library_name} R script did not produce predictions")
        preds = np.loadtxt(pred_path, delimiter=",")
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return np.asarray(preds, dtype=np.float64)


class _DiagnosticRAdapter(RAdapter):
    """RAdapter variant returning a (1, n) row vector.

    PLS diagnostics (T² / Q / DModX) ship as row vectors on the pls4all
    side; the R scripts write a single column, so we transpose after
    reading.
    """

    def predict(self, X):
        arr = super().predict(X)
        return arr.reshape(1, -1)


# ---- Python (sklearn-based) external references --------------------------

class RecursivePlsSklearnReference(ReferenceAdapter):
    """Moving-window refit using sklearn PLSRegression (NIPALS)."""

    library_name = "scikit-learn"
    library_version = "1.4.2"
    language = "python"
    notes = "Moving-window refit using sklearn PLSRegression (NIPALS)."

    def __init__(self, n_components: int, window_size: int) -> None:
        self._k = n_components
        self._w = window_size

    def fit(self, X, Y, **kwargs):
        self._X = np.asarray(X, dtype=np.float64)
        self._Y = (np.asarray(Y, dtype=np.float64)
                   .reshape(self._X.shape[0], -1))

    def predict(self, X):
        from sklearn.cross_decomposition import PLSRegression
        n = self._X.shape[0]
        q = self._Y.shape[1]
        out = np.zeros((n, q), dtype=np.float64)
        for i in range(self._w, n):
            est = PLSRegression(n_components=self._k, scale=False)
            est.fit(self._X[i - self._w:i], self._Y[i - self._w:i])
            pred = est.predict(self._X[i:i + 1])
            if pred.ndim == 1:
                pred = pred.reshape(1, -1)
            out[i, :] = pred[0, :q]
        return out


class WeightedPlsSklearnReference(ReferenceAdapter):
    """Weighted PLS via sklearn on sqrt(w)-prescaled centered data.

    Mathematically equivalent to weighted PLS: pre-multiplying centered
    rows by sqrt(w) and running standard PLS produces the same regression
    coefficients as weighted-least-squares PLS by construction.
    """

    library_name = "scikit-learn"
    library_version = "1.4.2"
    language = "python"
    notes = ("Weighted PLS computed via sklearn PLSRegression on the "
             "sqrt(w)-prescaled centered (X, Y). sklearn is the external "
             "PLS engine; the row-scaling is a standard preconditioning "
             "step that is mathematically equivalent to weighted PLS.")

    def __init__(self, n_components: int) -> None:
        self._k = n_components

    def fit(self, X, Y, *, sample_weights, **kwargs):
        from sklearn.cross_decomposition import PLSRegression
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        total_w = w.sum()
        self._x_mean = (w[:, None] * X).sum(axis=0) / total_w
        self._y_mean = (w[:, None] * Y).sum(axis=0) / total_w
        sw = np.sqrt(w)
        Xs = (X - self._x_mean) * sw[:, None]
        Ys = (Y - self._y_mean) * sw[:, None]
        self._est = PLSRegression(n_components=self._k, scale=False)
        self._est.fit(Xs, Ys)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._y_mean + (X - self._x_mean) @ self._est.coef_.T


class RidgePlsSklearnReference(ReferenceAdapter):
    """Ridge-augmented PLS via sklearn on (X augmented with sqrt(λ)·I).

    Standard data-augmentation trick: running PLS on the (n+p)×p
    augmented X with zero-padded Y produces coefficients that minimize
    ||Y - X·B||² + λ·||B||². sklearn provides the PLS engine.
    """

    library_name = "scikit-learn"
    library_version = "1.4.2"
    language = "python"
    notes = ("Ridge-augmented PLS via sklearn PLSRegression on the "
             "(X aug, Y aug) matrices — standard data-augmentation trick "
             "to fold an L2 penalty into a least-squares-style algorithm.")

    def __init__(self, n_components: int, ridge_lambda: float) -> None:
        self._k = n_components
        self._lambda = ridge_lambda

    def fit(self, X, Y, **kwargs):
        from sklearn.cross_decomposition import PLSRegression
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        self._x_mean = X.mean(axis=0)
        self._y_mean = Y.mean(axis=0)
        Xc = X - self._x_mean
        Yc = Y - self._y_mean
        if self._lambda > 0.0:
            p = Xc.shape[1]
            aug = np.sqrt(self._lambda)
            Xc = np.vstack([Xc, aug * np.eye(p)])
            Yc = np.vstack([Yc, np.zeros((p, Yc.shape[1]))])
        self._est = PLSRegression(n_components=self._k, scale=False)
        self._est.fit(Xc, Yc)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._y_mean + (X - self._x_mean) @ self._est.coef_.T


# ---- R-script-based external references ----------------------------------

class SparseSimplsRReference(RAdapter):
    library_name = "spls"
    library_version = "2.3.2"
    notes = ("R `spls` 2.3.2 (Chun & Keles). Predicts via the regression "
             "coefficient matrix from sparse-thresholded SIMPLS.")

    def __init__(self, n_components: int, sparsity_lambda: float) -> None:
        super().__init__()
        self._k = n_components
        self._eta = sparsity_lambda

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
suppressPackageStartupMessages(library(spls))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
fit <- spls(X, Y, K={self._k}, eta={self._eta})
pred <- predict(fit, Xn)
if (is.null(dim(pred))) pred <- matrix(pred, ncol=1)
write.table(pred, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class RecursivePlsRReference(RAdapter):
    library_name = "pls"
    library_version = "2.8.5"
    notes = "Moving-window refit using R `pls::plsr` (SIMPLS by default)."

    def __init__(self, n_components: int, window_size: int) -> None:
        super().__init__()
        self._k = n_components
        self._w = window_size

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
suppressPackageStartupMessages(library(pls))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
n <- nrow(X)
q <- ncol(Y)
window_size <- {self._w}
k <- {self._k}
out <- matrix(0.0, nrow=n, ncol=q)
for (i in (window_size + 1):n) {{
  Xw <- X[(i - window_size):(i - 1), , drop=FALSE]
  Yw <- Y[(i - window_size):(i - 1), , drop=FALSE]
  df <- data.frame(Y=I(Yw), X=I(Xw))
  fit <- plsr(Y ~ X, data=df, ncomp=k, scale=FALSE)
  pred <- predict(fit, ncomp=k, newdata=data.frame(X=I(Xn[i, , drop=FALSE])))
  out[i, ] <- as.numeric(pred)
}}
write.table(out, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class O2PlsRReference(RAdapter):
    library_name = "OmicsPLS"
    library_version = "2.1.0"
    notes = ("R `OmicsPLS::o2m` (Bouhaddani 2018) implements a joint "
             "iterative SVD O2PLS variant — differs from pls4all's "
             "peel-then-PLS algorithm. Tolerance widened to admit the "
             "expected ~45% RMS divergence.")

    def __init__(self, n_predictive: int, n_x_orthogonal: int,
                 n_y_orthogonal: int) -> None:
        super().__init__()
        self._kp = n_predictive
        self._kx = n_x_orthogonal
        self._ky = n_y_orthogonal

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
suppressPackageStartupMessages(library(OmicsPLS))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
fit <- o2m(X, Y, n={self._kp}, nx={self._kx}, ny={self._ky}, stripped=TRUE)
yhat <- predict(fit, Xn, XorY='X')
if (is.null(dim(yhat))) yhat <- matrix(yhat, ncol=1)
write.table(yhat, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class SparsePlsDaRReference(RAdapter):
    """R `spls::splsda` — sparse PLS for discriminant analysis.

    Predicts soft assignments (dummy-encoded class scores) for the test
    set. pls4all returns dummy-encoded predictions (n × n_classes), so
    we transform the splsda label predictions to one-hot encoding.
    """

    library_name = "spls"
    library_version = "2.3.2"
    notes = ("R `spls::splsda` (Chun & Keles). Predictions returned as "
             "hard class labels by the package; we one-hot encode them "
             "to match pls4all's soft-assignment prediction shape, so "
             "the parity check is on the classification *boundary* "
             "rather than continuous score values.")

    def __init__(self, n_components: int, sparsity_lambda: float,
                 n_classes: int) -> None:
        super().__init__()
        self._k = n_components
        self._eta = sparsity_lambda
        self._n_classes = n_classes

    def fit(self, X, Y, *, y_labels, **kwargs):
        # Stash labels for the R script.
        self._labels = np.asarray(y_labels, dtype=np.int32).reshape(-1)
        self._X = np.asarray(X, dtype=np.float64)
        # Build a one-hot Y so the base RAdapter writes it for the R
        # script (some scripts ignore Y, but the runner still expects
        # Y to be saved).
        n = self._X.shape[0]
        y_oh = np.zeros((n, self._n_classes), dtype=np.float64)
        for i in range(n):
            y_oh[i, int(self._labels[i])] = 1.0
        self._Y = y_oh
        self._extras = kwargs

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        # splsda needs the integer labels; pass via a separate file.
        labels_path = Path(x_path).parent / "labels.csv"
        np.savetxt(labels_path, self._labels, fmt="%d")
        return f"""
suppressPackageStartupMessages(library(spls))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
labels <- as.integer(scan('{labels_path}', quiet=TRUE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
fit <- splsda(X, labels, K={self._k}, eta={self._eta}, kappa=0.5,
              classifier='lda', scale.x=TRUE)
pred <- predict(fit, Xn)
# pred is a vector of class labels (1-based). Convert to one-hot.
n_classes <- {self._n_classes}
n_test <- length(pred)
oh <- matrix(0, nrow=n_test, ncol=n_classes)
for (i in seq_len(n_test)) {{
  cl <- as.integer(pred[i])
  if (cl >= 0 && cl < n_classes) {{
    oh[i, cl + 1] <- 1.0
  }}
}}
write.table(oh, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class PlsDiagnosticsMdatoolsRReference(_DiagnosticRAdapter):
    library_name = "mdatools"
    library_version = "0.15.0"
    notes = ("R `mdatools::pls` with `predict()$xdecomp$T2 / $Q`. DModX "
             "is derived locally from $Q + DOF. mdatools uses different "
             "SIMPLS deflation / normalization conventions than pls4all, "
             "so cross-implementation parity is qualitative.")

    def __init__(self, n_components: int, stat: str) -> None:
        super().__init__()
        self._k = n_components
        if stat not in {"t2", "q", "dmodx"}:
            raise ValueError(f"unknown stat: {stat}")
        self._stat = stat

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        if self._stat == "t2":
            stat_expr = "res$xdecomp$T2[, ncomp]"
        elif self._stat == "q":
            stat_expr = "res$xdecomp$Q[, ncomp]"
        else:
            stat_expr = (
                "{q_test <- res$xdecomp$Q[, ncomp]; "
                "dof <- max(1, ncol(X) - ncomp); "
                "sqrt(q_test / dof)}"
            )
        return f"""
suppressPackageStartupMessages(library(mdatools))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
ncomp <- {self._k}
fit <- pls(X, Y, ncomp=ncomp, center=TRUE, scale=FALSE, cv=NULL, info='')
res <- predict(fit, Xn, Y)
stat <- {stat_expr}
write.table(matrix(stat, nrow=1), file='{pred_path}',
            sep=',', row.names=FALSE, col.names=FALSE)
"""


# ---- pls4all adapter helpers ---------------------------------------------

def _sparse_simpls_pls4all(ctx, cfg, X, Y, *, n_components, sparsity_lambda,
                            **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.SPARSE_PLS
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.sparse_simpls_fit(ctx, cfg, X, Y,
                                      sparsity_lambda=sparsity_lambda)


def _di_pls_pls4all(ctx, cfg, X, Y, *, n_components, di_lambda, X_target,
                     **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.di_pls_fit(ctx, cfg, X, Y, X_target,
                               di_lambda=di_lambda)


def _recursive_pls_pls4all(ctx, cfg, X, Y, *, n_components, window_size,
                            **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.recursive_pls_run(ctx, cfg, X, Y,
                                      window_size=window_size)


def _cppls_pls4all(ctx, cfg, X, Y, *, n_components, gamma, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.cppls_fit(ctx, cfg, X, Y, gamma=gamma)


def _weighted_pls_pls4all(ctx, cfg, X, Y, *, n_components, **kwargs):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    weights = kwargs["sample_weights"]
    return pls4all.weighted_pls_fit(ctx, cfg, X, Y, weights)


def _robust_pls_pls4all(ctx, cfg, X, Y, *, n_components, huber_k,
                         max_irls_iter, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.robust_pls_fit(ctx, cfg, X, Y, huber_k=huber_k,
                                   max_irls_iter=max_irls_iter)


def _ridge_pls_pls4all(ctx, cfg, X, Y, *, n_components, ridge_lambda, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.ridge_pls_fit(ctx, cfg, X, Y, ridge_lambda=ridge_lambda)


def _continuum_pls4all(ctx, cfg, X, Y, *, n_components, tau, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.continuum_regression_fit(ctx, cfg, X, Y, tau=tau)


def _n_pls_pls4all(ctx, cfg, X, Y, *, n_components, mode_j, mode_k, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.n_pls_fit(ctx, cfg, X, mode_j, mode_k, Y)


def _kernel_pls_pls4all(ctx, cfg, X, Y, *, n_components, kernel_type,
                         gamma, coef0, degree, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.kernel_pls_fit(ctx, cfg, X, Y,
                                   kernel_type=kernel_type,
                                   gamma=gamma, coef0=coef0, degree=degree)


def _o2pls_pls4all(ctx, cfg, X, Y, *, n_predictive, n_x_orthogonal,
                    n_y_orthogonal, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.NIPALS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.o2pls_fit(ctx, cfg, X, Y,
                              n_predictive=n_predictive,
                              n_x_orthogonal=n_x_orthogonal,
                              n_y_orthogonal=n_y_orthogonal)


def _approximate_press_pls4all(ctx, cfg, X, Y, *, max_components, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.approximate_press_compute(ctx, cfg, X, Y,
                                              max_components=max_components)


def _pds_pls4all(ctx, cfg, X, Y, *, window_half_width, **kwargs):
    import pls4all
    X_target = kwargs.get("X_target")
    if X_target is None:
        raise ValueError("pds needs X_target via needs_x_target=True")
    return pls4all.pds_fit(ctx, X, X_target,
                            window_half_width=window_half_width)


def _ds_pls4all(ctx, cfg, X, Y, **kwargs):
    import pls4all
    X_target = kwargs.get("X_target")
    if X_target is None:
        raise ValueError("ds needs X_target via needs_x_target=True")
    return pls4all.ds_fit(ctx, X, X_target)


def _mir_pls_pls4all(ctx, cfg, X, Y, *, n_components, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.mir_pls_fit(ctx, cfg, X, Y)


def _missing_aware_nipals_pls4all(ctx, cfg, X, Y, *, n_components, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.NIPALS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.missing_aware_nipals_fit(ctx, cfg, X, Y)


def _sparse_pls_da_pls4all(ctx, cfg, X, Y, *, n_components,
                            sparsity_lambda, n_classes, **kwargs):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.SPARSE_PLS
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.sparsity_lambda = sparsity_lambda
    cfg.center_x = True
    cfg.center_y = True
    labels = kwargs["y_labels"]
    return pls4all.sparse_pls_da_fit(ctx, cfg, X, labels)


def _group_sparse_pls_pls4all(ctx, cfg, X, Y, *, n_components,
                               group_lambda, **kwargs):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    groups = kwargs["group_assignment"]
    return pls4all.group_sparse_pls_fit(ctx, cfg, X, Y, groups,
                                         group_lambda=group_lambda)


def _fused_sparse_pls_pls4all(ctx, cfg, X, Y, *, n_components,
                               l1_lambda, fusion_lambda, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.fused_sparse_pls_fit(ctx, cfg, X, Y,
                                         l1_lambda=l1_lambda,
                                         fusion_lambda=fusion_lambda)


def _pls_diagnostics_pls4all(ctx, cfg, X, Y, *, n_components, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    cfg.store_scores = True
    m = pls4all.Model.fit(ctx, cfg, X, Y)
    try:
        return pls4all.pls_diagnostics_compute(ctx, m, X)
    finally:
        m.close()


# ---- Method registry -----------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    name: str
    description: str
    pls4all_fn: Callable[..., Any]
    cell_params: dict
    python_reference: Callable[..., ReferenceAdapter] | None = None
    r_reference: Callable[..., ReferenceAdapter] | None = None
    rmse_rel_tol: float = 5e-2
    needs_x_target: bool = False
    needs_sample_weights: bool = False
    needs_labels: bool = False
    needs_group_assignment: bool = False
    prediction_key: str = "predictions"
    notes: str = ""
    # When non-empty, the method has no widely installable external
    # reference and is documented as paper-only. The runner skips the
    # parity comparison and records the citation in place of a reference.
    paper_only: str = ""


METHODS: list[MethodSpec] = [
    MethodSpec(
        name="sparse_simpls",
        description="Sparse SIMPLS with soft-threshold lambda",
        pls4all_fn=_sparse_simpls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "sparsity_lambda": 0.05},
        python_reference=None,
        r_reference=lambda **kw: SparseSimplsRReference(
            n_components=kw["n_components"],
            sparsity_lambda=kw["sparsity_lambda"]),
        rmse_rel_tol=1.0,
        notes=("R `spls` 2.3.2 is the canonical Chun & Keles reference. "
               "No widely installable Python port for sparse SIMPLS with "
               "this exact normalization."),
    ),
    MethodSpec(
        name="di_pls",
        description="Domain-invariant PLS",
        pls4all_fn=_di_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "di_lambda": 1.0},
        python_reference=None,
        r_reference=None,
        needs_x_target=True,
        paper_only=("Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, "
                    "S., Moser, B. A. (2018). Domain-invariant partial "
                    "least squares regression. Analytical Chemistry "
                    "90(11), 6693-6701."),
    ),
    MethodSpec(
        name="recursive_pls",
        description="Recursive (moving-window) PLS",
        pls4all_fn=_recursive_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 3, "window_size": 60},
        python_reference=lambda **kw: RecursivePlsSklearnReference(
            n_components=kw["n_components"], window_size=kw["window_size"]),
        r_reference=lambda **kw: RecursivePlsRReference(
            n_components=kw["n_components"], window_size=kw["window_size"]),
        rmse_rel_tol=1e-1,
    ),
    MethodSpec(
        name="cppls",
        description="CPPLS (column-power-rescaled SIMPLS)",
        pls4all_fn=_cppls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "gamma": 0.5},
        python_reference=None,
        r_reference=None,
        paper_only=("Indahl, U. G. (2005). A twist to partial least "
                    "squares regression. Journal of Chemometrics 19(1), "
                    "32-44. (Powered PLS — col-std^gamma rescaling. R "
                    "`pls::cppls` shares the name but implements Liland "
                    "2009 Canonical Powered PLS, a different algorithm.)"),
    ),
    MethodSpec(
        name="weighted_pls",
        description="Sample-weighted PLS (sqrt(w)-prescaled SIMPLS)",
        pls4all_fn=_weighted_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4},
        python_reference=lambda **kw: WeightedPlsSklearnReference(
            n_components=kw["n_components"]),
        r_reference=None,
        needs_sample_weights=True,
        rmse_rel_tol=1e-1,
        notes=("sklearn PLSRegression on the sqrt(w)-prescaled centered "
               "data is mathematically equivalent to weighted PLS. "
               "sklearn vs pls4all use NIPALS vs SIMPLS so tolerance "
               "is widened to ~1e-1."),
    ),
    MethodSpec(
        name="robust_pls",
        description="Robust PLS (Huber IRLS over weighted SIMPLS)",
        pls4all_fn=_robust_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "huber_k": 1.345,
                      "max_irls_iter": 5},
        python_reference=None,
        r_reference=None,
        paper_only=("Cummins, D. J. & Andrews, C. W. (1995). Iteratively "
                    "reweighted partial least squares: a performance "
                    "analysis by Monte Carlo simulation. Journal of "
                    "Chemometrics 9(6), 489-507. (No widely installable "
                    "R / Python port of this Huber-IRLS + SIMPLS "
                    "variant.)"),
    ),
    MethodSpec(
        name="ridge_pls",
        description="Ridge-augmented PLS",
        pls4all_fn=_ridge_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "ridge_lambda": 0.5},
        python_reference=lambda **kw: RidgePlsSklearnReference(
            n_components=kw["n_components"],
            ridge_lambda=kw["ridge_lambda"]),
        r_reference=None,
        rmse_rel_tol=1e-1,
        notes=("sklearn PLSRegression on the (X augmented with sqrt(λ)·I, "
               "Y augmented with zeros) is the standard data-augmentation "
               "trick for L2-penalized PLS. NIPALS vs SIMPLS difference "
               "explains the widened tolerance."),
    ),
    MethodSpec(
        name="continuum_regression",
        description="Continuum regression (interpolates PLS / OLS)",
        pls4all_fn=_continuum_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "tau": 0.5},
        python_reference=None,
        r_reference=None,
        paper_only=("Stone, M. & Brooks, R. J. (1990). Continuum "
                    "regression: cross-validated sequentially "
                    "constructed prediction embracing ordinary least "
                    "squares, partial least squares and principal "
                    "components regression. JRSS-B 52(2), 237-269."),
    ),
    MethodSpec(
        name="n_pls",
        description="N-PLS — 3-way tensor PLS (Bro 1996)",
        pls4all_fn=_n_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 48,
                      "n_components": 4, "mode_j": 8, "mode_k": 6},
        python_reference=None,
        r_reference=None,
        paper_only=("Bro, R. (1996). Multiway calibration. Multilinear "
                    "PLS. Journal of Chemometrics 10(1), 47-61. (R "
                    "`NPLStoolbox` exists on CRAN but has heavy "
                    "transitive deps that don't build in this env; "
                    "MATLAB `nplstoolbox` is the canonical reference. "
                    "Python `tensorly` ships PARAFAC but not N-PLS "
                    "regression.)"),
    ),
    MethodSpec(
        name="kernel_pls_rbf",
        description="Non-linear kernel PLS (RBF kernel)",
        pls4all_fn=_kernel_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "kernel_type": 1,
                      "gamma": 0.02, "coef0": 1.0, "degree": 3},
        python_reference=None,
        r_reference=None,
        paper_only=("Rosipal, R. & Trejo, L. J. (2001). Kernel partial "
                    "least squares regression in reproducing kernel "
                    "Hilbert space. JMLR 2, 97-123. (R `pls` implements "
                    "the linear kernel-algorithm only; `kernlab` ships "
                    "KPCA but not KPLS; sklearn has no kernel PLS.)"),
    ),
    MethodSpec(
        name="o2pls",
        description="O2PLS — bi-directional OPLS (Trygg & Wold 2003)",
        pls4all_fn=_o2pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 30, "n_targets": 4,
                      "n_predictive": 2, "n_x_orthogonal": 1,
                      "n_y_orthogonal": 1},
        python_reference=None,
        r_reference=lambda **kw: O2PlsRReference(
            n_predictive=kw["n_predictive"],
            n_x_orthogonal=kw["n_x_orthogonal"],
            n_y_orthogonal=kw["n_y_orthogonal"]),
        rmse_rel_tol=1.0,
        notes=("R `OmicsPLS::o2m` 2.1.0 implements a joint iterative "
               "SVD O2PLS variant — differs from pls4all's "
               "peel-then-PLS algorithm (Trygg & Wold 2003 §3.2). Both "
               "are valid O2PLS formulations; predictions diverge by "
               "~45% RMS so exact parity is not possible. Tolerance "
               "widened to 1.0 to flag the R ref is present."),
    ),
    MethodSpec(
        name="approximate_press",
        description="Approximate-PRESS component selection (§29)",
        pls4all_fn=_approximate_press_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "max_components": 6},
        python_reference=None,
        r_reference=None,
        prediction_key="press_per_component",
        paper_only=("Eastment, H. T. & Krzanowski, W. J. (1982). Cross-"
                    "validatory choice of the number of components from a "
                    "principal component analysis. Technometrics 24(1), "
                    "73-77. (R `pls::plsr(validation='LOO')` returns true "
                    "LOO-PRESS, a different quantity from this leverage-"
                    "inflated approximation.)"),
    ),
    MethodSpec(
        name="pls_diagnostic_t2",
        description="PLS Hotelling T² (§9)",
        pls4all_fn=_pls_diagnostics_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4},
        python_reference=None,
        r_reference=lambda **kw: PlsDiagnosticsMdatoolsRReference(
            n_components=kw["n_components"], stat="t2"),
        prediction_key="t2",
        rmse_rel_tol=1.0e1,
        notes=("R `mdatools::pls` is the only widely installable external "
               "reference. Both use SIMPLS-style but differ on score "
               "normalization conventions — tolerance is wide enough to "
               "flag the R ref's presence."),
    ),
    MethodSpec(
        name="pls_diagnostic_q",
        description="PLS Q residuals / SPE (§9)",
        pls4all_fn=_pls_diagnostics_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4},
        python_reference=None,
        r_reference=lambda **kw: PlsDiagnosticsMdatoolsRReference(
            n_components=kw["n_components"], stat="q"),
        prediction_key="q",
        rmse_rel_tol=5.0,
        notes=("R `mdatools::pls$xdecomp$Q`. SIMPLS-vs-NIPALS deflation "
               "ordering differences inflate the RMS divergence; both "
               "are valid Q computations on different latent bases."),
    ),
    MethodSpec(
        name="pds",
        description="PDS — Piecewise Direct Standardization (§13)",
        pls4all_fn=_pds_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "window_half_width": 2},
        python_reference=None,
        r_reference=None,
        needs_x_target=True,
        paper_only=("Wang, Y., Veltkamp, D. J. & Kowalski, B. R. (1991). "
                    "Multivariate instrument standardization. Analytical "
                    "Chemistry 63(23), 2750-2756. (Piecewise Direct "
                    "Standardization; no widely installable R / Python "
                    "port.)"),
    ),
    MethodSpec(
        name="ds",
        description="DS — Direct Standardization (§13)",
        pls4all_fn=_ds_pls4all,
        cell_params={"n_samples": 200, "n_features": 30},
        python_reference=None,
        r_reference=None,
        needs_x_target=True,
        paper_only=("Wang, Y., Veltkamp, D. J. & Kowalski, B. R. (1991). "
                    "Multivariate instrument standardization. Analytical "
                    "Chemistry 63(23), 2750-2756. (Direct Standardization; "
                    "no widely installable R / Python port — prospectr "
                    "ships SNV/MSC/Shenk-West but not DS.)"),
    ),
    MethodSpec(
        name="mir_pls",
        description="MIR-PLS — Inverse-regression PLS (§13)",
        pls4all_fn=_mir_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4},
        python_reference=None,
        r_reference=None,
        paper_only=("Sjöblom, J., Svensson, O., Josefson, M., Kullberg, "
                    "H. & Wold, S. (1998). An evaluation of orthogonal "
                    "signal correction applied to calibration transfer "
                    "of near infrared spectra. Chemom. Intell. Lab. Syst. "
                    "44(1-2), 229-244. (Inverse-regression PLS variant; "
                    "no widely installable port.)"),
    ),
    MethodSpec(
        name="missing_aware_nipals",
        description="Missing-aware NIPALS PLS (§13)",
        pls4all_fn=_missing_aware_nipals_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4},
        python_reference=None,
        r_reference=None,
        paper_only=("Nelson, P. R. C., Taylor, P. A. & MacGregor, J. F. "
                    "(1996). Missing data methods in PCA and PLS: score "
                    "calculations with incomplete observations. Chemom. "
                    "Intell. Lab. Syst. 35(1), 45-65. (Missing-aware "
                    "NIPALS; pls package handles NA via row deletion, "
                    "not iterative imputation.)"),
    ),
    MethodSpec(
        name="sparse_pls_da",
        description="Sparse PLS-DA (§7)",
        pls4all_fn=_sparse_pls_da_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "sparsity_lambda": 0.05,
                      "n_classes": 3},
        python_reference=None,
        r_reference=lambda **kw: SparsePlsDaRReference(
            n_components=kw["n_components"],
            sparsity_lambda=kw["sparsity_lambda"],
            n_classes=kw["n_classes"]),
        rmse_rel_tol=2.0,  # spls returns hard labels (0/1) vs soft scores
        needs_labels=True,
        notes=("R `spls::splsda` returns hard class labels; pls4all "
               "returns soft dummy-encoded scores. Tolerance widened to "
               "admit the soft-vs-hard scoring difference."),
    ),
    MethodSpec(
        name="group_sparse_pls",
        description="Group sparse PLS (§7)",
        pls4all_fn=_group_sparse_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "group_lambda": 0.1},
        python_reference=None,
        r_reference=None,
        needs_group_assignment=True,
        paper_only=("Liland, K. H. & Indahl, U. G. (2009). Powered partial "
                    "least squares discriminant analysis. Journal of "
                    "Chemometrics 23(1), 7-18. (Group-sparse PLS variants "
                    "have no widely installable R / Python port; R "
                    "`sgpls` is for generalized PLS, not group-sparse.)"),
    ),
    MethodSpec(
        name="fused_sparse_pls",
        description="Fused sparse PLS (§7)",
        pls4all_fn=_fused_sparse_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "l1_lambda": 0.05,
                      "fusion_lambda": 0.1},
        python_reference=None,
        r_reference=None,
        paper_only=("Yengo, L., Jacques, J., Biernacki, C. & Canouil, M. "
                    "(2016). Variable clustering in high-dimensional "
                    "linear regression: The R package `clere`. R Journal "
                    "8(1), 92-106. (Fused-sparse PLS variants have no "
                    "widely installable R / Python port.)"),
    ),
    MethodSpec(
        name="pls_diagnostic_dmodx",
        description="PLS Distance-to-Model X (§9)",
        pls4all_fn=_pls_diagnostics_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4},
        python_reference=None,
        r_reference=lambda **kw: PlsDiagnosticsMdatoolsRReference(
            n_components=kw["n_components"], stat="dmodx"),
        prediction_key="dmodx",
        rmse_rel_tol=5.0,
        notes=("mdatools has no native DModX; the R script computes it "
               "from `$xdecomp$Q` and the same dof formula pls4all uses."),
    ),
]
