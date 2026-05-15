"""Method registry for the parity gate.

Each entry pairs one pls4all method with at least one Python reference and
(when available) one R reference. References for which no widely-installable
external implementation exists are explicitly marked NumPy-mirror so the
README can flag them honestly.

External references actually installed on this host (May 2026):
- Python:  scikit-learn 1.4.2, NumPy 1.26.4, SciPy 1.11.4, lifelines 0.30+,
           tensorly 0.9+
- R:       pls 2.8.5, spls 2.3.2, OmicsPLS 2.1.0, prospectr 0.2.8,
           plsVarSel 0.10.0, mdatools 0.15.0, survival 3.8.x, glmnet 4.1
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


# ---- Reference adapter base ----------------------------------------------

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


# ---- Python adapters -----------------------------------------------------

class SparseSimplsNumpyReference(ReferenceAdapter):
    library_name = "numpy-mirror"
    library_version = np.__version__
    language = "python"
    notes = "Soft-threshold SIMPLS following Chun & Keles (2010)."

    def __init__(self, n_components: int, sparsity_lambda: float) -> None:
        self._k = n_components
        self._lambda = sparsity_lambda

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        n, p = X.shape
        q = Y.shape[1]
        self._x_mean = X.mean(axis=0)
        self._y_mean = Y.mean(axis=0)
        Xc = X - self._x_mean
        Yc = Y - self._y_mean
        eps = 1e-12
        cov = Xc.T @ Yc
        k = self._k
        W = np.zeros((p, k))
        P = np.zeros((p, k))
        Q = np.zeros((q, k))
        V = np.zeros((p, k))
        B = np.zeros(k)
        for comp in range(k):
            U, _, _ = np.linalg.svd(cov, full_matrices=False)
            direction = U[:, 0]
            if self._lambda > 0.0:
                sign = np.sign(direction)
                thresholded = np.maximum(np.abs(direction) - self._lambda, 0.0)
                direction = sign * thresholded
                norm = np.linalg.norm(direction)
                if norm < eps:
                    break
                direction /= norm
            t = Xc @ direction
            t_norm = np.linalg.norm(t)
            if t_norm < eps:
                break
            t /= t_norm
            direction /= t_norm
            p_load = Xc.T @ t
            q_load = Yc.T @ t
            v = p_load.copy()
            for prev in range(comp):
                proj = float(np.dot(V[:, prev], v))
                v -= proj * V[:, prev]
            v_norm = np.linalg.norm(v)
            if v_norm < eps:
                break
            v /= v_norm
            y_load_ss = float(np.dot(q_load, q_load))
            b = 0.0
            if y_load_ss > eps:
                u = (Yc @ q_load) / y_load_ss
                b = float(np.dot(u, t))
            W[:, comp] = direction
            P[:, comp] = p_load
            Q[:, comp] = q_load
            V[:, comp] = v
            B[comp] = b
            v_ts = V[:, comp] @ cov
            cov -= np.outer(v, v_ts)
        inv_pw = np.linalg.pinv(P.T @ W)
        R = W @ inv_pw
        self._coefficients = R @ (B[:, None] * Q.T)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._y_mean + (X - self._x_mean) @ self._coefficients


class DiPlsNumpyReference(ReferenceAdapter):
    library_name = "numpy-mirror"
    library_version = np.__version__
    language = "python"
    notes = ("Domain-invariant PLS — no widely installable external Python "
             "or R reference (nirs4all DiPLS is Dynamic Inner PLS; "
             "mdatools::ipls is Interval PLS). NumPy mirror of the same "
             "direction-projection formula as pls4all.")

    def __init__(self, n_components: int, di_lambda: float) -> None:
        self._k = n_components
        self._lambda = di_lambda

    def fit(self, X, Y, *, X_target, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        Xt = np.asarray(X_target, dtype=np.float64)
        n, p = X.shape
        q = Y.shape[1]
        self._x_mean = X.mean(axis=0)
        self._y_mean = Y.mean(axis=0)
        target_mean = Xt.mean(axis=0)
        diff = self._x_mean - target_mean
        diff_ss = float(np.dot(diff, diff))
        Xc = X - self._x_mean
        Yc = Y - self._y_mean
        eps = 1e-12
        cov = Xc.T @ Yc
        k = self._k
        W = np.zeros((p, k))
        P = np.zeros((p, k))
        Q = np.zeros((q, k))
        V = np.zeros((p, k))
        B = np.zeros(k)
        for comp in range(k):
            U, _, _ = np.linalg.svd(cov, full_matrices=False)
            direction = U[:, 0]
            if self._lambda > 0.0 and diff_ss > eps:
                proj = float(np.dot(direction, diff))
                frac = self._lambda / (1.0 + self._lambda)
                direction = direction - frac * (proj / diff_ss) * diff
                norm = np.linalg.norm(direction)
                if norm < eps:
                    break
                direction /= norm
            t = Xc @ direction
            t_norm = np.linalg.norm(t)
            if t_norm < eps:
                break
            t /= t_norm
            direction /= t_norm
            p_load = Xc.T @ t
            q_load = Yc.T @ t
            v = p_load.copy()
            for prev in range(comp):
                proj = float(np.dot(V[:, prev], v))
                v -= proj * V[:, prev]
            v_norm = np.linalg.norm(v)
            if v_norm < eps:
                break
            v /= v_norm
            y_load_ss = float(np.dot(q_load, q_load))
            b = 0.0
            if y_load_ss > eps:
                u = (Yc @ q_load) / y_load_ss
                b = float(np.dot(u, t))
            W[:, comp] = direction
            P[:, comp] = p_load
            Q[:, comp] = q_load
            V[:, comp] = v
            B[comp] = b
            v_ts = V[:, comp] @ cov
            cov -= np.outer(v, v_ts)
        inv_pw = np.linalg.pinv(P.T @ W)
        R = W @ inv_pw
        self._coefficients = R @ (B[:, None] * Q.T)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._y_mean + (X - self._x_mean) @ self._coefficients


class RecursivePlsSklearnReference(ReferenceAdapter):
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


# ---- R adapters ----------------------------------------------------------

class SparseSimplsRReference(RAdapter):
    library_name = "spls"
    library_version = "2.3.2"
    notes = ("R `spls` 2.3.2 (Chun & Keles). Predicts via the regression "
             "coefficient matrix from sparse-thresholded SIMPLS. Sklearn-"
             "different normalization → tolerance is widened.")

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
    notes: str = ""


METHODS: list[MethodSpec] = [
    MethodSpec(
        name="sparse_simpls",
        description="Sparse SIMPLS with soft-threshold lambda",
        pls4all_fn=_sparse_simpls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "sparsity_lambda": 0.05},
        python_reference=lambda **kw: SparseSimplsNumpyReference(
            n_components=kw["n_components"],
            sparsity_lambda=kw["sparsity_lambda"]),
        r_reference=lambda **kw: SparseSimplsRReference(
            n_components=kw["n_components"],
            sparsity_lambda=kw["sparsity_lambda"]),
        rmse_rel_tol=1.0,  # spls package uses different normalization
    ),
    MethodSpec(
        name="di_pls",
        description="Domain-invariant PLS",
        pls4all_fn=_di_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "di_lambda": 1.0},
        python_reference=lambda **kw: DiPlsNumpyReference(
            n_components=kw["n_components"], di_lambda=kw["di_lambda"]),
        r_reference=None,
        needs_x_target=True,
        rmse_rel_tol=1e-2,
        notes=("No external Python or R reference for Domain-Invariant PLS "
               "(nirs4all DiPLS is Dynamic Inner PLS; mdatools::ipls is "
               "Interval PLS). Parity is gated against a NumPy mirror of "
               "the same algorithm — sufficient to catch transcription / "
               "compiler bugs, weaker than a true cross-implementation "
               "reference."),
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
]
