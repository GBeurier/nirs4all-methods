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


# ---- Shared NumPy SIMPLS helper -----------------------------------------
#
# All §26 / §1 numpy-mirrors compose SIMPLS with a per-column or per-row
# rescaling step. Factor SIMPLS out once so the mirrors stay readable and
# their results match the C++ kernels (which use the same algorithm
# internally).

def _numpy_simpls(Xc: np.ndarray, Yc: np.ndarray, n_components: int
                   ) -> np.ndarray:
    """Return the SIMPLS regression coefficient matrix B (p x q) for the
    centered (Xc, Yc) such that Yhat = Xc @ B. Mirrors the loop in
    `cpp/src/core/extra_pls.cpp::simple_simpls`."""
    n, p = Xc.shape
    q = Yc.shape[1]
    k = max(1, min(n_components, min(n - 1, p)))
    eps = 1e-12
    cov = Xc.T @ Yc
    W = np.zeros((p, k))
    P = np.zeros((p, k))
    Q = np.zeros((q, k))
    V = np.zeros((p, k))
    B_scalars = np.zeros(k)
    for comp in range(k):
        U, _, _ = np.linalg.svd(cov, full_matrices=False)
        direction = U[:, 0]
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
            v -= float(np.dot(V[:, prev], v)) * V[:, prev]
        v_norm = np.linalg.norm(v)
        if v_norm < eps:
            break
        v /= v_norm
        y_ss = float(np.dot(q_load, q_load))
        b = 0.0
        if y_ss > eps:
            u = (Yc @ q_load) / y_ss
            b = float(np.dot(u, t))
        W[:, comp] = direction
        P[:, comp] = p_load
        Q[:, comp] = q_load
        V[:, comp] = v
        B_scalars[comp] = b
        cov -= np.outer(v, V[:, comp] @ cov)
    inv_pw = np.linalg.pinv(P.T @ W)
    R = W @ inv_pw
    return R @ (B_scalars[:, None] * Q.T)


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


class CpplsNumpyReference(ReferenceAdapter):
    library_name = "numpy-mirror"
    library_version = np.__version__
    language = "python"
    notes = ("Column-power-rescaled SIMPLS — pls4all CPPLS scales each X "
             "column by std^gamma before SIMPLS then unscales coefficients. "
             "R `pls::cppls` is named similarly but implements Liland 2009 "
             "Canonical Powered PLS, a different algorithm.")

    def __init__(self, n_components: int, gamma: float) -> None:
        self._k = n_components
        self._gamma = gamma

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        self._x_mean = X.mean(axis=0)
        self._y_mean = Y.mean(axis=0)
        Xc = X - self._x_mean
        Yc = Y - self._y_mean
        n = X.shape[0]
        col_std = np.sqrt((Xc * Xc).mean(axis=0))
        eps = 1e-12
        scale = np.where(col_std > eps, col_std ** self._gamma, 1.0)
        Xs = Xc / scale
        B = _numpy_simpls(Xs, Yc, self._k)
        # Unscale coefficients back to original X space.
        self._coefficients = B / scale[:, None]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._y_mean + (X - self._x_mean) @ self._coefficients


class WeightedPlsNumpyReference(ReferenceAdapter):
    library_name = "numpy-mirror"
    library_version = np.__version__
    language = "python"
    notes = ("Sample-weighted SIMPLS — pre-multiplies centered rows by "
             "sqrt(w) and runs SIMPLS. R `pls::plsr` does not accept a "
             "`weights` argument and no widely installable Python or R port "
             "for this exact algorithm exists.")

    def __init__(self, n_components: int) -> None:
        self._k = n_components

    def fit(self, X, Y, *, sample_weights, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        total_w = w.sum()
        self._x_mean = (w[:, None] * X).sum(axis=0) / total_w
        self._y_mean = (w[:, None] * Y).sum(axis=0) / total_w
        sw = np.sqrt(w)
        Xc = (X - self._x_mean) * sw[:, None]
        Yc = (Y - self._y_mean) * sw[:, None]
        self._coefficients = _numpy_simpls(Xc, Yc, self._k)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._y_mean + (X - self._x_mean) @ self._coefficients


class RobustPlsNumpyReference(ReferenceAdapter):
    library_name = "numpy-mirror"
    library_version = np.__version__
    language = "python"
    notes = ("Huber IRLS wrapped around weighted SIMPLS. Mirrors "
             "pls4all::fit_robust_pls. R `chemometrics` ships robust PCR "
             "but no widely installable robust-PLS C / R port of this "
             "specific IRLS+SIMPLS variant.")

    def __init__(self, n_components: int, huber_k: float,
                 max_irls_iter: int) -> None:
        self._k = n_components
        self._huber_k = huber_k
        self._max_iter = max_irls_iter

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        n = X.shape[0]
        w = np.ones(n, dtype=np.float64)
        eps = 1e-12
        for _ in range(self._max_iter):
            total_w = w.sum()
            x_mean = (w[:, None] * X).sum(axis=0) / total_w
            y_mean = (w[:, None] * Y).sum(axis=0) / total_w
            sw = np.sqrt(w)
            Xc = (X - x_mean) * sw[:, None]
            Yc = (Y - y_mean) * sw[:, None]
            coefs = _numpy_simpls(Xc, Yc, self._k)
            pred = y_mean + (X - x_mean) @ coefs
            resid = np.linalg.norm(Y - pred, axis=1)
            mad = np.median(resid)
            if mad < eps:
                mad = 1.0
            scale = 1.4826 * mad
            u = resid / (scale * self._huber_k)
            w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, eps))
        self._x_mean = x_mean
        self._y_mean = y_mean
        self._coefficients = coefs

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._y_mean + (X - self._x_mean) @ self._coefficients


class RidgePlsNumpyReference(ReferenceAdapter):
    library_name = "numpy-mirror"
    library_version = np.__version__
    language = "python"
    notes = ("Ridge-augmented SIMPLS — augments (X, Y) with sqrt(lambda)*I "
             "and zero blocks then runs SIMPLS. No widely installable R / "
             "Python port for this specific variant.")

    def __init__(self, n_components: int, ridge_lambda: float) -> None:
        self._k = n_components
        self._lambda = ridge_lambda

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        self._x_mean = X.mean(axis=0)
        self._y_mean = Y.mean(axis=0)
        Xc = X - self._x_mean
        Yc = Y - self._y_mean
        if self._lambda > 0.0:
            aug = np.sqrt(self._lambda)
            p = Xc.shape[1]
            Xc = np.vstack([Xc, aug * np.eye(p)])
            Yc = np.vstack([Yc, np.zeros((p, Yc.shape[1]))])
        self._coefficients = _numpy_simpls(Xc, Yc, self._k)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._y_mean + (X - self._x_mean) @ self._coefficients


class NPlsNumpyReference(ReferenceAdapter):
    library_name = "numpy-mirror"
    library_version = np.__version__
    language = "python"
    notes = ("3-way N-PLS (Bro 1996). NumPy mirror of pls4all::fit_n_pls. "
             "MATLAB `nplstoolbox` is the canonical reference; no widely "
             "installable R or Python port exists for the rank-1 Khatri-"
             "Rao decomposition variant.")

    def __init__(self, n_components: int, mode_j: int, mode_k: int) -> None:
        self._k = n_components
        self._J = mode_j
        self._K = mode_k

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        n = X.shape[0]
        J, K, a = self._J, self._K, self._k
        JK = J * K
        q = Y.shape[1]
        x_mean = X.mean(axis=0)
        y_mean = Y.mean(axis=0)
        Xc = X - x_mean
        Yc = Y - y_mean
        Xr = Xc.copy()
        Yr = Yc.copy()
        w_j_all = np.zeros((J, a))
        w_k_all = np.zeros((K, a))
        T = np.zeros((n, a))
        Q_load = np.zeros((q, a))
        B = np.zeros(a)
        P_load = np.zeros((JK, a))
        eps = 1e-12
        for comp in range(a):
            Z = Xr.T @ Yr   # JK x q
            if np.linalg.norm(Z) < eps:
                break
            U, _, _ = np.linalg.svd(Z, full_matrices=False)
            w_full = U[:, 0]
            W_mat = w_full.reshape(J, K)
            Uw, _, Vw = np.linalg.svd(W_mat, full_matrices=False)
            w_j = Uw[:, 0]
            w_k = Vw[0, :]
            if (np.linalg.norm(w_j) < eps or np.linalg.norm(w_k) < eps):
                break
            t = (Xr.reshape(n, J, K) * np.outer(w_j, w_k)[None]).sum(
                axis=(1, 2))
            tt = float(t @ t)
            if tt < eps:
                break
            q_load = (Yr.T @ t) / tt
            cc = float(q_load @ q_load)
            b = 0.0
            if cc > eps:
                u = (Yr @ q_load) / cc
                b = float(u @ t) / tt
            p_load = (Xr.T @ t) / tt
            w_j_all[:, comp] = w_j
            w_k_all[:, comp] = w_k
            T[:, comp] = t
            Q_load[:, comp] = q_load
            B[comp] = b
            P_load[:, comp] = p_load
            Xr = Xr - np.outer(t, p_load)
            Yr = Yr - b * np.outer(t, q_load)
        W = np.zeros((JK, a))
        for comp in range(a):
            W[:, comp] = np.outer(w_j_all[:, comp],
                                   w_k_all[:, comp]).reshape(JK)
        inv_pw = np.linalg.pinv(P_load.T @ W)
        coefs = W @ inv_pw @ (B[:, None] * Q_load.T)
        self._x_mean = x_mean
        self._y_mean = y_mean
        self._coefficients = coefs

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._y_mean + (X - self._x_mean) @ self._coefficients


class KernelPlsNumpyReference(ReferenceAdapter):
    library_name = "numpy-mirror"
    library_version = np.__version__
    language = "python"
    notes = ("Kernel NIPALS (Rosipal & Trejo 2001) mirror. sklearn does "
             "not ship a non-linear kernel PLS; R `pls` only covers linear "
             "kernel-algorithm. No widely installable external reference "
             "for RBF / polynomial / sigmoid kernel PLS exists.")

    KERNELS = {0: "linear", 1: "rbf", 2: "polynomial", 3: "sigmoid"}

    def __init__(self, n_components: int, kernel_type: int,
                 gamma: float, coef0: float, degree: int) -> None:
        self._k = n_components
        self._kt = kernel_type
        self._gamma = gamma
        self._coef0 = coef0
        self._degree = degree

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self._kt == 0:
            return A @ B.T
        if self._kt == 1:
            sq_norms_a = (A * A).sum(axis=1)[:, None]
            sq_norms_b = (B * B).sum(axis=1)[None, :]
            d2 = sq_norms_a + sq_norms_b - 2.0 * (A @ B.T)
            return np.exp(-self._gamma * d2)
        if self._kt == 2:
            return (self._gamma * (A @ B.T) + self._coef0) ** self._degree
        return np.tanh(self._gamma * (A @ B.T) + self._coef0)

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        n, p = X.shape
        q = Y.shape[1]
        # pls4all clamps gamma <= 0 to 1/p before the fit; mirror that here.
        if self._gamma <= 0.0:
            self._gamma = 1.0 / p
        y_mean = Y.mean(axis=0)
        Yc = Y - y_mean
        K = self._kernel(X, X)
        row_means = K.mean(axis=1)
        global_mean = float(K.mean())
        Kc = K - row_means[:, None] - row_means[None, :] + global_mean
        a = self._k
        Yr = Yc.copy()
        Kr = Kc.copy()
        alpha = np.zeros((n, q))
        eps = 1e-12
        for _ in range(a):
            u = Yr[:, 0].copy()
            t = np.zeros(n)
            c = np.zeros(q)
            for _ in range(100):
                t = Kr @ u
                t_norm = float(np.linalg.norm(t))
                if t_norm < eps:
                    break
                t /= t_norm
                c = Yr.T @ t
                cc = float(c @ c)
                if cc < eps:
                    break
                u_new = (Yr @ c) / cc
                if np.linalg.norm(u_new - u) < 1e-7:
                    u = u_new
                    break
                u = u_new
            tt = float(t @ t)
            if tt < eps:
                break
            denom = float(t @ (Kr @ t))
            if abs(denom) < eps:
                break
            alpha += np.outer(u, c) / denom
            Yr = Yr - np.outer(t, c)
            Kt = Kr @ t
            tKt = denom
            Kr = Kr - np.outer(t, Kt) - np.outer(Kt, t) + tKt * np.outer(t, t)
        self._X_train = X
        self._row_means = row_means
        self._global_mean = global_mean
        self._y_mean = y_mean
        self._alpha = alpha

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        K_test = self._kernel(X, self._X_train)
        col_means = K_test.mean(axis=1)
        # Match pls4all centering: K_c[i,j] = K_test[i,j] - row_mean_test[i]
        # - col_mean_train[j] + global_mean_train, where col_mean_train is
        # the saved row_means.
        K_c = K_test - col_means[:, None] - self._row_means[None, :] + \
            self._global_mean
        return self._y_mean + K_c @ self._alpha


class ContinuumNumpyReference(ReferenceAdapter):
    library_name = "numpy-mirror"
    library_version = np.__version__
    language = "python"
    notes = ("Column-power-rescaled SIMPLS interpolating PLS (tau=0) and "
             "OLS (tau=1). R `chemometrics::pls.cv` covers regular PLS; "
             "no widely installable continuum-regression variant matches "
             "this exact rescaling formulation.")

    def __init__(self, n_components: int, tau: float) -> None:
        self._k = n_components
        self._tau = tau

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        self._x_mean = X.mean(axis=0)
        self._y_mean = Y.mean(axis=0)
        Xc = X - self._x_mean
        Yc = Y - self._y_mean
        col_std = np.sqrt((Xc * Xc).mean(axis=0))
        eps = 1e-12
        scale = np.where(col_std > eps, col_std ** self._tau, 1.0)
        Xs = Xc / scale
        B = _numpy_simpls(Xs, Yc, self._k)
        self._coefficients = B / scale[:, None]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._y_mean + (X - self._x_mean) @ self._coefficients


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
    MethodSpec(
        name="cppls",
        description="CPPLS (column-power-rescaled SIMPLS)",
        pls4all_fn=_cppls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "gamma": 0.5},
        python_reference=lambda **kw: CpplsNumpyReference(
            n_components=kw["n_components"], gamma=kw["gamma"]),
        r_reference=None,
        rmse_rel_tol=5e-2,
        notes=("pls4all CPPLS uses col-std^gamma rescaling. R `pls::cppls` "
               "implements Liland 2009 Canonical Powered PLS (different "
               "algorithm); no widely installable R port of this rescaling "
               "variant exists."),
    ),
    MethodSpec(
        name="weighted_pls",
        description="Sample-weighted PLS (sqrt(w)-prescaled SIMPLS)",
        pls4all_fn=_weighted_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4},
        python_reference=lambda **kw: WeightedPlsNumpyReference(
            n_components=kw["n_components"]),
        r_reference=None,
        needs_sample_weights=True,
        rmse_rel_tol=5e-2,
        notes=("R `pls::plsr` does not accept a `weights` argument; no "
               "widely installable Python or R port of weighted SIMPLS "
               "with this exact formulation exists."),
    ),
    MethodSpec(
        name="robust_pls",
        description="Robust PLS (Huber IRLS over weighted SIMPLS)",
        pls4all_fn=_robust_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "huber_k": 1.345,
                      "max_irls_iter": 5},
        python_reference=lambda **kw: RobustPlsNumpyReference(
            n_components=kw["n_components"], huber_k=kw["huber_k"],
            max_irls_iter=kw["max_irls_iter"]),
        r_reference=None,
        rmse_rel_tol=5e-2,
        notes=("R `chemometrics` ships robust PCR but no widely installable "
               "robust-PLS port of this specific Huber-IRLS+SIMPLS variant."),
    ),
    MethodSpec(
        name="ridge_pls",
        description="Ridge-augmented PLS",
        pls4all_fn=_ridge_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "ridge_lambda": 0.5},
        python_reference=lambda **kw: RidgePlsNumpyReference(
            n_components=kw["n_components"],
            ridge_lambda=kw["ridge_lambda"]),
        r_reference=None,
        rmse_rel_tol=5e-2,
        notes=("No widely installable R / Python port for sqrt(lambda)*I-"
               "augmented PLS exists."),
    ),
    MethodSpec(
        name="continuum_regression",
        description="Continuum regression (interpolates PLS / OLS)",
        pls4all_fn=_continuum_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "tau": 0.5},
        python_reference=lambda **kw: ContinuumNumpyReference(
            n_components=kw["n_components"], tau=kw["tau"]),
        r_reference=None,
        rmse_rel_tol=5e-2,
        notes=("No widely installable R / Python port for this col-std^tau-"
               "rescaled continuum regression formulation exists."),
    ),
    MethodSpec(
        name="n_pls",
        description="N-PLS — 3-way tensor PLS (Bro 1996)",
        pls4all_fn=_n_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 48,
                      "n_components": 4, "mode_j": 8, "mode_k": 6},
        python_reference=lambda **kw: NPlsNumpyReference(
            n_components=kw["n_components"],
            mode_j=kw["mode_j"], mode_k=kw["mode_k"]),
        r_reference=None,
        rmse_rel_tol=5e-2,
        notes=("MATLAB `nplstoolbox` (Bro) is the canonical N-PLS "
               "reference. No widely installable R / Python port exists "
               "for the exact rank-1 Khatri-Rao deflation used here."),
    ),
    MethodSpec(
        name="kernel_pls_rbf",
        description="Non-linear kernel PLS (RBF kernel)",
        pls4all_fn=_kernel_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "kernel_type": 1,
                      "gamma": 0.02, "coef0": 1.0, "degree": 3},
        python_reference=lambda **kw: KernelPlsNumpyReference(
            n_components=kw["n_components"], kernel_type=kw["kernel_type"],
            gamma=kw["gamma"], coef0=kw["coef0"], degree=kw["degree"]),
        r_reference=None,
        rmse_rel_tol=5e-2,
        notes=("sklearn does not ship non-linear kernel PLS. R `pls` only "
               "implements the linear `kernel-algorithm` solver. No widely "
               "installable RBF / poly / sigmoid kernel PLS reference."),
    ),
]
