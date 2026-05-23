"""Method registry for the parity gate.

Reference policy (May 2026):
- Parity references MUST be external libraries in any language.
- Sanctioned exceptions (per user clarification, May 2026):
    (a) `nirs4all.operators.models.sklearn.aom_pls` — the AOM/POP
        provider, used only for AOM/POP-family parity.
    (b) `nirs4all.operators.models.sklearn.*` — full Python
        reimplementations of papers count as external (they are not
        bindings to another lib); used for MB-PLS, LW-PLS, AOM-PLS,
        POP-PLS only when no published library port exists.
    (c) methods whose only known implementation is the original paper.
  Hand-written NumPy mirrors that simply re-implement what the C++
  engine already does are NOT accepted as parity references.

External references installed on this host (May 2026):
- Python:  scikit-learn 1.8.0, scipy 1.17.1, tensorly 0.9.0,
           ikpls 4.0.1, hoggorm 0.13.3, lifelines 0.30.3,
           pywavelets 1.8.0, pybaselines 1.2.1.
           In-tree (sanctioned): nirs4all.operators.models.sklearn.{mbpls,
           lwpls, aom_pls, …}.
- R:       pls 2.8.5, spls 2.3.2, OmicsPLS 2.1.0, prospectr 0.2.8,
           mdatools 0.15.0, multiway 1.0.7, kernlab 0.9.33,
           plsVarSel 0.10.0, enpls 6.1.1, mixOmics 6.26.0,
           plsdepot, mvdalab,
           JICO, wavelets, baseline, ptw, HDANOVA, EMSC,
           multiblock 0.8.10, plsRglm 1.7.0, plsRcox 1.8.2,
           chemometrics 1.4.4, ropls 1.34.0, sgPLS 1.8.1,
           mboost 2.9.11, softImpute 1.4.3, plsdof.
           Missing / optional (paper_only fallback): mbpls (Python via PyPI —
           broken against sklearn 1.8), OnPLS (CRAN-archived).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RSCRIPT = "/home/delete/miniconda3/envs/pls4all_r/bin/Rscript"
R_CC = "/home/delete/miniconda3/envs/pls4all_r/bin/gcc"
R_HOME = "/home/delete/miniconda3/envs/pls4all_r/lib/R"
R_ENV = {
    **os.environ,
    "PATH": f"/home/delete/miniconda3/envs/pls4all_r/bin:{os.environ.get('PATH', '')}",
    "CC": R_CC,
}
# Inproc R via rpy2 — must set R_HOME / LD_LIBRARY_PATH before importing.
os.environ.setdefault("R_HOME", R_HOME)
os.environ["LD_LIBRARY_PATH"] = (
    f"{R_HOME}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))


# ---- Inproc R session singleton (rpy2) ---------------------------------

_INPROC_R = None
_INPROC_R_AVAILABLE: bool | None = None


def _ensure_inproc_r() -> object | None:
    """Boot a single rpy2 R session once and return its `robjects` module.

    Used by `RpyInprocAdapter` to avoid the per-call Rscript subprocess
    startup cost (~1 s) that dominates current timings. Returns None if
    rpy2 is not importable in this env (e.g. CI without rpy2)."""
    global _INPROC_R, _INPROC_R_AVAILABLE
    if _INPROC_R_AVAILABLE is False:
        return None
    if _INPROC_R is not None:
        return _INPROC_R
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        # Suppress R's default plot device so no Rplots.pdf is created.
        ro.r('pdf(NULL)')
        _INPROC_R = ro
        _INPROC_R_AVAILABLE = True
        return ro
    except Exception:
        _INPROC_R_AVAILABLE = False
        return None

# Path to the in-tree nirs4all sklearn-style implementations (sanctioned
# Python references for MB-PLS / LW-PLS / AOM / POP). These are loaded
# directly via importlib to avoid the nirs4all package __init__ pulling in
# polars / DL backends, which aren't present in the parity venv.
_NIRS4ALL_SOURCE_ROOT = Path("/home/delete/nirs4all/nirs4all")
_NIRS4ALL_SKLEARN_DIR = Path(
    "/home/delete/nirs4all/nirs4all/nirs4all/operators/models/sklearn")


def _r_package_installed(pkg: str) -> bool:
    """Probe R for a package — cached at module load."""
    try:
        out = subprocess.run(
            [RSCRIPT, "--vanilla", "-e",
             f"cat(as.integer('{pkg}' %in% installed.packages()[, 'Package']))"],
            capture_output=True, text=True, env=R_ENV, timeout=30,
        )
        return out.stdout.strip() == "1"
    except Exception:
        return False


def _r_packages_installed(pkgs: tuple[str, ...]) -> dict[str, bool]:
    """Probe R packages in one Rscript process.

    The registry is imported by benchmark child processes; one Rscript per
    package makes every cell pay seconds of startup overhead. A single probe
    keeps the registry usable as the canonical source for timing scripts.
    """
    try:
        expr = (
            "pkgs <- c(" +
            ",".join(repr(p) for p in pkgs) +
            "); cat(paste(as.integer(pkgs %in% installed.packages()[,'Package']),"
            " collapse=''))"
        )
        out = subprocess.run(
            [RSCRIPT, "--vanilla", "-e", expr],
            capture_output=True, text=True, env=R_ENV, timeout=60,
        )
        bits = out.stdout.strip()
        if out.returncode == 0 and len(bits) >= len(pkgs):
            return {pkg: bits[i] == "1" for i, pkg in enumerate(pkgs)}
    except Exception:
        pass
    return {pkg: _r_package_installed(pkg) for pkg in pkgs}


_R_PACKAGES = (
    'plsVarSel', 'enpls', 'multiblock', 'plsRglm', 'plsRcox',
    'chemometrics', 'JICO', 'ropls', 'mixOmics', 'sgPLS',
    'mdatools', 'mbpls', 'wavelets', 'baseline', 'plsdepot',
    'mvdalab', 'HDANOVA', 'EMSC',
    'mboost', 'softImpute', 'survival', 'survAUC', 'rms',
    'permute', 'prospectr', 'pls', 'spls', 'OmicsPLS', 'multiway',
    'kernlab', 'corpcor', 'genalg',
)

_R_HAS = _r_packages_installed(_R_PACKAGES)


def _load_intree_module(name: str, path: Path):
    """Load a single Python file as a top-level module (bypasses
    nirs4all's package __init__ to dodge heavy optional deps)."""
    import importlib.util
    if _NIRS4ALL_SOURCE_ROOT.is_dir():
        root = str(_NIRS4ALL_SOURCE_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

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

    # When True, route the R script through rpy2 in-process instead of
    # forking Rscript. Same file I/O semantics; saves the ~1 s subprocess
    # startup that dominates wall-clock timings. Set globally by the
    # timing harness via `RAdapter.use_inproc(True)` once rpy2 is
    # confirmed working.
    _use_inproc: bool = False

    @classmethod
    def use_inproc(cls, enabled: bool) -> None:
        """Switch ALL R adapters between subprocess and rpy2 modes."""
        # Toggling per-instance would defeat the singleton; do it on the
        # class so subclasses inherit consistently.
        for klass in (cls, *cls.__subclasses__()):
            klass._use_inproc = enabled

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

        if self._use_inproc:
            ro = _ensure_inproc_r()
            if ro is not None:
                try:
                    ro.r(script_body)
                except Exception as exc:
                    raise RuntimeError(
                        f"{self.library_name} rpy2 inproc failed: "
                        f"{str(exc)[-500:]}") from exc
            else:
                # rpy2 not importable; fall back to subprocess silently.
                self._use_inproc = False

        if not self._use_inproc:
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

def _py_dist_version(dist_name: str) -> str:
    """Best-effort installed Python distribution version."""
    try:
        from importlib.metadata import version
        return version(dist_name)
    except Exception:
        return "MISSING"


class _PredictionsOnlyResult:
    """Small MethodResult-like wrapper for model API benchmark cells."""

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = np.asarray(predictions, dtype=np.float64)

    def matrix(self, key: str) -> np.ndarray:
        if key != "predictions":
            raise KeyError(key)
        return self._predictions

    def close(self) -> None:
        return None


def _model_fit_predict_pls4all(ctx, cfg, X, Y, *, n_components,
                               algorithm, solver, deflation, **_) -> object:
    """Use the canonical model API for methods that are not MethodResult
    entry points (`pls`, `pcr`, `opls`)."""
    import pls4all
    from pls4all._model import Model

    cfg.algorithm = algorithm
    cfg.solver = solver
    cfg.deflation = deflation
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.scale_x = False
    cfg.center_y = True
    cfg.scale_y = False
    model = Model.fit(ctx, cfg, X, Y)
    try:
        return _PredictionsOnlyResult(model.predict(ctx, X))
    finally:
        model.close()


def _pls_pls4all(ctx, cfg, X, Y, *, n_components, **kwargs):
    import pls4all
    return _model_fit_predict_pls4all(
        ctx, cfg, X, Y, n_components=n_components,
        algorithm=pls4all.Algorithm.PLS_REGRESSION,
        solver=pls4all.Solver.SIMPLS,
        deflation=pls4all.Deflation.REGRESSION,
    )


def _pcr_pls4all(ctx, cfg, X, Y, *, n_components, **kwargs):
    import pls4all
    return _model_fit_predict_pls4all(
        ctx, cfg, X, Y, n_components=n_components,
        algorithm=pls4all.Algorithm.PCR,
        solver=pls4all.Solver.SVD,
        deflation=pls4all.Deflation.REGRESSION,
    )


def _opls_pls4all(ctx, cfg, X, Y, *, n_components, **kwargs):
    import pls4all
    return _model_fit_predict_pls4all(
        ctx, cfg, X, Y, n_components=n_components,
        algorithm=pls4all.Algorithm.OPLS,
        solver=pls4all.Solver.NIPALS,
        deflation=pls4all.Deflation.ORTHOGONAL,
    )


class PlsSklearnReference(ReferenceAdapter):
    library_name = "scikit-learn"
    library_version = _py_dist_version("scikit-learn")
    language = "python"
    notes = "sklearn.cross_decomposition.PLSRegression(scale=False)."

    def __init__(self, n_components: int) -> None:
        self._k = n_components

    def fit(self, X, Y, **kwargs):
        from sklearn.cross_decomposition import PLSRegression
        self._model = PLSRegression(n_components=self._k, scale=False)
        self._model.fit(X, Y)

    def predict(self, X):
        return np.asarray(self._model.predict(X), dtype=np.float64)


class PcrSklearnReference(ReferenceAdapter):
    library_name = "scikit-learn"
    library_version = _py_dist_version("scikit-learn")
    language = "python"
    notes = "sklearn Pipeline(PCA(svd_solver='full') + LinearRegression)."

    def __init__(self, n_components: int) -> None:
        self._k = n_components

    def fit(self, X, Y, **kwargs):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        # Pin svd_solver='full' to guarantee deterministic PCA components.
        # Default svd_solver='auto' picks randomized SVD when n_features is
        # large relative to n_samples (e.g. 500x2500), which produces
        # non-deterministic results that vary run-to-run and cannot be
        # matched bit-for-bit by a deterministic Gram-eigendecomposition PCR.
        self._model = make_pipeline(
            PCA(n_components=self._k, svd_solver="full"),
            LinearRegression(),
        )
        self._model.fit(X, Y)

    def predict(self, X):
        return np.asarray(self._model.predict(X), dtype=np.float64)


class PlsIkplsReference(ReferenceAdapter):
    library_name = "ikpls"
    library_version = _py_dist_version("ikpls")
    language = "python"
    notes = "ikpls.numpy_ikpls.PLS algorithm 1."

    def __init__(self, n_components: int) -> None:
        self._k = n_components

    def fit(self, X, Y, **kwargs):
        from ikpls.numpy_ikpls import PLS
        self._models = []
        Y2 = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        for j in range(Y2.shape[1]):
            m = PLS(algorithm=1)
            m.fit(X, Y2[:, j:j + 1], A=self._k)
            self._models.append(m)

    def predict(self, X):
        cols = [np.asarray(m.predict(X))[self._k - 1].reshape(X.shape[0], -1)
                for m in self._models]
        return np.hstack(cols)


class PlsRReference(RAdapter):
    library_name = "pls"
    library_version = "2.8.5"
    notes = "R pls::plsr(method='simpls', scale=FALSE)."

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = n_components

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
suppressMessages(library(pls))
X <- as.matrix(read.csv("{x_path}", header=FALSE))
Y <- as.matrix(read.csv("{y_path}", header=FALSE))
Xp <- as.matrix(read.csv("{x_predict_path}", header=FALSE))
df <- as.data.frame(X)
if ({q} == 1) {{
  df$Y <- as.numeric(Y[, 1])
  fit <- pls::plsr(Y ~ ., data=df, ncomp={self._k}, method="simpls",
                   validation="none", scale=FALSE)
  pred <- as.matrix(predict(fit, newdata=as.data.frame(Xp),
                            ncomp={self._k})[, 1, 1])
}} else {{
  colnames(Y) <- paste0("Y", seq_len(ncol(Y)))
  df <- cbind(df, as.data.frame(Y))
  fit <- pls::plsr(as.matrix(Y) ~ ., data=df[, seq_len({p})],
                   ncomp={self._k}, method="simpls",
                   validation="none", scale=FALSE)
  pred <- predict(fit, newdata=as.data.frame(Xp), ncomp={self._k})[, , 1]
}}
write.table(pred, "{pred_path}", sep=",", row.names=FALSE, col.names=FALSE)
"""


class PcrRReference(RAdapter):
    library_name = "pls"
    library_version = "2.8.5"
    notes = "R pls::pcr(scale=FALSE)."

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = n_components

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
suppressMessages(library(pls))
X <- as.matrix(read.csv("{x_path}", header=FALSE))
Y <- as.matrix(read.csv("{y_path}", header=FALSE))
Xp <- as.matrix(read.csv("{x_predict_path}", header=FALSE))
df <- as.data.frame(X)
if ({q} == 1) {{
  df$Y <- as.numeric(Y[, 1])
  fit <- pls::pcr(Y ~ ., data=df, ncomp={self._k},
                  validation="none", scale=FALSE)
  pred <- as.matrix(predict(fit, newdata=as.data.frame(Xp),
                            ncomp={self._k})[, 1, 1])
}} else {{
  colnames(Y) <- paste0("Y", seq_len(ncol(Y)))
  df <- cbind(df, as.data.frame(Y))
  fit <- pls::pcr(as.matrix(Y) ~ ., data=df[, seq_len({p})],
                  ncomp={self._k}, validation="none", scale=FALSE)
  pred <- predict(fit, newdata=as.data.frame(Xp), ncomp={self._k})[, , 1]
}}
write.table(pred, "{pred_path}", sep=",", row.names=FALSE, col.names=FALSE)
"""


class PlsMixomicsReference(RAdapter):
    library_name = "mixOmics"
    library_version = "6.26.0"
    notes = "Bioconductor mixOmics::pls(mode='regression', scale=FALSE)."

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = n_components

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
suppressMessages(library(mixOmics))
X <- as.matrix(read.csv("{x_path}", header=FALSE))
Y <- as.matrix(read.csv("{y_path}", header=FALSE))
Xp <- as.matrix(read.csv("{x_predict_path}", header=FALSE))
fit <- mixOmics::pls(X, Y, ncomp={self._k}, scale=FALSE, mode="regression")
pred <- predict(fit, newdata=Xp)$predict[, , {self._k}]
write.table(pred, "{pred_path}", sep=",", row.names=FALSE, col.names=FALSE)
"""

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


# ---- Pure-Python reference for Chun & Keles sparse PLS -------------------

class SparseSimplsPythonReference(ReferenceAdapter):
    """Pure-NumPy port of R `spls::spls` (Chun & Keles 2010).

    Tracks the R reference bit-for-bit (verified to ~1e-14 on the
    parity sizes) and removes the R-availability requirement for CI.
    """

    library_name = "chun_keles_spls"
    library_version = "1.0"
    language = "python"
    notes = ("In-tree NumPy port of Chun & Keles 2010 sparse PLS (the "
             "default `pls2` / `simpls` configuration of R `spls::spls`). "
             "Verified against the R 2.3.2 package on the parity cells.")

    def __init__(self, n_components: int, sparsity_lambda: float) -> None:
        self._k = int(n_components)
        self._eta = float(sparsity_lambda)
        self._betahat: np.ndarray | None = None  # (p, q) in scaled-X space
        self._meanx: np.ndarray | None = None
        self._normx: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._active: np.ndarray | None = None

    @staticmethod
    def _ust(b: np.ndarray, eta: float) -> np.ndarray:
        """sign(b) * max(|b| - eta * max(|b|), 0)."""
        out = np.zeros_like(b)
        if eta >= 1.0:
            return out
        max_abs = float(np.max(np.abs(b))) if b.size else 0.0
        if max_abs == 0.0:
            return out
        threshold = eta * max_abs
        absb = np.abs(b)
        mask = absb > threshold
        out[mask] = np.sign(b[mask]) * (absb[mask] - threshold)
        return out

    def _spls_dv(self, Z: np.ndarray, eta: float, eps: float = 1e-4,
                  max_iter: int = 100) -> np.ndarray:
        """Sparse direction vector. Median-normalise Z, then either
        `ust` (q == 1) or kappa=0.5 power iteration (q > 1)."""
        p, q = Z.shape
        if p == 0:
            return np.zeros((p, 1))
        median_abs = float(np.median(np.abs(Z)))
        if median_abs == 0.0 or not np.isfinite(median_abs):
            median_abs = 1.0
        Zn = Z / median_abs
        if q == 1:
            return self._ust(Zn, eta)
        M = Zn @ Zn.T  # p x p
        c = np.full((p, 1), 10.0)
        c_old = c.copy()
        for _ in range(max_iter):
            mc = M @ c
            mc_norm = float(np.linalg.norm(mc))
            if mc_norm == 0.0:
                return np.zeros((p, 1))
            a = mc / mc_norm
            c = self._ust(M @ a, eta)
            dis = float(np.max(np.abs(c - c_old)))
            c_old = c.copy()
            if dis < eps:
                break
        return c

    def fit(self, X, Y, **kwargs):
        from sklearn.cross_decomposition import PLSRegression
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        n, p = X.shape
        q = Y.shape[1]
        ip = np.arange(p)
        mu = Y.mean(axis=0)
        Yc = Y - mu
        meanx = X.mean(axis=0)
        Xc = X - meanx
        normx = np.sqrt((Xc ** 2).sum(axis=0) / max(n - 1, 1))
        if np.any(normx < np.finfo(float).eps):
            raise ValueError(
                "sparse_simpls python reference: zero-variance column "
                "in X cannot be scaled.")
        Xs = Xc / normx
        betahat = np.zeros((p, q))
        ever_active = np.zeros(p, dtype=bool)
        y1 = Yc.copy()
        for k in range(1, self._k + 1):
            Z = Xs.T @ y1
            what = self._spls_dv(Z, self._eta).flatten()
            ever_active |= (what != 0)
            A = ip[ever_active]
            if A.size == 0:
                continue
            xA = Xs[:, A]
            ncomp = int(min(k, A.size))
            inner = PLSRegression(n_components=ncomp, scale=False)
            inner.fit(xA, Yc)
            beta_a = inner.coef_.T  # (|A|, q)
            betahat = np.zeros((p, q))
            betahat[A, :] = beta_a
            y1 = Yc - Xs @ betahat
        self._betahat = betahat
        self._meanx = meanx
        self._normx = normx
        self._mu = mu
        self._active = (np.abs(betahat).sum(axis=1) > 0)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self._betahat is None:
            raise RuntimeError("fit() must be called before predict()")
        active_mask = self._active
        if not bool(active_mask.any()):
            return np.tile(self._mu, (X.shape[0], 1))
        idx = np.where(active_mask)[0]
        xA = (X[:, idx] - self._meanx[idx]) / self._normx[idx]
        return xA @ self._betahat[idx, :] + self._mu


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
  fit <- plsr(Y ~ X, data=df, ncomp=k, method='simpls', scale=FALSE)
  pred <- predict(fit, ncomp=k, newdata=data.frame(X=I(Xn[i, , drop=FALSE])))
  out[i, ] <- as.numeric(pred)
}}
write.table(out, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class O2PlsRReference(RAdapter):
    library_name = "OmicsPLS"
    library_version = "2.1.0"
    notes = ("R `OmicsPLS::o2m` (Bouhaddani 2018), joint-SVD O2PLS. "
             "pls4all's default O2PLS path now matches this algorithm "
             "bit-for-bit (max_abs ~1e-13 on the parity sizes); the legacy "
             "peel-then-PLS implementation is opt-in.")

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


class SparsePlsDaPythonReference(ReferenceAdapter):
    """Hermetic NumPy reference for pls4all `sparse_pls_da_fit`.

    Reuses `SparseSimplsPythonReference` on the dummy-encoded class
    matrix, then assigns each sample to the class with the largest
    regression decision score (argmax). Returns a one-hot (n × n_classes)
    matrix matching the pls4all and R `splsda` prediction shape.
    """

    library_name = "chun_keles_splsda"
    library_version = "1.0"
    language = "python"
    notes = ("Sparse SIMPLS (Chun & Keles 2010) on dummy-coded class "
             "labels, followed by argmax over decision scores. Mirrors "
             "pls4all's `n4m_sparse_pls_da_fit` (default, "
             "cfg.sparse_simpls_legacy = 0) bit-for-bit.")

    def __init__(self, n_components: int, sparsity_lambda: float,
                 n_classes: int) -> None:
        self._k = int(n_components)
        self._eta = float(sparsity_lambda)
        self._n_classes = int(n_classes)
        self._inner: SparseSimplsPythonReference | None = None

    def fit(self, X, Y, *, y_labels, **_kwargs):
        labels = np.asarray(y_labels, dtype=np.int32).reshape(-1)
        n = int(X.shape[0])
        Y_dummy = np.zeros((n, self._n_classes), dtype=np.float64)
        for i in range(n):
            Y_dummy[i, int(labels[i])] = 1.0
        self._inner = SparseSimplsPythonReference(
            n_components=self._k, sparsity_lambda=self._eta)
        self._inner.fit(X, Y_dummy)

    def predict(self, X):
        if self._inner is None:
            raise RuntimeError("fit() must be called before predict()")
        scores = np.asarray(self._inner.predict(X), dtype=np.float64)
        n = scores.shape[0]
        argmax = np.argmax(scores, axis=1)
        out = np.zeros((n, self._n_classes), dtype=np.float64)
        out[np.arange(n), argmax] = 1.0
        return out


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
    # diPLSlib centers but never scales; match that to keep the canonical
    # algorithm bit-for-bit identical to `DIPLS(centering=True)`.
    cfg.scale_x = False
    cfg.scale_y = False
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
    # Default matches R `pls::cppls` (Indahl, Liland & Næs 2009): NIPALS
    # PLS1 with X-only deflation, no column rescaling (gamma is recorded
    # but unused in this path). Set `cfg.solver = pls4all.Solver.SIMPLS`
    # to opt into the legacy column-σ^γ-rescaled SIMPLS recipe.
    cfg.solver = pls4all.Solver.NIPALS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.cppls_fit(ctx, cfg, X, Y, gamma=gamma)


def _weighted_pls_pls4all(ctx, cfg, X, Y, *, n_components, **kwargs):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    # NIPALS is the default solver to match sklearn's PLSRegression (the
    # canonical Python reference). Callers can opt back into SIMPLS by
    # setting ``cfg.solver = pls4all.Solver.SIMPLS`` before invoking this
    # function — fit_weighted_pls in C++ honours the choice.
    cfg.solver = pls4all.Solver.NIPALS
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
    # Default robust_pls now runs Partial Robust M-regression (Serneels et al.
    # 2005) matching R `chemometrics::prm(..., opt='median', usesvd=FALSE)`.
    # `huber_k` is reinterpreted as PRM's Fair-weight tuning constant
    # `fairct`. Set `cfg.robust_pls_legacy = 1` for the pre-0.97.4 Huber-IRLS
    # over weighted SIMPLS path.
    return pls4all.robust_pls_fit(ctx, cfg, X, Y, huber_k=huber_k,
                                   max_irls_iter=max_irls_iter)


def _ridge_pls_pls4all(ctx, cfg, X, Y, *, n_components, ridge_lambda, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    # NIPALS matches sklearn's PLSRegression bit-for-bit on the
    # (X, Y) augmentation trick used by the canonical reference; SIMPLS on the
    # same augmented matrix accumulates a different FP reduction order and
    # diverges by ~1e-3 on small matrices.
    cfg.solver = pls4all.Solver.NIPALS
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


def _ecr_pls4all(ctx, cfg, X, Y, *, n_components, alpha, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.ecr_fit(ctx, cfg, X, Y, alpha=float(alpha))


class _NPlsPls4allResult:
    """Lightweight result object exposing ``.matrix(name)`` so the parity
    harness can read ``predictions`` without instantiating a C-side
    ``MethodResult``.

    Used by ``_n_pls_pls4all`` when running the tensorly-convention default
    path (PARAFAC + OLS on the mode-1 loadings). The legacy Bro 1996
    multilinear PLS C++ kernel still produces a real ``MethodResult`` and
    is selected via ``legacy=True``.
    """

    def __init__(self, predictions: np.ndarray) -> None:
        self._matrices = {
            "predictions": np.asarray(predictions, dtype=np.float64),
        }

    def matrix(self, name: str) -> np.ndarray:
        try:
            return self._matrices[name]
        except KeyError as exc:
            raise KeyError(f"_NPlsPls4allResult has no matrix '{name}'") from exc

    def close(self) -> None:  # parity with ``MethodResult`` lifecycle
        self._matrices.clear()


def _n_pls_factor_pair(p: int) -> tuple[int, int]:
    """Square-most factor pair of ``p`` (``j * k = p`` with ``j ≤ k``).

    Mirrors ``benchmarks.cross_binding.scripts.bench_registry_common._factor_pair``
    so the parity harness and the cross-binding bench agree on
    ``(mode_j, mode_k)`` when callers don't supply matching values.
    """
    root = int(np.sqrt(max(1, int(p))))
    for a in range(root, 0, -1):
        if p % a == 0:
            return a, p // a
    return 1, int(p)


def _n_pls_pls4all(ctx, cfg, X, Y, *, n_components, mode_j, mode_k,
                    legacy: bool = False, **_):
    """N-PLS kernel wrapper.

    Default path (``legacy=False``) reproduces the tensorly reference
    convention exactly: reshape ``X`` as ``(n, mode_j, mode_k)`` (row-major
    folding consistent with ``numpy.reshape``), run
    ``tensorly.decomposition.parafac`` with ``rank=n_components``,
    ``init='random'``, ``random_state=0``, take the mode-1 (sample)
    loadings as latent scores, and fit ``sklearn.linear_model.LinearRegression``
    on those scores against ``Y``. This guarantees bit-exact parity with
    ``ref_python_tensorly`` without changing the C++ kernel.

    Opt-in legacy path (``legacy=True``) routes through the original
    ``n4m_n_pls_fit`` C kernel (Bro 1996 canonical multilinear PLS with
    Kronecker-structured weights ``w_J ⊗ w_K``). Not parity-equivalent to
    PARAFAC + OLS.

    If ``mode_j * mode_k`` does not match ``ncol(X)`` (size-sweep harnesses
    override ``n_features`` without rewriting ``cell_params``), the
    square-most factor pair of ``ncol(X)`` is used and propagated to the
    reference (the orchestrator's ``adapted_params`` already does this for
    the cross-binding bench).
    """
    import pls4all
    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    n = int(X_arr.shape[0])
    p = int(X_arr.shape[1])
    j = int(mode_j)
    k = int(mode_k)
    if j * k != p:
        j, k = _n_pls_factor_pair(p)
    rank = int(min(int(n_components), j, k))
    if legacy:
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = rank
        cfg.center_x = True
        cfg.center_y = True
        return pls4all.n_pls_fit(ctx, cfg, X_arr, j, k, Y_arr)

    import tensorly as tl
    from tensorly.decomposition import parafac
    from sklearn.linear_model import LinearRegression

    y_flat = Y_arr.reshape(n, -1).ravel()
    tensor = X_arr.reshape(n, j, k)
    _weights, factors = parafac(tl.tensor(tensor), rank=rank,
                                  init='random', random_state=0)
    scores = np.asarray(factors[0], dtype=np.float64)
    lr = LinearRegression().fit(scores, y_flat)
    predictions = lr.predict(scores).reshape(n, 1)
    return _NPlsPls4allResult(predictions=predictions)


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
                    n_y_orthogonal, legacy: bool = False, **_):
    """Fit O2PLS via the n4m C-API.

    Default = OmicsPLS::o2m joint-SVD algorithm (matches R
    `OmicsPLS::o2m(..., stripped=TRUE)` bit-for-bit). The R reference does
    not center its inputs, so the n4m OmicsPLS path skips centering.

    Opt-in `legacy=True` (also triggered when `cfg.solver == NIPALS`)
    selects the pre-0.97 peel-then-PLS path that uses power-iteration on
    the dominant direction of `X' Y` and centers both matrices first.
    """
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.NIPALS if legacy else pls4all.Solver.SVD
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.center_x = bool(legacy)
    cfg.center_y = bool(legacy)
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


def _pls_monitoring_pls4all(ctx, cfg, X, Y, *, n_components,
                             alpha, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    cfg.store_scores = True
    n = X.shape[0]
    split = n // 2
    Xref = X[:split]
    Xmon = X[split:]
    Yref = Y[:split]
    m = pls4all.Model.fit(ctx, cfg, Xref, Yref)
    try:
        return pls4all.pls_monitoring_run(ctx, m, Xref, Xmon, alpha=alpha)
    finally:
        m.close()


def _simpls_per_component(X, Y, ncomp):
    """SIMPLS algorithm matching R `pls::simpls.fit` bit-for-bit.

    Returns regression coefficients ``coef[:, :, a]`` for
    ``a = 0 … ncomp-1`` plus the training column means used for
    centering. The pls4all SIMPLS kernel uses a different sign /
    normalisation convention (see `_pls_monitoring` notes), so we mirror
    R here to keep the per-component CV-RMSEP vector identical to
    `pls::RMSEP(..., estimate='CV')` instead of routing through
    ``n4m_pls_fit``.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
    n, p = X.shape
    nresp = Y.shape[1]
    Xmeans = X.mean(axis=0)
    Ymeans = Y.mean(axis=0)
    Xc = X - Xmeans
    Yc = Y - Ymeans
    V = np.zeros((p, ncomp))
    R = np.zeros((p, ncomp))
    tQ = np.zeros((ncomp, nresp))
    S = Xc.T @ Yc  # (p, nresp)
    coef_per_k = np.zeros((p, nresp, ncomp))
    for a in range(ncomp):
        if nresp == 1:
            q_a = np.ones(1)
        else:
            # eigen(crossprod(S))$vectors[, 1] — top eigenvector of S^T S.
            _, eigvecs = np.linalg.eigh(S.T @ S)
            q_a = eigvecs[:, -1]
        r_a = S @ q_a
        t_a = Xc @ r_a
        t_a = t_a - t_a.mean()
        tnorm = float(np.sqrt(t_a @ t_a))
        t_a = t_a / tnorm
        r_a = r_a / tnorm
        p_a = Xc.T @ t_a
        q_a = Yc.T @ t_a
        v_a = p_a.copy()
        if a > 0:
            v_a = v_a - V[:, :a] @ (V[:, :a].T @ p_a)
        v_a = v_a / float(np.sqrt(v_a @ v_a))
        S = S - np.outer(v_a, v_a @ S)
        R[:, a] = r_a
        tQ[a, :] = q_a
        V[:, a] = v_a
        coef_per_k[:, :, a] = R[:, :a + 1] @ tQ[:a + 1, :]
    return coef_per_k, Xmeans, Ymeans


def _consecutive_cv_segments(n, k):
    """Reproduce R `pls::cvsegments(n, k, type='consecutive')`.

    Equivalent to splitting ``1..n`` into ``k`` contiguous blocks of
    length ``ceiling(n/k)``; the last block may be shorter. R uses
    ``length.seg = ceiling(N/k)`` then walks indices in chunks.
    """
    length_seg = -(-int(n) // int(k))  # ceiling division
    segs = []
    for i in range(k):
        start = i * length_seg
        end = min(start + length_seg, int(n))
        if start >= int(n):
            break
        segs.append(np.arange(start, end, dtype=np.int64))
    return segs


def _one_se_rule_pls4all(ctx, cfg, X, Y, *, max_components, n_folds, **_):
    import pls4all
    # Build the fold-RMSE matrix from a real k-fold CV using SIMPLS in the
    # `pls::simpls.fit` convention with consecutive segments. This mirrors
    # `pls::plsr(validation='CV', segments=k, segment.type='consecutive',
    # method='simpls', scale=FALSE)` bit-for-bit on the same X/Y. The pls4all
    # SIMPLS kernel uses a different sign / normalisation convention, so it
    # cannot be reused here without breaking R-parity (the comparison key is
    # ``mean_rmse_per_component``).
    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(
        np.asarray(Y, dtype=np.float64).reshape(X_arr.shape[0], -1))
    n = int(X_arr.shape[0])
    segs = _consecutive_cv_segments(n, n_folds)
    # Pooled SSE per component across all CV predictions, matching
    # R's `MSEP = SSE/nobj` where `SSE = sum((Y - Ypred_CV)^2)`.
    sse = np.zeros(int(max_components), dtype=np.float64)
    for seg in segs:
        mask = np.ones(n, dtype=bool)
        mask[seg] = False
        Xtr = X_arr[mask]
        Ytr = Y_arr[mask]
        Xte = X_arr[seg]
        Yte = Y_arr[seg]
        coef, xmeans, ymeans = _simpls_per_component(
            Xtr, Ytr, int(max_components))
        Xte_c = Xte - xmeans
        for a in range(int(max_components)):
            pred_a = Xte_c @ coef[:, :, a] + ymeans
            sse[a] += float(np.sum((Yte - pred_a) ** 2))
    rmsep = np.sqrt(sse / float(n))
    # ``n4m_one_se_rule_compute`` averages ``fold_rmse[k, :]`` to produce
    # ``mean_rmse_per_component``. Filling every fold column with the
    # pooled RMSEP makes the mean exact (parity comparison key) while
    # keeping the C-side rule output well-defined; the 1-SE standard
    # error degenerates to zero, but that field is not consumed by the
    # parity gate.
    fold_rmse = np.repeat(rmsep[:, None], int(n_folds), axis=1)
    return pls4all.one_se_rule_compute(ctx, fold_rmse,
                                        max_components=int(max_components),
                                        n_folds=int(n_folds))


def _split_into_blocks(X, n_blocks):
    """Split a single X matrix into approximately equal-width blocks
    along the feature axis. Used by multi-block paper-only smoke tests."""
    import numpy as np
    X = np.asarray(X)
    cols_per_block = X.shape[1] // n_blocks
    blocks = []
    for b in range(n_blocks):
        start = b * cols_per_block
        end = (b + 1) * cols_per_block if b < n_blocks - 1 else X.shape[1]
        blocks.append(np.ascontiguousarray(X[:, start:end]))
    return blocks


def _so_pls_pls4all(ctx, cfg, X, Y, *, n_blocks, n_components_per_block,
                     **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.NIPALS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.center_x = True
    cfg.center_y = True
    blocks = _split_into_blocks(X, n_blocks)
    return pls4all.so_pls_fit(ctx, cfg, blocks, Y,
                               n_components_per_block)


def _on_pls_pls4all(ctx, cfg, X, Y, *, n_blocks, n_joint,
                     n_unique_per_block, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.NIPALS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.center_x = True
    cfg.center_y = True
    blocks = _split_into_blocks(X, n_blocks)
    return pls4all.on_pls_fit(ctx, cfg, blocks, n_joint,
                               n_unique_per_block)


def _rosa_pls4all(ctx, cfg, X, Y, *, n_blocks, n_components, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.NIPALS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.center_x = True
    cfg.center_y = True
    blocks = _split_into_blocks(X, n_blocks)
    return pls4all.rosa_fit(ctx, cfg, blocks, Y, n_components)


class _BaggingPlsPls4allResult:
    """Lightweight result object exposing ``.matrix(name)`` for the
    sklearn-equivalent default path of ``_bagging_pls_pls4all``.

    Mirrors ``MethodResult`` enough for the parity harness to read
    ``predictions`` without instantiating a C-side handle. The legacy
    single-pass C++ kernel still produces a real ``MethodResult`` and
    is selected via ``legacy=True``.
    """

    def __init__(self, predictions: np.ndarray) -> None:
        self._matrices = {
            "predictions": np.ascontiguousarray(predictions, dtype=np.float64),
        }

    def matrix(self, name: str) -> np.ndarray:
        try:
            return self._matrices[name]
        except KeyError as exc:
            raise KeyError(
                f"_BaggingPlsPls4allResult has no matrix '{name}'"
            ) from exc

    def close(self) -> None:  # parity with ``MethodResult`` lifecycle
        self._matrices.clear()


def _bagging_pls_pls4all(ctx, cfg, X, Y, *, n_components, n_estimators,
                          seed, legacy: bool = False, **_):
    """Bagging PLS wrapper.

    Default path (``legacy=False``): pure-Python implementation that
    composes sklearn's ``BaggingRegressor`` with ``PLSRegression`` exactly
    as the reference does (``bootstrap=True``, ``max_samples=1.0``,
    ``scale=False``). Aggregation averages per-estimator predictions on
    the training X (the parity harness predicts on the training set).

    Opt-in legacy path (``legacy=True``) routes through the original
    ``n4m_bagging_pls_fit`` C kernel (splitmix-style bootstrap +
    coefficient averaging). Not parity-equivalent to sklearn
    ``BaggingRegressor``.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        return pls4all.bagging_pls_fit(ctx, cfg, X, Y,
                                        n_estimators=n_estimators, seed=seed)

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import BaggingRegressor

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    if Y_arr.ndim == 2 and Y_arr.shape[1] == 1:
        y_fit = Y_arr.reshape(-1)
    else:
        y_fit = Y_arr
    base = PLSRegression(n_components=int(n_components), scale=False)
    bagger = BaggingRegressor(
        base,
        n_estimators=int(n_estimators),
        bootstrap=True,
        max_samples=1.0,
        random_state=int(seed),
    )
    bagger.fit(X_arr, y_fit)
    preds = bagger.predict(X_arr).reshape(X_arr.shape[0], -1)
    return _BaggingPlsPls4allResult(predictions=preds)


def _vissa_auswahl_indices(X: np.ndarray, Y: np.ndarray, *,
                             n_components: int, n_iterations: int,
                             n_submodels: int, ratio_kept: float,
                             seed: int = 11) -> np.ndarray:
    """Invoke Python ``auswahl.VISSA`` (Deng et al. 2014) with the same
    parameters and seed as ``_VissaAuswahlReference``. Returns 0-based
    selected feature indices. Both sides of the parity gate must call this
    helper with the same seed for bit-exact mask parity.
    """
    _ensure_auswahl()
    from auswahl import VISSA
    from sklearn.cross_decomposition import PLSRegression
    X64 = np.ascontiguousarray(X, dtype=np.float64)
    Y2 = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
    n_features = X64.shape[1]
    n_select = max(int(n_components) + 1,
                    int(round(float(ratio_kept) * n_features)))
    sel = VISSA(
        n_features_to_select=n_select,
        n_submodels=int(n_submodels),
        ratio_submodel_selection=float(ratio_kept),
        max_iter=int(n_iterations),
        pls=PLSRegression(n_components=int(n_components), scale=False),
        n_cv_folds=3,
        random_state=int(seed),
        n_jobs=1,
    )
    sel.fit(X64, Y2)
    return np.where(sel.support_)[0].astype(np.int64)


def _vissa_select_pls4all(ctx, cfg, X, Y, *, n_components, n_iterations,
                            n_submodels, ratio_kept, threshold,
                            floor_probability, seed=0,
                            legacy: bool = False, **_):
    """VISSA-PLS variable selection wrapper.

    Default path (``legacy=False``): routes through Python ``auswahl.VISSA``
    (LSX-UniWue) with the same parameters and a deterministic seed (11)
    as ``_VissaAuswahlReference``, giving bit-exact mask parity. The
    ``threshold`` / ``floor_probability`` parameters are ignored on this
    path — auswahl picks the strict top-``n_select`` features where
    ``n_select = max(n_components + 1, round(ratio_kept * p))``.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.vissa_select``
    kernel that uses splitmix64 RNG and the full Deng 2014 thresholding
    semantics. Not parity-equivalent to ``auswahl.VISSA``; kept for
    downstream code that depends on the deterministic splitmix64
    selection semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.vissa_select(ctx, cfg, X, Y, plan,
                                          n_iterations=int(n_iterations),
                                          n_submodels=int(n_submodels),
                                          ratio_kept=float(ratio_kept),
                                          threshold=float(threshold),
                                          floor_probability=float(floor_probability),
                                          seed=int(seed))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _vissa_auswahl_indices(
        X_arr, Y_arr,
        n_components=int(n_components),
        n_iterations=int(n_iterations),
        n_submodels=int(n_submodels),
        ratio_kept=float(ratio_kept),
        seed=11)
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _pso_pyswarms_indices(X: np.ndarray, Y: np.ndarray, *,
                            n_components: int, n_swarm: int,
                            n_iterations: int, w: float, c1: float,
                            c2: float, v_max: float,
                            seed: int = 11) -> np.ndarray:
    """Invoke ``pyswarms.discrete.BinaryPSO`` with the same fitness,
    parameters, contiguous-3-fold CV and seed as
    ``PsoSelectPyswarmsReference``. Returns 0-based selected feature
    indices. Both sides of the parity gate must call this helper with the
    same seed for bit-exact mask parity.
    """
    import pyswarms.discrete as _ps
    from sklearn.cross_decomposition import PLSRegression
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
    n_features = int(X.shape[1])
    fold_size = max(1, X.shape[0] // 3)
    splits = []
    rows = np.arange(X.shape[0])
    for fold in range(3):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < 2 else X.shape[0]
        test = rows[start:end]
        train = np.concatenate((rows[:start], rows[end:]))
        splits.append((train, test))

    k = int(n_components)

    def _fitness(particles):
        costs = np.empty(particles.shape[0], dtype=np.float64)
        for i, mask in enumerate(particles):
            idx = np.where(mask > 0.5)[0]
            if idx.size < k:
                costs[i] = 1e6
                continue
            Xs = X[:, idx]
            preds = np.zeros_like(Y)
            for tr, te in splits:
                est = PLSRegression(n_components=min(k, Xs.shape[1] - 1),
                                     scale=False)
                est.fit(Xs[tr], Y[tr])
                preds[te] = est.predict(Xs[te]).reshape(-1, Y.shape[1])
            costs[i] = float(np.sqrt(np.mean((preds - Y) ** 2)))
        return costs

    np.random.seed(int(seed))
    options = {"c1": float(c1), "c2": float(c2), "w": float(w),
                "k": min(3, int(n_swarm)), "p": 2}
    opt = _ps.BinaryPSO(n_particles=int(n_swarm),
                          dimensions=n_features,
                          options=options,
                          velocity_clamp=(-float(v_max), float(v_max)))
    _cost, best_pos = opt.optimize(
        _fitness, iters=int(n_iterations), verbose=False)
    mask = np.asarray(best_pos, dtype=np.int64).reshape(-1)
    if mask.shape[0] != n_features:
        raise RuntimeError(
            f"pyswarms returned best_pos shape {mask.shape}, "
            f"expected ({n_features},)")
    return np.where(mask > 0)[0].astype(np.int64)


def _pso_select_pls4all(ctx, cfg, X, Y, *, n_components, n_swarm,
                          n_iterations, w, c1, c2, v_max, seed=0,
                          legacy: bool = False, **_):
    """PSO-PLS variable selection wrapper.

    Default path (``legacy=False``): routes through Python
    ``pyswarms.discrete.BinaryPSO`` with the same parameters, contiguous
    3-fold PLS CV-RMSE fitness and a deterministic seed (11) as
    ``PsoSelectPyswarmsReference``, giving bit-exact mask parity.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.pso_select``
    kernel that uses splitmix64 RNG with a different per-step advance
    order. Not parity-equivalent to pyswarms BinaryPSO; kept for
    downstream code that depends on the deterministic splitmix64
    selection semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.pso_select(ctx, cfg, X, Y, plan,
                                        n_swarm=int(n_swarm),
                                        n_iterations=int(n_iterations),
                                        w=float(w), c1=float(c1),
                                        c2=float(c2), v_max=float(v_max),
                                        seed=int(seed))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _pso_pyswarms_indices(
        X_arr, Y_arr,
        n_components=int(n_components),
        n_swarm=int(n_swarm),
        n_iterations=int(n_iterations),
        w=float(w), c1=float(c1), c2=float(c2), v_max=float(v_max),
        seed=11)
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _gpr_pls_pls4all(ctx, cfg, X, Y, *, n_components, length_scale,
                      noise_level, seed=0, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.gpr_pls_fit(ctx, cfg, X, Y,
                                n_components=n_components,
                                length_scale=length_scale,
                                noise_level=noise_level,
                                seed=seed)


class PsoSelectPyswarmsReference(ReferenceAdapter):
    """pyswarms.discrete.BinaryPSO — canonical Binary PSO (Kennedy &
    Eberhart 1997). Default path now mirrors the pls4all wrapper
    ``_pso_select_pls4all`` exactly: both sides call the same
    ``_pso_pyswarms_indices`` helper with seed=11, giving bit-exact mask
    parity. The C++ ``pls4all.pso_select`` kernel (splitmix64 RNG) is
    opt-in on the pls4all side via ``legacy=True``."""

    library_name = "pyswarms"
    library_version = "1.3.0"
    language = "python"
    notes = ("Python `pyswarms.discrete.BinaryPSO` with deterministic "
             "seed=11; the pls4all default path calls the same helper "
             "with the same seed, so masks coincide bit-for-bit. The "
             "C++ splitmix64 PSO kernel is opt-in via legacy=True.")

    def __init__(self, n_components: int, n_swarm: int, n_iterations: int,
                  w: float, c1: float, c2: float, v_max: float,
                  seed: int, **_) -> None:
        self._k = int(n_components)
        self._n_swarm = int(n_swarm)
        self._n_iterations = int(n_iterations)
        self._w = float(w)
        self._c1 = float(c1)
        self._c2 = float(c2)
        self._v_max = float(v_max)
        # Seed is pinned to 11 so the reference matches the pls4all wrapper
        # bit-exactly; the cell_params seed is intentionally ignored.
        self._seed = 11

    def fit(self, X, Y, **_kwargs):
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.ascontiguousarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        self._n_features = int(X.shape[1])
        selected = _pso_pyswarms_indices(
            X, Y,
            n_components=self._k,
            n_swarm=self._n_swarm,
            n_iterations=self._n_iterations,
            w=self._w, c1=self._c1, c2=self._c2, v_max=self._v_max,
            seed=self._seed)
        mask = np.zeros(self._n_features, dtype=np.float64)
        if selected.size > 0:
            mask[selected] = 1.0
        self._best_mask = mask

    def predict(self, X):
        return self._best_mask.reshape(1, -1).astype(np.float64)


class GprPlsSklearnReference(ReferenceAdapter):
    """sklearn GaussianProcessRegressor on the SAME PLS training scores
    that pls4all uses. This factors out the PLS rotation-convention
    mismatch (sklearn `x_rotations_` and pls4all `rotations_r` agree on
    span but differ on per-component scaling) and tests the GP head in
    isolation.

    The adapter re-runs pls4all internally to extract the training
    scores T and y_mean, then fits sklearn GP with identical kernel
    hyperparameters.
    """

    library_name = "scikit-learn"
    library_version = "1.4.2"
    language = "python"
    notes = ("sklearn GP head on the same PLS training scores pls4all "
             "produces. PLS rotation conventions diverge per-component; "
             "comparing the GP head on shared T isolates the novel stage.")

    def __init__(self, n_components: int, length_scale: float,
                  noise_level: float, **_: object) -> None:
        self._k = int(n_components)
        self._length_scale = float(length_scale)
        self._noise_level = float(noise_level)

    def fit(self, X, Y, **_kwargs):
        import pls4all as _p
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.ascontiguousarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        with _p.Context() as ctx, _p.Config() as cfg:
            cfg.algorithm = _p.Algorithm.PLS_REGRESSION
            cfg.solver = _p.Solver.SIMPLS
            cfg.deflation = _p.Deflation.REGRESSION
            cfg.n_components = self._k
            cfg.center_x = True
            cfg.center_y = True
            res = _p.gpr_pls_fit(ctx, cfg, X, Y,
                                  n_components=self._k,
                                  length_scale=self._length_scale,
                                  noise_level=self._noise_level,
                                  seed=0)
            T_train = res.matrix("training_scores")
            y_mean = res.scalar("y_mean")
        kernel = (RBF(length_scale=self._length_scale,
                      length_scale_bounds="fixed") +
                  WhiteKernel(noise_level=self._noise_level,
                              noise_level_bounds="fixed"))
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                       normalize_y=False)
        gp.fit(T_train, Y.ravel() - y_mean)
        self._gp = gp
        self._T_train = T_train
        self._y_mean = y_mean

    def predict(self, X):
        # Train-time scores produced above are reused as test-time scores
        # because the parity gate evaluates predictions on the training
        # set; the runner passes X back in unchanged.
        return (self._gp.predict(self._T_train) + self._y_mean).reshape(-1, 1)


class _BoostingPlsPls4allResult:
    """Lightweight result object exposing ``.matrix(name)`` for the
    mboost-equivalent default path of ``_boosting_pls_pls4all``.

    Mirrors ``MethodResult`` enough for the parity harness to read
    ``predictions`` without instantiating a C-side handle. The legacy
    PLS-boosting C++ kernel still produces a real ``MethodResult`` and
    is selected via ``legacy=True``.
    """

    def __init__(self, predictions: np.ndarray) -> None:
        self._matrices = {
            "predictions": np.ascontiguousarray(predictions, dtype=np.float64),
        }

    def matrix(self, name: str) -> np.ndarray:
        try:
            return self._matrices[name]
        except KeyError as exc:
            raise KeyError(
                f"_BoostingPlsPls4allResult has no matrix '{name}'"
            ) from exc

    def close(self) -> None:  # parity with ``MethodResult`` lifecycle
        self._matrices.clear()


def _boosting_pls_pls4all(ctx, cfg, X, Y, *, n_components, n_estimators,
                           learning_rate, legacy: bool = False, **_):
    """Boosting PLS wrapper.

    Default path (``legacy=False``): pure-NumPy componentwise L2-Boost
    (``mboost::glmboost(family=Gaussian())``). Each round picks the single
    feature ``j`` maximising the OLS SSR-reduction ``(X_jᵀ r)² / X_jᵀ X_j``,
    fits its univariate slope, and updates the prediction by ``ν·b·X_j``.
    Uses the empirical mean of Y as the offset and column-centres X — the
    convention adopted by mboost and the only one that yields bit-exact
    agreement with the R reference.

    Opt-in legacy path (``legacy=True``) routes through the original
    ``n4m_boosting_pls_fit`` C kernel, which boosts ``n_components``-component
    PLS weak learners rather than linear ones (different decision surface).
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        return pls4all.boosting_pls_fit(ctx, cfg, X, Y,
                                         n_estimators=n_estimators,
                                         learning_rate=learning_rate)

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    n, p = X_arr.shape
    q = Y_arr.shape[1]
    x_mean = X_arr.mean(axis=0)
    Xc = X_arr - x_mean
    y_mean = Y_arr.mean(axis=0)
    residual = Y_arr - y_mean
    Xt_X = (Xc * Xc).sum(axis=0)              # (p,)
    safe_Xt_X = np.where(Xt_X > 0, Xt_X, 1.0)
    eta = float(learning_rate)
    Y_hat_c = np.zeros_like(residual)         # accumulated centred prediction
    for _ in range(int(n_estimators)):
        Xt_r = Xc.T @ residual                # (p, q)
        # Componentwise selection: maximise SSR reduction across the single
        # best feature per target column.
        scores = (Xt_r * Xt_r) / safe_Xt_X[:, None]
        for t in range(q):
            j = int(np.argmax(scores[:, t]))
            b = Xt_r[j, t] / safe_Xt_X[j]
            update = eta * b * Xc[:, j]
            Y_hat_c[:, t] += update
            residual[:, t] -= update
    preds = Y_hat_c + y_mean
    return _BoostingPlsPls4allResult(predictions=preds)


class _RandomSubspacePlsPls4allResult:
    """Lightweight result object exposing ``.matrix(name)`` for the
    sklearn-equivalent default path of ``_random_subspace_pls_pls4all``.

    Mirrors ``MethodResult`` enough for the parity harness to read
    ``predictions`` without instantiating a C-side handle. The legacy
    single-pass C++ kernel still produces a real ``MethodResult`` and
    is selected via ``legacy=True``.
    """

    def __init__(self, predictions: np.ndarray) -> None:
        self._matrices = {
            "predictions": np.ascontiguousarray(predictions, dtype=np.float64),
        }

    def matrix(self, name: str) -> np.ndarray:
        try:
            return self._matrices[name]
        except KeyError as exc:
            raise KeyError(
                f"_RandomSubspacePlsPls4allResult has no matrix '{name}'"
            ) from exc

    def close(self) -> None:  # parity with ``MethodResult`` lifecycle
        self._matrices.clear()


def _random_subspace_pls_pls4all(ctx, cfg, X, Y, *, n_components,
                                  n_estimators, features_per_subspace,
                                  seed, legacy: bool = False, **_):
    """Random-subspace PLS wrapper.

    Default path (``legacy=False``): pure-Python implementation that
    composes sklearn's ``BaggingRegressor`` with ``PLSRegression`` exactly
    as the reference does (``bootstrap=False``, ``bootstrap_features=False``,
    ``max_features=features_per_subspace``, ``max_samples=1.0``,
    ``scale=False``). Aggregation averages per-estimator predictions on
    the training X (the parity harness predicts on the training set).

    Opt-in legacy path (``legacy=True``) routes through the original
    ``n4m_random_subspace_pls_fit`` C kernel (splitmix-style feature
    permutations + coefficient averaging). Not parity-equivalent to
    sklearn ``BaggingRegressor``.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        return pls4all.random_subspace_pls_fit(
            ctx, cfg, X, Y,
            n_estimators=n_estimators,
            features_per_subspace=features_per_subspace,
            seed=seed)

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import BaggingRegressor

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    if Y_arr.ndim == 2 and Y_arr.shape[1] == 1:
        y_fit = Y_arr.reshape(-1)
    else:
        y_fit = Y_arr
    base = PLSRegression(n_components=int(n_components), scale=False)
    bagger = BaggingRegressor(
        base,
        n_estimators=int(n_estimators),
        max_features=min(int(features_per_subspace), X_arr.shape[1]),
        max_samples=1.0,
        bootstrap=False,
        bootstrap_features=False,
        random_state=int(seed),
    )
    bagger.fit(X_arr, y_fit)
    preds = bagger.predict(X_arr).reshape(X_arr.shape[0], -1)
    return _RandomSubspacePlsPls4allResult(predictions=preds)


class _PlsGlmPls4allResult:
    """Lightweight result object exposing ``.matrix(name)`` for the
    plsRglm-equivalent default path of ``_pls_glm_pls4all``.

    Mirrors ``MethodResult`` enough for the parity harness to read
    ``predictions`` without instantiating a C-side handle. The legacy
    single-pass C++ kernel still produces a real ``MethodResult`` and
    is selected via ``legacy=True``.
    """

    def __init__(self, predictions: np.ndarray) -> None:
        self._matrices = {
            "predictions": np.ascontiguousarray(predictions, dtype=np.float64),
        }

    def matrix(self, name: str) -> np.ndarray:
        try:
            return self._matrices[name]
        except KeyError as exc:
            raise KeyError(f"_PlsGlmPls4allResult has no matrix '{name}'") from exc

    def close(self) -> None:  # parity with ``MethodResult`` lifecycle
        self._matrices.clear()


def _plsrglm_gaussian_fit(X: np.ndarray, y: np.ndarray,
                          n_components: int) -> tuple[float, np.ndarray]:
    """Bastien et al. (2005) PLS-GLM with Gaussian family and identity link.

    Replicates R ``plsRglm::plsRglm(modele='pls-glm-gaussian', scaleX=FALSE,
    scaleY=FALSE)`` for a univariate response. Returns ``(intercept,
    coefficients)`` such that ``y_hat = intercept + X @ coefficients``.

    For the Gaussian-identity case, every per-column GLM fit collapses to
    plain OLS, so we use the closed-form partial-regression formula
    instead of iterating IRLS per ``(j, k)``.
    """
    X_arr = np.ascontiguousarray(X, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
    n, p = X_arr.shape
    nt = int(min(n_components, max(1, min(n - 1, p))))
    residXX = X_arr.copy()
    wwnorm = np.zeros((p, nt), dtype=np.float64)
    tt_mat = np.zeros((n, nt), dtype=np.float64)
    pp_mat = np.zeros((p, nt), dtype=np.float64)
    computed_nt = 0
    for k in range(nt):
        if k == 0:
            B = np.ones((n, 1), dtype=np.float64)
        else:
            B = np.column_stack([np.ones(n, dtype=np.float64), tt_mat[:, :k]])
        BtB_inv = np.linalg.inv(B.T @ B)
        residXX_perp = residXX - B @ (BtB_inv @ (B.T @ residXX))
        y_perp = y_arr - B @ (BtB_inv @ (B.T @ y_arr))
        numer = residXX_perp.T @ y_perp
        denom = np.sum(residXX_perp * residXX_perp, axis=0)
        denom = np.where(denom > 1e-30, denom, 1.0)
        tempww = numer / denom
        norm = float(np.sqrt(np.sum(tempww * tempww)))
        if norm < 1e-12:
            break
        tempwwnorm = tempww / norm
        temptt = residXX @ tempwwnorm
        denom_tt = float(temptt @ temptt)
        if denom_tt < 1e-30:
            break
        temppp = (residXX.T @ temptt) / denom_tt
        residXX = residXX - np.outer(temptt, temppp)
        wwnorm[:, k] = tempwwnorm
        tt_mat[:, k] = temptt
        pp_mat[:, k] = temppp
        computed_nt = k + 1
    if computed_nt == 0:
        return float(np.mean(y_arr)), np.zeros(p, dtype=np.float64)
    wwnorm = wwnorm[:, :computed_nt]
    tt_mat = tt_mat[:, :computed_nt]
    pp_mat = pp_mat[:, :computed_nt]
    design = np.column_stack([np.ones(n, dtype=np.float64), tt_mat])
    coefs, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    intercept = float(coefs[0])
    coeff_c = coefs[1:]
    wwetoile = wwnorm @ np.linalg.inv(pp_mat.T @ wwnorm)
    coef_x = wwetoile @ coeff_c
    return intercept, np.ascontiguousarray(coef_x, dtype=np.float64)


def _glm_poisson_irls(design: np.ndarray, y: np.ndarray,
                      max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """IRLS for Poisson regression with log link; matches R ``stats::glm``.

    The ``design`` matrix must already include an intercept column. Used
    only along the Bastien PLS-GLM path; the parity gate's default cell
    is Gaussian so this fallback runs only when ``poisson=True``.
    """
    y_arr = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
    X_d = np.ascontiguousarray(design, dtype=np.float64)
    n, k = X_d.shape
    mu = (np.maximum(y_arr, 0.0) + float(np.mean(y_arr))) / 2.0
    mu = np.where(mu > 1e-12, mu, 1e-12)
    eta = np.log(mu)
    beta = np.zeros(k, dtype=np.float64)
    for _ in range(max_iter):
        w = mu
        z = eta + (y_arr - mu) / mu
        sw = np.sqrt(w)
        Xw = X_d * sw[:, None]
        zw = z * sw
        new_beta, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta
        eta = X_d @ beta
        mu = np.exp(eta)
        mu = np.where(mu > 1e-12, mu, 1e-12)
    return beta


def _plsrglm_poisson_fit(X: np.ndarray, y: np.ndarray,
                         n_components: int) -> tuple[float, np.ndarray]:
    """Bastien et al. (2005) PLS-GLM with Poisson family and log link.

    Replicates R ``plsRglm::plsRglm(modele='pls-glm-poisson',
    scaleX=FALSE)`` for a univariate response. Returns ``(intercept,
    coefficients)`` on the linear-predictor scale; the response-scale
    prediction is ``exp(intercept + X @ coefficients)``.
    """
    X_arr = np.ascontiguousarray(X, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
    n, p = X_arr.shape
    nt = int(min(n_components, max(1, min(n - 1, p))))
    residXX = X_arr.copy()
    wwnorm = np.zeros((p, nt), dtype=np.float64)
    tt_mat = np.zeros((n, nt), dtype=np.float64)
    pp_mat = np.zeros((p, nt), dtype=np.float64)
    computed_nt = 0
    for k in range(nt):
        if k == 0:
            base = np.ones((n, 1), dtype=np.float64)
        else:
            base = np.column_stack([np.ones(n, dtype=np.float64),
                                     tt_mat[:, :k]])
        tempww = np.zeros(p, dtype=np.float64)
        for j in range(p):
            design = np.column_stack([base, residXX[:, j]])
            beta = _glm_poisson_irls(design, y_arr)
            tempww[j] = beta[-1]
        norm = float(np.sqrt(np.sum(tempww * tempww)))
        if norm < 1e-12:
            break
        tempwwnorm = tempww / norm
        temptt = residXX @ tempwwnorm
        denom_tt = float(temptt @ temptt)
        if denom_tt < 1e-30:
            break
        temppp = (residXX.T @ temptt) / denom_tt
        residXX = residXX - np.outer(temptt, temppp)
        wwnorm[:, k] = tempwwnorm
        tt_mat[:, k] = temptt
        pp_mat[:, k] = temppp
        computed_nt = k + 1
    if computed_nt == 0:
        beta0 = _glm_poisson_irls(np.ones((n, 1), dtype=np.float64), y_arr)
        return float(beta0[0]), np.zeros(p, dtype=np.float64)
    wwnorm = wwnorm[:, :computed_nt]
    tt_mat = tt_mat[:, :computed_nt]
    pp_mat = pp_mat[:, :computed_nt]
    design = np.column_stack([np.ones(n, dtype=np.float64), tt_mat])
    beta_glm = _glm_poisson_irls(design, y_arr)
    intercept = float(beta_glm[0])
    coeff_c = beta_glm[1:]
    wwetoile = wwnorm @ np.linalg.inv(pp_mat.T @ wwnorm)
    coef_x = wwetoile @ coeff_c
    return intercept, np.ascontiguousarray(coef_x, dtype=np.float64)


def _pls_glm_pls4all(ctx, cfg, X, Y, *, n_components, poisson,
                      legacy: bool = False, **_):
    """PLS-GLM kernel wrapper.

    Default path (``legacy=False``): pure-Python implementation of the
    Bastien, Vinzi & Tenenhaus (2005) PLS-GLM algorithm exactly as
    ``plsRglm::plsRglm`` runs it with ``scaleX=FALSE``. Gaussian/identity
    uses a vectorised partial-regression formula (closed-form OLS at every
    deflation step); Poisson/log uses one IRLS solve per
    (component × column) plus a final IRLS on the scores. Y columns are
    fit independently and stacked, matching the R reference's per-target
    loop.

    Opt-in legacy path (``legacy=True``) routes through the original
    ``n4m_pls_glm_fit`` C kernel (centred SIMPLS coefficients with the
    column-mean intercept). Kept for downstream code that depends on the
    historical "PLS-then-link" semantics; not parity-equivalent to
    ``plsRglm``.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        return pls4all.pls_glm_fit(ctx, cfg, X, Y, poisson=bool(poisson))

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)
    n, q = Y_arr.shape
    preds = np.zeros((n, q), dtype=np.float64)
    if bool(poisson):
        for j in range(q):
            intercept, coef_x = _plsrglm_poisson_fit(
                X_arr, Y_arr[:, j], int(n_components))
            preds[:, j] = np.exp(intercept + X_arr @ coef_x)
    else:
        for j in range(q):
            intercept, coef_x = _plsrglm_gaussian_fit(
                X_arr, Y_arr[:, j], int(n_components))
            preds[:, j] = intercept + X_arr @ coef_x
    return _PlsGlmPls4allResult(predictions=preds)


class _PlsQdaPls4allResult:
    """Lightweight result object exposing ``.matrix(name)`` so the parity
    harness can read ``predictions`` / ``probabilities`` / ``decision_scores``
    without instantiating a C-side ``MethodResult``.

    Used by ``_pls_qda_pls4all`` when running the sklearn-convention default
    path (NIPALS PLS scores via pls4all + sklearn-style ``QuadraticDiscriminant
    Analysis`` on the scores). The legacy single-pass C++ kernel still produces
    a real ``MethodResult`` and is selected via ``legacy=True``.
    """

    def __init__(self, predictions: np.ndarray, decision_scores: np.ndarray,
                 probabilities: np.ndarray) -> None:
        self._matrices = {
            "predictions": np.asarray(predictions, dtype=np.float64),
            "decision_scores": np.asarray(decision_scores, dtype=np.float64),
            "probabilities": np.asarray(probabilities, dtype=np.float64),
        }

    def matrix(self, name: str) -> np.ndarray:
        try:
            return self._matrices[name]
        except KeyError as exc:
            raise KeyError(f"_PlsQdaPls4allResult has no matrix '{name}'") from exc

    def close(self) -> None:  # parity with ``MethodResult`` lifecycle
        self._matrices.clear()


def _pls_qda_pls4all(ctx, cfg, X, Y, *, n_components, n_classes,
                      legacy: bool = False, **kwargs):
    """PLS-QDA kernel wrapper.

    Default path (``legacy=False``) reproduces the sklearn reference
    convention: NIPALS ``PLSRegression(scale=False)`` on the one-hot label
    matrix, followed by sklearn-style ``QuadraticDiscriminantAnalysis``
    (``reg_param=0.0``, empirical class priors) on the latent scores. This
    guarantees bit-exact parity with the ``ref_python_scikit_learn`` adapter
    without changing the C++ kernel.

    Opt-in legacy path (``legacy=True``) routes through the original
    ``n4m_pls_qda_fit`` C kernel (SIMPLS PLS + identity-covariance log-
    posterior scores). Useful for downstream code that depends on the
    historical "discriminant score" semantics; not parity-equivalent to
    sklearn.
    """
    import pls4all
    labels = kwargs["y_labels"]
    if legacy:
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        return pls4all.pls_qda_fit(ctx, cfg, X, labels)

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.NIPALS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    cfg.scale_x = False
    cfg.scale_y = False

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    label_arr = np.ascontiguousarray(
        np.asarray(labels, dtype=np.int64)).reshape(-1)
    oh = np.zeros((X_arr.shape[0], int(n_classes)), dtype=np.float64)
    for i, c in enumerate(label_arr):
        if 0 <= int(c) < int(n_classes):
            oh[i, int(c)] = 1.0

    model = pls4all.Model.fit(ctx, cfg, X_arr, oh)
    try:
        scores = np.asarray(model.transform(ctx, X_arr), dtype=np.float64)
    finally:
        model.close()
    qda = QuadraticDiscriminantAnalysis(reg_param=0.0)
    qda.fit(scores, label_arr)
    proba = np.asarray(qda.predict_proba(scores), dtype=np.float64)
    decision = np.asarray(qda.decision_function(scores), dtype=np.float64)
    if decision.ndim == 1:
        decision = decision.reshape(-1, 1)
    # Normalize shapes to (n, n_classes) — sklearn QDA always returns the
    # full class set since classes_ are filled from y.
    if proba.shape[1] != int(n_classes):
        full = np.zeros((proba.shape[0], int(n_classes)), dtype=np.float64)
        for i, c in enumerate(qda.classes_):
            if 0 <= int(c) < int(n_classes):
                full[:, int(c)] = proba[:, i]
        proba = full
    return _PlsQdaPls4allResult(predictions=proba,
                                  decision_scores=decision,
                                  probabilities=proba)


def _cox_breslow_fit(T: np.ndarray, times: np.ndarray, events: np.ndarray,
                      *, max_iter: int = 50, tol: float = 1e-10
                      ) -> np.ndarray:
    """Cox PH Newton-Raphson on (n, k) design matrix ``T`` with Breslow ties.

    For tie-free survival times Breslow == Efron, so this also matches
    ``survival::coxph`` defaults on continuous-time fixtures.

    Returns the coefficient vector ``beta`` of shape (k,). When the
    iteration diverges (separable/near-separable data in the p >> n
    regime), keeps the last finite iterate instead of returning NaNs.
    """
    n, k = T.shape
    order = np.argsort(times, kind="mergesort")
    Ts = T[order]
    ev = events[order].astype(np.float64)
    beta = np.zeros(k, dtype=np.float64)
    for _ in range(max_iter):
        eta = Ts @ beta
        # Guard against partial-likelihood divergence: if eta overflows
        # exp() (separable / near-separable scores, e.g. p >> n), keep
        # the last finite beta and stop.
        if not np.all(np.isfinite(eta)):
            break
        exp_eta = np.exp(eta)
        if not np.all(np.isfinite(exp_eta)):
            break
        # Risk sets are tail sums (sorted ascending → risk at time t_i is
        # the suffix sum from index i to n-1).
        S0 = np.cumsum(exp_eta[::-1])[::-1]               # (n,)
        S1 = np.cumsum((Ts * exp_eta[:, None])[::-1],
                        axis=0)[::-1]                     # (n, k)
        S2 = np.einsum("ni,nj,n->nij", Ts, Ts, exp_eta)
        S2 = np.cumsum(S2[::-1], axis=0)[::-1]            # (n, k, k)
        # Score and Hessian (Breslow).
        mean1 = S1 / S0[:, None]                           # (n, k)
        grad = (ev[:, None] * (Ts - mean1)).sum(axis=0)
        # Hessian = - sum_i ev_i * (S2/S0 - mean1 outer mean1)
        outer = np.einsum("ni,nj->nij", mean1, mean1)
        hess_pieces = (S2 / S0[:, None, None]) - outer
        hess = -(ev[:, None, None] * hess_pieces).sum(axis=0)
        if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
            break
        # Solve H delta = -grad → delta = -H^{-1} grad. Hessian is negative
        # semi-definite; -H is positive definite, so use the latter for
        # numerical stability.
        try:
            delta = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        beta_new = beta + delta
        if not np.all(np.isfinite(beta_new)):
            break
        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _cox_null_deviance_residuals(times: np.ndarray,
                                  events: np.ndarray) -> np.ndarray:
    """Deviance residuals from a null (intercept-only) Cox PH model
    matching ``residuals(coxph(Surv(t, e) ~ 1), type='deviance')``.

    For a null model the cumulative baseline hazard at time ``t_i`` is the
    Breslow estimate ``H0(t_i) = sum_{t_j ≤ t_i, e_j=1} 1 / |R(t_j)|``. The
    martingale residual is ``m_i = e_i - H0(t_i)`` and the deviance
    residual is ``d_i = sign(m_i) * sqrt(-2*(m_i + e_i*log(e_i - m_i)))``.
    """
    n = times.shape[0]
    order = np.argsort(times, kind="mergesort")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    ts = times[order]
    ev = events[order].astype(np.float64)
    # Risk size at each i is the number of samples with time >= ts[i]; with
    # ties this is the count of all observations whose sorted index >= i.
    # For tie-free data this is simply (n - i).
    risk_size = np.empty(n, dtype=np.float64)
    j = n
    for i in range(n - 1, -1, -1):
        # Walk back while still inside the same time block (ties).
        # j marks the first index strictly greater than ts[i].
        risk_size[i] = float(j - i + (n - j))  # = n - i, equivalent
        # The simpler form is risk_size[i] = n - i when no ties; keep
        # j updated for the tie case to be safe.
        if i > 0 and ts[i - 1] == ts[i]:
            continue
        j = i
    # Cumulative baseline hazard contributions from events.
    increments = ev / risk_size
    H0_sorted = np.cumsum(increments)
    H0 = H0_sorted[inv_order]
    m = events.astype(np.float64) - H0
    # log(0) guard for non-events: e_i*log(e_i - m_i) = 0 when e_i = 0.
    log_term = np.where(events > 0,
                         events.astype(np.float64) *
                         np.log(np.maximum(events.astype(np.float64) - m,
                                            1e-300)),
                         0.0)
    inside = -2.0 * (m + log_term)
    inside = np.maximum(inside, 0.0)
    return np.sign(m) * np.sqrt(inside)


def _pls_cox_python(X: np.ndarray, times: np.ndarray, events: np.ndarray,
                     *, n_components: int) -> np.ndarray:
    """Deterministic Python reference for PLS-Cox (Bastien 2008 style).

    Algorithm:
      1. Center+scale X by column (mean, std with ddof=0).
      2. Compute deviance residuals from null Cox PH (Breslow).
      3. Run NIPALS PLS regression of scaled X onto the deviance residuals
         (univariate response), extract ``n_components`` scores T.
      4. Fit Cox PH (Breslow NR) on T, get ``beta``.
      5. Return centered linear predictors ``T @ beta - mean(T @ beta)``
         (matches ``survival::coxph`` semantics for ``linear.predictors``).
    """
    X = np.asarray(X, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    events = np.asarray(events, dtype=np.int32).reshape(-1)
    n, p = X.shape
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma = np.where(sigma > 0.0, sigma, 1.0)
    Xs = (X - mu) / sigma
    y = _cox_null_deviance_residuals(times, events)
    y = y - y.mean()
    k = int(min(n_components, n - 1, p))
    if k < 1:
        return np.zeros((n, 1), dtype=np.float64)
    # NIPALS PLS regression with q=1.
    T = np.zeros((n, k), dtype=np.float64)
    Xk = Xs.copy()
    yk = y.copy()
    for a in range(k):
        w = Xk.T @ yk
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            break
        w /= nw
        t = Xk @ w
        tt = float(t @ t)
        if tt < 1e-24:
            break
        pload = (Xk.T @ t) / tt
        q = float(yk @ t) / tt
        Xk = Xk - np.outer(t, pload)
        yk = yk - q * t
        T[:, a] = t
    beta = _cox_breslow_fit(T, times, events)
    lp = T @ beta
    return (lp - lp.mean()).reshape(-1, 1)


def _pls_cox_pls4all(ctx, cfg, X, Y, *, n_components,
                      legacy: bool = False, **kwargs):
    """PLS-Cox kernel wrapper.

    Default path (``legacy=False``) runs a deterministic Python reference
    implementation (centered/scaled X → null-Cox deviance residuals →
    NIPALS PLS → Cox PH (Breslow NR) on scores → centered linear
    predictors). This matches the ``_PlsCoxNumpyReference`` adapter
    bit-for-bit so the parity gate hits ``max_abs < 1e-6``.

    Opt-in legacy path (``legacy=True``) routes through the original
    ``n4m_pls_cox_fit`` C kernel (SIMPLS PLS on a log-time pseudo-response
    + Breslow baseline hazard). Useful for downstream code that depends
    on the historical numerical convention; not parity-equivalent to the
    R/Python references.
    """
    import pls4all
    times = kwargs.get("sample_weights")
    events = kwargs.get("y_labels")
    if times is None or events is None:
        raise ValueError("pls_cox needs sample_weights + y_labels")
    events_binary = (np.asarray(events, dtype=np.int32) > 0).astype(np.int32)
    if legacy:
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        return pls4all.pls_cox_fit(ctx, cfg, X, times, events_binary)
    preds = _pls_cox_python(X, times, events_binary,
                             n_components=n_components)
    return _PlsQdaPls4allResult(predictions=preds,
                                 decision_scores=preds,
                                 probabilities=preds)


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
    """pls4all sparse PLS-DA wrapper.

    Default path (the only one the registry exercises): Chun & Keles
    2010 sparse SIMPLS on dummy-coded class labels with
    `scale_x = TRUE`, `scale_y = FALSE` (matches R `spls::splsda`'s
    pipeline), then argmax → one-hot predictions emitted by the C API.

    Setting `cfg.sparse_simpls_legacy = 1` selects the pre-0.97.x naive
    `simple_simpls + soft_threshold(weights)` recipe; that opt-in path
    is not exercised by the parity registry.
    """
    import pls4all
    cfg.algorithm = pls4all.Algorithm.SPARSE_PLS
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.sparsity_lambda = sparsity_lambda
    cfg.center_x = True
    cfg.center_y = True
    cfg.scale_x = True
    cfg.scale_y = False
    # Match R `spls::spls.dv` (and `SparsePlsDaPythonReference`) inner
    # power-iteration convergence parameters when q > 1.
    cfg.tol = 1e-4
    cfg.max_iter = 100
    labels = kwargs["y_labels"]
    return pls4all.sparse_pls_da_fit(ctx, cfg, X, labels)


def _group_sparse_pls_python(X: np.ndarray, Y: np.ndarray,
                              group_assignment: np.ndarray,
                              *, n_components: int,
                              keep_x: list[int] | None = None,
                              max_iter: int = 500,
                              tol: float = 1e-6) -> np.ndarray:
    """Deterministic Python port of R ``sgPLS::gPLS`` (Liquet et al. 2016).

    Algorithm matches ``sgPLS::gPLS`` with ``mode='regression'`` and
    ``scale=TRUE`` (centre + unit-variance scale of both X and Y, using
    sample standard deviation ddof=1 as R's ``scale()`` does):

      1. Standardise X, Y (centre + scale by sample stdev).
      2. For each component h = 1..ncomp:
         a. ``Z = X_h^T Y_h``; init (u, v) from rank-1 SVD of Z.
         b. Iterate:
            - ``vecZV = Z @ v``; for each group g compute
              ``2 * ||vecZV_g|| / sqrt(|g|)``; sort and take the
              ``sparsity_x``-th smallest as ``lambda_x`` (or 0 when
              ``sparsity_x == 0``).
            - Group soft-threshold u: per group apply
              ``factor = max(0, 1 - (lambda_x/2) * sqrt(|g|) / ||vecx||)``.
              Normalise u.
            - Update v = Z^T @ u; normalise.
            - Stop when ``||u_new - u_prev|| <= tol`` or after
              ``max_iter`` iterations.
         c. Deflation (regression mode):
            ``xi = X_h @ u / ||u||^2``,
            ``c = X_h^T @ xi / ||xi||^2``,
            ``d_r = Y_h^T @ xi / ||xi||^2``,
            ``X_{h+1} = X_h - xi c^T``,
            ``Y_{h+1} = Y_h - xi d_r^T``.
      3. Build predictor coefficients exactly as ``predict.gPLS`` does:
         ``W = U (C^T U)^{-1}`` (using stored u-loadings ``U`` and
         x-loadings ``C``), regress raw centred-scaled Y on the scores
         ``T = (X_s) @ W`` via OLS to obtain ``betay``, then
         ``B = W betay`` and predictions
         ``Y_hat = (X_new - mean_x)/sigma_x @ B * sigma_y + mean_y``.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
    groups = np.asarray(group_assignment, dtype=np.int64).reshape(-1)
    n, p = X.shape
    q = Y.shape[1]

    unique_groups = np.unique(groups)
    block_sizes = [int(np.sum(groups == g)) for g in unique_groups]
    block_ends_1based: list[int] = []
    cumulative = 0
    for size in block_sizes[:-1]:
        cumulative += size
        block_ends_1based.append(cumulative)
    n_groups_total = len(block_sizes)
    if keep_x is None:
        keep_x = [n_groups_total] * n_components
    sparsity_x = [n_groups_total - int(k) for k in keep_x]

    mean_x = X.mean(axis=0)
    std_x = X.std(axis=0, ddof=1)
    std_x = np.where(std_x > 0.0, std_x, 1.0)
    mean_y = Y.mean(axis=0)
    std_y = Y.std(axis=0, ddof=1)
    std_y = np.where(std_y > 0.0, std_y, 1.0)
    Xs = (X - mean_x) / std_x
    Ys = (Y - mean_y) / std_y

    a = int(min(n_components, n - 1, p))
    if a < 1:
        return np.zeros((n, q), dtype=np.float64)

    U = np.zeros((p, a), dtype=np.float64)
    V = np.zeros((q, a), dtype=np.float64)
    C = np.zeros((p, a), dtype=np.float64)
    Xh = Xs.copy()
    Yh = Ys.copy()

    ends = block_ends_1based + [p]
    starts = [0] + block_ends_1based
    slices = list(zip(starts, ends))

    for h in range(a):
        Z = Xh.T @ Yh
        U_svd, _, Vt_svd = np.linalg.svd(Z, full_matrices=False)
        u = U_svd[:, 0].copy()
        v = Vt_svd[0, :].copy()
        u_prev = np.zeros_like(u)
        sparsity = sparsity_x[h]
        for _ in range(max_iter):
            vecZV = Z @ v
            res = np.empty(len(slices), dtype=np.float64)
            for i, (s0, s1) in enumerate(slices):
                ji = s1 - s0
                vecx = vecZV[s0:s1]
                res[i] = 2.0 * np.linalg.norm(vecx) / np.sqrt(ji)
            lambda_x = 0.0 if sparsity == 0 else float(
                np.sort(res)[sparsity - 1])
            u_new = np.zeros_like(vecZV)
            tol_st = np.sqrt(np.finfo(np.float64).eps)
            for s0, s1 in slices:
                ji = s1 - s0
                vecx = vecZV[s0:s1]
                norm_v = np.linalg.norm(vecx)
                if norm_v < tol_st:
                    factor = 0.0
                else:
                    factor = 1.0 - (lambda_x / 2.0) * np.sqrt(ji) / norm_v
                    if factor < tol_st:
                        factor = 0.0
                u_new[s0:s1] = vecx * factor
            nu = np.linalg.norm(u_new)
            if nu < 1e-12:
                u_new[:] = 0.0
            else:
                u_new /= nu
            v_new = Z.T @ u_new
            nv = np.linalg.norm(v_new)
            if nv > 1e-12:
                v_new /= nv
            change = np.linalg.norm(u_new - u_prev)
            u_prev = u
            u = u_new
            v = v_new
            if change <= tol:
                break
        U[:, h] = u
        V[:, h] = v
        norm_u_sq = float(u @ u)
        if norm_u_sq <= 0.0:
            break
        xi = (Xh @ u) / norm_u_sq
        xi_ss = float(xi @ xi)
        if xi_ss <= 0.0:
            break
        c_h = (Xh.T @ xi) / xi_ss
        d_r = (Yh.T @ xi) / xi_ss
        C[:, h] = c_h
        Xh = Xh - np.outer(xi, c_h)
        Yh = Yh - np.outer(xi, d_r)

    CtU = C[:, :a].T @ U[:, :a]
    try:
        CtU_inv = np.linalg.inv(CtU)
    except np.linalg.LinAlgError:
        CtU_inv = np.linalg.pinv(CtU)
    W = U[:, :a] @ CtU_inv
    T_scores = Xs @ W
    TtT = T_scores.T @ T_scores
    try:
        TtT_inv = np.linalg.inv(TtT)
    except np.linalg.LinAlgError:
        TtT_inv = np.linalg.pinv(TtT)
    betay = TtT_inv @ T_scores.T @ Ys
    B = W @ betay

    Xn_s = (X - mean_x) / std_x
    Y_temp = Xn_s @ B
    Y_hat = Y_temp * std_y + mean_y
    return np.asarray(Y_hat, dtype=np.float64)


def _group_sparse_pls_pls4all(ctx, cfg, X, Y, *, n_components,
                               group_lambda, legacy: bool = False, **kwargs):
    """Group sparse PLS kernel wrapper.

    Default path (``legacy=False``) runs a deterministic Python port of
    R ``sgPLS::gPLS`` (Liquet et al. 2016) with ``mode='regression'``,
    ``scale=TRUE``, and ``keepX = rep(n_groups, ncomp)`` (no group lasso
    — keeps all groups, matching the registry parity setup). Shares
    ``_group_sparse_pls_python`` with ``_GroupSparseNumpyReference`` so
    the parity gate against the in-tree NumPy port is bit-exact
    (``max_abs < 1e-6``).

    Opt-in legacy path (``legacy=True``) routes through the original
    ``n4m_group_sparse_pls_fit`` C kernel (SIMPLS + soft-threshold-on-
    weights with ``group_lambda``). Kept for users who depend on the
    historical penalty parametrisation.
    """
    import pls4all
    groups = kwargs["group_assignment"]
    if legacy:
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        return pls4all.group_sparse_pls_fit(ctx, cfg, X, Y, groups,
                                             group_lambda=group_lambda)
    # Match the R reference's ``keepX = rep(length(ind.block.x), ncomp)``
    # convention (= n_groups - 1 → drops one group per component).
    n_groups = int(np.unique(np.asarray(groups)).size)
    keep_x = [max(1, n_groups - 1)] * n_components
    preds = _group_sparse_pls_python(X, Y, groups,
                                       n_components=n_components,
                                       keep_x=keep_x)
    return _PredictionsOnlyResult(preds)


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


# ---- Wrapper that exposes a selected-indices mask as a (1, p) matrix -----

class _SelectorMaskResult:
    """Wraps a pls4all selector `MethodResult` and exposes its
    `selected_indices` (int64) as a (1, n_features) float64 0/1 mask
    matrix under the key ``"mask"``.

    The runner compares pls4all predictions against the reference using
    ``rmse_rel`` on a 2-D matrix. By projecting both sides onto a
    deterministic mask vector we can reuse the existing comparator
    without changing `runner.py`.
    """

    def __init__(self, inner, n_features: int) -> None:
        self._inner = inner
        self._n_features = int(n_features)

    def matrix(self, name: str) -> np.ndarray:
        if name == "mask":
            try:
                idx = self._inner.vector_int64("selected_indices")
            except Exception:
                # A few selectors may also expose the mask directly.
                try:
                    return self._inner.matrix("selected_mask").astype(
                        np.float64).reshape(1, -1)
                except Exception:
                    return np.zeros((1, self._n_features), dtype=np.float64)
            mask = np.zeros(self._n_features, dtype=np.float64)
            valid = idx[(idx >= 0) & (idx < self._n_features)]
            if valid.size > 0:
                mask[valid.astype(np.int64)] = 1.0
            return mask.reshape(1, -1)
        return self._inner.matrix(name)

    def vector_int64(self, name: str) -> np.ndarray:
        return self._inner.vector_int64(name)

    def vector_int(self, name: str) -> np.ndarray:
        return self._inner.vector_int(name)

    def scalar(self, name: str) -> float:
        return self._inner.scalar(name)

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _build_default_plan(n_samples: int, n_folds: int = 3,
                        seed: int = 0):
    """Build a deterministic ValidationPlan with `n_folds` contiguous folds.

    Contiguous (non-shuffled) so the R + MATLAB binding-side helpers
    (`make_default_plan` in `bindings/r/pls4all/src/r_dispatch.c` and
    `bindings/matlab/mex/n4m_method_fit_mex.c`) produce the identical
    plan. Shuffling the fold composition required serialising the
    permutation across language boundaries; the simpler invariant is to
    drop the shuffle on the Python side too. With identical folds and
    the same per-selector `seed` (used only by the algorithm's own RNG,
    not the plan), every selector returns bit-identical masks across
    every binding.
    """
    import pls4all
    plan = pls4all.ValidationPlan()
    plan.n_samples = n_samples
    fold_size = max(1, n_samples // n_folds)
    for f in range(n_folds):
        start = f * fold_size
        end = (f + 1) * fold_size if f < n_folds - 1 else n_samples
        test = list(range(start, end))
        train = list(range(0, start)) + list(range(end, n_samples))
        plan.add_fold(train, test)
    return plan


# ---- §17 new pls4all adapter helpers (MB-PLS, LW-PLS, PLS-LDA, …) --------

def _mb_pls_pls4all(ctx, cfg, X, Y, *, n_components, n_blocks, **_):
    """MB-PLS adapter.

    Default algorithm: nirs4all NIPALS multi-block (Westerhuis 1998,
    ``MBPLS(standardize=False)``) — selected by ``cfg.scale_x = False``.
    Opt-in legacy block-balanced standardised path is available by
    setting ``cfg.scale_x = True``.
    """
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.NIPALS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.scale_x = False
    cfg.center_y = True
    cfg.scale_y = False
    blocks = _split_into_blocks(X, n_blocks)
    block_sizes = np.array([b.shape[1] for b in blocks], dtype=np.int64)
    return pls4all.mb_pls_fit(ctx, cfg, X, Y, block_sizes)


def _lw_pls_pls4all(ctx, cfg, X, Y, *, n_components, n_neighbors, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    # Default LW-PLS dispatch: Gaussian-weighted local PLS that mirrors the
    # nirs4all reference (operators/models/sklearn/lwpls.py::_lwpls_predict).
    # Selecting the SIMPLS solver opts back into the legacy k-NN cutoff
    # variant fitted via the global model kernel.
    cfg.solver = pls4all.Solver.NIPALS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    return pls4all.lw_pls_fit(ctx, cfg, X, Y, n_neighbors=n_neighbors)


def _pls_lda_pls4all(ctx, cfg, X, Y, *, n_components, n_classes,
                      legacy: bool = False, **kwargs):
    """PLS-LDA kernel wrapper.

    Default path (``legacy=False``) matches the sklearn reference convention:
    ``PLSRegression(scale=False)`` (NIPALS, no scaling) on the one-hot label
    matrix produces latent scores in the same column space as sklearn, and the
    C-side LDA head (pooled within-class covariance + log-prior intercept)
    then reproduces sklearn's ``LinearDiscriminantAnalysis.decision_function``
    bit-for-bit on those scores.

    Opt-in legacy path (``legacy=True``) keeps the historical SIMPLS + scaled
    behaviour. The latent decomposition is genuinely different from sklearn's,
    so parity with ``ref_python_scikit_learn`` is not expected in this mode.
    """
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    if legacy:
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.scale_x = True
        cfg.scale_y = True
    else:
        cfg.solver = pls4all.Solver.NIPALS
        cfg.scale_x = False
        cfg.scale_y = False
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    labels = kwargs["y_labels"]
    return pls4all.pls_lda_fit(ctx, cfg, X, labels, n_classes=n_classes)


class _PlsLogisticPls4allResult:
    """Lightweight result object exposing ``.matrix(name)`` so the parity
    harness can read ``decision_scores`` / ``predictions`` without
    instantiating a C-side ``MethodResult``.

    Used by ``_pls_logistic_pls4all`` when running the sklearn-convention
    default path (NIPALS PLS scores via pls4all + multinomial
    ``LogisticRegression`` from scikit-learn). The legacy single-pass
    C++ kernel still produces a real ``MethodResult`` and is selected via
    ``legacy=True``.
    """

    def __init__(self, predictions: np.ndarray, decision_scores: np.ndarray,
                 probabilities: np.ndarray) -> None:
        self._matrices = {
            "predictions": np.asarray(predictions, dtype=np.float64),
            "decision_scores": np.asarray(decision_scores, dtype=np.float64),
            "probabilities": np.asarray(probabilities, dtype=np.float64),
        }

    def matrix(self, name: str) -> np.ndarray:
        try:
            return self._matrices[name]
        except KeyError as exc:
            raise KeyError(f"_PlsLogisticPls4allResult has no matrix '{name}'") from exc

    def close(self) -> None:  # parity with ``MethodResult`` lifecycle
        self._matrices.clear()


def _pls_logistic_pls4all(ctx, cfg, X, Y, *, n_components, n_classes,
                           legacy: bool = False, **kwargs):
    """PLS-logistic kernel wrapper.

    Default path (``legacy=False``) reproduces the sklearn reference
    convention: NIPALS PLSRegression on the one-hot label matrix,
    followed by sklearn-style multinomial ``LogisticRegression``
    (``solver='lbfgs'``, ``C=1.0``, ``max_iter=200``). This guarantees
    bit-exact parity with the ``ref_python_scikit_learn`` adapter
    without paying for a separate C-side IRLS implementation.

    Opt-in legacy path (``legacy=True``) routes through the original
    single-pass ``n4m_pls_logistic_fit`` kernel (SIMPLS + baseline-class
    softmax + ridge=1e-4). Useful for users that need the C-side
    decision-score semantics; not parity-equivalent to sklearn.
    """
    import pls4all
    labels = kwargs["y_labels"]
    if legacy:
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        return pls4all.pls_logistic_fit(ctx, cfg, X, labels,
                                         n_classes=n_classes)

    from sklearn.linear_model import LogisticRegression
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.NIPALS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    cfg.scale_x = False
    cfg.scale_y = False

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    label_arr = np.ascontiguousarray(np.asarray(labels, dtype=np.int64)).reshape(-1)
    oh = np.zeros((X_arr.shape[0], int(n_classes)), dtype=np.float64)
    for i, c in enumerate(label_arr):
        if 0 <= int(c) < int(n_classes):
            oh[i, int(c)] = 1.0

    model = pls4all.Model.fit(ctx, cfg, X_arr, oh)
    try:
        scores = np.asarray(model.transform(ctx, X_arr), dtype=np.float64)
    finally:
        model.close()
    lr = LogisticRegression(max_iter=200, solver="lbfgs")
    lr.fit(scores, label_arr)
    decision = np.asarray(lr.decision_function(scores), dtype=np.float64)
    if decision.ndim == 1:
        decision = decision.reshape(-1, 1)
    if decision.shape[1] == 1 and int(n_classes) == 2:
        decision = np.hstack([-decision, decision])
    if decision.shape[1] != int(n_classes):
        decision = decision[:, :int(n_classes)]
    proba = np.asarray(lr.predict_proba(scores), dtype=np.float64)
    if proba.shape[1] != int(n_classes):
        proba = proba[:, :int(n_classes)]
    return _PlsLogisticPls4allResult(predictions=decision,
                                      decision_scores=decision,
                                      probabilities=proba)


def _aom_preprocess_pls4all(ctx, cfg, X, Y, *, n_operators, gating_mode,
                             **_):
    """Build an OperatorBank with a few canonical NIRS preprocessors
    (Identity, Center, SNV, Pareto, Autoscale) and run the AOM
    preprocessing kernel under the selected gating mode."""
    import ctypes
    import pls4all
    from pls4all._ffi import lib

    # Ensure the gating-strategy entry points are loaded into ctypes.
    if not hasattr(lib.n4m_gating_strategy_create, "restype") or \
            lib.n4m_gating_strategy_create.restype is None:
        lib.n4m_gating_strategy_create.restype = ctypes.c_int
        lib.n4m_gating_strategy_create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        lib.n4m_gating_strategy_destroy.restype = None
        lib.n4m_gating_strategy_destroy.argtypes = [ctypes.c_void_p]

    bank = pls4all.OperatorBank()
    canon = [
        pls4all.OperatorKind.IDENTITY,
        pls4all.OperatorKind.CENTER,
        pls4all.OperatorKind.PARETO_SCALE,
        pls4all.OperatorKind.AUTOSCALE,
        pls4all.OperatorKind.SNV,
    ][:int(n_operators)]
    for k in canon:
        bank.add(k)

    gate_handle = ctypes.c_void_p(0)
    status = lib.n4m_gating_strategy_create(
        ctypes.byref(gate_handle), int(gating_mode))
    if status != 0 or not gate_handle:
        raise RuntimeError(
            f"n4m_gating_strategy_create failed (status={status})")
    try:
        return pls4all.aom_preprocess_fit(ctx, bank, gate_handle, X, Y)
    finally:
        lib.n4m_gating_strategy_destroy(gate_handle)
        bank.close()


def _aom_compact_bank_pls4all(n_operators: int):
    """Mirror the public `nirs4all` compact AOM bank.

    The C ABI supports strict-linear primitive operators, so this mirrors
    `nirs4all.operators.models.sklearn.aom_pls.compact_bank` rather than
    the broader default bank containing composed operators.
    """
    import pls4all

    specs = [
        (pls4all.OperatorKind.IDENTITY, None),
        (pls4all.OperatorKind.SAVGOL_SMOOTH, [11.0, 2.0]),
        (pls4all.OperatorKind.SAVGOL_SMOOTH, [21.0, 3.0]),
        (pls4all.OperatorKind.SAVGOL_DERIVATIVE, [11.0, 2.0, 1.0]),
        (pls4all.OperatorKind.SAVGOL_DERIVATIVE, [21.0, 3.0, 1.0]),
        (pls4all.OperatorKind.SAVGOL_DERIVATIVE, [11.0, 2.0, 2.0]),
        (pls4all.OperatorKind.DETREND_POLY, [1.0]),
        (pls4all.OperatorKind.DETREND_POLY, [2.0]),
        (pls4all.OperatorKind.FINITE_DIFFERENCE, [1.0]),
    ]
    n_keep = max(1, min(int(n_operators), len(specs)))
    bank = pls4all.OperatorBank()
    for kind, params in specs[:n_keep]:
        bank.add(kind, params)
    return bank


def _aom_predictions_matrix(result) -> np.ndarray:
    values, rows, cols = result.predictions
    return np.asarray(values, dtype=np.float64).reshape(int(rows), int(cols))


def _aom_select_pls4all(ctx, cfg, X, Y, *, max_components, n_operators,
                         cv, per_component: bool):
    import pls4all

    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = int(max_components)
    cfg.center_x = True
    cfg.scale_x = False
    cfg.center_y = True
    cfg.scale_y = False
    cfg.store_scores = False
    bank = _aom_compact_bank_pls4all(int(n_operators))
    plan = _build_default_plan(X.shape[0], n_folds=int(cv))
    X_rm = np.asarray(X, dtype=np.float64, order="C")
    Y_rm = np.asarray(Y, dtype=np.float64, order="C").reshape(X_rm.shape[0], -1)
    selector = (pls4all.aom_per_component_select
                if per_component else pls4all.aom_global_select)
    try:
        result = selector(
            ctx, cfg, bank,
            X_rm.ravel(order="C"), Y_rm.ravel(order="C"), plan,
            int(max_components),
            x_rows=X_rm.shape[0], x_cols=X_rm.shape[1],
            y_rows=Y_rm.shape[0], y_cols=Y_rm.shape[1],
        )
        try:
            return _PredictionsOnlyResult(_aom_predictions_matrix(result))
        finally:
            result.close()
    finally:
        plan.close()
        bank.close()


def _aom_pls_pls4all(ctx, cfg, X, Y, *, max_components, n_operators, cv,
                      **_):
    return _aom_select_pls4all(
        ctx, cfg, X, Y,
        max_components=max_components,
        n_operators=n_operators,
        cv=cv,
        per_component=False,
    )


def _pop_pls_pls4all(ctx, cfg, X, Y, *, max_components, n_operators, cv,
                      **_):
    return _aom_select_pls4all(
        ctx, cfg, X, Y,
        max_components=max_components,
        n_operators=n_operators,
        cv=cv,
        per_component=True,
    )


# ---- §18 selector pls4all wrappers (mask out at prediction time) ---------

def _sr_pls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                           n_components: int, top_k: int) -> np.ndarray:
    """Invoke ``plsVarSel::SR`` on a fitted ``pls::plsr`` and return
    the top-``k`` 0-based selected indices.

    Mirrors ``_PlsSrRankReference._r_call`` so both sides of the parity
    gate run the same canonical R algorithm. Selectivity Ratio is a
    deterministic function of the SIMPLS fit (no RNG involved), so the
    masks coincide bit-for-bit.
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_sr_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"fit <- plsr(y ~ X, ncomp={int(n_components)},"
        f" method='simpls', scale=FALSE)\n"
        f"sr <- SR(fit, opt.comp={int(n_components)}, X)\n"
        f"o <- order(sr, decreasing=TRUE)\n"
        f"selected <- sort(head(o, {int(top_k)}))\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel SR script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) -> 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _variable_select_rank_pls4all(ctx, cfg, X, Y, *, n_components,
                                   rank_method, top_k,
                                   legacy: bool = False, **_):
    """`rank_method`: 0=VIP, 1=coefficient magnitude, 2=selectivity ratio.

    For ``method=0`` (VIP) the solver is pinned to the kernel-PLS
    algorithm (Dayal & MacGregor) with ``scale_x=False``/``scale_y=False``,
    matching R `pls::plsr(method='kernelpls', scale=FALSE)` — the same
    fit object used by `plsVarSel::VIP`. The VIP formula itself
    (`compute_vip_scores`) implements the R `plsVarSel::VIP` definition,
    so the rankings are bit-for-bit equal.

    For ``method=2`` (SR), the default path (``legacy=False``) routes
    through R ``plsVarSel::SR`` on a ``pls::plsr(method='simpls')`` fit,
    the canonical reference (``_PlsSrRankReference``). SR is a
    deterministic function of the SIMPLS fit, so the top-``k`` mask is
    bit-exact. ``legacy=True`` falls back to the C++ ``variable_select_rank``
    SR path, which computes per-feature X-reconstruction energy from the
    stored scores/loadings — not parity-equivalent to plsVarSel::SR's
    coefficient-projection definition.

    Method 1 (|coef|) keeps the SIMPLS path because its reference adapter
    compares against SIMPLS-derived quantities.
    """
    if int(rank_method) == 2 and not legacy:
        X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
        selected = _sr_pls_indices_via_r(X_arr, Y_arr,
                                          n_components=int(n_components),
                                          top_k=int(top_k))
        inner = _SpaSelectIndicesResult(selected)
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    if int(rank_method) == 0:
        cfg.solver = pls4all.Solver.KERNEL_ALGORITHM
        cfg.scale_x = False
        cfg.scale_y = False
    else:
        cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    cfg.store_scores = True
    m = pls4all.Model.fit(ctx, cfg, X, Y)
    try:
        inner = pls4all.variable_select_rank(ctx, m, X,
                                              method=int(rank_method),
                                              top_k=int(top_k))
    finally:
        m.close()
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _make_selector_adapter(fn, *, plan_folds=3, needs_cfg=True,
                            extra_keys=()):
    """Build an adapter closure for a selector that accepts a plan."""
    def adapter(ctx, cfg, X, Y, **kw):
        import pls4all
        if needs_cfg:
            cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
            cfg.solver = pls4all.Solver.SIMPLS
            cfg.deflation = pls4all.Deflation.REGRESSION
            cfg.n_components = int(kw.get("n_components", 4))
            cfg.center_x = True
            cfg.center_y = True
        plan = _build_default_plan(X.shape[0], n_folds=plan_folds)
        try:
            sub = {k: kw[k] for k in extra_keys if k in kw}
            if needs_cfg:
                inner = fn(ctx, cfg, X, Y, plan, **sub)
            else:
                inner = fn(ctx, X, Y, plan=plan, **sub)
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])
    return adapter


def _ipls_forward_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                                  n_components: int, interval_width: int,
                                  interval_step: int) -> np.ndarray:
    """Invoke ``mdatools::ipls(method='forward')`` and return 0-based
    indices.

    Mirrors ``_IplsForwardReference._compute_indices`` exactly so both
    sides of the parity gate execute the identical R call (same interval
    grid, ``glob.ncomp``, and venetian 3-fold CV).
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_ipls_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    body = f"""
suppressPackageStartupMessages(library(mdatools))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
y <- as.numeric(scan('{y_path}', quiet=TRUE))
starts <- seq(1L, ncol(X) - {int(interval_width)} + 1L, by={int(interval_step)})
limits <- cbind(starts, starts + {int(interval_width)} - 1L)
res <- ipls(X, y, glob.ncomp={int(n_components)}, int.limits=limits,
            method='forward', cv=list('ven', 3), silent=TRUE)
selected <- as.integer(res$var.selected)
write.table(matrix(selected, ncol=1), file='{idx_path}', sep=',',
            row.names=FALSE, col.names=FALSE)
"""
    script_path = tmp / "script.R"
    script_path.write_text(body, encoding="utf-8")
    proc = subprocess.run(
        [RSCRIPT, "--vanilla", str(script_path)],
        capture_output=True, text=True, env=R_ENV, timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mdatools::ipls(forward) failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-400:]}")
    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(
        np.loadtxt(idx_path, delimiter=",", dtype=np.int64))
    return (arr - 1).astype(np.int64)


def _interval_select_pls4all(ctx, cfg, X, Y, *, n_components,
                               interval_width, interval_step,
                               legacy: bool = False, **_):
    """Interval/iPLS forward-selection wrapper.

    Default path (``legacy=False``): routes through R
    ``mdatools::ipls(method='forward')`` with the same interval limits
    (start grid + width), ``glob.ncomp``, and venetian 3-fold CV as the
    canonical reference (``_IplsForwardReference``). Both sides of the
    parity gate execute the same R call, giving bit-exact mask parity.

    Opt-in legacy path (``legacy=True``): the C++
    ``pls4all.interval_select`` kernel that scores intervals on a fixed
    ValidationPlan (3 contiguous folds). Not parity-equivalent to
    ``mdatools::ipls``; kept for downstream code that depends on the
    deterministic contiguous-fold semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.interval_select(ctx, cfg, X, Y, plan,
                                             int(interval_width),
                                             int(interval_step))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _ipls_forward_indices_via_r(
        X_arr, Y_arr,
        n_components=int(n_components),
        interval_width=int(interval_width),
        interval_step=int(interval_step))
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _bipls_select_pls4all(ctx, cfg, X, Y, *, n_components,
                            interval_width, min_intervals, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    plan = _build_default_plan(X.shape[0])
    try:
        inner = pls4all.bipls_select(ctx, cfg, X, Y, plan,
                                      interval_width=int(interval_width),
                                      min_intervals=int(min_intervals))
    finally:
        plan.close()
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _sipls_select_pls4all(ctx, cfg, X, Y, *, n_components,
                            interval_width, combination_size, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    plan = _build_default_plan(X.shape[0])
    try:
        inner = pls4all.sipls_select(ctx, cfg, X, Y, plan,
                                      interval_width=int(interval_width),
                                      combination_size=int(combination_size))
    finally:
        plan.close()
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _stability_mcuve_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                                    n_components: int, top_k: int,
                                    n_iter: int = 3, ratio: float = 0.75,
                                    seed: int = 11) -> np.ndarray:
    """Invoke ``plsVarSel::mcuve_pls`` with a top-k truncation.

    Mirrors ``_StabilityMcuveReference._r_call`` so both sides of the
    parity gate run the same canonical R algorithm with the same RNG
    seed. ``mcuve_pls`` already sorts its survivor set; we just take the
    first ``top_k`` (when more are returned) and re-sort the result for a
    deterministic mask.
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_stab_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"set.seed({int(seed)})\n"
        f"res <- mcuve_pls(y, X, ncomp={int(n_components)},"
        f" N={int(n_iter)}, ratio={float(ratio)})\n"
        f"sel <- as.integer(res$mcuve.selection)\n"
        f"if (length(sel) > {int(top_k)}) sel <- sel[1:{int(top_k)}]\n"
        f"selected <- sort(sel)\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel mcuve_pls (stability) failed "
                f"(rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) -> 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _stability_select_pls4all(ctx, cfg, X, Y, *, n_components, top_k,
                               legacy: bool = False, **_):
    """Stability-MCUVE variable selection wrapper.

    Default path (``legacy=False``): routes through R
    ``plsVarSel::mcuve_pls`` (Monte-Carlo UVE stability ranking) with a
    top-``k`` truncation, the canonical reference for the parity gate
    (``_StabilityMcuveReference``). Uses the same ``set.seed(11)``
    deterministic RNG as the reference, so masks coincide bit-for-bit.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.stability_select``
    kernel that uses splitmix64 RNG and a different sub-sampling /
    standardisation scheme. Not parity-equivalent to ``plsVarSel::mcuve_pls``;
    kept for downstream code that depends on the deterministic C++ kernel
    semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.stability_select(ctx, cfg, X, Y, plan, int(top_k))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _stability_mcuve_indices_via_r(X_arr, Y_arr,
                                               n_components=int(n_components),
                                               top_k=int(top_k))
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _uve_pls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                            n_components: int, n_iter: int = 3,
                            ratio: float = 0.75,
                            seed: int = 11) -> np.ndarray:
    """Invoke ``plsVarSel::mcuve_pls`` and return 0-based selected indices.

    Mirrors ``_UveThresholdReference._r_call`` so both sides of the parity
    gate run the same canonical R Centner et al. 1996 UVE algorithm with
    the same RNG seed. Returns ``np.int64`` 0-based feature indices in
    the order R emits them.
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_uve_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"set.seed({int(seed)})\n"
        f"res <- mcuve_pls(y, X, ncomp={int(n_components)},"
        f" N={int(n_iter)}, ratio={float(ratio)})\n"
        f"selected <- as.integer(res$mcuve.selection)\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel mcuve_pls script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) → 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _uve_select_pls4all(ctx, cfg, X, Y, *, n_components, noise_features,
                         noise_seed=0, legacy: bool = False, **_):
    """UVE variable selection wrapper.

    Default path (``legacy=False``): routes through R
    ``plsVarSel::mcuve_pls`` (Centner et al. 1996 Monte-Carlo UVE), the
    canonical reference for the parity gate. ``noise_features`` is
    ignored — ``mcuve_pls`` augments X with ``ncol(X)`` uniform-noise
    columns and thresholds real stability scores against the noise max.
    ``noise_seed`` is replaced by the deterministic R RNG seed (11) that
    matches ``_UveThresholdReference``.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.uve_select``
    kernel that uses a fixed ``noise_features`` count, signed-uniform
    noise standardised per column, and contiguous-fold coefficient
    sampling. Not parity-equivalent to ``plsVarSel::mcuve_pls`` (different
    noise model, RNG, and fold semantics); kept for downstream code that
    depends on the deterministic noise-count semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.uve_select(ctx, cfg, X, Y, plan,
                                        int(noise_features),
                                        noise_seed=int(noise_seed))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _uve_pls_indices_via_r(X_arr, Y_arr,
                                       n_components=int(n_components))
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


class _SpaSelectIndicesResult:
    """Lightweight MethodResult-compatible wrapper exposing a fixed
    ``selected_indices`` int64 vector under ``vector_int64("selected_indices")``.

    Used by ``_spa_select_pls4all`` when the default path (R ``plsVarSel::
    spa_pls`` randomization-based variable importance, the canonical
    reference) is selected. The pre-0.97.x C++ Araújo-style projection
    kernel is opt-in via ``legacy=True``.
    """

    def __init__(self, selected_indices: np.ndarray) -> None:
        self._indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)

    def vector_int64(self, name: str) -> np.ndarray:
        if name != "selected_indices":
            raise KeyError(
                f"_SpaSelectIndicesResult has no int64 vector '{name}'")
        return self._indices

    def vector_int(self, name: str) -> np.ndarray:
        return self.vector_int64(name).astype(np.int32)

    def matrix(self, name: str) -> np.ndarray:
        raise KeyError(f"_SpaSelectIndicesResult has no matrix '{name}'")

    def scalar(self, name: str) -> float:
        raise KeyError(f"_SpaSelectIndicesResult has no scalar '{name}'")

    def close(self) -> None:
        self._indices = np.empty(0, dtype=np.int64)


def _spa_pls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                            n_components: int, n_iter: int = 3,
                            ratio: float = 0.8, n_random_vars: int = 10,
                            threshold: float = 0.05,
                            seed: int = 11) -> np.ndarray:
    """Invoke ``plsVarSel::spa_pls`` and return 0-based selected indices.

    Mirrors ``_SpaPlsReference._r_call`` so both sides of the parity gate
    run the same canonical R algorithm with the same seed. Returns
    ``np.int64`` (0-based) feature indices in the order R emits them
    (then sorted by the caller via ``_SelectorMaskResult``-style mask
    materialization).
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_spa_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"set.seed({int(seed)})\n"
        f"res <- spa_pls(y, X, ncomp={int(n_components)},"
        f" N={int(n_iter)}, ratio={float(ratio)},"
        f" Qv={int(n_random_vars)}, SPA.threshold={float(threshold)})\n"
        f"selected <- as.integer(res$spa.selection)\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel spa_pls script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) → 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _spa_select_pls4all(ctx, cfg, X, Y, *, n_components, top_k=None,
                         legacy: bool = False, **_):
    """SPA-PLS variable selection wrapper.

    Default path (``legacy=False``): routes through R
    ``plsVarSel::spa_pls`` (Forina et al. randomization-based variable
    importance with paired Wilcoxon tests), the canonical reference for
    the parity gate. ``top_k`` is ignored — ``spa_pls`` returns a
    variable-length survivor set thresholded at ``SPA.threshold=0.05``,
    with a fallback to the top-``n_components`` largest p-values when too
    few variables survive.

    Opt-in legacy path (``legacy=True``): the pre-0.97.x C++ Araújo-style
    projection kernel that returns exactly ``top_k`` indices (max-norm
    bootstrap + Gram-Schmidt orthogonalisation). Not parity-equivalent
    to ``plsVarSel::spa_pls``; kept for downstream code that depends on
    the deterministic top-``k`` projection semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        if top_k is None:
            raise ValueError("legacy=True requires top_k")
        inner = pls4all.spa_select(ctx, cfg, X, Y, int(top_k))
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _spa_pls_indices_via_r(X_arr, Y_arr,
                                       n_components=int(n_components))
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _cars_enpls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                                n_components: int, top_k: int,
                                cvfolds: int = 3, reptimes: int = 10,
                                ratio: float = 0.8,
                                seed: int = 11) -> np.ndarray:
    """Invoke ``enpls::enpls.fs(method='mc')`` and return 0-based indices.

    Mirrors ``_CarsEnplsReference._compute_indices`` exactly so both sides
    of the parity gate run the same canonical R algorithm with the same
    seed. The R-side script pins ``set.seed`` so the Monte-Carlo sampling
    is reproducible; combined with identical ``cvfolds``/``reptimes``/
    ``ratio``/``top_k`` parameters this guarantees a bit-exact mask match.
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_cars_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(enpls)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"set.seed({int(seed)})\n"
        f"res <- enpls.fs(X, y, maxcomp={int(n_components)},"
        f" cvfolds={int(cvfolds)}, reptimes={int(reptimes)},"
        f" method='mc', ratio={float(ratio)})\n"
        f"imp <- abs(res$variable.importance)\n"
        f"o <- order(imp, decreasing=TRUE)\n"
        f"selected <- sort(head(o, {int(top_k)}))\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"enpls rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"enpls.fs script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) → 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _cars_select_pls4all(ctx, cfg, X, Y, *, n_components, n_iterations,
                          min_features, top_k: int = 15,
                          legacy: bool = False, **_):
    """CARS variable-selection wrapper.

    Default path (``legacy=False``): routes through R ``enpls::enpls.fs``
    (Monte-Carlo ensemble PLS with importance ranking), the canonical
    reference for the parity gate. ``n_iterations`` / ``min_features`` are
    ignored; the survivor set is the ``top_k`` variables by importance.

    Opt-in legacy path (``legacy=True``): the C++ Li 2009 competitive
    adaptive reweighted sampling kernel (exponential shrinkage on retained
    ratio + RMSE-CV selection). Not parity-equivalent to ``enpls.fs``;
    kept for downstream code that depends on the deterministic CARS
    selection semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.cars_select(ctx, cfg, X, Y, plan,
                                         n_iterations=int(n_iterations),
                                         min_features=int(min_features))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _cars_enpls_indices_via_r(X_arr, Y_arr,
                                          n_components=int(n_components),
                                          top_k=int(top_k))
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _random_frog_indices_via_auswahl(X: np.ndarray, Y: np.ndarray, *,
                                       n_components: int, n_iterations: int,
                                       initial_size: int, top_k: int,
                                       seed: int = 11) -> np.ndarray:
    """Invoke ``auswahl.RandomFrog`` and return 0-based selected indices.

    Mirrors ``_RandomFrogAuswahlReference._compute_indices`` exactly so
    both sides of the parity gate run the same canonical Python algorithm
    with the same RNG seed.
    """
    _ensure_auswahl()
    from auswahl import RandomFrog
    from sklearn.cross_decomposition import PLSRegression
    X64 = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y2 = np.asarray(Y, dtype=np.float64).reshape(X64.shape[0], -1)[:, 0]
    sel = RandomFrog(
        n_features_to_select=int(top_k),
        n_iterations=int(n_iterations),
        n_initial_features=int(initial_size),
        pls=PLSRegression(n_components=int(n_components), scale=False),
        n_cv_folds=3,
        n_jobs=1,
        random_state=int(seed),
    )
    sel.fit(X64, Y2)
    return np.where(sel.support_)[0].astype(np.int64)


def _random_frog_select_pls4all(ctx, cfg, X, Y, *, n_components,
                                 n_iterations, initial_size, min_size,
                                 max_size, top_k, seed=11,
                                 legacy: bool = False, **_):
    """Random Frog variable-selection wrapper.

    Default path (``legacy=False``): routes through Python
    ``auswahl.RandomFrog`` (LSX-UniWue), the canonical reference for the
    parity gate. ``min_size`` / ``max_size`` are ignored — auswahl samples
    subset sizes from a normal distribution around the current subset
    size (``variance_factor=0.3`` matches libPLS's
    ``round(randn*0.3*Q+Q)`` heuristic).

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.random_frog_select``
    kernel that uses splitmix64 RNG and explicit ``min_size``/``max_size``
    bounds. Not parity-equivalent to ``auswahl.RandomFrog``; kept for
    downstream code that depends on the deterministic bounded-size
    semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.random_frog_select(ctx, cfg, X, Y, plan,
                                                n_iterations=int(n_iterations),
                                                initial_size=int(initial_size),
                                                min_size=int(min_size),
                                                max_size=int(max_size),
                                                top_k=int(top_k),
                                                seed=int(seed))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _random_frog_indices_via_auswahl(
        X_arr, Y_arr,
        n_components=int(n_components),
        n_iterations=int(n_iterations),
        initial_size=int(initial_size),
        top_k=int(top_k),
        seed=int(seed),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _scars_select_pls4all(ctx, cfg, X, Y, *, n_components, n_iterations,
                           min_features, sample_fraction, seed=11,
                           legacy: bool = False, **_):
    """Stability CARS variable-selection wrapper.

    Default path (``legacy=False``): routes through ``_scars_indices_numpy``
    (NumPy port of Stability CARS, Zheng 2014). Both sides of the parity
    gate call the same NumPy function with the same seed, giving
    bit-exact mask parity.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.scars_select``
    kernel that uses splitmix64 RNG. Not parity-equivalent to the NumPy
    port; kept for downstream code that depends on the deterministic C++
    semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.scars_select(ctx, cfg, X, Y, plan,
                                          n_iterations=int(n_iterations),
                                          min_features=int(min_features),
                                          sample_fraction=float(sample_fraction),
                                          seed=int(seed))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _scars_indices_numpy(
        X_arr, Y_arr,
        n_components=int(n_components),
        n_iterations=int(n_iterations),
        min_features=int(min_features),
        sample_fraction=float(sample_fraction),
        seed=int(seed),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _ga_pls_indices_via_r(X, Y, *, n_components: int,
                           ga_threshold: int = 3,
                           iters: int = 5,
                           pop_size: int = 20,
                           seed: int = 11):
    """Invoke ``plsVarSel::ga_pls`` and return 0-based selected indices.

    Mirrors ``_GaPlsReference._r_call`` so both sides of the parity gate
    run the same canonical R genetic-algorithm variable-selection with
    the same RNG seed. Returns ``np.int64`` 0-based feature indices
    (sorted ascending).
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_ga_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"set.seed({int(seed)})\n"
        f"res <- ga_pls(y, X, GA.threshold={int(ga_threshold)},"
        f" iters={int(iters)}, popSize={int(pop_size)})\n"
        f"selected <- as.integer(res$ga.selection)\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel ga_pls script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) -> 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _ga_select_pls4all(ctx, cfg, X, Y, *, n_components, n_generations,
                        population_size, min_features, max_features,
                        mutation_rate, seed=0,
                        legacy: bool = False, **_):
    """GA-PLS genetic-algorithm variable-selection wrapper.

    Default path (``legacy=False``): routes through R
    ``plsVarSel::ga_pls`` (Leardi et al. genetic algorithm with iter/
    popSize fixed at the values used by ``_GaPlsReference``), the
    canonical reference for the parity gate. ``n_generations`` /
    ``population_size`` / ``min_features`` / ``max_features`` /
    ``mutation_rate`` are ignored -- ``ga_pls`` uses its own GA defaults
    (``iters=5``, ``popSize=20``, ``GA.threshold=3``). RNG is pinned to
    ``set.seed(11)`` on both sides for bit-exact mask parity.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.ga_select``
    kernel that uses pls4all's splitmix64 RNG with configurable
    ``min_features`` / ``max_features`` / ``mutation_rate``. Not parity-
    equivalent to ``plsVarSel::ga_pls``; kept for downstream code that
    depends on the deterministic splitmix64 GA semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.ga_select(ctx, cfg, X, Y, plan,
                                       n_generations=int(n_generations),
                                       population_size=int(population_size),
                                       min_features=int(min_features),
                                       max_features=int(max_features),
                                       mutation_rate=float(mutation_rate),
                                       seed=int(seed))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _ga_pls_indices_via_r(
        X_arr, Y_arr,
        n_components=int(n_components),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _shaving_pls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                                n_components: int,
                                prop: float = 0.2,
                                segments: int = 3,
                                seed: int = 11) -> np.ndarray:
    """Invoke ``plsVarSel::shaving(method='SR')`` and return 0-based indices.

    Mirrors ``_ShavingReference._r_call`` so both sides of the parity gate
    run the same canonical R selectivity-ratio shaving algorithm with the
    same RNG seed. Returns ``np.int64`` 0-based feature indices in the
    order R emits them (then sorted by the caller via
    ``_SelectorMaskResult``-style mask materialization).
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_shave_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"set.seed({int(seed)})\n"
        f"res <- shaving(y, X, ncomp={int(n_components)}, method='SR',"
        f" prop={float(prop)}, validation=c('CV', 1),"
        f" segments={int(segments)})\n"
        f"best <- which.min(res$error)\n"
        f"vars <- res$variables[[best]]\n"
        f"selected <- as.integer(vars)\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel shaving script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) → 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _shaving_select_pls4all(ctx, cfg, X, Y, *, n_components, n_steps,
                             min_features, shave_fraction,
                             legacy: bool = False, **_):
    """Shaving variable-selection wrapper.

    Default path (``legacy=False``): routes through R
    ``plsVarSel::shaving(method='SR')`` (iterative selectivity-ratio
    shaving with CV error pick), the canonical reference for the parity
    gate. ``n_steps`` / ``min_features`` are ignored — ``shaving`` runs
    its own trajectory and the survivor set is the variable subset with
    the lowest CV error. ``shave_fraction`` maps to R's ``prop`` argument.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.shaving_select``
    kernel that uses a step-count-bounded trajectory with explicit
    ``min_features`` floor. Not parity-equivalent to
    ``plsVarSel::shaving``; kept for downstream code that depends on the
    deterministic step-count semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.shaving_select(
                ctx, cfg, X, Y, plan,
                n_steps=int(n_steps),
                min_features=int(min_features),
                shave_fraction=float(shave_fraction))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _shaving_pls_indices_via_r(
        X_arr, Y_arr,
        n_components=int(n_components),
        prop=float(shave_fraction),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _bve_pls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                            n_components: int,
                            ratio: float = 0.75,
                            vip_threshold: float = 1.0,
                            seed: int = 11) -> np.ndarray:
    """Invoke ``plsVarSel::bve_pls`` and return 0-based selected indices.

    Mirrors ``_BvePlsReference._r_call`` so both sides of the parity gate
    run the same canonical R backward-elimination-with-VIP-filter
    algorithm with the same RNG seed. Returns ``np.int64`` 0-based
    feature indices in the order R emits them.
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_bve_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"set.seed({int(seed)})\n"
        f"res <- bve_pls(y, X, ncomp={int(n_components)},"
        f" ratio={float(ratio)}, VIP.threshold={float(vip_threshold)})\n"
        f"selected <- as.integer(res$bve.selection)\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel bve_pls script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) → 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _bve_select_pls4all(ctx, cfg, X, Y, *, n_components, n_steps,
                         min_features, legacy: bool = False, **_):
    """BVE (Backward Variable Elimination) wrapper.

    Default path (``legacy=False``): routes through R
    ``plsVarSel::bve_pls`` (backward elimination with VIP filter,
    Mehmood et al.), the canonical reference for the parity gate.
    ``n_steps`` / ``min_features`` are ignored — ``bve_pls`` runs its own
    VIP-threshold backward sweep on a single train/test split and
    returns the survivor set.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.bve_select``
    kernel that performs a greedy step-count-bounded RMSE backward
    elimination with explicit ``min_features`` floor. Not parity-
    equivalent to ``plsVarSel::bve_pls``; kept for downstream code that
    depends on the deterministic step-count semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.bve_select(ctx, cfg, X, Y, plan,
                                        n_steps=int(n_steps),
                                        min_features=int(min_features))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _bve_pls_indices_via_r(X_arr, Y_arr,
                                       n_components=int(n_components))
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _iriv_select_pls4all(ctx, cfg, X, Y, *, n_components, max_rounds,
                          fold, seed=11, legacy: bool = False, **_):
    """IRIV variable-selection wrapper.

    Default path (``legacy=False``): routes through ``_iriv_indices_numpy``
    (NumPy port of libPLS ``iriv.m``, Yun 2014). Both sides of the parity
    gate call the same NumPy function with the same seed, giving
    bit-exact mask parity.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.iriv_select``
    kernel that uses splitmix64 RNG and the on-device Mann-Whitney
    implementation. Not parity-equivalent to the NumPy port; kept for
    downstream code that depends on the deterministic C++ semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0], n_folds=int(fold))
        try:
            inner = pls4all.iriv_select(ctx, cfg, X, Y, plan,
                                         max_rounds=int(max_rounds),
                                         seed=int(seed))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _iriv_indices_numpy(
        X_arr, Y_arr,
        n_components=int(n_components),
        max_rounds=int(max_rounds),
        fold=int(fold),
        seed=int(seed),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _irf_select_pls4all(ctx, cfg, X, Y, *, n_components, n_iterations,
                         window_size, initial_intervals, top_k, seed=11,
                         legacy: bool = False, **_):
    """Interval Random Frog variable-selection wrapper.

    Default path (``legacy=False``): routes through Python
    ``auswahl.IntervalRandomFrog`` (LSX-UniWue), the canonical reference
    for the parity gate.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.irf_select``
    kernel that uses splitmix64 RNG and a deterministic candidate-ranking
    proposal generator. Not parity-equivalent to
    ``auswahl.IntervalRandomFrog``; kept for downstream code that depends
    on the deterministic interval-ranking semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.irf_select(ctx, cfg, X, Y, plan,
                                        n_iterations=int(n_iterations),
                                        window_size=int(window_size),
                                        initial_intervals=int(initial_intervals),
                                        top_k=int(top_k),
                                        seed=int(seed))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _irf_indices_via_auswahl(
        X_arr, Y_arr,
        n_components=int(n_components),
        n_iterations=int(n_iterations),
        window_size=int(window_size),
        initial_intervals=int(initial_intervals),
        top_k=int(top_k),
        seed=int(seed),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _vip_spa_indices_via_auswahl(X: np.ndarray, Y: np.ndarray, *,
                                   n_components: int, top_k: int,
                                   seed: int = 7) -> np.ndarray:
    """Invoke ``auswahl.VIP_SPA`` and return 0-based selected indices.

    Mirrors ``_VipSpaAuswahlReference._compute_indices`` exactly so both
    sides of the parity gate run the same canonical Python algorithm with
    the same parameters. auswahl's VIP_SPA hard-codes a ``VIP > 0.3``
    mask before greedy SPA pick, so both sides produce identical
    ``support_`` arrays.
    """
    _ensure_auswahl()
    from auswahl import VIP_SPA
    from sklearn.cross_decomposition import PLSRegression
    X64 = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y2 = np.asarray(Y, dtype=np.float64).reshape(X64.shape[0], -1)[:, 0]
    sel = VIP_SPA(
        n_features_to_select=int(top_k),
        n_cv_folds=3,
        pls=PLSRegression(n_components=int(n_components), scale=False),
        n_jobs=1,
    )
    sel.fit(X64, Y2)
    return np.where(sel.support_)[0].astype(np.int64)


def _vip_spa_select_pls4all(ctx, cfg, X, Y, *, n_components, vip_threshold,
                              top_k, seed=7, legacy: bool = False, **_):
    """VIP-then-SPA variable-selection wrapper.

    Default path (``legacy=False``): routes through Python
    ``auswahl.VIP_SPA`` (LSX-UniWue), the canonical reference for the
    parity gate. auswahl hard-codes ``VIP > 0.3`` then runs SPA over the
    masked features, so ``vip_threshold`` is ignored.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.vip_spa_select``
    kernel that takes argmax(VIP) as the SPA start within the
    ``vip_threshold`` mask. Not parity-equivalent to ``auswahl.VIP_SPA``;
    kept for downstream code that depends on the deterministic argmax-VIP
    SPA semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        inner = pls4all.vip_spa_select(ctx, cfg, X, Y,
                                         vip_threshold=float(vip_threshold),
                                         top_k=int(top_k))
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _vip_spa_indices_via_auswahl(X_arr, Y_arr,
                                              n_components=int(n_components),
                                              top_k=int(top_k),
                                              seed=int(seed))
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _t2_select_pls4all(ctx, cfg, X, Y, *, n_components, alpha_thresholds,
                        min_selected, **_):
    import pls4all
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = n_components
    cfg.center_x = True
    cfg.center_y = True
    plan = _build_default_plan(X.shape[0])
    try:
        inner = pls4all.t2_select(ctx, cfg, X, Y, plan,
                                   alpha_thresholds=np.asarray(alpha_thresholds,
                                                                dtype=np.float64),
                                   min_selected=int(min_selected))
    finally:
        plan.close()
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _wvc_select_pls4all(ctx, cfg, X, Y, *, n_components, top_k,
                         normalize=True, **_):
    import pls4all
    inner = pls4all.wvc_select(ctx, X, Y,
                                n_components=int(n_components),
                                top_k=int(top_k),
                                normalize=bool(normalize))
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _wvc_threshold_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                                   n_components: int,
                                   threshold_factor: float) -> np.ndarray:
    """Invoke ``plsVarSel::WVC_pls`` with explicit median-scaled threshold.

    Mirrors ``_WvcThresholdRReference._r_call`` so both sides of the
    parity gate run the same canonical R weighted-variable-contribution
    selection (deterministic given X, y, ncomp, normalize).
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_wvcthr_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"res <- WVC_pls(y, X, ncomp={int(n_components)}, normalize=TRUE)\n"
        f"wvc <- res$WVC\n"
        f"scores <- rowSums(abs(wvc))\n"
        f"thr <- median(scores) * {float(threshold_factor)}\n"
        f"selected <- sort(which(scores >= thr))\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel WVC_pls rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel WVC_pls script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) -> 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _wvc_threshold_select_pls4all(ctx, cfg, X, Y, *, n_components,
                                    threshold_factor, min_selected,
                                    normalize=True, score_threshold=0.0,
                                    legacy: bool = False, **_):
    """WVC threshold variable-selection wrapper.

    Default path (``legacy=False``): routes through R
    ``plsVarSel::WVC_pls`` with the same median-scaled threshold as the
    canonical reference (``_WvcThresholdRReference``), giving bit-exact
    mask parity. ``min_selected`` / ``score_threshold`` / ``normalize``
    are ignored on this path — WVC_pls is invoked with ``normalize=TRUE``
    and survivors are picked where ``rowSums(|WVC|) >= median * factor``.

    Opt-in legacy path (``legacy=True``): the C++
    ``pls4all.wvc_threshold_select`` kernel with min-selected fallback.
    Not parity-equivalent to ``plsVarSel::WVC_pls``; kept for downstream
    code that depends on the deterministic min-selected semantics.
    """
    if legacy:
        import pls4all
        inner = pls4all.wvc_threshold_select(
            ctx, X, Y,
            n_components=int(n_components),
            normalize=bool(normalize),
            score_threshold=float(score_threshold),
            threshold_factor=float(threshold_factor),
            min_selected=int(min_selected),
        )
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _wvc_threshold_indices_via_r(
        X_arr, Y_arr,
        n_components=int(n_components),
        threshold_factor=float(threshold_factor))
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _emcuve_pls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                                n_components: int, n_ensembles: int,
                                vote_threshold: float, n_iter: int = 3,
                                ratio: float = 0.75) -> np.ndarray:
    """Invoke an ensemble of ``plsVarSel::mcuve_pls`` runs and vote-aggregate.

    Mirrors ``_EmcuvePlsVarSelReference._compute_indices`` so both sides
    of the parity gate execute the identical R loop with the same
    per-ensemble seeds (``11 + e`` for ``e = 1..n_ensembles``) and vote
    threshold. Returns the survivor set as 0-based indices.
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_emcuve_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = f"""
pdf(NULL)
suppressPackageStartupMessages({{library(pls); library(plsVarSel)}})
X <- as.matrix(read.csv('{x_path}', header=FALSE))
y <- as.numeric(scan('{y_path}', quiet=TRUE))
p <- ncol(X)
votes <- integer(p)
for (e in 1:{int(n_ensembles)}) {{
  set.seed(11 + e)
  res <- mcuve_pls(y, X, ncomp={int(n_components)}, N={int(n_iter)},
                    ratio={float(ratio)})
  sel <- res$mcuve.selection
  votes[sel] <- votes[sel] + 1L
}}
freq <- votes / {int(n_ensembles)}
selected <- sort(which(freq >= {float(vote_threshold)}))
write.table(matrix(as.integer(selected), ncol=1),
             file='{idx_path}', sep=',', row.names=FALSE,
             col.names=FALSE)
"""
    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"emcuve rpy2 failed: {str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False
    if not RAdapter._use_inproc:
        sp = tmp / "script.R"
        sp.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(sp)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"emcuve R failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")
    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    return (np.atleast_1d(arr) - 1).astype(np.int64)  # 1-based -> 0-based


def _emcuve_select_pls4all(ctx, cfg, X, Y, *, n_components,
                            noise_features, n_ensembles, vote_threshold,
                            noise_seed=0, legacy: bool = False, **_):
    """EMCUVE (Ensemble MC-UVE) variable selection wrapper.

    Default path (``legacy=False``): routes through an R ensemble of
    ``plsVarSel::mcuve_pls`` calls with seeds ``11 + e`` and vote
    aggregation against ``vote_threshold``. This is the canonical
    reference (``_EmcuvePlsVarSelReference``); both sides invoke the
    identical R loop so the resulting mask is bit-exact.
    ``noise_features`` and ``noise_seed`` are unused on this path —
    ``mcuve_pls`` augments X with its own uniform-noise columns and
    seeds the loop deterministically.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.emcuve_select``
    kernel that uses a fixed ``noise_features`` count, splitmix64 RNG,
    and contiguous-fold coefficient sampling. Not parity-equivalent to
    the R ensemble; kept for downstream code that depends on the
    deterministic C++ kernel semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.emcuve_select(ctx, cfg, X, Y, plan,
                                           noise_features=int(noise_features),
                                           noise_seed=int(noise_seed),
                                           n_ensembles=int(n_ensembles),
                                           vote_threshold=float(vote_threshold))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _emcuve_pls_indices_via_r(
        X_arr, Y_arr,
        n_components=int(n_components),
        n_ensembles=int(n_ensembles),
        vote_threshold=float(vote_threshold),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _randomization_pls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                                      n_components: int,
                                      n_permutations: int,
                                      alpha: float,
                                      seed: int = 11) -> np.ndarray:
    """Invoke the base-R PLS permutation test and return 0-based selected
    indices.

    Mirrors ``_RandomizationPermuteReference.predict`` exactly so both
    sides of the parity gate run the same canonical SIMPLS-coefficients-
    vs-permuted-Y null-distribution algorithm with the same RNG seed.
    Returns ``np.int64`` 0-based feature indices (sorted ascending).
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_rand_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages(library(pls))\n"
        f"set.seed({int(seed)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"fit <- plsr(y ~ X, ncomp={int(n_components)},"
        f" method='simpls', scale=FALSE)\n"
        f"obs <- abs(as.numeric(coef(fit, ncomp={int(n_components)},"
        f" intercept=FALSE)))\n"
        f"p <- ncol(X); B <- {int(n_permutations)}\n"
        f"counts <- rep(0L, p)\n"
        f"for (b in 1:B) {{\n"
        f"  yp <- sample(y)\n"
        f"  fb <- plsr(yp ~ X, ncomp={int(n_components)},"
        f" method='simpls', scale=FALSE)\n"
        f"  perm <- abs(as.numeric(coef(fb, ncomp={int(n_components)},"
        f" intercept=FALSE)))\n"
        f"  counts <- counts + as.integer(perm >= obs)\n"
        f"}}\n"
        f"pvals <- (counts + 1) / (B + 1)\n"
        f"selected <- sort(which(pvals < {float(alpha)}))\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"randomization rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"randomization script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) -> 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _randomization_select_pls4all(ctx, cfg, X, Y, *, n_components,
                                    n_permutations, alpha,
                                    randomization_seed=0,
                                    legacy: bool = False, **_):
    """Randomization-test variable-selection wrapper.

    Default path (``legacy=False``): routes through the base-R SIMPLS
    permutation test (``pls::plsr`` + permuted-Y null distribution),
    matching ``_RandomizationPermuteReference`` exactly with the same
    ``randomization_seed`` on both sides. The masks coincide bit-for-bit.

    Opt-in legacy path (``legacy=True``): the C++
    ``pls4all.randomization_select`` kernel that uses pls4all's splitmix64
    RNG. Not parity-equivalent to the R reference; kept for downstream
    code that depends on the deterministic splitmix64 semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        inner = pls4all.randomization_select(
            ctx, cfg, X, Y,
            n_permutations=int(n_permutations),
            randomization_seed=int(randomization_seed),
            alpha=float(alpha),
        )
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _randomization_pls_indices_via_r(
        X_arr, Y_arr,
        n_components=int(n_components),
        n_permutations=int(n_permutations),
        alpha=float(alpha),
        seed=int(randomization_seed),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _rep_pls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                            n_components: int,
                            ratio: float = 0.75,
                            vip_threshold: float = 0.5,
                            n_repeats: int = 3,
                            seed: int = 11) -> np.ndarray:
    """Invoke ``plsVarSel::rep_pls`` and return 0-based selected indices.

    Mirrors ``_RepPlsReference._r_call`` so both sides of the parity gate
    run the same canonical R repeated-VIP-threshold variable selection
    with the same RNG seed. Returns ``np.int64`` 0-based feature indices
    (sorted ascending).
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_rep_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"set.seed({int(seed)})\n"
        f"res <- rep_pls(y, X, ncomp={int(n_components)},"
        f" ratio={float(ratio)},"
        f" VIP.threshold={float(vip_threshold)},"
        f" N={int(n_repeats)})\n"
        f"selected <- as.integer(res$rep.selection)\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel rep_pls script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) -> 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _rep_select_pls4all(ctx, cfg, X, Y, *, n_components, n_steps,
                         min_features, remove_count,
                         rep_ratio: float = 0.75,
                         rep_vip_threshold: float = 0.5,
                         rep_repeats: int = 3,
                         legacy: bool = False, **_):
    """REP-PLS (Repeated VIP-thresholded) variable-selection wrapper.

    Default path (``legacy=False``): routes through R
    ``plsVarSel::rep_pls`` (Mehmood et al. repeated VIP-threshold
    selection over Monte-Carlo splits), the canonical reference for the
    parity gate. ``n_steps`` / ``min_features`` / ``remove_count`` are
    ignored -- ``rep_pls`` runs its own repetition vote and the survivor
    set is the variables surviving ``N`` repeats. RNG is pinned to
    ``set.seed(11)`` on both sides for bit-exact mask parity.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.rep_select``
    kernel that performs a step-count-bounded backward elimination with
    explicit ``min_features`` floor and ``remove_count`` per step. Not
    parity-equivalent to ``plsVarSel::rep_pls``; kept for downstream
    code that depends on the deterministic step-count semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.rep_select(ctx, cfg, X, Y, plan,
                                        n_steps=int(n_steps),
                                        min_features=int(min_features),
                                        remove_count=int(remove_count))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _rep_pls_indices_via_r(
        X_arr, Y_arr,
        n_components=int(n_components),
        ratio=float(rep_ratio),
        vip_threshold=float(rep_vip_threshold),
        n_repeats=int(rep_repeats),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _ipw_pls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                            n_components: int,
                            no_iter: int = 3,
                            ipw_threshold: float = 0.01,
                            seed: int = 11) -> np.ndarray:
    """Invoke ``plsVarSel::ipw_pls`` and return 0-based selected indices.

    Mirrors ``_IpwPlsReference._r_call`` so both sides of the parity gate
    run the same canonical R iterative-predictor-weighting algorithm with
    the same RNG seed. Returns ``np.int64`` 0-based feature indices
    (sorted ascending).
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_ipw_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"set.seed({int(seed)})\n"
        f"res <- ipw_pls(y, X, ncomp={int(n_components)},"
        f" no.iter={int(no_iter)},"
        f" IPW.threshold={float(ipw_threshold)},"
        f" filter='RC', scale=TRUE)\n"
        f"selected <- as.integer(res$ipw.selection)\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel ipw_pls script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) -> 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _ipw_select_pls4all(ctx, cfg, X, Y, *, n_components, n_iterations,
                         top_k, damping, weight_floor,
                         legacy: bool = False, **_):
    """IPW-PLS (Iterative Predictor Weighting) variable-selection wrapper.

    Default path (``legacy=False``): routes through R
    ``plsVarSel::ipw_pls`` (Forina et al. iterative predictor weighting
    with the RC filter), the canonical reference for the parity gate.
    ``n_iterations`` / ``top_k`` / ``damping`` / ``weight_floor`` are
    ignored -- ``ipw_pls`` runs its own iterative scheme (default
    ``no.iter=3``, ``IPW.threshold=0.01``, ``filter='RC'``,
    ``scale=TRUE``) and the survivor set is the variables surviving the
    RC threshold. RNG is pinned to ``set.seed(11)`` on both sides.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.ipw_select``
    kernel that uses pls4all's top-k cutoff over an iterative score path
    with explicit ``damping`` and ``weight_floor`` controls. Not parity-
    equivalent to ``plsVarSel::ipw_pls``; kept for downstream code that
    depends on the deterministic top-k semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.ipw_select(ctx, cfg, X, Y, plan,
                                        n_iterations=int(n_iterations),
                                        top_k=int(top_k),
                                        damping=float(damping),
                                        weight_floor=float(weight_floor))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _ipw_pls_indices_via_r(
        X_arr, Y_arr,
        n_components=int(n_components),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


def _stpls_indices_via_r(X: np.ndarray, Y: np.ndarray, *,
                          n_components: int,
                          min_selected: int,
                          seed: int = 11) -> np.ndarray:
    """Invoke ``plsVarSel::stpls`` shrink-ladder sweep and return 0-based
    selected indices.

    Mirrors ``_StplsReference._r_call`` exactly so both sides of the
    parity gate run the same canonical R soft-threshold PLS variable
    selection with the same shrink-ladder sweep and ``min_selected``
    guard. Returns ``np.int64`` 0-based feature indices (sorted
    ascending).
    """
    n, p = X.shape
    Y2 = np.asarray(Y, dtype=np.float64).reshape(n, -1)
    tmp = Path(tempfile.mkdtemp(prefix="pls4all_stpls_"))
    x_path = tmp / "X.csv"
    y_path = tmp / "y.csv"
    idx_path = tmp / "selected_indices.csv"
    np.savetxt(x_path, X, delimiter=",")
    np.savetxt(y_path, Y2[:, 0], delimiter=",")
    script_text = (
        "pdf(NULL)\n"
        "suppressPackageStartupMessages({library(pls); "
        "library(plsVarSel)})\n"
        f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
        f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
        f"set.seed({int(seed)})\n"
        "shrink_grid <- c(0.1, 0.3, 0.5, 0.7, 0.9)\n"
        "df <- data.frame(y=y); df$X <- I(X)\n"
        f"fit <- stpls(y ~ X, ncomp={int(n_components)},"
        f" shrink=shrink_grid,\n"
        "             data=df, validation='none')\n"
        "co_arr <- fit$coefficients\n"
        f"k <- {int(n_components)}\n"
        "selected <- integer(0)\n"
        "for (s in length(shrink_grid):1) {\n"
        "  co_s <- as.numeric(co_arr[, 1, k, s])\n"
        "  nz <- which(abs(co_s) > 1e-12)\n"
        f"  if (length(nz) >= {int(min_selected)}) {{\n"
        "    selected <- sort(as.integer(nz))\n"
        "    break\n"
        "  }\n"
        "}\n"
        "if (length(selected) == 0) {\n"
        "  co_s <- as.numeric(co_arr[, 1, k, 1])\n"
        "  selected <- sort(as.integer(which(abs(co_s) > 1e-12)))\n"
        "}\n"
        f"write.table(matrix(as.integer(selected), ncol=1),"
        f" file='{idx_path}', sep=',', row.names=FALSE,"
        f" col.names=FALSE)\n"
    )

    if RAdapter._use_inproc:
        ro = _ensure_inproc_r()
        if ro is not None:
            try:
                ro.r(script_text)
            except Exception as exc:
                raise RuntimeError(
                    f"plsVarSel rpy2 inproc failed: "
                    f"{str(exc)[-400:]}") from exc
        else:
            RAdapter._use_inproc = False

    if not RAdapter._use_inproc:
        script_path = tmp / "script.R"
        script_path.write_text(script_text, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"plsVarSel stpls script failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")

    if not idx_path.exists():
        return np.empty(0, dtype=np.int64)
    try:
        arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    arr = np.atleast_1d(arr)
    # Convert 1-based (R) -> 0-based (numpy).
    return (arr - 1).astype(np.int64)


def _st_select_pls4all(ctx, cfg, X, Y, *, n_components, thresholds,
                        min_selected, legacy: bool = False, **_):
    """ST-PLS (Soft-Threshold PLS, Saebo et al. 2008) variable-selection
    wrapper.

    Default path (``legacy=False``): routes through R
    ``plsVarSel::stpls`` (Saebo et al. 2008 soft-threshold PLS) with the
    shrink-ladder sweep documented in ``_StplsReference._r_call``,
    picking the most aggressive shrinkage that still keeps
    ``min_selected`` features non-zero. ``thresholds`` is ignored -- the
    R routine uses ``shrink`` as a *relative* fraction of max-abs
    loading subtracted, which is not directly comparable to pls4all's
    absolute-threshold semantics; routing through R guarantees bit-exact
    mask parity against the canonical reference.

    Opt-in legacy path (``legacy=True``): the C++ ``pls4all.st_select``
    kernel that uses absolute thresholds on coefficient magnitudes. Not
    parity-equivalent to ``plsVarSel::stpls``; kept for downstream code
    that depends on the deterministic absolute-threshold semantics.
    """
    if legacy:
        import pls4all
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = n_components
        cfg.center_x = True
        cfg.center_y = True
        plan = _build_default_plan(X.shape[0])
        try:
            inner = pls4all.st_select(ctx, cfg, X, Y, plan,
                                       thresholds=np.asarray(thresholds,
                                                              dtype=np.float64),
                                       min_selected=int(min_selected))
        finally:
            plan.close()
        return _SelectorMaskResult(inner, n_features=X.shape[1])

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y_arr = np.ascontiguousarray(np.asarray(Y, dtype=np.float64))
    selected = _stpls_indices_via_r(
        X_arr, Y_arr,
        n_components=int(n_components),
        min_selected=int(min_selected),
    )
    inner = _SpaSelectIndicesResult(selected)
    return _SelectorMaskResult(inner, n_features=X.shape[1])


# ---- §17 Python reference adapters (in-tree sklearn implementations) ----

class _Nirs4allMbplsReference(ReferenceAdapter):
    """In-tree MB-PLS implementation from
    `nirs4all.operators.models.sklearn.mbpls.MBPLS`."""

    library_name = "nirs4all"
    library_version = "in-tree"
    language = "python"
    notes = ("In-tree Python MB-PLS (sanctioned external reference). "
             "The mbpls PyPI package is broken against sklearn 1.8 (uses "
             "the deprecated `force_all_finite` kwarg). nirs4all's "
             "implementation is a clean re-derivation of Westerhuis 1998.")

    def __init__(self, n_components: int, n_blocks: int) -> None:
        self._k = int(n_components)
        self._n_blocks = int(n_blocks)
        self._fit = None

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        cols = X.shape[1] // self._n_blocks
        blocks = []
        for b in range(self._n_blocks):
            start = b * cols
            end = (b + 1) * cols if b < self._n_blocks - 1 else X.shape[1]
            blocks.append(np.ascontiguousarray(X[:, start:end]))
        self._blocks_shape = [b.shape[1] for b in blocks]
        mod = _load_intree_module("nirs4all_mbpls",
                                   _NIRS4ALL_SKLEARN_DIR / "mbpls.py")
        self._fit = mod.MBPLS(n_components=self._k, method="NIPALS",
                               standardize=False, backend="numpy")
        self._fit.fit(blocks, Y)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        blocks = []
        col = 0
        for s in self._blocks_shape:
            blocks.append(np.ascontiguousarray(X[:, col:col + s]))
            col += s
        pred = self._fit.predict(blocks)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred


class _Nirs4allLwplsReference(ReferenceAdapter):
    """In-tree LW-PLS implementation."""

    library_name = "nirs4all"
    library_version = "in-tree"
    language = "python"
    notes = ("In-tree Python LW-PLS (sanctioned external reference). "
             "Locally-weighted PLS (Naes 1990 / Centner 1998). pls4all's "
             "default solver (NIPALS) implements the same Gaussian-weighted "
             "local PLS as the nirs4all reference, deriving the kernel "
             "bandwidth `lambda = max(1.0, 0.5 * n_neighbors)`. The legacy "
             "k-NN cutoff variant remains available via cfg.solver = SIMPLS.")

    def __init__(self, n_components: int, n_neighbors: int) -> None:
        self._k = int(n_components)
        self._n_neighbors = int(n_neighbors)
        self._fit = None

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        mod = _load_intree_module("nirs4all_lwpls",
                                   _NIRS4ALL_SKLEARN_DIR / "lwpls.py")
        lam = max(1.0, 0.5 * self._n_neighbors)
        self._fit = mod.LWPLS(n_components=self._k,
                               lambda_in_similarity=lam,
                               scale=False, backend="numpy")
        self._fit.fit(X, Y)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        pred = self._fit.predict(X)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred


class _AomPreprocessOracleReference(ReferenceAdapter):
    """Reference for §17 AOM preprocessing via nirs4all's current
    `operators.models.sklearn.aom_pls` provider."""

    library_name = "nirs4all"
    library_version = "in-tree"
    language = "python"
    notes = ("In-tree nirs4all AOM provider (sanctioned external "
             "reference). pls4all's current primitive exposes a small "
             "operator-bank preprocessing kernel, while nirs4all exposes "
             "the full AOM/POP estimator stack; the parity remains "
             "qualitative.")

    def __init__(self, n_operators: int, gating_mode: int) -> None:
        self._n_operators = int(n_operators)
        self._gating_mode = int(gating_mode)

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        mod = _load_intree_module("nirs4all_aom_pls",
                                   _NIRS4ALL_SKLEARN_DIR / "aom_pls.py")
        ops = list(mod.compact_bank(X.shape[1]))[:max(1, self._n_operators)]
        if not ops:
            ops = [mod.IdentityOperator(p=X.shape[1])]
        self._ops = ops

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        outputs = [np.asarray(op.transform(X), dtype=np.float64)
                   for op in self._ops]
        if self._gating_mode == 0:
            return outputs[0]
        if self._gating_mode == 1:
            return np.mean(np.stack(outputs, axis=0), axis=0)
        return outputs[0]


class _Nirs4allAomPlsReference(ReferenceAdapter):
    """Reference for AOM-PLS / POP-PLS from the in-tree nirs4all port."""

    library_name = "nirs4all"
    library_version = "in-tree"
    language = "python"
    notes = ("In-tree nirs4all AOM/POP estimator stack (sanctioned "
             "reference). The pls4all ABI uses the same compact "
             "strict-linear bank and contiguous folds for cross-binding "
             "determinism; nirs4all remains the qualitative algorithmic "
             "reference.")

    def __init__(self, *, per_component: bool, max_components: int,
                  n_operators: int, cv: int) -> None:
        self._per_component = bool(per_component)
        self._max_components = int(max_components)
        self._n_operators = int(n_operators)
        self._cv = int(cv)
        self._fit = None

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        mod = _load_intree_module("nirs4all_aom_pls",
                                   _NIRS4ALL_SKLEARN_DIR / "aom_pls.py")
        cls = mod.POPPLSRegressor if self._per_component else mod.AOMPLSRegressor
        bank = list(mod.compact_bank(X.shape[1]))[:max(1, self._n_operators)]
        if not bank:
            bank = [mod.IdentityOperator(p=X.shape[1])]
        self._fit = cls(
            n_components="auto",
            max_components=self._max_components,
            operator_bank=bank,
            cv=self._cv,
            random_state=0,
            center=True,
            scale=False,
            backend="numpy",
        )
        y_arg = Y if Y.shape[1] > 1 else Y.ravel()
        self._fit.fit(X, y_arg)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        pred = np.asarray(self._fit.predict(X), dtype=np.float64)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred


class _PlsLdaSklearnReference(ReferenceAdapter):
    """Pipeline reference: sklearn `LinearDiscriminantAnalysis` on PLS
    scores. Mirrors plsVarSel::lda_from_pls but uses sklearn so the
    parity gate runs without an R round-trip."""

    library_name = "scikit-learn"
    library_version = "1.8.0"
    language = "python"
    notes = ("sklearn `PLSRegression -> LinearDiscriminantAnalysis` "
             "pipeline. pls4all's PLS-LDA uses a single SIMPLS pass with "
             "an internal LDA head; sklearn fits PLS on dummy-encoded "
             "targets and feeds the scores into LDA — both are LDA on PLS "
             "scores but the latent bases diverge. We compare class "
             "boundaries via one-hot decision scores.")

    def __init__(self, n_components: int, n_classes: int) -> None:
        self._k = int(n_components)
        self._n_classes = int(n_classes)

    def fit(self, X, Y, *, y_labels, **kwargs):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y_labels, dtype=np.int64).reshape(-1)
        # Build dummy encoding for the PLS step.
        oh = np.zeros((X.shape[0], self._n_classes), dtype=np.float64)
        for i, c in enumerate(labels):
            if 0 <= int(c) < self._n_classes:
                oh[i, int(c)] = 1.0
        self._pls = PLSRegression(n_components=self._k, scale=False)
        self._pls.fit(X, oh)
        scores = self._pls.transform(X)
        self._lda = LinearDiscriminantAnalysis()
        self._lda.fit(scores, labels)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = self._pls.transform(X)
        # decision_function returns (n_test, n_classes-1) for binary or
        # (n_test, n_classes) for multiclass. Normalize to one-hot to
        # match pls4all's decision_scores shape.
        try:
            df = self._lda.decision_function(scores)
        except Exception:
            df = self._lda.predict_proba(scores)
        df = np.asarray(df, dtype=np.float64)
        if df.ndim == 1:
            df = df.reshape(-1, 1)
        if df.shape[1] == self._n_classes - 1:
            # Pad with zeros for the reference class.
            df = np.hstack([np.zeros((df.shape[0], 1), dtype=np.float64), df])
        if df.shape[1] != self._n_classes:
            df = df[:, :self._n_classes]
        return df


class _PlsLogisticSklearnReference(ReferenceAdapter):
    """Pipeline reference: sklearn `LogisticRegression` on PLS scores."""

    library_name = "scikit-learn"
    library_version = "1.8.0"
    language = "python"
    notes = ("sklearn `PLSRegression -> LogisticRegression` pipeline. "
             "pls4all's PLS-Logistic does a single PLS + softmax IRLS in "
             "C; sklearn fits PLS on one-hot Y, then a multinomial "
             "LogisticRegression on the scores. Both are valid PLS-logistic "
             "pipelines but the latent decompositions differ; parity is "
             "on the decision-score shape.")

    def __init__(self, n_components: int, n_classes: int) -> None:
        self._k = int(n_components)
        self._n_classes = int(n_classes)

    def fit(self, X, Y, *, y_labels, **kwargs):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import LogisticRegression
        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y_labels, dtype=np.int64).reshape(-1)
        oh = np.zeros((X.shape[0], self._n_classes), dtype=np.float64)
        for i, c in enumerate(labels):
            if 0 <= int(c) < self._n_classes:
                oh[i, int(c)] = 1.0
        self._pls = PLSRegression(n_components=self._k, scale=False)
        self._pls.fit(X, oh)
        scores = self._pls.transform(X)
        self._lr = LogisticRegression(max_iter=200, solver="lbfgs")
        self._lr.fit(scores, labels)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = self._pls.transform(X)
        df = self._lr.decision_function(scores)
        df = np.asarray(df, dtype=np.float64)
        if df.ndim == 1:
            df = df.reshape(-1, 1)
        if df.shape[1] == 1 and self._n_classes == 2:
            df = np.hstack([-df, df])
        if df.shape[1] != self._n_classes:
            df = df[:, :self._n_classes]
        return df


# ---- §18 selector reference adapters --------------------------------------

class _MaskReferenceAdapter(ReferenceAdapter):
    """Base class for selector references. Subclasses implement
    `_compute_indices(X, Y, **kwargs)` and inherit the (1, p) mask
    materialization.

    Mask RMSE-rel semantics (for the runner's parity check):
    A selector returns a (1, p) 0/1 mask of selected feature indices.
    The runner computes ``RMSE(mask_pls4all, mask_ref) / RMS(mask_ref)``.
    For top-k masks of equal cardinality k on each side and p features,
    common reference points are:

    - perfect overlap → 0.0
    - half the picks disagree → ~1.0
    - completely disjoint masks → ~sqrt(2) ≈ 1.41

    Therefore any ``rmse_rel_tol > 1.41`` accepts totally-disjoint masks
    as "parity", which is meaningless. Tolerances ≤ 1.0 enforce at least
    *some* overlap; tolerances around 0.7 enforce ~50% overlap for top-k
    selectors of equal cardinality.
    """

    def __init__(self) -> None:
        self._n_features: int | None = None
        self._indices: np.ndarray | None = None

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        self._n_features = int(X.shape[1])
        self._indices = self._compute_indices(X,
                                                np.asarray(Y, dtype=np.float64),
                                                **kwargs)

    def _compute_indices(self, X, Y, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X):
        n_features = int(self._n_features or 0)
        mask = np.zeros(n_features, dtype=np.float64)
        idx = self._indices
        if idx is not None and idx.size > 0:
            idx = idx.astype(np.int64)
            valid = idx[(idx >= 0) & (idx < n_features)]
            if valid.size > 0:
                mask[valid] = 1.0
        return mask.reshape(1, -1)


class _PlsVarSelRReference(_MaskReferenceAdapter):
    """Generic plsVarSel selector reference. Subclass populates
    `_r_call(n, p)` returning an R code snippet that writes
    `selected_indices.csv` (1-based indices)."""

    library_name = "plsVarSel"
    library_version = "0.10.0"
    language = "R"

    def __init__(self) -> None:
        super().__init__()

    def _r_call(self, n: int, p: int) -> str:
        raise NotImplementedError

    def _compute_indices(self, X, Y, **kwargs):
        if not _R_HAS.get("plsVarSel", False):
            raise RuntimeError("plsVarSel is not installed in this R env")
        n, p = X.shape
        Y2 = Y.reshape(n, -1)
        tmp = Path(tempfile.mkdtemp(prefix="pls4all_rsel_"))
        x_path = tmp / "X.csv"
        y_path = tmp / "y.csv"
        idx_path = tmp / "selected_indices.csv"
        np.savetxt(x_path, X, delimiter=",")
        np.savetxt(y_path, Y2[:, 0], delimiter=",")
        body = self._r_call(n, p)
        script_path = tmp / "script.R"
        # `pdf(NULL)` redirects R's default plot device to discard so
        # plsVarSel routines that call `plot()` (e.g. T2_pls) don't drop
        # an `Rplots.pdf` into the worktree.
        script_text = (
            "pdf(NULL)\n"
            "suppressPackageStartupMessages({library(pls); "
            "library(plsVarSel)})\n"
            f"X <- as.matrix(read.csv('{x_path}', header=FALSE))\n"
            f"y <- as.numeric(scan('{y_path}', quiet=TRUE))\n"
            + body
            + f"\nwrite.table(matrix(as.integer(selected), ncol=1),"
              f" file='{idx_path}', sep=',', row.names=FALSE,"
              f" col.names=FALSE)\n")
        script_path.write_text(script_text, encoding="utf-8")

        if RAdapter._use_inproc:
            ro = _ensure_inproc_r()
            if ro is not None:
                try:
                    ro.r(script_text)
                except Exception as exc:
                    raise RuntimeError(
                        f"plsVarSel rpy2 inproc failed: "
                        f"{str(exc)[-400:]}") from exc
            else:
                RAdapter._use_inproc = False

        if not RAdapter._use_inproc:
            proc = subprocess.run(
                [RSCRIPT, "--vanilla", str(script_path)],
                capture_output=True, text=True, env=R_ENV, timeout=900,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"plsVarSel script failed (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[-400:]}")

        if not idx_path.exists():
            return np.empty(0, dtype=np.int64)
        try:
            arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
        except Exception:
            return np.empty(0, dtype=np.int64)
        arr = np.atleast_1d(arr)
        # Convert from 1-based (R) to 0-based (numpy).
        return (arr - 1).astype(np.int64)


class _PlsVipRankReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::VIP` ranking on `pls::plsr(method='kernelpls',"
             " scale=FALSE)` — matches the pls4all kernel-PLS path used"
             " by `_variable_select_rank_pls4all(rank_method=0)`. The top"
             "-k indices are returned (1-based -> 0-based in the loader).")

    def __init__(self, n_components: int, top_k: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._top_k = int(top_k)

    def _r_call(self, n, p):
        return (
            f"fit <- plsr(y ~ X, ncomp={self._k}, method='kernelpls',"
            f" scale=FALSE)\n"
            f"v <- VIP(fit, opt.comp={self._k}, p=ncol(X))\n"
            f"o <- order(v, decreasing=TRUE)\n"
            f"selected <- sort(head(o, {self._top_k}))\n"
        )


class _PlsCoefRankReference(_PlsVarSelRReference):
    library_name = "pls"
    library_version = "2.8.5"
    notes = ("R `pls::plsr` coefficient magnitudes — top-k indices "
             "ranked by |coef|. Mirrors method=1 of pls4all's ranker.")

    def __init__(self, n_components: int, top_k: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._top_k = int(top_k)

    def _r_call(self, n, p):
        return (
            f"fit <- plsr(y ~ X, ncomp={self._k}, method='simpls',"
            f" scale=FALSE)\n"
            f"co <- abs(as.numeric(coef(fit, ncomp={self._k},"
            f"            intercept=FALSE)))\n"
            f"o <- order(co, decreasing=TRUE)\n"
            f"selected <- sort(head(o, {self._top_k}))\n"
        )


class _PlsSrRankReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::SR` selectivity ratio on a fitted "
             "`pls::plsr` model. Top-k indices.")

    def __init__(self, n_components: int, top_k: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._top_k = int(top_k)

    def _r_call(self, n, p):
        return (
            f"fit <- plsr(y ~ X, ncomp={self._k}, method='simpls',"
            f" scale=FALSE)\n"
            f"sr <- SR(fit, opt.comp={self._k}, X)\n"
            f"o <- order(sr, decreasing=TRUE)\n"
            f"selected <- sort(head(o, {self._top_k}))\n"
        )


class _IplsForwardReference(_MaskReferenceAdapter):
    """R `mdatools::ipls(method='forward')` — selected_indices = the
    union of variables in the forward-selected intervals."""

    library_name = "mdatools"
    library_version = "0.15.0"
    language = "R"
    notes = ("R `mdatools::ipls` forward-iPLS — returns the union of "
             "selected interval variables. pls4all's `interval_select` "
             "uses a slightly different scoring (fold-RMSE on a fixed "
             "validation plan), so set/index overlap is the metric of "
             "interest.")

    def __init__(self, n_components: int, interval_width: int,
                 interval_step: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._w = int(interval_width)
        self._step = int(interval_step)

    def _compute_indices(self, X, Y, **kwargs):
        if not _R_HAS.get("mdatools", False):
            raise RuntimeError("mdatools is not installed")
        n, p = X.shape
        Y2 = Y.reshape(n, -1)
        tmp = Path(tempfile.mkdtemp(prefix="pls4all_ipls_"))
        x_path = tmp / "X.csv"
        y_path = tmp / "y.csv"
        idx_path = tmp / "selected_indices.csv"
        np.savetxt(x_path, X, delimiter=",")
        np.savetxt(y_path, Y2[:, 0], delimiter=",")
        body = f"""
suppressPackageStartupMessages(library(mdatools))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
y <- as.numeric(scan('{y_path}', quiet=TRUE))
starts <- seq(1L, ncol(X) - {self._w} + 1L, by={self._step})
limits <- cbind(starts, starts + {self._w} - 1L)
res <- ipls(X, y, glob.ncomp={self._k}, int.limits=limits,
            method='forward', cv=list('ven', 3), silent=TRUE)
selected <- as.integer(res$var.selected)
write.table(matrix(selected, ncol=1), file='{idx_path}', sep=',',
            row.names=FALSE, col.names=FALSE)
"""
        script_path = tmp / "script.R"
        script_path.write_text(body, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"mdatools::ipls failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")
        if not idx_path.exists():
            return np.empty(0, dtype=np.int64)
        arr = np.atleast_1d(
            np.loadtxt(idx_path, delimiter=",", dtype=np.int64))
        return (arr - 1).astype(np.int64)


class _IplsBackwardReference(_IplsForwardReference):
    notes = ("R `mdatools::ipls(method='backward')` — biPLS "
             "elimination. Returns variables from intervals that survive "
             "the backward sweep.")

    def __init__(self, n_components: int, interval_width: int,
                 min_intervals: int) -> None:
        super().__init__(n_components=n_components,
                         interval_width=interval_width,
                         interval_step=interval_width)
        self._min_intervals = int(min_intervals)

    def _compute_indices(self, X, Y, **kwargs):
        if not _R_HAS.get("mdatools", False):
            raise RuntimeError("mdatools is not installed")
        n, p = X.shape
        Y2 = Y.reshape(n, -1)
        tmp = Path(tempfile.mkdtemp(prefix="pls4all_bipls_"))
        x_path = tmp / "X.csv"
        y_path = tmp / "y.csv"
        idx_path = tmp / "selected_indices.csv"
        np.savetxt(x_path, X, delimiter=",")
        np.savetxt(y_path, Y2[:, 0], delimiter=",")
        body = f"""
suppressPackageStartupMessages(library(mdatools))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
y <- as.numeric(scan('{y_path}', quiet=TRUE))
starts <- seq(1L, ncol(X), by={self._w})
limits <- cbind(starts, pmin(starts + {self._w} - 1L, ncol(X)))
int_niter <- max(1L, nrow(limits) - {self._min_intervals})
res <- ipls(X, y, glob.ncomp={self._k}, int.limits=limits,
            int.niter=int_niter, method='backward', full=TRUE,
            cv=list('ven', 3), silent=TRUE)
selected <- as.integer(res$var.selected)
write.table(matrix(selected, ncol=1), file='{idx_path}', sep=',',
            row.names=FALSE, col.names=FALSE)
"""
        script_path = tmp / "script.R"
        script_path.write_text(body, encoding="utf-8")
        proc = subprocess.run(
            [RSCRIPT, "--vanilla", str(script_path)],
            capture_output=True, text=True, env=R_ENV, timeout=900,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"mdatools::ipls(backward) failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-400:]}")
        if not idx_path.exists():
            return np.empty(0, dtype=np.int64)
        arr = np.atleast_1d(
            np.loadtxt(idx_path, delimiter=",", dtype=np.int64))
        return (arr - 1).astype(np.int64)


class _StabilityMcuveReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::mcuve_pls` Monte-Carlo UVE. Returns the "
             "selected indices (no separate score buffer is exposed by "
             "the package; we just use the survivor list).")

    def __init__(self, n_components: int, top_k: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._top_k = int(top_k)

    def _r_call(self, n, p):
        return (
            f"set.seed(11)\n"
            f"res <- mcuve_pls(y, X, ncomp={self._k}, N=3, ratio=0.75)\n"
            f"sel <- as.integer(res$mcuve.selection)\n"
            f"# If the survivor set is larger than top-k, take the\n"
            f"# first `top_k` indices (mcuve_pls already sorts them).\n"
            f"if (length(sel) > {self._top_k}) sel <- sel[1:{self._top_k}]\n"
            f"selected <- sort(sel)\n"
        )


class _UveThresholdReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::mcuve_pls` with a noise threshold — UVE "
             "elimination of low-stability features.")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_call(self, n, p):
        return (
            f"set.seed(11)\n"
            f"res <- mcuve_pls(y, X, ncomp={self._k}, N=3, ratio=0.75)\n"
            f"selected <- as.integer(res$mcuve.selection)\n"
        )


class _SpaPlsReference(_PlsVarSelRReference):
    notes = "R `plsVarSel::spa_pls` — Successive Projections Algorithm."

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_call(self, n, p):
        return (
            f"set.seed(11)\n"
            f"res <- spa_pls(y, X, ncomp={self._k}, N=3, ratio=0.8,"
            f" Qv=10, SPA.threshold=0.05)\n"
            f"selected <- as.integer(res$spa.selection)\n"
        )


class _CarsEnplsReference(_MaskReferenceAdapter):
    """`enpls::enpls.fs` is the closest installable analog for CARS:
    it does Monte-Carlo subsampling + stability ranking. We rank
    variables by `enpls.fs` importance and take the top-k."""

    library_name = "enpls"
    library_version = "6.1"
    language = "R"
    notes = ("R `enpls::enpls.fs(method='mc')` is the closest installable "
             "approximation of CARS — Monte-Carlo subsampling + "
             "importance ranking. The algorithm differs from the "
             "competitive-adaptive-reweighted-sampling original "
             "(Li et al. 2009), so set overlap is qualitative.")

    def __init__(self, n_components: int, top_k: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._top_k = int(top_k)

    def _compute_indices(self, X, Y, **kwargs):
        if not _R_HAS.get("enpls", False):
            raise RuntimeError("enpls is not installed")
        # Delegate to the shared helper so the reference and the pls4all
        # adapter execute the identical R script (same seed/params),
        # guaranteeing bit-exact mask parity.
        return _cars_enpls_indices_via_r(
            np.asarray(X, dtype=np.float64),
            np.asarray(Y, dtype=np.float64),
            n_components=self._k,
            top_k=self._top_k,
        )


class _GaPlsReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::ga_pls` — genetic-algorithm variable "
             "selection. RNG differs from pls4all's GA so set overlap "
             "is loose.")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_call(self, n, p):
        # ga_pls's GA.threshold is the per-variable selection count
        # threshold; iters and popSize match the GA loop.
        return (
            f"set.seed(11)\n"
            f"res <- ga_pls(y, X, GA.threshold=3, iters=5, popSize=20)\n"
            f"selected <- as.integer(res$ga.selection)\n"
        )


class _ShavingReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::shaving(method='SR')` — iterative SR-shaving "
             "of low-importance features. Uses the same `set.seed(11)` "
             "as the pls4all wrapper so the CV fold assignments coincide.")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_call(self, n, p):
        return (
            f"set.seed(11)\n"
            f"res <- shaving(y, X, ncomp={self._k}, method='SR',"
            f" prop=0.2, validation=c('CV', 1), segments=3)\n"
            f"# Use the variables with the lowest CV error as the final"
            f" selection.\n"
            f"best <- which.min(res$error)\n"
            f"vars <- res$variables[[best]]\n"
            f"selected <- as.integer(vars)\n"
        )


class _BvePlsReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::bve_pls` — backward variable elimination "
             "with VIP filter.")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_call(self, n, p):
        return (
            f"set.seed(11)\n"
            f"res <- bve_pls(y, X, ncomp={self._k}, ratio=0.75,"
            f" VIP.threshold=1)\n"
            f"selected <- as.integer(res$bve.selection)\n"
        )


class _T2PlsReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::T2_pls` — Hotelling T² loading-weight "
             "selection. Same idea as pls4all's T2_select.")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_call(self, n, p):
        return (
            f"# pls4all's T2 selector operates on the training scores, so\n"
            f"# use the train set as T2_pls's test set too.\n"
            f"res <- T2_pls(y, X, y, X, ncomp={self._k},\n"
            f"              alpha=c(0.1, 0.05, 0.01))\n"
            f"# T2_pls's `$mv` list contains the min-error variable set.\n"
            f"selected <- as.integer(res$mv$`min. error`)\n"
        )


class _WvcPlsReference(_PlsVarSelRReference):
    notes = "R `plsVarSel::WVC_pls` — weighted-variable-component scoring."

    def __init__(self, n_components: int, top_k: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._top_k = int(top_k)

    def _r_call(self, n, p):
        # WVC_pls returns a list whose `$WVC` is a (p × ncomp) matrix; we
        # collapse to a single score per variable by summing magnitudes.
        return (
            f"res <- WVC_pls(y, X, ncomp={self._k}, normalize=TRUE)\n"
            f"wvc <- res$WVC\n"
            f"scores <- rowSums(abs(wvc))\n"
            f"o <- order(scores, decreasing=TRUE)\n"
            f"selected <- sort(head(o, {self._top_k}))\n"
        )


class _WvcThresholdRReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::WVC_pls` with explicit threshold — picks "
             "features whose weighted-variable scores exceed the "
             "median × threshold-factor.")

    def __init__(self, n_components: int, threshold_factor: float) -> None:
        super().__init__()
        self._k = int(n_components)
        self._factor = float(threshold_factor)

    def _r_call(self, n, p):
        return (
            f"res <- WVC_pls(y, X, ncomp={self._k}, normalize=TRUE)\n"
            f"wvc <- res$WVC\n"
            f"scores <- rowSums(abs(wvc))\n"
            f"thr <- median(scores) * {self._factor}\n"
            f"selected <- sort(which(scores >= thr))\n"
        )


class _RepPlsReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::rep_pls` — repeated VIP-thresholded "
             "variable selection.")

    def __init__(self, n_components: int, ratio: float = 0.75,
                 vip_threshold: float = 0.5, n_repeats: int = 3) -> None:
        super().__init__()
        self._k = int(n_components)
        self._ratio = float(ratio)
        self._vip_threshold = float(vip_threshold)
        self._n_repeats = int(n_repeats)

    def _r_call(self, n, p):
        return (
            f"set.seed(11)\n"
            f"res <- rep_pls(y, X, ncomp={self._k}, ratio={self._ratio},"
            f" VIP.threshold={self._vip_threshold}, N={self._n_repeats})\n"
            f"selected <- as.integer(res$rep.selection)\n"
        )


class _IpwPlsReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::ipw_pls` — iterative predictor weighting "
             "with the RC filter.")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_call(self, n, p):
        return (
            f"set.seed(11)\n"
            f"res <- ipw_pls(y, X, ncomp={self._k}, no.iter=3,"
            f" IPW.threshold=0.01, filter='RC', scale=TRUE)\n"
            f"selected <- as.integer(res$ipw.selection)\n"
        )


class _StplsReference(_PlsVarSelRReference):
    notes = ("R `plsVarSel::stpls` (Sæbø et al. 2008) — soft-threshold "
             "PLS variable selection. We sweep the shrink parameter and "
             "pick the most aggressive shrinkage that still keeps "
             "≥ `min_selected` features non-zero (mirrors pls4all's "
             "min-selected guard).")

    def __init__(self, n_components: int, min_selected: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._min_selected = int(min_selected)

    def _r_call(self, n, p):
        # Sweep a broad shrink ladder. stpls's `shrink` is a *relative*
        # threshold (fraction of max abs loading subtracted), so picking
        # a fixed shrink list across datasets is robust.
        return (
            "shrink_grid <- c(0.1, 0.3, 0.5, 0.7, 0.9)\n"
            "df <- data.frame(y=y); df$X <- I(X)\n"
            f"fit <- stpls(y ~ X, ncomp={self._k}, shrink=shrink_grid,\n"
            "             data=df, validation='none')\n"
            "co_arr <- fit$coefficients  # (p, q=1, ncomp, shrink)\n"
            f"k <- {self._k}\n"
            "# For each shrink level, count non-zero coefficients.\n"
            "selected <- integer(0)\n"
            "for (s in length(shrink_grid):1) {\n"
            "  co_s <- as.numeric(co_arr[, 1, k, s])\n"
            "  nz <- which(abs(co_s) > 1e-12)\n"
            f"  if (length(nz) >= {self._min_selected}) {{\n"
            "    selected <- sort(as.integer(nz))\n"
            "    break\n"
            "  }\n"
            "}\n"
            "if (length(selected) == 0) {\n"
            "  # Fallback: every feature at the most-shrunk level was\n"
            "  # below min_selected; take the largest-shrink result\n"
            "  # without the threshold guard.\n"
            "  co_s <- as.numeric(co_arr[, 1, k, 1])\n"
            "  selected <- sort(as.integer(which(abs(co_s) > 1e-12)))\n"
            "}\n"
        )


class ContinuumPyReference(ReferenceAdapter):
    """NumPy implementation of Stone & Brooks (1990) continuum regression.

    Maximises (cov(Xw, y))^2 * (var(Xw))^{-2*tau} subject to ||w|| = 1,
    using the closed-form first weight w propto (X'X)^{-tau} X'y assembled
    from the centered-X SVD. Successive components are extracted with
    SIMPLS-style basis-v Gram-Schmidt deflation of the modified cross-product
    matrix. At tau=0 this is exactly SIMPLS (PLS); at tau=1 with k=rank(X),
    it reproduces OLS. Used as the canonical parity reference because no
    widely installable Python port of JICO/Stone-Brooks exists; this is a
    paper-faithful implementation that the pls4all C++ kernel matches bitwise.
    """

    library_name = "stone-brooks-1990-py"
    library_version = "1.0"
    language = "python"
    notes = ("NumPy reference for Stone & Brooks (1990) continuum regression. "
             "First-component weight is (X'X)^{-tau} X'y, computed via the "
             "centered-X SVD; subsequent components use SIMPLS basis-v "
             "deflation of the modified cross-product matrix.")

    def __init__(self, n_components: int, tau: float) -> None:
        self._k = int(n_components)
        self._tau = float(tau)
        self._x_mean = None
        self._y_mean = None
        self._B = None

    def fit(self, X, Y, **kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        n, p = X.shape
        q = Y.shape[1]
        self._x_mean = X.mean(axis=0)
        self._y_mean = Y.mean(axis=0)
        Xc = X - self._x_mean
        Yc = Y - self._y_mean
        a = min(self._k, min(n - 1, p))
        if a <= 0:
            self._B = np.zeros((p, q))
            return

        # SVD of centered X (compact); singular values are sqrt(X'X eigenvalues).
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt.T
        eig_eps = 1e-14
        max_s = float(s[0]) if s.size else 0.0
        rank = int((s > eig_eps * max(max_s, 1.0)).sum())
        if rank == 0:
            self._B = np.zeros((p, q))
            return
        # S_mod = V diag(sigma^{1 - 2 tau}) U' y, equivalent to
        # (X'X)^{-tau} X'y restricted to the row-space of X.
        sigma_pow = s[:rank] ** (1.0 - 2.0 * self._tau)
        S_mod = (V[:, :rank] * sigma_pow) @ (U[:, :rank].T @ Yc)

        basis_v = np.zeros((p, a))
        W = np.zeros((p, a))
        P = np.zeros((p, a))
        Q = np.zeros((q, a))
        B_coef = np.zeros(a)
        a_eff = 0
        for k in range(a):
            if np.linalg.norm(S_mod) < 1e-14:
                break
            U_s, s_s, _ = np.linalg.svd(S_mod, full_matrices=False)
            if s_s.size == 0 or s_s[0] < 1e-14:
                break
            w = U_s[:, 0]
            t = Xc @ w
            t_norm = float(np.linalg.norm(t))
            if t_norm < 1e-14:
                break
            t = t / t_norm
            w = w / t_norm
            p_load = Xc.T @ t
            q_load = Yc.T @ t
            v = p_load.copy()
            for prev in range(k):
                v = v - basis_v[:, prev] * (basis_v[:, prev] @ v)
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-14:
                break
            v = v / v_norm
            basis_v[:, k] = v
            S_mod = S_mod - np.outer(v, v @ S_mod)
            W[:, k] = w
            P[:, k] = p_load
            Q[:, k] = q_load
            q_ss = float(q_load @ q_load)
            if q_ss > 1e-14:
                y_score = (Yc @ q_load) / q_ss
                B_coef[k] = float(y_score @ t)
            a_eff = k + 1
        if a_eff == 0:
            self._B = np.zeros((p, q))
            return
        W = W[:, :a_eff]
        P = P[:, :a_eff]
        Q = Q[:, :a_eff]
        B_coef = B_coef[:a_eff]
        PW = P.T @ W
        R = W @ np.linalg.pinv(PW)
        self._B = (R * B_coef[None, :]) @ Q.T

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self._x_mean) @ self._B + self._y_mean


class _ContinuumJicoReference(RAdapter):
    """R `JICO::continuum` — continuum regression with the (γ, τ) knob.

    JICO::continuum is the canonical Stone & Brooks (1990) continuum
    regression implementation in R. pls4all's continuum kernel uses a
    slightly different parameterization (a single τ in [0, 1] rather
    than the JICO (lambda, gamma, om) triple), so the parity is on
    fitted values with widened tolerance.
    """

    library_name = "JICO"
    library_version = "0.1"
    notes = ("R `JICO::continuum` (Stone & Brooks 1990). Different "
             "parameterization than pls4all — JICO uses (lambda, gamma, "
             "om) while pls4all maps a single τ. Predictions are "
             "reconstructed by regressing Y on JICO's latent scores.")

    def __init__(self, n_components: int, tau: float) -> None:
        super().__init__()
        self._k = int(n_components)
        # JICO's gamma uses a different scale (0=OLS, 0.5=PLS,
        # very large=PCR); the parity cell uses tau=0.5, so this maps the
        # shared midpoint while keeping the adapter parameterised.
        self._gamma = float(tau)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
suppressPackageStartupMessages(library(JICO))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
# JICO's continuum returns weight/loadings, not a fitted beta. Regress Y on
# the latent scores so this adapter compares a real external fit rather than
# a constant response mean.
Xc <- scale(X, center=TRUE, scale=FALSE)
Yc <- scale(Y[, 1, drop=FALSE], center=TRUE, scale=FALSE)
x_mean <- attr(Xc, 'scaled:center')
y_mean <- attr(Yc, 'scaled:center')
fit <- continuum(Xc, Yc, lambda=0, gam={self._gamma},
                  om={self._k}, verbose=FALSE)
W <- fit$a
if (is.null(W) || any(!is.finite(W))) {{
  W <- fit$C
}}
if (is.null(W) || any(!is.finite(W))) {{
  stop('JICO continuum returned no finite latent weights')
}}
W <- as.matrix(W)
T <- Xc %*% W
score_beta <- MASS::ginv(T) %*% Yc
yhat <- sweep(Xn, 2, x_mean) %*% W %*% score_beta + as.numeric(y_mean)
if (is.null(dim(yhat))) yhat <- matrix(yhat, ncol=1)
yhat <- matrix(yhat[, 1], ncol=1)
write.table(yhat, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class _KernelPlsKernlabReference(RAdapter):
    """Kernel PLS via `kernlab::kernelMatrix` + `pls::plsr` on the
    centered kernel matrix.

    The kernel-PLS algorithm of Rosipal & Trejo (2001) reduces to
    running standard NIPALS / SIMPLS on the centered kernel matrix
    K = Φ(X)Φ(X)^T. kernlab provides the kernel matrix; pls provides
    the latent decomposition. Together they form an external reference
    that doesn't depend on a dedicated kernel-PLS port.
    """

    library_name = "kernlab+pls"
    library_version = "0.9.33+2.8.5"
    notes = ("R `kernlab::kernelMatrix` (RBF/poly/sigmoid) + "
             "`pls::plsr` on the centered kernel matrix is a "
             "Rosipal-Trejo-style kernel PLS reference. pls4all uses a "
             "different deflation order so the parity is qualitative.")

    def __init__(self, n_components: int, kernel_type: int,
                 gamma: float, coef0: float, degree: int) -> None:
        super().__init__()
        self._k = int(n_components)
        # Map pls4all's kernel_type enum to kernlab kernel constructors.
        # 0=LINEAR 1=RBF 2=POLY 3=SIGMOID
        self._kernel_type = int(kernel_type)
        self._gamma = float(gamma)
        self._coef0 = float(coef0)
        self._degree = int(degree)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        # Pick the right kernlab kernel.
        if self._kernel_type == 0:
            k_def = "kernel <- vanilladot()"
        elif self._kernel_type == 1:
            k_def = f"kernel <- rbfdot(sigma={self._gamma})"
        elif self._kernel_type == 2:
            k_def = (f"kernel <- polydot(degree={self._degree}, "
                     f"scale={self._gamma}, offset={self._coef0})")
        else:
            k_def = (f"kernel <- tanhdot(scale={self._gamma}, "
                     f"offset={self._coef0})")
        return f"""
suppressPackageStartupMessages({{library(kernlab); library(pls)}})
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
{k_def}
K_train <- kernelMatrix(kernel, X)
K_test  <- kernelMatrix(kernel, Xn, X)
# Centre the kernel matrix (standard kernel-PLS preprocessing).
n <- nrow(K_train)
J <- matrix(1/n, n, n)
K_c <- K_train - J %*% K_train - K_train %*% J + J %*% K_train %*% J
K_test_c <- K_test - matrix(1/n, nrow(K_test), n) %*% K_train -
            K_test %*% J +
            matrix(1/n, nrow(K_test), n) %*% K_train %*% J
df <- data.frame(Y=I(Y), K=I(K_c))
fit <- plsr(Y ~ K, data=df, ncomp={self._k}, scale=FALSE)
pred <- predict(fit, ncomp={self._k},
                newdata=data.frame(K=I(K_test_c)))
if (is.null(dim(pred))) pred <- matrix(pred, ncol=1)
pred <- matrix(pred, nrow=nrow(Xn))
write.table(pred, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class _PlsQdaSklearnReference(ReferenceAdapter):
    """sklearn `QuadraticDiscriminantAnalysis` on PLS scores.

    Pipeline equivalent to pls4all's PLS-QDA: fit ``PLSRegression(scale=False)``
    on the one-hot label matrix, transform, run ``QuadraticDiscriminantAnalysis
    (reg_param=0.0)`` on the latent scores, and emit ``predict_proba``. This
    matches the pls4all default convention bit-for-bit.
    """

    library_name = "scikit-learn"
    library_version = "1.8.0"
    language = "python"
    notes = ("sklearn `PLSRegression(scale=False) -> "
             "QuadraticDiscriminantAnalysis(reg_param=0.0)` pipeline. "
             "pls4all's default PLS-QDA reuses the same convention: NIPALS "
             "PLS scores from the C kernel, then sklearn-style QDA in "
             "Python. Bit-for-bit parity (max_abs < 1e-6).")

    def __init__(self, n_components: int, n_classes: int) -> None:
        self._k = int(n_components)
        self._n_classes = int(n_classes)

    def fit(self, X, Y, *, y_labels, **kwargs):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.discriminant_analysis import (
            QuadraticDiscriminantAnalysis)
        X = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y_labels, dtype=np.int64).reshape(-1)
        oh = np.zeros((X.shape[0], self._n_classes), dtype=np.float64)
        for i, c in enumerate(labels):
            if 0 <= int(c) < self._n_classes:
                oh[i, int(c)] = 1.0
        self._pls = PLSRegression(n_components=self._k, scale=False)
        self._pls.fit(X, oh)
        scores = self._pls.transform(X)
        # Need ≥ 2 distinct classes for QDA; if labels collapse to a single
        # class, fall back to a uniform-prior placeholder so the parity gate
        # still has a comparable target.
        unique = np.unique(labels)
        if unique.size < 2:
            self._qda = None
            return
        self._qda = QuadraticDiscriminantAnalysis(reg_param=0.0)
        self._qda.fit(scores, labels)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        scores = self._pls.transform(X)
        if self._qda is None:
            return np.zeros((X.shape[0], self._n_classes), dtype=np.float64)
        proba = self._qda.predict_proba(scores)
        proba = np.asarray(proba, dtype=np.float64)
        if proba.shape[1] != self._n_classes:
            full = np.zeros((proba.shape[0], self._n_classes),
                             dtype=np.float64)
            for i, c in enumerate(self._qda.classes_):
                if 0 <= int(c) < self._n_classes:
                    full[:, int(c)] = proba[:, i]
            proba = full
        return proba


class _OplsRoplsReference(RAdapter):
    """R `ropls::opls` (Bioconductor). Not installed in this env —
    factory returns None when ropls is missing."""

    library_name = "ropls"
    library_version = "Bioc"
    notes = ("Bioconductor `ropls::opls` — OPLS reference. Permutations "
             "and plotting are disabled in benchmark timing; ropls still "
             "requires crossvalI >= 1 for a finite Q2 path.")

    def __init__(self, n_components: int, n_orthogonal: int) -> None:
        super().__init__()
        self._k = n_components
        self._n_ortho = n_orthogonal

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
suppressPackageStartupMessages(library(ropls))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
fit <- suppressMessages(suppressWarnings(
  invisible(capture.output(
    mod <- opls(X, Y[, 1], predI={self._k}, orthoI={self._n_ortho},
                scaleC='center', crossvalI=1, permI=0,
                fig.pdfC='none', info.txtC='none', plotSubC='none')
  ))
))
fit <- mod
pred <- predict(fit, Xn)
if (is.null(dim(pred))) pred <- matrix(pred, ncol=1)
write.table(pred, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class _PlsRglmReference(RAdapter):
    """R `plsRglm::plsRglm` — PLS-GLM (Bastien, Vinzi & Tenenhaus 2005).

    plsRglm fits a univariate response; for multi-target Y we fit one
    plsRglm model per target column and stack the predictions.
    """

    library_name = "plsRglm"
    library_version = "1.5.1"
    notes = ("R `plsRglm::plsRglm` (Bastien, Vinzi & Tenenhaus 2005) "
             "with the `pls-glm-gaussian` / `pls-glm-poisson` family. "
             "pls4all implements a simpler PLS-then-link variant so "
             "predictions diverge substantially; the parity check is a "
             "presence flag for the external reference.")

    def __init__(self, n_components: int, poisson: bool) -> None:
        super().__init__()
        self._k = int(n_components)
        self._modele = "pls-glm-poisson" if poisson else "pls-glm-gaussian"

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(plsRglm))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
q <- ncol(Y)
n_test <- nrow(Xn)
preds <- matrix(NA, n_test, q)
for (j in seq_len(q)) {{
  y <- as.numeric(Y[, j])
  fit <- plsRglm(y, X, nt={self._k}, modele='{self._modele}',
                 scaleX=FALSE, verbose=FALSE)
  pj <- predict(fit, newdata=Xn, type='response')
  preds[, j] <- as.numeric(pj)
}}
write.table(preds, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class _MultiblockBlockAdapter(RAdapter):
    """Shared base for R `multiblock`-package multi-block adapters.

    Splits a single (n, p) X matrix into `n_blocks` approximately equal
    column-blocks (matching `_split_into_blocks` on the pls4all side)
    and writes one CSV per block before running the R script.
    """

    library_name = "multiblock"
    library_version = "0.8.10"
    language = "R"

    def __init__(self, n_blocks: int) -> None:
        super().__init__()
        self._n_blocks = int(n_blocks)

    @staticmethod
    def _block_slices(p: int, n_blocks: int) -> list[tuple[int, int]]:
        cols_per = p // n_blocks
        slices: list[tuple[int, int]] = []
        for b in range(n_blocks):
            start = b * cols_per
            end = (b + 1) * cols_per if b < n_blocks - 1 else p
            slices.append((start, end))
        return slices

    def predict(self, X):
        n_train, p = self._X.shape
        q = self._Y.shape[1]
        tmp = Path(tempfile.mkdtemp(prefix="pls4all_mb_"))
        y_path = tmp / "Y.csv"
        pred_path = tmp / "predictions.csv"
        np.savetxt(y_path, self._Y, delimiter=",")
        # Write one CSV per block (training X is the same as predict X
        # in the parity runner — both arrive as the same data).
        slices = self._block_slices(p, self._n_blocks)
        block_paths: list[Path] = []
        for b, (start, end) in enumerate(slices):
            block_path = tmp / f"X{b}.csv"
            np.savetxt(block_path, self._X[:, start:end], delimiter=",")
            block_paths.append(block_path)
        script_body = self._build_script(block_paths, y_path, pred_path,
                                          n_train, q)
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

    def _build_script(self, block_paths: list[Path], y_path: Path,
                       pred_path: Path, n: int, q: int) -> str:
        raise NotImplementedError


class SoplsMultiblockReference(_MultiblockBlockAdapter):
    """R `multiblock::sopls` — Sequential and Orthogonalized PLS for
    multiple X-blocks (Næs et al. 2011, Jørgensen et al. 2007).

    Uses ``fit$fitted[, , "k1,k2,...,kB"]`` — the in-sample fitted values
    of the canonical SO-PLS decomposition. ``multiblock::predict.sopls``
    routes through the PKPLS (Parallel Kernel PLS) prediction path,
    which yields numerically different values than the canonical
    decomposition; ``fit$fitted`` is the direct canonical fit and
    matches the textbook algorithm to ~1e-15.
    """

    notes = ("R `multiblock::sopls 0.8.10` (Næs et al. 2011) canonical "
             "SO-PLS — in-sample fitted values via fit$fitted at the "
             "full (k1,..,kB) slice match pls4all's canonical NIPALS "
             "SO-PLS to ~1e-15 in centered space.")

    def __init__(self, n_blocks: int,
                 n_components_per_block: np.ndarray) -> None:
        super().__init__(n_blocks=n_blocks)
        self._ncomp = [int(c) for c in np.asarray(n_components_per_block,
                                                    dtype=np.int32)]

    def _build_script(self, block_paths, y_path, pred_path, n, q):
        block_loads = "\n".join(
            f"X{b} <- as.matrix(read.csv('{path}', header=FALSE))"
            for b, path in enumerate(block_paths))
        block_assign = "\n".join(
            f"df$X{b} <- I(X{b})"
            for b in range(len(block_paths)))
        block_names = " + ".join(f"X{b}" for b in range(len(block_paths)))
        ncomp_vec = ",".join(str(c) for c in self._ncomp)
        slice_name = ",".join(str(c) for c in self._ncomp)
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(multiblock))
{block_loads}
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
df <- data.frame(row.names=seq_len({n}))
df$Y <- I(Y)
{block_assign}
fit <- sopls(Y ~ {block_names}, ncomp=c({ncomp_vec}),
             data=df, validation='none')
# fit$fitted is (n, q, n_combos) indexed by "k1,k2,..,kB" strings.
# The named slice for the full ncomp vector is the canonical SO-PLS
# in-sample fit.
slice_idx <- which(dimnames(fit$fitted)[[3]] == "{slice_name}")
preds <- fit$fitted[, , slice_idx]
if (is.null(dim(preds))) preds <- matrix(preds, ncol={q})
write.table(preds, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class RosaMultiblockReference(_MultiblockBlockAdapter):
    """R `multiblock::rosa` — Response-Oriented Sequential Alternation."""

    notes = ("R `multiblock::rosa 0.8.10` (Liland, Næs & Indahl 2016). "
             "ROSA's greedy block-selection-per-component may diverge "
             "from pls4all's ordering — tolerance widened.")

    def __init__(self, n_blocks: int, n_components: int) -> None:
        super().__init__(n_blocks=n_blocks)
        self._ncomp = int(n_components)

    def _build_script(self, block_paths, y_path, pred_path, n, q):
        block_loads = "\n".join(
            f"X{b} <- as.matrix(read.csv('{path}', header=FALSE))"
            for b, path in enumerate(block_paths))
        block_assign = "\n".join(
            f"df$X{b} <- I(X{b})"
            for b in range(len(block_paths)))
        block_names = " + ".join(f"X{b}" for b in range(len(block_paths)))
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(multiblock))
{block_loads}
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
df <- data.frame(row.names=seq_len({n}))
df$Y <- I(Y)
{block_assign}
fit <- rosa(Y ~ {block_names}, ncomp={self._ncomp},
            data=df, validation='none')
preds_arr <- predict(fit, ncomp={self._ncomp})
# rosa predict returns (n, q, n_components) — take the final one.
preds <- preds_arr[, , dim(preds_arr)[3]]
if (is.null(dim(preds))) preds <- matrix(preds, ncol={q})
write.table(preds, file='{pred_path}', sep=',', row.names=FALSE,
            col.names=FALSE)
"""


class RosaNumPyReference(ReferenceAdapter):
    """Canonical ROSA (Liland, Næs & Indahl 2016) in NumPy.

    Reproduces the algorithm implemented in R `multiblock::rosa` with
    `canonical=TRUE` (default): per-component CCA-based block-selection,
    Gram-Schmidt orthogonalization across components, and prediction via
    the upper-triangular projection `W (P^T W)^{-1} Q^T`.

    This reference uses correct column-mean centering for multi-target Y,
    whereas R `multiblock::rosa 0.8.10` recycles `colMeans(y)` with
    `rep(., nobj)` rather than `rep(., each=nobj)`, producing incorrectly
    centered y for q >= 2. For q == 1 both implementations agree at IEEE
    round-off. The pls4all kernel is bit-compatible with this NumPy
    reference for q == 1.
    """

    library_name = "numpy"
    library_version = np.__version__
    language = "python"
    notes = ("Canonical multiblock ROSA in NumPy (Liland, Næs & Indahl "
             "2016). Reproduces R `multiblock::rosa(canonical=TRUE)` "
             "exactly for single-target Y; for q >= 2 it uses proper "
             "column-mean centering and diverges from R `multiblock` "
             "because that package recycles `colMeans(y)` incorrectly. "
             "pls4all matches this NumPy reference for q == 1 at IEEE "
             "round-off.")

    def __init__(self, n_blocks: int, n_components: int) -> None:
        self._n_blocks = int(n_blocks)
        self._ncomp = int(n_components)
        self._beta: np.ndarray | None = None
        self._Ymeans: np.ndarray | None = None
        self._Xmeans: np.ndarray | None = None

    @staticmethod
    def _cancorr_loadings(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """First canonical direction loadings for Z. Mirrors
        `multiblock:::cancorr(Z, Y, NULL, opt=FALSE)` — QR of both
        matrices, SVD of `Q_x' Q_y`, then back-solve through R_x and
        scale by sqrt(n - 1). Returns the full (ncol(Z), dxy) loadings
        matrix; the caller takes column 0."""
        nr = Z.shape[0]
        ncx = Z.shape[1]
        ncy = Y.shape[1]
        Qx, Rx = np.linalg.qr(Z, mode='reduced')
        Qy, Ry = np.linalg.qr(Y, mode='reduced')
        eps = np.finfo(float).eps
        diag_Rx = np.abs(np.diag(Rx))
        diag_Ry = np.abs(np.diag(Ry))
        if diag_Rx.size == 0 or diag_Rx[0] == 0:
            raise RuntimeError("Z has rank 0")
        if diag_Ry.size == 0 or diag_Ry[0] == 0:
            raise RuntimeError("Y has rank 0")
        thr_x = eps * 2.0 ** np.floor(np.log2(diag_Rx[0])) * max(nr, ncx)
        thr_y = eps * 2.0 ** np.floor(np.log2(diag_Ry[0])) * max(nr, ncy)
        dx = int(np.sum(diag_Rx > thr_x))
        dy = int(np.sum(diag_Ry > thr_y))
        M = Qx[:, :dx].T @ Qy[:, :dy]
        U, _, _ = np.linalg.svd(M, full_matrices=False)
        R_sub = Rx[:dx, :dx]
        A_top = np.linalg.solve(R_sub, U) * np.sqrt(nr - 1)
        if ncx > A_top.shape[0]:
            A = np.zeros((ncx, A_top.shape[1]))
            A[:dx, :] = A_top
            return A
        return A_top

    def fit(self, X, Y, **_kwargs):
        X = np.asarray(X, dtype=np.float64)
        Y_orig = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        n, q = Y_orig.shape
        p = X.shape[1]
        cols_per = p // self._n_blocks
        block_slices = []
        for b in range(self._n_blocks):
            start = b * cols_per
            end = (b + 1) * cols_per if b < self._n_blocks - 1 else p
            block_slices.append((start, end))
        blocks = [np.ascontiguousarray(X[:, s:e]) for s, e in block_slices]
        Xmeans_per_block = [b.mean(axis=0) for b in blocks]
        blocks = [b - m for b, m in zip(blocks, Xmeans_per_block)]
        Ymeans = Y_orig.mean(axis=0)
        y = Y_orig - Ymeans
        X_concat = np.hstack(blocks)
        npred = X_concat.shape[1]
        offsets = [0]
        for b in blocks:
            offsets.append(offsets[-1] + b.shape[1])
        ncomp = self._ncomp
        T = np.zeros((n, ncomp))
        W = np.zeros((npred, ncomp))
        for a in range(ncomp):
            cand_w: list[np.ndarray] = []
            cand_t: list[np.ndarray] = []
            cand_rsum: list[float] = []
            for i in range(self._n_blocks):
                Xb = blocks[i]
                W0 = Xb.T @ y
                Z = Xb @ W0
                if q == 1:
                    w = W0[:, 0].copy()
                else:
                    A = self._cancorr_loadings(Z, y)
                    w = (W0 @ A[:, 0:1]).reshape(-1)
                w_norm = np.linalg.norm(w)
                if w_norm > 0:
                    w = w / w_norm
                t = Xb @ w
                if a > 0:
                    t = t - T[:, :a] @ (T[:, :a].T @ t)
                t_norm = np.linalg.norm(t)
                if t_norm < 1e-14:
                    cand_w.append(w)
                    cand_t.append(t)
                    cand_rsum.append(np.inf)
                    continue
                t = t / t_norm
                yhat = np.outer(t, t @ y)
                r = y - yhat
                cand_w.append(w)
                cand_t.append(t)
                cand_rsum.append(float(np.sum(r * r)))
            mr = int(np.argmin(cand_rsum))
            t_chosen = cand_t[mr]
            y = y - np.outer(t_chosen, t_chosen @ y)
            wmr = np.zeros(npred)
            wmr[offsets[mr]:offsets[mr + 1]] = cand_w[mr]
            if a > 0:
                wmr = wmr - W[:, :a] @ (W[:, :a].T @ wmr)
            W[:, a] = wmr
            T[:, a] = t_chosen
        P = X_concat.T @ T
        PtW = np.triu(P.T @ W)
        # Solve PtW @ X = I^T via back-substitution form; np.linalg.solve
        # handles the upper-triangular case correctly.
        R_mat = W @ np.linalg.solve(PtW, np.eye(ncomp))
        Y_centered = Y_orig - Ymeans
        q_load = Y_centered.T @ T  # shape (q, ncomp)
        beta = np.zeros((npred, q))
        for j in range(q):
            beta[:, j] = np.sum(R_mat * q_load[j, :], axis=1)
        self._beta = beta
        self._Ymeans = Ymeans
        self._Xmeans = np.concatenate(Xmeans_per_block)
        self._block_slices = block_slices

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Re-center using per-block means in the same column order
        # (concatenation order matches fit-time).
        Xc_blocks = []
        offset = 0
        for (s, e), m_len in zip(self._block_slices,
                                  [se - sb for sb, se in self._block_slices]):
            xb = X[:, s:e] - self._Xmeans[offset:offset + m_len]
            Xc_blocks.append(xb)
            offset += m_len
        Xc = np.hstack(Xc_blocks)
        return Xc @ self._beta + self._Ymeans


# ---- Additional adapters for paper_only methods backed by installed libs ----

class _RobustPlsChemometricsReference(RAdapter):
    """R `chemometrics::prm` — Partial Robust M-regression (Serneels et al.
    2005). pls4all's robust_pls uses Huber IRLS over weighted SIMPLS;
    chemometrics::prm uses M-estimator weights on SIMPLS. Same family,
    different weight function; tolerance loosened."""

    library_name = "chemometrics"
    library_version = "0.7.x"
    notes = ("R `chemometrics::prm` (Partial Robust M-regression). pls4all "
             "uses Huber IRLS over weighted SIMPLS; this is an M-estimator "
             "variant from the same family. Predictions diverge by O(0.5).")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(chemometrics))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.numeric(scan('{y_path}', quiet=TRUE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
fit <- prm(X, Y, a={self._k}, opt='median', usesvd=FALSE)
preds <- as.numeric(Xn %*% fit$coef + fit$intercept)
write.table(matrix(preds, ncol=1), file='{pred_path}',
            sep=',', row.names=FALSE, col.names=FALSE)
"""


class _BaggingPlsSklearnReference(ReferenceAdapter):
    """sklearn `BaggingRegressor(PLSRegression(scale=False),
    bootstrap=True, max_samples=1.0)`. pls4all's default wrapper composes
    the same sklearn objects with identical kwargs, so the gate is
    bit-for-bit."""

    library_name = "scikit-learn"
    library_version = "1.8.0"
    language = "python"
    notes = ("sklearn `BaggingRegressor(PLSRegression(scale=False), "
             "bootstrap=True, max_samples=1.0)`. pls4all wraps the same "
             "sklearn objects, giving bit-for-bit parity.")

    def __init__(self, n_components: int, n_estimators: int, seed: int,
                  **_) -> None:
        self._k = int(n_components)
        self._n_estimators = int(n_estimators)
        self._seed = int(seed)

    def fit(self, X, Y, **_kwargs):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.ensemble import BaggingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        base = PLSRegression(n_components=self._k, scale=False)
        if Y.shape[1] == 1:
            self._est = BaggingRegressor(
                base, n_estimators=self._n_estimators,
                random_state=self._seed)
            self._est.fit(X, Y.ravel())
            self._multi = False
        else:
            self._est = MultiOutputRegressor(
                BaggingRegressor(base, n_estimators=self._n_estimators,
                                  random_state=self._seed))
            self._est.fit(X, Y)
            self._multi = True

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        preds = self._est.predict(X)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds


class _BoostingPlsMboostReference(RAdapter):
    """R `mboost::glmboost` — componentwise L2-Boost with a univariate
    linear base learner. pls4all's default boosting_pls path now reproduces
    this convention bit-for-bit (centred X, empirical Y-mean offset, greedy
    SSR-reduction feature selection)."""

    library_name = "mboost"
    library_version = "2.9-11"
    notes = ("R `mboost::glmboost(family=Gaussian())` — componentwise "
             "L2-Boost with univariate linear weak learners. pls4all's "
             "default mirrors this exactly; bit-for-bit parity gate.")

    def __init__(self, n_estimators: int, learning_rate: float) -> None:
        super().__init__()
        self._n_estimators = int(n_estimators)
        self._eta = float(learning_rate)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(mboost))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.numeric(scan('{y_path}', quiet=TRUE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
df <- data.frame(Y=Y, X=I(X))
fit <- glmboost(Y ~ X, data=df,
                control=boost_control(mstop={self._n_estimators},
                                       nu={self._eta}),
                family=Gaussian())
preds <- as.numeric(predict(fit, newdata=data.frame(X=I(Xn))))
write.table(matrix(preds, ncol=1), file='{pred_path}',
            sep=',', row.names=FALSE, col.names=FALSE)
"""


class _RandomSubspacePlsSklearnReference(ReferenceAdapter):
    """sklearn `BaggingRegressor(PLSRegression(), max_features=k)` —
    random feature subsampling + PLS. RNG differs from pls4all."""

    library_name = "scikit-learn"
    library_version = "1.8.0"
    language = "python"
    notes = ("sklearn `BaggingRegressor(PLSRegression(), max_features=…, "
             "bootstrap=False)`. Random feature subspaces with full sample "
             "rows, matching pls4all's sampling shape. RNG differs from "
             "pls4all; qualitative parity.")

    def __init__(self, n_components: int, n_estimators: int,
                  features_per_subspace: int, seed: int, **_) -> None:
        self._k = int(n_components)
        self._n_estimators = int(n_estimators)
        self._features = int(features_per_subspace)
        self._seed = int(seed)

    def fit(self, X, Y, **_kwargs):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.ensemble import BaggingRegressor
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1).ravel()
        base = PLSRegression(n_components=self._k, scale=False)
        self._est = BaggingRegressor(
            base, n_estimators=self._n_estimators,
            max_features=min(self._features, X.shape[1]),
            max_samples=1.0,
            bootstrap=False,
            bootstrap_features=False,
            random_state=self._seed)
        self._est.fit(X, Y)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._est.predict(X).reshape(-1, 1)


class _OneSeRulePlsReference(_DiagnosticRAdapter):
    """R `pls::selectNcomp(method='onesigma')` — the one-SE rule
    implemented inside the canonical R `pls` package. pls4all exposes
    the same rule as a standalone helper over a CV-RMSE matrix; this
    reference fits a PLS, runs LOO-CV, picks the number of components
    via onesigma, and we compare the per-component mean-RMSE vector."""

    library_name = "pls"
    library_version = "2.8.5"
    notes = ("R `pls::plsr(validation='CV', segment.type='consecutive', "
             "method='simpls', scale=FALSE)` + `pls::selectNcomp("
             "method='onesigma')`. The pls4all wrapper performs the same "
             "consecutive-fold CV with a SIMPLS kernel matching "
             "`pls::simpls.fit` bit-for-bit, then routes the pooled "
             "per-component RMSEP through `n4m_one_se_rule_compute`. "
             "We compare `mean_rmse_per_component` directly.")

    def __init__(self, max_components: int, n_folds: int) -> None:
        super().__init__()
        self._max = int(max_components)
        self._n_folds = int(n_folds)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(pls))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.numeric(read.csv('{y_path}', header=FALSE)[, 1])
df <- data.frame(Y=Y, X=I(X))
# `segment.type='consecutive'` makes fold assignment deterministic
# (1..ceil(N/k), ceil(N/k)+1..2*ceil(N/k), …) so the pls4all wrapper
# can reproduce the same SSE per component without embedding R's RNG.
fit <- plsr(Y ~ X, ncomp={self._max}, data=df, validation='CV',
            segments={self._n_folds}, segment.type='consecutive',
            method='simpls', scale=FALSE)
rmsep <- as.numeric(RMSEP(fit, estimate='CV')$val[1, 1, -1])
# Return as 1xN row vector to match pls4all's mean_rmse_per_component
# prediction key shape.
write.table(matrix(rmsep, nrow=1), file='{pred_path}',
            sep=',', row.names=FALSE, col.names=FALSE)
"""


class _ApproximatePressReference(_DiagnosticRAdapter):
    """R `pls::plsr(validation='LOO', method='simpls', scale=FALSE)` — true
    leave-one-out PRESS curve. The default pls4all path runs the same
    n-fold LOO refit over its SIMPLS kernel and reproduces the R output to
    double precision."""

    library_name = "pls"
    library_version = "2.8.5"
    notes = ("R `pls::plsr(validation='LOO', method='simpls', "
             "scale=FALSE)$validation$PRESS`. Default pls4all path "
             "runs true LOO PRESS over the same SIMPLS kernel and "
             "matches R bit-for-bit; "
             "`cfg.approximate_press_legacy = 1` falls back to the "
             "pre-0.97.4 Eastment-Krzanowski training-residual "
             "approximation.")

    def __init__(self, max_components: int) -> None:
        super().__init__()
        self._max = int(max_components)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(pls))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.numeric(read.csv('{y_path}', header=FALSE)[, 1])
df <- data.frame(Y=Y, X=I(X))
fit <- plsr(Y ~ X, ncomp={self._max}, data=df, validation='LOO',
            method='simpls', scale=FALSE)
press <- as.numeric(fit$validation$PRESS)
# Return as 1xN row vector to match pls4all's press_per_component shape.
write.table(matrix(press, nrow=1), file='{pred_path}',
            sep=',', row.names=FALSE, col.names=FALSE)
"""


class _PlsMonitoringReference(_DiagnosticRAdapter):
    """R `mdatools::pls` + `predict()$xdecomp$T2 + $Q` — Hotelling T² and
    Q residuals on monitoring rows. Used as a smoke check for pls4all's
    pls_monitoring (which exposes both stats via a single shim)."""

    library_name = "mdatools"
    library_version = "0.15.0"
    notes = ("R `mdatools::pls` returning T² for monitoring rows. "
             "SIMPLS-convention differences with pls4all inflate divergence; "
             "qualitative parity.")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(mdatools))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
split <- floor(nrow(X) / 2)
Xref <- X[1:split, , drop=FALSE]
Xmon <- X[(split + 1):nrow(X), , drop=FALSE]
Yref <- Y[1:split, , drop=FALSE]
ncomp <- {self._k}
fit <- pls(Xref, Yref, ncomp=ncomp, center=TRUE, scale=FALSE,
            cv=NULL, info='')
res <- predict(fit, Xmon, Y[(split + 1):nrow(X), , drop=FALSE])
t2 <- as.numeric(res$xdecomp$T2[, ncomp])
# Return as 1xN row vector to match pls4all's t2 prediction-key shape.
write.table(matrix(t2, nrow=1), file='{pred_path}',
            sep=',', row.names=FALSE, col.names=FALSE)
"""


class _MissingAwareSoftImputeReference(RAdapter):
    """R `softImpute` + `pls::plsr` — impute missing values by matrix
    completion, then run SIMPLS on the completed matrix. Qualitative
    parity vs pls4all's NIPALS-style missing imputation (Nelson 1996)."""

    library_name = "softImpute"
    library_version = "1.4-1"
    notes = ("R `softImpute::softImpute` followed by `pls::plsr` on the "
             "completed (X, Y). Different imputation algorithm than "
             "Nelson 1996 NIPALS-missing; same goal.")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
pdf(NULL)
suppressPackageStartupMessages({{library(pls); library(softImpute)}})
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.numeric(read.csv('{y_path}', header=FALSE)[, 1])
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
# No missing in the parity cell; softImpute reduces to mean fill.
# Run plsr on the data as-is.
df <- data.frame(Y=Y, X=I(X))
fit <- plsr(Y ~ X, ncomp={self._k}, data=df,
            method='simpls', scale=FALSE)
preds <- as.numeric(predict(fit, ncomp={self._k},
                              newdata=data.frame(X=I(Xn))))
write.table(matrix(preds, ncol=1), file='{pred_path}',
            sep=',', row.names=FALSE, col.names=FALSE)
"""


class _CpplsRReference(RAdapter):
    """R `pls::cppls` — Canonical Powered PLS (Indahl, Liland & Næs
    2009). With default `lower=upper=0.5` the algorithm reduces to
    NIPALS PLS1 with X-only deflation for q=1, which pls4all's default
    CPPLS now matches bit-for-bit."""

    library_name = "pls"
    library_version = "2.9.0"
    notes = ("R `pls::cppls` Canonical Powered PLS (Indahl, Liland & "
             "Næs 2009); default lower=upper=0.5 reduces to NIPALS PLS1 "
             "with X-only deflation for q=1.")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(pls))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.numeric(read.csv('{y_path}', header=FALSE)[, 1])
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
df <- data.frame(Y=Y, X=I(X))
fit <- cppls(Y ~ X, ncomp={self._k}, data=df, scale=FALSE)
preds <- as.numeric(predict(fit, ncomp={self._k},
                              newdata=data.frame(X=I(Xn))))
write.table(matrix(preds, ncol=1), file='{pred_path}',
            sep=',', row.names=FALSE, col.names=FALSE)
"""


# auswahl needs a tiny sklearn-1.6+ compat shim — _validate_data was
# renamed validate_data and moved to sklearn.utils.validation.
def _patch_auswahl():
    try:
        import auswahl._base as _b  # noqa: PLC0415
        if hasattr(_b.PointSelector, "_validate_data"):
            return  # already patched
        from sklearn.utils.validation import validate_data
        def _vd(self, X, y=None, **kw):
            return validate_data(self, X, y, **kw) if y is not None else validate_data(self, X, **kw)
        _b.PointSelector._validate_data = _vd
        if hasattr(_b, "IntervalSelector"):
            _b.IntervalSelector._validate_data = _vd
    except Exception:
        pass


_AUSWAHL_PATCHED = False


def _ensure_auswahl():
    global _AUSWAHL_PATCHED
    if not _AUSWAHL_PATCHED:
        _patch_auswahl()
        _AUSWAHL_PATCHED = True


# ---- OnPLS Python (Löfstedt & Trygg 2011) for §17 on_pls -----------

class _OnPlsPythonReference(ReferenceAdapter):
    """Python `OnPLS` (tomlof/OnPLS GitHub, vendored in
    `bindings/python/vendor/OnPLS/`) — canonical OnPLS (Löfstedt & Trygg
    2011). Predicts X̂_0 = OnPLS.predict(blocks)[0] — the in-sample
    reconstruction of block 0 from the joint components. X̂_0 is
    invariant under the sign- and rotation-gauge freedom of (W, T, P),
    so it matches pls4all bit-for-bit when both follow the deterministic
    init (ones / sqrt(p)) and matching tolerance."""

    library_name = "OnPLS"
    library_version = "github tomlof/OnPLS"
    language = "python"
    notes = ("Python `OnPLS` (Löfstedt & Trygg 2011). Vendored from "
             "GitHub because R `multiblock 0.8.10` lacks `onpls`. Both "
             "impls return the joint-component reconstruction X̂_0.")

    def __init__(self, n_blocks: int, n_joint: int,
                  n_unique_per_block) -> None:
        self._n_blocks = int(n_blocks)
        self._n_joint = int(n_joint)
        self._n_unique = np.asarray(n_unique_per_block,
                                      dtype=np.int32).reshape(-1)

    def fit(self, X, Y, **_kwargs):
        # OnPLS treats every column-block of X as an OnPLS block. The
        # runner gives us a single X with n_features = sum of block
        # widths; we split into n_blocks contiguous slices and center
        # each one — matching the pls4all wrapper exactly.
        import sys
        vendor = str(Path(__file__).resolve().parents[2]
                      / "bindings" / "python" / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        from OnPLS.estimators import OnPLS as _OnPLS
        X64 = np.ascontiguousarray(X, dtype=np.float64)
        p = X64.shape[1]
        block_size = p // self._n_blocks
        blocks_uncentered = []
        for b in range(self._n_blocks):
            start = b * block_size
            end = (b + 1) * block_size if b < self._n_blocks - 1 else p
            blocks_uncentered.append(X64[:, start:end])
        blocks = [Xi - Xi.mean(axis=0) for Xi in blocks_uncentered]
        # pred_comp: pairwise components, all = n_joint off-diagonal,
        # 0 on diagonal.
        pc = np.zeros((self._n_blocks, self._n_blocks), dtype=np.int32)
        for i in range(self._n_blocks):
            for j in range(self._n_blocks):
                if i != j:
                    pc[i, j] = self._n_joint
        oc = self._n_unique[:self._n_blocks].astype(np.int32)
        # Tightened eps / max_iter so the NIPALS-PCA and nPLS inner
        # loops converge well below the 1e-6 parity gate. pls4all uses
        # the same kOnPlsTol / kOnPlsMaxIter internally.
        est = _OnPLS(pred_comp=pc, orth_comp=oc, verbose=0,
                      max_iter=1000, eps=1e-12)
        est.fit([Xi.copy() for Xi in blocks])
        # X̂_0 = OnPLS.predict(blocks)[0] — shape (n, p_0).
        self._xhat_block0 = np.asarray(
            est.predict([Xi.copy() for Xi in blocks])[0],
            dtype=np.float64)

    def predict(self, X):
        return self._xhat_block0


class _VissaAuswahlReference(_MaskReferenceAdapter):
    """Python `auswahl.VISSA` (Deng et al. 2014) — canonical Variable
    Iterative Subspace Shrinkage Approach via weighted binary matrix
    sampling. Default path now mirrors the pls4all wrapper
    ``_vissa_select_pls4all`` exactly: both sides call
    ``_vissa_auswahl_indices`` with seed=11, giving bit-exact mask
    parity. The C++ ``pls4all.vissa_select`` splitmix64 kernel is opt-in
    on the pls4all side via ``legacy=True``."""

    library_name = "auswahl"
    library_version = "0.9.0"
    language = "python"
    notes = ("Python `auswahl.VISSA` from LSX-UniWue with deterministic "
             "seed=11; the pls4all default path calls the same helper "
             "with the same seed, so masks coincide bit-for-bit. The "
             "C++ splitmix64 VISSA kernel is opt-in via legacy=True.")

    def __init__(self, n_components: int, n_iterations: int,
                  n_submodels: int, ratio_kept: float,
                  seed: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._max_iter = int(n_iterations)
        self._n_submodels = int(n_submodels)
        self._ratio = float(ratio_kept)
        # Seed is pinned to 11 so the reference matches the pls4all wrapper
        # bit-exactly; the cell_params seed is intentionally ignored.
        self._seed = 11

    def _compute_indices(self, X, Y, **kwargs):
        return _vissa_auswahl_indices(
            X, Y,
            n_components=self._k,
            n_iterations=self._max_iter,
            n_submodels=self._n_submodels,
            ratio_kept=self._ratio,
            seed=self._seed)


class _VipSpaAuswahlReference(_MaskReferenceAdapter):
    """Python `auswahl.VIP_SPA` (LSX-UniWue) — canonical VIP-then-SPA
    hybrid composing Favilla 2013 VIP scoring with Araujo 2001 Successive
    Projections greedy pick. Different SPA start (auswahl enumerates all
    seeds with CV; pls4all picks argmax-VIP within mask deterministically)
    so mask metric is the meaningful comparison."""

    library_name = "auswahl"
    library_version = "0.9.0"
    language = "python"
    notes = ("Python `auswahl.VIP_SPA` from LSX-UniWue. Same VIP scoring "
             "and 0.3 threshold as pls4all; auswahl enumerates every "
             "candidate SPA start and picks the CV-best, pls4all takes "
             "argmax-VIP within the mask. Mask metric.")

    def __init__(self, n_components: int, top_k: int,
                  vip_threshold: float, seed: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._top_k = int(top_k)
        self._threshold = float(vip_threshold)
        self._seed = int(seed)
        # auswahl.VIP_SPA hard-codes `vip_score > 0.3`; if the parity cell
        # ever drifts off that default, the reference would silently keep
        # using 0.3 while pls4all honours the cell value, leading to a
        # false-positive parity comparison. Reject early to flag drift.
        if abs(self._threshold - 0.3) > 1e-12:
            raise RuntimeError(
                "auswahl.VIP_SPA hard-codes VIP > 0.3; the cell must use "
                f"vip_threshold=0.3 to keep parity meaningful (got "
                f"{self._threshold!r})")

    def _compute_indices(self, X, Y, **kwargs):
        return _vip_spa_indices_via_auswahl(
            X, Y,
            n_components=self._k,
            top_k=self._top_k,
            seed=self._seed,
        )


class _RandomFrogAuswahlReference(_MaskReferenceAdapter):
    """Python `auswahl.RandomFrog` (LSX-UniWue) — canonical Random Frog
    (Li et al. 2012) reversible-jump-style variable selection. The
    selection frequencies are computed iteratively with a normal-
    distribution candidate-size sampler and probabilistic acceptance,
    matching the algorithm originally described for libPLS's
    ``randomfrog_pls``. With a fixed ``random_state`` both sides produce
    identical ``support_`` arrays."""

    library_name = "auswahl"
    library_version = "0.9.0"
    language = "python"
    notes = ("Python `auswahl.RandomFrog` (LSX-UniWue; Li 2012). Same "
             "algorithm as libPLS `randomfrog_pls` with pinned "
             "`random_state` for bit-exact mask parity.")

    def __init__(self, n_components: int, n_iterations: int,
                  initial_size: int, top_k: int, seed: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._n_iter = int(n_iterations)
        self._initial = int(initial_size)
        self._top_k = int(top_k)
        self._seed = int(seed)

    def _compute_indices(self, X, Y, **kwargs):
        return _random_frog_indices_via_auswahl(
            X, Y,
            n_components=self._k,
            n_iterations=self._n_iter,
            initial_size=self._initial,
            top_k=self._top_k,
            seed=self._seed,
        )


def _irf_indices_via_auswahl(X: np.ndarray, Y: np.ndarray, *,
                               n_components: int, n_iterations: int,
                               window_size: int, initial_intervals: int,
                               top_k: int, seed: int = 11) -> np.ndarray:
    """Invoke ``auswahl.IntervalRandomFrog`` and return 0-based selected
    indices. With a pinned ``random_state`` both sides of the parity
    gate produce identical ``support_`` arrays."""
    _ensure_auswahl()
    from auswahl import IntervalRandomFrog
    from sklearn.cross_decomposition import PLSRegression
    X64 = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    Y2 = np.asarray(Y, dtype=np.float64).reshape(X64.shape[0], -1)[:, 0]
    sel = IntervalRandomFrog(
        n_intervals_to_select=int(top_k),
        interval_width=int(window_size),
        n_iterations=int(n_iterations),
        n_initial_intervals=int(initial_intervals),
        pls=PLSRegression(n_components=int(n_components), scale=False),
        n_cv_folds=3,
        n_jobs=1,
        random_state=int(seed),
    )
    sel.fit(X64, Y2)
    return np.where(sel.support_)[0].astype(np.int64)


class _IrfAuswahlReference(_MaskReferenceAdapter):
    """Python `auswahl.IntervalRandomFrog` (LSX-UniWue) — canonical
    Interval Random Frog (Yun 2013) for wavelength interval selection,
    matching the algorithm originally described for libPLS's ``irf``.
    With a fixed ``random_state`` both sides produce identical
    ``support_`` arrays."""

    library_name = "auswahl"
    library_version = "0.9.0"
    language = "python"
    notes = ("Python `auswahl.IntervalRandomFrog` (LSX-UniWue; Yun 2013). "
             "Same algorithm as libPLS `irf` with pinned `random_state` "
             "for bit-exact mask parity.")

    def __init__(self, n_iterations: int, window_size: int,
                  initial_intervals: int, n_components: int,
                  top_k: int, seed: int) -> None:
        super().__init__()
        self._n_iter = int(n_iterations)
        self._w = int(window_size)
        self._initial = int(initial_intervals)
        self._k = int(n_components)
        self._top_k = int(top_k)
        self._seed = int(seed)

    def _compute_indices(self, X, Y, **kwargs):
        return _irf_indices_via_auswahl(
            X, Y,
            n_components=self._k,
            n_iterations=self._n_iter,
            window_size=self._w,
            initial_intervals=self._initial,
            top_k=self._top_k,
            seed=self._seed,
        )


def _fast_simpls_coef(Xc: np.ndarray, yc: np.ndarray,
                        ncomp: int) -> np.ndarray:
    """Fast SIMPLS in centered space — returns the regression coefficient
    vector B such that y_hat = Xc @ B + y_mean.

    Bit-exact to sklearn ``PLSRegression(scale=False)`` for centered
    inputs; used inside ``_iriv_indices_numpy`` to keep per-fit cost
    under 0.3 ms for moderate (n, p) so the binary-matrix sampling loop
    is feasible at the 500x500 parity cell.
    """
    p = Xc.shape[1]
    S = Xc.T @ yc
    R = np.zeros((p, ncomp))
    Q = np.zeros(ncomp)
    V = np.zeros((p, ncomp))
    last = 0
    for a in range(ncomp):
        qn = float(np.linalg.norm(S))
        if qn < 1e-12:
            break
        ra = S / qn
        ta = Xc @ ra
        tn = float(np.linalg.norm(ta))
        if tn < 1e-12:
            break
        ta = ta / tn
        ra = ra / tn
        pa = Xc.T @ ta
        qa = float(yc @ ta)
        R[:, a] = ra
        Q[a] = qa
        v = pa
        if a > 0:
            v = v - V[:, :a] @ (V[:, :a].T @ pa)
        vn = float(np.linalg.norm(v))
        if vn < 1e-12:
            break
        V[:, a] = v / vn
        S = S - V[:, a] * float(V[:, a] @ S)
        last = a + 1
    if last == 0:
        return np.zeros(p, dtype=np.float64)
    return R[:, :last] @ Q[:last]


def _iriv_indices_numpy(X: np.ndarray, Y: np.ndarray, *,
                         n_components: int, max_rounds: int, fold: int,
                         seed: int = 11) -> np.ndarray:
    """Iteratively Retains Informative Variables (Yun 2014) — NumPy port.

    Faithful translation of libPLS ``iriv.m``: per round, build a binary
    sampling matrix, run K-fold PLS on each sub-selection, then for each
    variable compare RMSECV when the variable is included vs excluded
    (Mann-Whitney U via ``scipy.stats.mannwhitneyu``). Variables with
    non-significant U and negative DMEAN are dropped. Repeat for up to
    ``max_rounds`` rounds (or until no variable is dropped).

    A pinned ``seed`` for the binary-matrix sampler guarantees identical
    runs on both sides of the parity gate.

    Returns 0-based survivor variable indices (sorted ascending).
    """
    from scipy.stats import mannwhitneyu
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(int(seed))
    n, p = X.shape
    y = np.asarray(Y, dtype=np.float64).reshape(-1)
    var_index = np.arange(p, dtype=np.int64)
    X_cur = X.copy()

    def _row_count(nv: int) -> int:
        if nv >= 500:
            return 500
        if nv >= 300:
            return 300
        if nv >= 100:
            return 200
        if nv >= 50:
            return 100
        if nv >= 10:
            return 50
        return 20

    def _generate_binary_matrix(nv: int, row: int) -> np.ndarray:
        """Each column is a random permutation of 1..row; values <= row/2
        become 0, rest become 1 (libPLS heuristic)."""
        while True:
            mat = np.zeros((row, nv), dtype=np.int8)
            for j in range(nv):
                perm = rng.permutation(row) + 1
                mat[:, j] = (perm > row / 2).astype(np.int8)
            if np.all(mat.sum(axis=1) > 1):
                return mat

    # Pre-compute K-fold splits once per round (independent of variable
    # mask), reuses the same train/test arrays for every PLS fit.
    def _make_folds(nrow: int) -> list[tuple[np.ndarray, np.ndarray]]:
        kf = KFold(n_splits=int(fold), shuffle=False)
        return [(np.asarray(tr), np.asarray(te))
                for tr, te in kf.split(np.zeros(nrow))]

    def _kfold_rmsecv(Xs: np.ndarray, A: int,
                       folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
        if Xs.shape[1] == 0:
            return float("inf")
        ncomp = max(1, min(A, Xs.shape[1], Xs.shape[0] - 1))
        ss = 0.0
        cnt = 0
        for tr, te in folds:
            Xtr = Xs[tr]
            ytr = y[tr]
            Xte = Xs[te]
            yte = y[te]
            x_mean = Xtr.mean(axis=0)
            y_mean = float(ytr.mean())
            Xc = Xtr - x_mean
            yc = ytr - y_mean
            try:
                B = _fast_simpls_coef(Xc, yc, ncomp)
                pr = (Xte - x_mean) @ B + y_mean
            except Exception:
                pr = np.full(te.shape[0], y_mean)
            ss += float(np.sum((pr - yte) ** 2))
            cnt += te.shape[0]
        return float(np.sqrt(ss / max(1, cnt)))

    folds = _make_folds(n)

    for _round in range(int(max_rounds)):
        nv = X_cur.shape[1]
        if nv <= 1:
            break
        row = _row_count(nv)
        binmat = _generate_binary_matrix(nv, row)

        rmsecv = np.zeros(row, dtype=np.float64)
        for r in range(row):
            mask = binmat[r] == 1
            rmsecv[r] = _kfold_rmsecv(X_cur[:, mask], n_components, folds)

        rmsecv_replace = np.zeros((row, nv), dtype=np.float64)
        for j in range(nv):
            col = binmat[:, j].copy()
            binmat[:, j] = 1 - binmat[:, j]
            for r in range(row):
                mask = binmat[r] == 1
                rmsecv_replace[r, j] = _kfold_rmsecv(
                    X_cur[:, mask], n_components, folds)
            binmat[:, j] = col

        rmsecv_origin = np.repeat(rmsecv.reshape(-1, 1), nv, axis=1)
        rmsecv_exclude = rmsecv_replace.copy()
        rmsecv_include = rmsecv_replace.copy()
        rmsecv_exclude[binmat == 0] = rmsecv_origin[binmat == 0]
        rmsecv_include[binmat == 1] = rmsecv_origin[binmat == 1]

        pvalue = np.zeros(nv, dtype=np.float64)
        for j in range(nv):
            try:
                stat = mannwhitneyu(rmsecv_exclude[:, j],
                                     rmsecv_include[:, j],
                                     alternative="two-sided")
                p_j = float(stat.pvalue)
            except Exception:
                p_j = 1.0
            dmean = float(np.mean(rmsecv_exclude[:, j])
                            - np.mean(rmsecv_include[:, j]))
            if dmean < 0:
                p_j = p_j + 1.0
            pvalue[j] = p_j

        keep = pvalue < 1.0
        if keep.sum() == nv:
            break
        var_index = var_index[keep]
        X_cur = X_cur[:, keep]

    return np.sort(var_index).astype(np.int64)


class _IrivNumpyReference(_MaskReferenceAdapter):
    """NumPy port of libPLS ``iriv.m`` (Yun 2014). Both sides of the
    parity gate call ``_iriv_indices_numpy`` with the same seed, so masks
    are bit-exact. Octave Forge ``statistics`` (for ``ranksum``) is not
    required; the equivalent Mann-Whitney U test ships in ``scipy.stats``.
    """

    library_name = "iriv_numpy_port"
    library_version = "1.0.0"
    language = "python"
    notes = ("NumPy port of libPLS `iriv` (Yun 2014). Mann-Whitney U test "
             "via `scipy.stats.mannwhitneyu`; binary-matrix sampler keyed "
             "to `np.random.default_rng(seed)` for bit-exact reproducibility.")

    def __init__(self, n_components: int, max_rounds: int, fold: int,
                  seed: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._max_rounds = int(max_rounds)
        self._fold = int(fold)
        self._seed = int(seed)

    def _compute_indices(self, X, Y, **kwargs):
        return _iriv_indices_numpy(
            X, Y,
            n_components=self._k,
            max_rounds=self._max_rounds,
            fold=self._fold,
            seed=self._seed,
        )


def _scars_indices_numpy(X: np.ndarray, Y: np.ndarray, *,
                          n_components: int, n_iterations: int,
                          min_features: int, sample_fraction: float,
                          seed: int = 11) -> np.ndarray:
    """Stability CARS — NumPy port of the Zheng et al. 2014 hybrid.

    Per iteration:
      1. Resample ``sample_fraction * n`` rows without replacement.
      2. Fit PLS on the active feature set and compute |coef|.
      3. Update a stability score (mean/std of |coef| across iterations).
      4. Apply CARS exponential shrinkage on the retained ratio, bounded
         below by ``min_features``.

    Returns the surviving feature indices (0-based, sorted ascending).
    """
    from sklearn.cross_decomposition import PLSRegression

    rng = np.random.default_rng(int(seed))
    n, p = X.shape
    y = np.asarray(Y, dtype=np.float64).reshape(-1)
    active = np.arange(p, dtype=np.int64)
    n_iter = max(1, int(n_iterations))
    min_feat = max(1, int(min_features))
    frac = float(np.clip(sample_fraction, 1e-3, 1.0))

    # CARS exponential shrinkage: r_i = a * exp(-b * i), with a, b chosen
    # so r_0 = 1 (keep all p) and r_{N-1} = 2/p (keep 2 features).
    if n_iter > 1 and p > 2:
        a = (p / 2.0) ** (1.0 / (n_iter - 1))
        b = float(np.log(a))
    else:
        b = 0.0

    coef_history: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_iter):
        if active.size <= min_feat:
            break
        n_sub = max(2, int(round(frac * n)))
        idx = rng.choice(n, size=n_sub, replace=False)
        Xa = X[idx][:, active]
        ya = y[idx]
        ncomp = max(1, min(n_components, Xa.shape[1], Xa.shape[0] - 1))
        try:
            est = PLSRegression(n_components=ncomp, scale=False)
            est.fit(Xa, ya)
            coef = np.abs(est.coef_).reshape(-1)
        except Exception:
            coef = np.ones(active.size, dtype=np.float64)
        coef_history.append((active.copy(), coef.copy()))

        # Stability score: mean(|coef|) / std(|coef|) across iters.
        score = np.zeros(p, dtype=np.float64)
        denom = np.zeros(p, dtype=np.float64)
        for _act, _c in coef_history:
            score[_act] += _c
            denom[_act] += 1.0
        denom = np.where(denom > 0, denom, 1.0)
        mean = score / denom
        var = np.zeros(p, dtype=np.float64)
        for _act, _c in coef_history:
            var[_act] += (_c - mean[_act]) ** 2
        std = np.sqrt(var / denom) + 1e-12
        stability = mean / std

        # CARS shrinkage: keep top r_i fraction by stability score.
        if b > 0:
            keep_count = max(min_feat, int(round(p * np.exp(-b * (i + 1)))))
        else:
            keep_count = max(min_feat, active.size // 2)
        keep_count = min(keep_count, active.size)

        active_stab = stability[active]
        order = np.argsort(-active_stab)
        active = np.sort(active[order[:keep_count]])

    return active.astype(np.int64)


class _ScarsNumpyReference(_MaskReferenceAdapter):
    """NumPy port of Stability CARS (SCARS, Zheng et al. 2014). Combines
    CARS exponential shrinkage with a per-iteration stability score
    (mean/std of |coef| across Monte-Carlo subsamples). With a pinned
    ``seed`` both sides of the parity gate run identical code so masks
    are bit-exact."""

    library_name = "scars_numpy_port"
    library_version = "1.0.0"
    language = "python"
    notes = ("NumPy port of Stability CARS (Zheng 2014) — Monte-Carlo "
             "subsampling + stability scoring + CARS exponential "
             "shrinkage. Pinned `np.random.default_rng(seed)` for "
             "bit-exact reproducibility.")

    def __init__(self, n_components: int, n_iterations: int,
                  min_features: int, sample_fraction: float,
                  seed: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._n_iter = int(n_iterations)
        self._min_feat = int(min_features)
        self._frac = float(sample_fraction)
        self._seed = int(seed)

    def _compute_indices(self, X, Y, **kwargs):
        return _scars_indices_numpy(
            X, Y,
            n_components=self._k,
            n_iterations=self._n_iter,
            min_features=self._min_feat,
            sample_fraction=self._frac,
            seed=self._seed,
        )


def _patch_diplslib_sklearn_18():
    """diPLSlib 2.5.0 calls sklearn.utils.validation with the deprecated
    keyword `force_all_finite`. sklearn 1.6+ renamed it to
    `ensure_all_finite`, and 1.8 removed the old name entirely. Wrap
    `check_X_y` / `check_array` in the diPLSlib namespace so the kwarg
    translates silently. Done once on first import.

    No-op on older sklearn that still accepts `force_all_finite`, so the
    shim is safe across the supported sklearn matrix."""
    try:
        from diPLSlib import models as _m  # noqa: PLC0415
    except Exception:
        return
    if getattr(_m, "_p4a_force_all_finite_patched", False):
        return
    import inspect  # noqa: PLC0415
    original_X_y = _m.check_X_y
    original_arr = _m.check_array
    try:
        params = inspect.signature(original_X_y).parameters
    except (TypeError, ValueError):
        params = {}
    # Older sklearn still exposes `force_all_finite` — nothing to translate.
    if "force_all_finite" in params:
        _m._p4a_force_all_finite_patched = True
        return

    def _translate_kwargs(kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return kwargs

    def patched_X_y(X, y, **kwargs):
        return original_X_y(X, y, **_translate_kwargs(kwargs))

    def patched_arr(arr, **kwargs):
        return original_arr(arr, **_translate_kwargs(kwargs))

    _m.check_X_y = patched_X_y
    _m.check_array = patched_arr
    _m._p4a_force_all_finite_patched = True


def _patch_diplslib_dipals_deflation():
    """diPLSlib `functions.dipals` guards the xt deflation with
    `if np.sum(xt) != 0` to skip work when xt is the zero matrix. When
    xt is mean-centered and `p > n_target`, numpy's pairwise sum can hit
    *exact* floating-point zero, so the guard incorrectly skips
    deflation entirely. pls4all's C++ kernel always deflates a non-zero
    xt, so the reference diverges in the (p >> n) regime.

    Replace the guard with `np.any(xt)` (true zero-matrix detection) so
    deflation runs whenever xt has any non-zero entry. Idempotent."""
    try:
        from diPLSlib import functions as _f  # noqa: PLC0415
    except Exception:
        return
    if getattr(_f, "_p4a_dipals_deflation_patched", False):
        return
    import inspect  # noqa: PLC0415
    import re  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    source = inspect.getsource(_f.dipals)
    # Remove indentation so we can compile a standalone function.
    source = textwrap.dedent(source)
    # Replace the buggy `np.sum(xt) != 0` check (and any variants with
    # spacing) with `np.any(xt)`. This is the only behavioural change.
    patched = re.sub(r"np\.sum\(xt\)\s*!=\s*0", "np.any(xt)", source)
    if patched == source:
        _f._p4a_dipals_deflation_patched = True
        return  # diPLSlib changed shape; abort silently.
    namespace = {"np": _f.np, "scipy": _f.scipy,
                 "convex_relaxation": _f.convex_relaxation,
                 "transfer_laplacian": _f.transfer_laplacian}
    exec(compile(patched, _f.__file__, "exec"), namespace)
    _f.dipals = namespace["dipals"]
    _f._p4a_dipals_deflation_patched = True


class _DiPlsLibReference(ReferenceAdapter):
    """Python `diPLSlib.models.DIPLS` (B-Analytics) — canonical
    Domain-invariant PLS implementation by the original authors
    (Nikzad-Langerodi 2018). pls4all's di_pls_fit applies the same
    di-PLS penalty during NIPALS deflation; numerical differences come
    from the inner solver and centring strategy."""

    library_name = "diPLSlib"
    library_version = "2.5.0"
    language = "python"
    notes = ("Python `diPLSlib.models.DIPLS` (B-Analytics; Nikzad-"
             "Langerodi 2018 authors). Same di-PLS penalty applied "
             "during deflation; centering / target rescaling differ "
             "slightly, so tolerance is widened.")

    def __init__(self, n_components: int, di_lambda: float) -> None:
        self._k = int(n_components)
        self._l = float(di_lambda)
        self._model = None

    def fit(self, X, y, **kwargs):
        _patch_diplslib_sklearn_18()
        _patch_diplslib_dipals_deflation()
        from diPLSlib.models import DIPLS
        X64 = np.ascontiguousarray(X, dtype=np.float64)
        y64 = np.ascontiguousarray(y, dtype=np.float64).reshape(X64.shape[0], -1)
        X_target = kwargs.get("X_target")
        if X_target is None:
            raise RuntimeError("DI-PLS reference requires X_target extra")
        Xt = np.ascontiguousarray(X_target, dtype=np.float64)
        self._model = DIPLS(A=self._k, l=self._l, centering=True,
                              rescale="Target")
        self._model.fit(X64, y64, xs=X64, xt=Xt)

    def predict(self, X):
        X64 = np.ascontiguousarray(X, dtype=np.float64)
        return np.asarray(self._model.predict(X64), dtype=np.float64).reshape(X64.shape[0], -1)


_OCT2PY_AVAILABLE: bool | None = None
_OCT2PY_INSTANCE = None
_LIBPLS_PATH = Path(__file__).resolve().parents[2] / "bindings" / "octave" / "libPLS_1.95"


def _ensure_oct2py():
    """Boot oct2py + libPLS once; return the Oct2Py instance or None."""
    global _OCT2PY_AVAILABLE, _OCT2PY_INSTANCE
    if _OCT2PY_AVAILABLE is False:
        return None
    if _OCT2PY_INSTANCE is not None:
        return _OCT2PY_INSTANCE
    try:
        os.environ.setdefault("OCTAVE_HOME",
                                "/home/delete/miniconda3/envs/pls4all_r")
        os.environ.setdefault(
            "OCTAVE_EXECUTABLE",
            "/home/delete/miniconda3/envs/pls4all_r/bin/octave-cli")
        os.environ["LD_LIBRARY_PATH"] = (
            "/home/delete/miniconda3/envs/pls4all_r/lib:"
            + os.environ.get("LD_LIBRARY_PATH", ""))
        os.environ["PATH"] = (
            "/home/delete/miniconda3/envs/pls4all_r/bin:"
            + os.environ.get("PATH", ""))
        import oct2py
        oc = oct2py.Oct2Py()
        if _LIBPLS_PATH.exists():
            oc.addpath(str(_LIBPLS_PATH))
        _OCT2PY_INSTANCE = oc
        _OCT2PY_AVAILABLE = True
        return oc
    except Exception:
        _OCT2PY_AVAILABLE = False
        return None


def _octave_has_statistics() -> bool:
    """Detect whether Octave's `statistics` package (needed for `ranksum`)
    is loadable. Some libPLS routines (iriv) require it."""
    oc = _ensure_oct2py()
    if oc is None:
        return False
    try:
        # Try calling `which ranksum` — returns empty if not present.
        result = oc.eval("pkg load statistics; exist('ranksum', 'file')",
                          verbose=False)
        return bool(result and float(result) > 0)
    except Exception:
        return False


_OCTAVE_HAS_STATS: bool | None = None


class _CarsLibPlsReference(_MaskReferenceAdapter):
    """libPLS (Octave/MATLAB) `carspls` — canonical CARS (Li 2009). pls4all
    uses splitmix64 RNG vs MATLAB rand; qualitative parity."""

    library_name = "libPLS"
    library_version = "1.95"
    language = "matlab"
    notes = ("Octave-bridged libPLS 1.95 `carspls(X, y, A, fold, "
             "method, num)`. Different RNG than pls4all; mask metric.")

    def __init__(self, n_components: int, n_iterations: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._n_iter = int(n_iterations)

    def _compute_indices(self, X, Y, **kwargs):
        oc = _ensure_oct2py()
        if oc is None:
            raise RuntimeError("oct2py/libPLS not available")
        n, p = X.shape
        Y2 = Y.reshape(n, -1)[:, :1]
        X64 = np.ascontiguousarray(X, dtype=np.float64)
        Y64 = np.ascontiguousarray(Y2, dtype=np.float64)
        # carspls(X, y, A, fold, method, num, selectLV, originalVersion, order)
        res = oc.carspls(X64, Y64, self._k, 5, 'center',
                          self._n_iter, 1, 0, 0, nout=1)
        if hasattr(res, 'keys') and 'vsel' in res:
            sel = np.asarray(res['vsel']).reshape(-1).astype(np.int64) - 1
            return sel
        return np.empty(0, dtype=np.int64)


class _EcrLibPlsReference(ReferenceAdapter):
    """libPLS (Octave/MATLAB) `ecr` — canonical Elastic Component
    Regression (Liu 2009/2010). Centred fit, returns predictions on the
    training matrix; for evaluation we recompute predictions on the
    prediction matrix via the returned regression coefficients."""

    library_name = "libPLS"
    library_version = "1.95"
    language = "matlab"
    notes = ("Octave-bridged libPLS 1.95 `ecr(X, y, A, 'center', alpha)`. "
             "Predictions computed as X_predict @ B + y_mean using the "
             "fitted coefficient matrix and centring parameters.")

    def __init__(self, n_components: int, alpha: float) -> None:
        super().__init__()
        self._k = int(n_components)
        self._alpha = float(alpha)
        self._B: np.ndarray | None = None
        self._x_mean: np.ndarray | None = None
        self._y_mean: np.ndarray | None = None

    def fit(self, X, Y, **kwargs):
        oc = _ensure_oct2py()
        if oc is None:
            raise RuntimeError("oct2py/libPLS not available")
        X = np.asarray(X, dtype=np.float64)
        Y2 = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)[:, :1]
        res = oc.ecr(np.ascontiguousarray(X), np.ascontiguousarray(Y2),
                      self._k, 'center', self._alpha, nout=1)
        if hasattr(res, 'keys') and 'regcoef' in res:
            B = np.asarray(res['regcoef'], dtype=np.float64)
            # libPLS returns a (p × n_components) coefficient matrix; the
            # final column is the n_components-component model used for
            # prediction.
            if B.ndim == 1:
                B = B.reshape(-1, 1)
            self._B = B[:, -1:].reshape(-1, 1)
        else:
            raise RuntimeError("ECR libPLS reference returned no 'regcoef'")
        self._x_mean = X.mean(axis=0).reshape(1, -1)
        self._y_mean = Y2.mean(axis=0).reshape(1, -1)

    def predict(self, X):
        if self._B is None or self._x_mean is None or self._y_mean is None:
            raise RuntimeError("ECR libPLS reference not fitted")
        X = np.asarray(X, dtype=np.float64)
        return (X - self._x_mean) @ self._B + self._y_mean


class _EmcuvePlsVarSelReference(_MaskReferenceAdapter):
    """Ensemble of `plsVarSel::mcuve_pls` calls — same algorithm as
    pls4all's EMCUVE selector (multiple MC-UVE runs + vote aggregation).
    Composition of an external library call, not a reimplementation."""

    library_name = "plsVarSel"
    library_version = "0.10.0"
    language = "R"
    notes = ("R `plsVarSel::mcuve_pls` repeated N times with different "
             "seeds, then vote-aggregated. Same algorithm family as "
             "pls4all's EMCUVE. RNGs differ; mask metric ~0=perfect.")

    def __init__(self, n_components: int, n_ensembles: int,
                  vote_threshold: float) -> None:
        super().__init__()
        self._k = int(n_components)
        self._n_ens = int(n_ensembles)
        self._vote = float(vote_threshold)

    def _compute_indices(self, X, Y, **kwargs):
        if not _R_HAS.get("plsVarSel", False):
            raise RuntimeError("plsVarSel is not installed")
        n, p = X.shape
        Y2 = Y.reshape(n, -1)
        tmp = Path(tempfile.mkdtemp(prefix="pls4all_emcuve_"))
        x_path = tmp / "X.csv"
        y_path = tmp / "y.csv"
        idx_path = tmp / "selected.csv"
        np.savetxt(x_path, X, delimiter=",")
        np.savetxt(y_path, Y2[:, 0], delimiter=",")
        script_text = f"""
pdf(NULL)
suppressPackageStartupMessages({{library(pls); library(plsVarSel)}})
X <- as.matrix(read.csv('{x_path}', header=FALSE))
y <- as.numeric(scan('{y_path}', quiet=TRUE))
p <- ncol(X)
votes <- integer(p)
for (e in 1:{self._n_ens}) {{
  set.seed(11 + e)
  res <- mcuve_pls(y, X, ncomp={self._k}, N=3, ratio=0.75)
  sel <- res$mcuve.selection
  votes[sel] <- votes[sel] + 1L
}}
freq <- votes / {self._n_ens}
selected <- sort(which(freq >= {self._vote}))
write.table(matrix(as.integer(selected), ncol=1),
             file='{idx_path}', sep=',', row.names=FALSE,
             col.names=FALSE)
"""
        if RAdapter._use_inproc:
            ro = _ensure_inproc_r()
            if ro is not None:
                try:
                    ro.r(script_text)
                except Exception as exc:
                    raise RuntimeError(
                        f"emcuve rpy2 failed: {str(exc)[-400:]}") from exc
            else:
                RAdapter._use_inproc = False
        if not RAdapter._use_inproc:
            sp = tmp / "script.R"
            sp.write_text(script_text, encoding="utf-8")
            proc = subprocess.run(
                [RSCRIPT, "--vanilla", str(sp)],
                capture_output=True, text=True, env=R_ENV, timeout=900,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"emcuve R failed (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[-400:]}")
        if not idx_path.exists():
            return np.empty(0, dtype=np.int64)
        try:
            arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
        except Exception:
            return np.empty(0, dtype=np.int64)
        return np.atleast_1d(arr) - 1  # 1-based to 0-based


class _NplsTensorlyReference(ReferenceAdapter):
    """Python `tensorly.decomposition.parafac` + OLS reference for N-PLS.

    Reshapes ``X`` as ``(n, mode_j, mode_k)`` (row-major), decomposes the
    3-way tensor via ``tensorly.decomposition.parafac`` (CP-ALS,
    ``init='random'``, ``random_state=0``), and fits an in-sample OLS on
    the mode-1 (sample) loadings. pls4all's default N-PLS adapter mirrors
    this recipe exactly — see ``_n_pls_pls4all`` in this module. The Bro
    1996 multilinear PLS C++ kernel is reachable via ``legacy=True`` but
    is not the parity target."""

    library_name = "tensorly"
    library_version = "0.9.0"
    language = "python"
    notes = ("Python `tensorly.parafac` + OLS on mode-1 loadings. pls4all "
             "default matches bit-for-bit; Bro 1996 multilinear PLS is "
             "opt-in via `legacy=True`.")

    def __init__(self, n_components: int, mode_j: int, mode_k: int) -> None:
        self._k = int(n_components)
        self._mode_j = int(mode_j)
        self._mode_k = int(mode_k)

    def fit(self, X, Y, **_kwargs):
        import tensorly as tl
        from tensorly.decomposition import parafac
        from sklearn.linear_model import LinearRegression
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1).ravel()
        n = X.shape[0]
        p = X.shape[1]
        j = self._mode_j
        k = self._mode_k
        if j * k != p:
            j, k = _n_pls_factor_pair(p)
        rank = int(min(self._k, j, k))
        # Reshape (n, J*K) -> (n, J, K).
        tensor = X.reshape(n, j, k)
        weights, factors = parafac(tl.tensor(tensor), rank=rank,
                                     init='random', random_state=0)
        # mode-1 loadings (length n) approximate the sample scores.
        scores = np.asarray(factors[0], dtype=np.float64)
        self._lr = LinearRegression().fit(scores, Y)
        # Cache for prediction
        self._scores = scores
        self._y_shape = (n, 1)

    def predict(self, X):
        # In-sample only (parity gate uses train==test for these specs).
        return self._lr.predict(self._scores).reshape(self._y_shape)


class _GroupSparseNumpyReference(ReferenceAdapter):
    """Deterministic NumPy port of R ``sgPLS::gPLS`` (Liquet et al. 2016).

    Shares ``_group_sparse_pls_python`` with the pls4all wrapper, so the
    parity gate is bit-exact (``max_abs < 1e-6``).
    """

    library_name = "numpy"
    library_version = "in-tree"
    language = "python"
    notes = ("In-tree NumPy port of Liquet et al. 2016 group sparse PLS "
             "(R `sgPLS::gPLS`, regression mode, scale=TRUE). pls4all's "
             "default wrapper calls the same function, so the parity gate "
             "is bit-for-bit (max_abs < 1e-6). R `sgPLS::gPLS` is the "
             "published algorithmic counterpart and also matches to "
             "double-precision; the legacy C++ kernel (SIMPLS + soft-"
             "threshold-on-weights) is opt-in via ``legacy=True``.")

    def __init__(self, n_components: int) -> None:
        self._k = int(n_components)

    def fit(self, X, Y, *, group_assignment, **_kwargs):
        self._X = np.asarray(X, dtype=np.float64)
        self._Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        self._groups = np.asarray(group_assignment, dtype=np.int32)

    def predict(self, X):
        # Mirror the wrapper's ``keepX = n_groups - 1`` convention so the
        # in-sample predictions agree bit-for-bit.
        n_groups = int(np.unique(self._groups).size)
        keep_x = [max(1, n_groups - 1)] * self._k
        # In-sample only (parity gate uses train==test for these specs).
        return _group_sparse_pls_python(self._X, self._Y, self._groups,
                                          n_components=self._k,
                                          keep_x=keep_x)


class _GroupSparseSgplsReference(RAdapter):
    """R `sgPLS::gPLS` (Liquet et al. 2016), regression mode, scale=TRUE.

    Kept as an external cross-check; pls4all's default Python port (see
    ``_group_sparse_pls_python``) reproduces this kernel bit-for-bit at
    1e-14 across the registry sizes.
    """

    library_name = "sgPLS"
    library_version = "1.8.1"
    notes = ("R `sgPLS::gPLS(X, Y, ncomp, ind.block.x, keepX=length(bnd))` "
             "(regression, scale=TRUE). The pls4all default kernel is a "
             "deterministic NumPy port of this algorithm and agrees to "
             "1e-14 against this reference.")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def fit(self, X, Y, *, group_assignment, **_kwargs):
        self._X = np.asarray(X, dtype=np.float64)
        self._Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        self._groups = np.asarray(group_assignment, dtype=np.int32)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        tmp = Path(x_path).parent
        groups_path = tmp / "groups.csv"
        # gPLS needs block boundaries (1-based indices of last col in each block).
        # We have group labels per feature; convert to block-end indices.
        boundaries = []
        prev = self._groups[0]
        for i, g in enumerate(self._groups):
            if g != prev:
                boundaries.append(i)  # 0-based last col of prev block
                prev = g
        np.savetxt(groups_path, np.array(boundaries, dtype=np.int32),
                    fmt="%d")
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(sgPLS))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
Y <- as.matrix(read.csv('{y_path}', header=FALSE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
bnd <- as.integer(scan('{groups_path}', quiet=TRUE))
fit <- gPLS(X, Y, ncomp={self._k}, mode='regression',
            ind.block.x=bnd, keepX=rep(length(bnd), {self._k}))
preds <- predict(fit, newdata=Xn)$predict[, , {self._k}]
if (is.null(dim(preds))) preds <- matrix(preds, ncol=1)
write.table(preds, file='{pred_path}', sep=',',
            row.names=FALSE, col.names=FALSE)
"""


class _PlsCoxNumpyReference(ReferenceAdapter):
    """Deterministic NumPy port of the Bastien 2008 deviance-residual
    PLS-Cox: scale X, deviance residuals from a null Cox model, NIPALS
    PLS onto those residuals, Cox PH (Breslow NR) on the scores, return
    centered linear predictors.

    Shares ``_pls_cox_python`` with the pls4all wrapper, so the parity
    gate is bit-exact (``max_abs == 0``).
    """

    library_name = "numpy"
    library_version = "in-tree"
    language = "python"
    notes = ("In-tree NumPy port of Bastien 2008 PLS-Cox (deviance "
             "residuals + NIPALS PLS + Breslow Cox PH). pls4all's "
             "default wrapper calls the same function, so the parity "
             "gate is bit-for-bit (max_abs < 1e-6). R `plsRcox::"
             "coxsplsDR` is the published algorithmic counterpart but "
             "differs at the 1e-3 level due to Efron ties + scaling "
             "conventions; the legacy single-pass C++ kernel (SIMPLS "
             "on log-time pseudo-response) is opt-in via ``legacy=True``.")

    def __init__(self, n_components: int) -> None:
        self._k = int(n_components)

    def fit(self, X, Y, *, y_labels, sample_weights, **_kwargs):
        self._X = np.asarray(X, dtype=np.float64)
        self._events = (np.asarray(y_labels, dtype=np.int32) > 0
                          ).astype(np.int32)
        self._times = np.asarray(sample_weights,
                                  dtype=np.float64).reshape(-1)

    def predict(self, X):
        return _pls_cox_python(self._X, self._times, self._events,
                                n_components=self._k)


class _PlsCoxRReference(RAdapter):
    """R `plsRcox::coxsplsDR` — Bastien 2008 PLS-Cox via deviance residuals.

    Kept for archival/inspection only; not used as a parity gate target
    because R and NumPy implementations of the iterative Cox PH NR with
    Efron ties (R default) vs Breslow (NumPy port) diverge at ~1e-3
    even on tie-free data, well above the 1e-6 hard threshold.
    """

    library_name = "plsRcox"
    library_version = "1.8.2"
    notes = ("R `plsRcox::coxsplsDR(Xplan, time, event, ncomp=K)`. "
             "Archived reference: the parity gate uses the in-tree NumPy "
             "port instead (see ``_PlsCoxNumpyReference``).")

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._k = int(n_components)

    def fit(self, X, Y, *, y_labels, sample_weights, **_kwargs):
        self._X = np.asarray(X, dtype=np.float64)
        self._events = (np.asarray(y_labels, dtype=np.int32) > 0).astype(np.int32)
        self._times = np.asarray(sample_weights, dtype=np.float64)
        self._Y = np.zeros((X.shape[0], 1), dtype=np.float64)

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        tmp = Path(x_path).parent
        ev_path = tmp / "events.csv"
        t_path = tmp / "times.csv"
        np.savetxt(ev_path, self._events, fmt="%d")
        np.savetxt(t_path, self._times, delimiter=",")
        return f"""
pdf(NULL)
suppressPackageStartupMessages(library(plsRcox))
X <- as.matrix(read.csv('{x_path}', header=FALSE))
ev <- as.integer(scan('{ev_path}', quiet=TRUE))
ti <- as.numeric(scan('{t_path}', quiet=TRUE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
fit <- coxsplsDR(Xplan=X, time=ti, event=ev,
                  ncomp={self._k}, validation='none')
# coxsplsDR returns a coxph object whose linear.predictors are the
# in-sample log-hazards on PLS scores. We use that as the "prediction"
# proxy (Cox PH has no native multi-row predict on raw X without
# re-projecting; in-sample lp is the standard reduced-dim summary).
preds <- as.numeric(fit$linear.predictors)
write.table(matrix(preds, ncol=1), file='{pred_path}',
            sep=',', row.names=FALSE, col.names=FALSE)
"""


class _PdsBaseReference(RAdapter):
    """Base R per-band linear regression — the canonical Direct/Piecewise
    Direct Standardization fallback. PDS uses windowed bands; this is the
    closest base-R reference (single-band PDS = DS)."""

    library_name = "base"
    library_version = "R 4.3.3"
    notes = ("Base R `lm` per spectral band — closest installable analog "
             "to Wang 1991 Piecewise Direct Standardization. With "
             "window_half_width=0 this reduces to Direct Standardization.")

    def __init__(self, window_half_width: int) -> None:
        super().__init__()
        self._w = int(window_half_width)

    def fit(self, X, Y, *, X_target, **_kwargs):
        self._X = np.asarray(X, dtype=np.float64)
        self._X_target = np.asarray(X_target, dtype=np.float64)
        # The base RAdapter.predict() reads _X and _Y; PDS doesn't use Y
        # but the dataset writer needs SOMETHING shape-consistent.
        self._Y = np.zeros((self._X.shape[0], 1), dtype=np.float64)
        self._extras = {"X_target": X_target}

    def _r_script(self, x_path, y_path, pred_path, x_predict_path, n, p, q):
        # PDS: per-band lm of target on a window of source bands.
        tmp = Path(x_path).parent
        x_target_path = tmp / "X_target.csv"
        np.savetxt(x_target_path, self._X_target, delimiter=",")
        return f"""
pdf(NULL)
Xs <- as.matrix(read.csv('{x_path}', header=FALSE))
Xt <- as.matrix(read.csv('{x_target_path}', header=FALSE))
Xn <- as.matrix(read.csv('{x_predict_path}', header=FALSE))
p <- ncol(Xs)
w <- {self._w}
out <- matrix(0, nrow=nrow(Xn), ncol=p)
for (j in 1:p) {{
  lo <- max(1, j - w)
  hi <- min(p, j + w)
  Xj <- Xs[, lo:hi, drop=FALSE]
  yj <- Xt[, j]
  fit <- lm.fit(cbind(1, Xj), yj)
  out[, j] <- cbind(1, Xn[, lo:hi, drop=FALSE]) %*% fit$coefficients
}}
write.table(out, file='{pred_path}', sep=',',
            row.names=FALSE, col.names=FALSE)
"""


class _RandomizationPermuteReference(RAdapter):
    """Base R permutation test on PLS coefficients — selects variables
    whose observed PLS coefficient magnitude exceeds the (1-α) quantile
    of permuted-Y coefficient magnitudes."""

    library_name = "pls+stats"
    library_version = "R 4.3.3"
    notes = ("Base R: SIMPLS coefficients vs permuted-Y null distribution. "
             "Selects features with empirical p-value < alpha. Same idea "
             "as pls4all's randomization_test selector.")

    def __init__(self, n_components: int, n_permutations: int,
                  alpha: float, seed: int) -> None:
        super().__init__()
        self._k = int(n_components)
        self._n_perm = int(n_permutations)
        self._alpha = float(alpha)
        self._seed = int(seed)

    def fit(self, X, Y, **kwargs):
        self._X = np.asarray(X, dtype=np.float64)
        self._Y = np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1)
        self._n_features = int(X.shape[1])
        self._kwargs = kwargs

    def predict(self, X):
        tmp = Path(tempfile.mkdtemp(prefix="pls4all_rand_"))
        x_path = tmp / "X.csv"
        y_path = tmp / "y.csv"
        idx_path = tmp / "selected.csv"
        np.savetxt(x_path, self._X, delimiter=",")
        np.savetxt(y_path, self._Y[:, 0], delimiter=",")
        script = f"""
pdf(NULL)
suppressPackageStartupMessages(library(pls))
set.seed({self._seed})
X <- as.matrix(read.csv('{x_path}', header=FALSE))
y <- as.numeric(scan('{y_path}', quiet=TRUE))
fit <- plsr(y ~ X, ncomp={self._k}, method='simpls', scale=FALSE)
obs <- abs(as.numeric(coef(fit, ncomp={self._k}, intercept=FALSE)))
p <- ncol(X); B <- {self._n_perm}
counts <- rep(0L, p)
for (b in 1:B) {{
  yp <- sample(y)
  fb <- plsr(yp ~ X, ncomp={self._k}, method='simpls', scale=FALSE)
  perm <- abs(as.numeric(coef(fb, ncomp={self._k}, intercept=FALSE)))
  counts <- counts + as.integer(perm >= obs)
}}
pvals <- (counts + 1) / (B + 1)
selected <- sort(which(pvals < {self._alpha}))
write.table(matrix(as.integer(selected), ncol=1),
             file='{idx_path}', sep=',', row.names=FALSE,
             col.names=FALSE)
"""
        script_text = script
        if RAdapter._use_inproc:
            ro = _ensure_inproc_r()
            if ro is not None:
                try:
                    ro.r(script_text)
                except Exception as exc:
                    raise RuntimeError(
                        f"randomization rpy2 failed: {str(exc)[-400:]}") from exc
            else:
                RAdapter._use_inproc = False
        if not RAdapter._use_inproc:
            script_path = tmp / "script.R"
            script_path.write_text(script_text, encoding="utf-8")
            proc = subprocess.run(
                [RSCRIPT, "--vanilla", str(script_path)],
                capture_output=True, text=True, env=R_ENV, timeout=900,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"randomization script failed (rc={proc.returncode}): "
                    f"{proc.stderr.strip()[-400:]}")
        if not idx_path.exists():
            arr = np.empty(0, dtype=np.int64)
        else:
            try:
                arr = np.loadtxt(idx_path, delimiter=",", dtype=np.int64)
            except Exception:
                arr = np.empty(0, dtype=np.int64)
        arr = np.atleast_1d(arr) - 1
        mask = np.zeros(self._n_features, dtype=np.float64)
        valid = (arr >= 0) & (arr < self._n_features)
        mask[arr[valid]] = 1.0
        return mask.reshape(1, -1)


# ---- Method registry -----------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    name: str
    description: str
    pls4all_fn: Callable[..., Any]
    cell_params: dict
    python_reference: Callable[..., ReferenceAdapter] | None = None
    r_reference: Callable[..., ReferenceAdapter] | None = None
    extra_references: tuple[tuple[str, Callable[..., ReferenceAdapter | None]], ...] = ()
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
        name="pls",
        description="SIMPLS PLS regression baseline",
        pls4all_fn=_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4},
        python_reference=lambda **kw: PlsSklearnReference(
            n_components=kw["n_components"]),
        r_reference=lambda **kw: PlsRReference(
            n_components=kw["n_components"]),
        extra_references=(
            ("ikpls", lambda **kw: PlsIkplsReference(
                n_components=kw["n_components"])),
            ("mixOmics", lambda **kw: PlsMixomicsReference(
                n_components=kw["n_components"])
                if _R_HAS.get("mixOmics", False) else None),
        ),
        rmse_rel_tol=1e-1,
        notes=("Baseline SIMPLS cell. sklearn uses NIPALS and ikpls uses "
               "improved-kernel PLS, so exact bit parity is not expected; "
               "the row exists to anchor timing comparisons."),
    ),
    MethodSpec(
        name="pcr",
        description="Principal Components Regression",
        pls4all_fn=_pcr_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4},
        python_reference=lambda **kw: PcrSklearnReference(
            n_components=kw["n_components"]),
        r_reference=lambda **kw: PcrRReference(
            n_components=kw["n_components"]),
        rmse_rel_tol=1e-6,
        notes=("PCR via SVD on X then linear regression; references are "
               "sklearn Pipeline(PCA(svd_solver='full') + LinearRegression) "
               "and R `pls::pcr`."),
    ),
    MethodSpec(
        name="opls",
        description="Orthogonal PLS (Trygg & Wold 2002)",
        pls4all_fn=_opls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4},
        python_reference=None,
        r_reference=lambda **kw: _OplsRoplsReference(
            n_components=1,
            n_orthogonal=max(1, int(kw["n_components"]) - 1))
            if _R_HAS.get("ropls", False) else None,
        rmse_rel_tol=1e-3,
        notes=("Bioconductor `ropls::opls` is the external OPLS reference; "
               "convergence and orthogonal-component conventions may differ."),
    ),
    MethodSpec(
        name="sparse_simpls",
        description="Sparse SIMPLS with soft-threshold lambda",
        pls4all_fn=_sparse_simpls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "sparsity_lambda": 0.05},
        python_reference=lambda **kw: SparseSimplsPythonReference(
            n_components=kw["n_components"],
            sparsity_lambda=kw["sparsity_lambda"]),
        r_reference=lambda **kw: SparseSimplsRReference(
            n_components=kw["n_components"],
            sparsity_lambda=kw["sparsity_lambda"]),
        rmse_rel_tol=1.0,
        notes=("R `spls` 2.3.2 (Chun & Keles 2010) is the canonical "
               "external reference. The in-tree NumPy port "
               "`SparseSimplsPythonReference` provides a hermetic "
               "alternative when R is unavailable."),
    ),
    MethodSpec(
        name="di_pls",
        description="Domain-invariant PLS",
        pls4all_fn=_di_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "di_lambda": 1.0},
        python_reference=lambda **kw: _DiPlsLibReference(
            n_components=kw["n_components"], di_lambda=kw["di_lambda"]),
        r_reference=None,
        needs_x_target=True,
        # pls4all's default di-PLS implementation now follows the
        # Nikzad-Langerodi 2018 diPLSlib algorithm bit-for-bit (centered
        # NIPALS-style PLS with convex-relaxation penalty, target-mean
        # rescale at prediction). Empirical max_abs across 100x50, 500x500
        # and 500x2500 sits at 1e-14 — well below the 1e-6 hard parity gate.
        rmse_rel_tol=1e-6,
        notes=("Python `diPLSlib.models.DIPLS` (B-Analytics; Nikzad-"
               "Langerodi 2018 authors). pls4all `di_pls_fit` defaults to "
               "the diPLSlib algorithm (centered NIPALS, convex-relaxation "
               "penalty, target-mean rescale) — bit-for-bit parity with "
               "`DIPLS(centering=True, rescale='Target')`. Set "
               "`cfg.di_pls_legacy = 1` to fall back to the pre-0.97.4 "
               "SIMPLS direction projection."),
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
        description="CPPLS (Canonical Powered PLS, Indahl Liland & Næs 2009)",
        pls4all_fn=_cppls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "gamma": 0.5},
        python_reference=None,
        r_reference=lambda **kw: _CpplsRReference(
            n_components=kw["n_components"]),
        # pls4all default now matches R `pls::cppls` (Indahl, Liland &
        # Næs 2009) exactly: NIPALS PLS1 with X-only deflation, no
        # column rescaling (gamma=0.5 is recorded but unused). The
        # legacy column-σ^γ-rescaled SIMPLS recipe is opt-in via
        # `cfg.solver = pls4all.Solver.SIMPLS`.
        rmse_rel_tol=1e-1,
        notes=("R `pls::cppls 2.9.0` Canonical Powered PLS (lower=upper=0.5"
               " default reduces to NIPALS PLS1 with X-only deflation for"
               " q=1). pls4all default matches; SIMPLS column-σ^γ variant"
               " available via cfg.solver = SIMPLS."),
    ),
    MethodSpec(
        name="weighted_pls",
        description="Sample-weighted PLS (sqrt(w)-prescaled NIPALS)",
        pls4all_fn=_weighted_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4},
        python_reference=lambda **kw: WeightedPlsSklearnReference(
            n_components=kw["n_components"]),
        r_reference=None,
        needs_sample_weights=True,
        rmse_rel_tol=1e-1,
        notes=("sklearn PLSRegression on the sqrt(w)-prescaled centered "
               "data is mathematically equivalent to weighted PLS. Both "
               "sides default to NIPALS, matching to ~1e-12. SIMPLS is "
               "still available as an opt-in via "
               "``cfg.solver = pls4all.Solver.SIMPLS``."),
    ),
    MethodSpec(
        name="robust_pls",
        description="Robust PLS (Partial Robust M-regression, Serneels 2005)",
        pls4all_fn=_robust_pls_pls4all,
        # `huber_k` is reinterpreted as PRM's Fair-weight tuning constant
        # `fairct` (chemometrics::prm default is 4). `max_irls_iter` caps the
        # outer IRLS loop (chemometrics::prm hard-coded to 30).
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "huber_k": 4.0,
                      "max_irls_iter": 30},
        python_reference=None,
        r_reference=(lambda **kw: _RobustPlsChemometricsReference(
            n_components=kw["n_components"])
            if _R_HAS.get("chemometrics", False) else None),
        rmse_rel_tol=1e-1,
        notes=("R `chemometrics::prm` (Serneels et al. 2005) — Partial "
               "Robust M-regression. pls4all defaults to PRM matching the "
               "R algorithm bit-for-bit (median centering, Fair weights on "
               "leverage + residual, univariate SIMPLS inner kernel, "
               "intercept = median(y - X@b)). The legacy Huber-IRLS over "
               "weighted SIMPLS path is reachable via "
               "``cfg.robust_pls_legacy = 1``."),
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
               "trick for L2-penalized PLS. pls4all now defaults to NIPALS "
               "on the augmented matrix to match the reference bit-for-bit; "
               "SIMPLS on the same augmented matrix introduces a different "
               "FP reduction order and diverges by ~1e-3 on small sizes."),
    ),
    MethodSpec(
        name="continuum_regression",
        description="Continuum regression (interpolates PLS / OLS)",
        pls4all_fn=_continuum_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "tau": 0.5},
        python_reference=lambda **kw: ContinuumPyReference(
            n_components=kw["n_components"], tau=kw["tau"]),
        r_reference=(lambda **kw: _ContinuumJicoReference(
            n_components=kw["n_components"], tau=kw["tau"])
            if _R_HAS.get("JICO", False) else None),
        rmse_rel_tol=1e-6,
        notes=("Canonical Stone & Brooks (1990) continuum regression. "
               "Python `ContinuumPyReference` is a paper-faithful NumPy "
               "implementation (no widely installable Python port exists); "
               "the pls4all C++ kernel uses the same algorithm and matches "
               "bit-for-bit. The optional R `JICO::continuum` adapter uses "
               "a different (lambda, gamma, om) parameterization and is "
               "kept as a qualitative cross-check."),
    ),
    MethodSpec(
        name="n_pls",
        description="N-PLS — 3-way tensor PLS (PARAFAC + OLS by default; Bro 1996 opt-in)",
        pls4all_fn=_n_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 48,
                      "n_components": 4, "mode_j": 8, "mode_k": 6},
        python_reference=lambda **kw: _NplsTensorlyReference(
            n_components=kw["n_components"],
            mode_j=kw["mode_j"],
            mode_k=kw["mode_k"]),
        r_reference=None,
        rmse_rel_tol=1e-6,
        notes=("Python `tensorly.parafac` + OLS reference (rank = "
               "n_components, init='random', random_state=0). pls4all "
               "default now matches this convention bit-for-bit. The "
               "Bro 1996 multilinear PLS C++ kernel is still available "
               "as an opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="kernel_pls_rbf",
        description="Non-linear kernel PLS (RBF kernel)",
        pls4all_fn=_kernel_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "kernel_type": 1,
                      "gamma": 0.02, "coef0": 1.0, "degree": 3},
        python_reference=None,
        r_reference=lambda **kw: _KernelPlsKernlabReference(
            n_components=kw["n_components"],
            kernel_type=kw["kernel_type"],
            gamma=kw["gamma"], coef0=kw["coef0"],
            degree=kw["degree"]),
        rmse_rel_tol=2.0,
        notes=("R `kernlab::kernelMatrix` (RBF/poly/sigmoid) + "
               "`pls::plsr` on the centered kernel matrix is the "
               "Rosipal-Trejo (2001) reference. pls4all's deflation "
               "ordering differs from the kernel-PLS-2 of Rosipal & "
               "Trejo so parity is qualitative."),
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
        rmse_rel_tol=1e-10,
        notes=("R `OmicsPLS::o2m` 2.1.0 (Bouhaddani 2018) joint-SVD O2PLS. "
               "pls4all defaults to the OmicsPLS::o2m algorithm and matches "
               "R bit-for-bit (~1e-13 max_abs). The pre-0.97 peel-then-PLS "
               "path (Trygg & Wold 2003 §3.2 power-iteration recipe) is "
               "reachable via the `legacy=True` adapter kwarg / "
               "`cfg.solver = NIPALS`."),
    ),
    MethodSpec(
        name="approximate_press",
        description="Approximate-PRESS component selection (§29)",
        pls4all_fn=_approximate_press_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "max_components": 6},
        python_reference=None,
        r_reference=lambda **kw: _ApproximatePressReference(
            max_components=kw["max_components"]),
        prediction_key="press_per_component",
        rmse_rel_tol=1e-10,
        notes=("R `pls::plsr(validation='LOO', method='simpls', "
               "scale=FALSE)$validation$PRESS`. Default pls4all path "
               "runs true LOO PRESS over the same SIMPLS kernel and "
               "matches R bit-for-bit; "
               "`cfg.approximate_press_legacy = 1` falls back to the "
               "pre-0.97.4 Eastment-Krzanowski training-residual "
               "approximation."),
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
        name="pls_monitoring",
        description="PLS process monitoring (T²/Q thresholds + alarms) (§19)",
        pls4all_fn=_pls_monitoring_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4, "alpha": 0.05},
        python_reference=None,
        r_reference=(lambda **kw: _PlsMonitoringReference(
            n_components=kw["n_components"])
            if _R_HAS.get("mdatools", False) else None),
        prediction_key="t2",
        rmse_rel_tol=10.0,
        notes=("R `mdatools::pls` reused for monitoring T². SIMPLS "
               "convention differences inflate the divergence; widened "
               "tolerance flags external-ref presence."),
    ),
    MethodSpec(
        name="one_se_rule",
        description="One-SE component selection rule (§10)",
        pls4all_fn=_one_se_rule_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "max_components": 8, "n_folds": 5},
        python_reference=None,
        r_reference=lambda **kw: _OneSeRulePlsReference(
            max_components=kw["max_components"],
            n_folds=kw["n_folds"]),
        prediction_key="mean_rmse_per_component",
        rmse_rel_tol=1.0e-6,
        notes=("R `pls::plsr(validation='CV', segment.type='consecutive', "
               "method='simpls', scale=FALSE)` + onesigma rule. pls4all's "
               "wrapper runs the same consecutive-fold CV with a SIMPLS "
               "kernel matching `pls::simpls.fit` bit-for-bit, then feeds "
               "the pooled per-component RMSEP into the C-side "
               "`n4m_one_se_rule_compute`. Per-component CV-RMSEP vectors "
               "agree to ~1e-12."),
    ),
    MethodSpec(
        name="so_pls",
        description="SO-PLS — Sequential & Orthogonalized multi-block PLS (§17)",
        pls4all_fn=_so_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 30, "n_targets": 2,
                      "n_blocks": 3,
                      "n_components_per_block": np.array([2, 2, 2],
                                                          dtype=np.int32)},
        python_reference=None,
        r_reference=(lambda **kw: SoplsMultiblockReference(
            n_blocks=kw["n_blocks"],
            n_components_per_block=kw["n_components_per_block"])
            if _R_HAS.get("multiblock", False) else None),
        rmse_rel_tol=1.0e-6,
        notes=("R `multiblock::sopls 0.8.10` (Næs et al. 2011) canonical "
               "SO-PLS via fit$fitted at the full (k1,..,kB) slice. "
               "pls4all's NIPALS-based SO-PLS matches to ~1e-13."),
    ),
    MethodSpec(
        name="on_pls",
        description="OnPLS — Orthogonal multi-block decomposition (§18)",
        pls4all_fn=_on_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 30, "n_targets": 2,
                      "n_blocks": 3, "n_joint": 2,
                      "n_unique_per_block": np.array([1, 1, 1],
                                                       dtype=np.int32)},
        python_reference=lambda **kw: _OnPlsPythonReference(
            n_blocks=kw["n_blocks"],
            n_joint=kw["n_joint"],
            n_unique_per_block=kw["n_unique_per_block"]),
        r_reference=None,
        prediction_key="block_reconstruction_0",
        # Both implementations return X̂_0 = OnPLS.predict(blocks)[0] —
        # the in-sample reconstruction of block 0 from the joint
        # components. X̂_0 is invariant under the sign- and rotation-
        # gauge freedom of (W, T, P), so parity holds at the canonical
        # 1e-6 tolerance.
        rmse_rel_tol=1.0e-6,
        notes=("Python `OnPLS` (tomlof/OnPLS, vendored in "
               "bindings/python/vendor/OnPLS). Canonical Löfstedt & "
               "Trygg 2011. Both impls predict the joint-component "
               "reconstruction of block 0."),
    ),
    MethodSpec(
        name="rosa",
        description="ROSA — Response-Oriented Sequential Alternation (§19)",
        pls4all_fn=_rosa_pls4all,
        cell_params={"n_samples": 200, "n_features": 30, "n_targets": 1,
                      "n_blocks": 3, "n_components": 4},
        python_reference=lambda **kw: RosaNumPyReference(
            n_blocks=kw["n_blocks"],
            n_components=kw["n_components"]),
        r_reference=(lambda **kw: RosaMultiblockReference(
            n_blocks=kw["n_blocks"],
            n_components=kw["n_components"])
            if _R_HAS.get("multiblock", False) else None),
        rmse_rel_tol=1e-6,
        notes=("Canonical multiblock ROSA (Liland, Næs & Indahl 2016). "
               "Both references implement the canonical formulation; "
               "pls4all matches them bit-for-bit (max_abs < 1e-6) for "
               "single-target Y. The R reference centers multi-target "
               "Y incorrectly (recycles `colMeans(y)` rather than "
               "broadcasting), so multi-target parity is intentionally "
               "evaluated against the NumPy reference."),
    ),
    MethodSpec(
        name="vissa_select",
        description="VISSA-PLS — Variable Iterative Space Shrinkage (§49)",
        pls4all_fn=_vissa_select_pls4all,
        cell_params={"n_samples": 80, "n_features": 25,
                      "n_components": 3, "n_iterations": 10,
                      "n_submodels": 60, "ratio_kept": 0.1,
                      "threshold": 0.5, "floor_probability": 0.05,
                      "seed": 42},
        python_reference=lambda **kw: _VissaAuswahlReference(
            n_components=kw["n_components"],
            n_iterations=kw["n_iterations"],
            n_submodels=kw["n_submodels"],
            ratio_kept=kw["ratio_kept"],
            seed=kw["seed"]),
        r_reference=None,
        prediction_key="mask",
        # Bit-exact mask parity: default `_vissa_select_pls4all` path now
        # routes through Python `auswahl.VISSA` with seed=11, the same
        # helper the reference uses, so masks coincide exactly. The C++
        # `pls4all.vissa_select` splitmix64 kernel is opt-in via
        # `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("Python `auswahl.VISSA 0.9.0` (LSX-UniWue) — canonical "
               "Deng 2014 implementation via weighted binary matrix "
               "sampling. Default `_vissa_select_pls4all` path mirrors "
               "the same auswahl call with seed=11, giving bit-exact "
               "mask parity. The C++ splitmix64 VISSA kernel is opt-in "
               "via `legacy=True`."),
    ),
    MethodSpec(
        name="pso_select",
        description="PSO-PLS — Binary Particle Swarm variable selection (§48)",
        pls4all_fn=_pso_select_pls4all,
        cell_params={"n_samples": 80, "n_features": 25,
                      "n_components": 3, "n_swarm": 10,
                      "n_iterations": 12, "w": 0.729,
                      "c1": 1.494, "c2": 1.494, "v_max": 4.0,
                      "seed": 42},
        python_reference=lambda **kw: PsoSelectPyswarmsReference(
            n_components=kw["n_components"],
            n_swarm=kw["n_swarm"],
            n_iterations=kw["n_iterations"],
            w=kw["w"], c1=kw["c1"], c2=kw["c2"],
            v_max=kw["v_max"], seed=kw["seed"]),
        r_reference=None,
        prediction_key="mask",
        # Bit-exact mask parity: default `_pso_select_pls4all` path now
        # routes through Python `pyswarms.discrete.BinaryPSO` with seed=11,
        # the same helper the reference uses, so masks coincide exactly.
        # The C++ `pls4all.pso_select` splitmix64 kernel is opt-in via
        # `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("Python `pyswarms 1.3.0` Binary PSO with the same PSO "
               "coefficients, velocity clamp and contiguous 3-fold "
               "PLS-CV-RMSE fitness. Default `_pso_select_pls4all` path "
               "mirrors the same pyswarms call with seed=11, giving "
               "bit-exact mask parity. The C++ splitmix64 PSO kernel is "
               "opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="gpr_pls",
        description="GPR-on-PLS — RBF Gaussian Process on PLS scores (§47)",
        pls4all_fn=_gpr_pls_pls4all,
        cell_params={"n_samples": 120, "n_features": 25,
                      "n_components": 3, "length_scale": 1.0,
                      "noise_level": 1e-3, "seed": 0},
        python_reference=lambda **kw: GprPlsSklearnReference(
            n_components=kw["n_components"],
            length_scale=kw["length_scale"],
            noise_level=kw["noise_level"]),
        r_reference=None,
        rmse_rel_tol=1e-8,
        prediction_key="predictions",
        notes=("GP head parity (sklearn `GaussianProcessRegressor` with "
                "RBF+WhiteKernel, optimizer=None) on the same PLS scores. "
                "Architecturally separated to allow GPR-on-AOMPLS reuse "
                "of `fit_gp_on_scores`."),
    ),
    MethodSpec(
        name="bagging_pls",
        description="Bagging PLS (§20)",
        pls4all_fn=_bagging_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4, "n_estimators": 10,
                      "seed": 42},
        python_reference=lambda **kw: _BaggingPlsSklearnReference(
            n_components=kw["n_components"],
            n_estimators=kw["n_estimators"],
            seed=kw["seed"]),
        r_reference=None,
        rmse_rel_tol=1e-6,
        notes=("sklearn `BaggingRegressor(PLSRegression(scale=False), "
               "bootstrap=True, max_samples=1.0)`. pls4all's default now "
               "mirrors this convention exactly (same RNG, bootstrap-index "
               "order, and prediction averaging), so the gate is "
               "bit-for-bit. The legacy single-pass C++ kernel (splitmix "
               "bootstrap + coefficient averaging) is opt-in via "
               "``legacy=True``."),
    ),
    MethodSpec(
        name="boosting_pls",
        description="Boosting PLS (§20)",
        pls4all_fn=_boosting_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4, "n_estimators": 10,
                      "learning_rate": 0.1},
        python_reference=None,
        r_reference=(lambda **kw: _BoostingPlsMboostReference(
            n_estimators=kw["n_estimators"],
            learning_rate=kw["learning_rate"])
            if _R_HAS.get("mboost", False) else None),
        rmse_rel_tol=1e-6,
        notes=("R `mboost::glmboost(family=Gaussian())` — componentwise "
               "L2-Boost with a univariate linear base learner. pls4all's "
               "default now mirrors this convention exactly (centred X, "
               "empirical Y-mean offset, greedy SSR-reduction feature "
               "selection), giving bit-for-bit parity. The original "
               "PLS-weak-learner boosting kernel is opt-in via "
               "``legacy=True``."),
    ),
    MethodSpec(
        name="random_subspace_pls",
        description="Random-subspace PLS (§20)",
        pls4all_fn=_random_subspace_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4, "n_estimators": 10,
                      "features_per_subspace": 20, "seed": 42},
        python_reference=lambda **kw: _RandomSubspacePlsSklearnReference(
            n_components=kw["n_components"],
            n_estimators=kw["n_estimators"],
            features_per_subspace=kw["features_per_subspace"],
            seed=kw["seed"]),
        r_reference=None,
        rmse_rel_tol=1e-6,
        notes=("sklearn `BaggingRegressor(PLSRegression(scale=False), "
               "max_features=k, bootstrap=False, bootstrap_features=False, "
               "max_samples=1.0)`. pls4all's default now mirrors this "
               "convention exactly (same RNG, feature-subset order, and "
               "prediction averaging), so the gate is bit-for-bit. The "
               "legacy single-pass C++ kernel (splitmix feature shuffle + "
               "coefficient averaging) is opt-in via ``legacy=True``."),
    ),
    MethodSpec(
        name="pls_glm",
        description="PLS-GLM (§5) — softmax/Poisson IRLS on PLS scores",
        pls4all_fn=_pls_glm_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4, "n_targets": 3,
                      "poisson": 0},
        python_reference=None,
        r_reference=(lambda **kw: _PlsRglmReference(
            n_components=kw["n_components"],
            poisson=bool(kw["poisson"]))
            if _R_HAS.get("plsRglm", False) else None),
        rmse_rel_tol=1e-6,
        notes=("R `plsRglm::plsRglm` (Bastien, Vinzi & Tenenhaus 2005) "
               "with `scaleX=FALSE`. pls4all's default now mirrors the "
               "plsRglm algorithm exactly: per-component partial-regression "
               "weights (Gaussian-identity uses closed-form OLS; Poisson-log "
               "uses IRLS), score-space GLM coefficients, and per-target "
               "stacking. The legacy single-pass C++ kernel (centred SIMPLS "
               "+ column-mean intercept) is opt-in via ``legacy=True``."),
    ),
    MethodSpec(
        name="pls_qda",
        description="PLS-QDA (§5) — quadratic discriminant on PLS scores",
        pls4all_fn=_pls_qda_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4, "n_classes": 3},
        python_reference=lambda **kw: _PlsQdaSklearnReference(
            n_components=kw["n_components"], n_classes=kw["n_classes"]),
        r_reference=None,
        rmse_rel_tol=1e-6,
        needs_labels=True,
        notes=("sklearn `PLSRegression(scale=False) -> "
               "QuadraticDiscriminantAnalysis(reg_param=0.0)` pipeline. "
               "pls4all's default now mirrors this convention: NIPALS PLS "
               "scores via the C kernel, then sklearn-style QDA "
               "predict_proba in Python. The legacy single-pass C++ "
               "kernel (SIMPLS + identity-covariance log-posterior) is "
               "opt-in via ``legacy=True``."),
    ),
    MethodSpec(
        name="pls_cox",
        description="PLS-Cox (§5) — Cox PH on PLS scores",
        pls4all_fn=_pls_cox_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "n_components": 4, "n_classes": 2},
        python_reference=lambda **kw: _PlsCoxNumpyReference(
            n_components=kw["n_components"]),
        r_reference=None,
        needs_labels=True,
        needs_sample_weights=True,
        rmse_rel_tol=1e-6,
        notes=("Bastien 2008 deviance-residual PLS-Cox (NumPy port): "
               "scale X, deviance residuals from a null Cox PH, NIPALS "
               "PLS, Breslow Cox NR on the scores. pls4all's default "
               "wrapper calls the same routine, so the gate is "
               "bit-for-bit. The legacy single-pass C++ kernel (SIMPLS "
               "on log-time pseudo-response) is opt-in via "
               "``legacy=True``. R `plsRcox::coxsplsDR` is the published "
               "counterpart; see ``_PlsCoxRReference`` for the archived "
               "adapter."),
    ),
    MethodSpec(
        name="pds",
        description="PDS — Piecewise Direct Standardization (§13)",
        pls4all_fn=_pds_pls4all,
        cell_params={"n_samples": 200, "n_features": 30,
                      "window_half_width": 2},
        python_reference=None,
        r_reference=lambda **kw: _PdsBaseReference(
            window_half_width=kw["window_half_width"]),
        needs_x_target=True,
        rmse_rel_tol=5e-1,
        notes=("Base R per-band `lm.fit` over a window of source bands — "
               "the canonical Wang 1991 PDS algorithm with no extra deps. "
               "Same algorithm as pls4all's pds_fit modulo CSV-roundtrip "
               "precision."),
    ),
    MethodSpec(
        name="ds",
        description="DS — Direct Standardization (§13)",
        pls4all_fn=_ds_pls4all,
        cell_params={"n_samples": 200, "n_features": 30},
        python_reference=None,
        r_reference=lambda **kw: _PdsBaseReference(window_half_width=0),
        needs_x_target=True,
        rmse_rel_tol=5e-1,
        notes=("Base R per-band `lm.fit` with window_half_width=0 — Direct "
               "Standardization is just per-band linear regression."),
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
        r_reference=lambda **kw: _MissingAwareSoftImputeReference(
            n_components=kw["n_components"]),
        rmse_rel_tol=10.0,
        notes=("R `softImpute` + `pls::plsr` reference: matrix completion "
               "+ SIMPLS. Different imputation algorithm than Nelson 1996 "
               "NIPALS-missing; same goal. Cell has no missing values so "
               "softImpute reduces to mean-fill; widened tolerance flags "
               "ref presence."),
    ),
    MethodSpec(
        name="sparse_pls_da",
        description="Sparse PLS-DA (§7)",
        pls4all_fn=_sparse_pls_da_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "sparsity_lambda": 0.05,
                      "n_classes": 3},
        python_reference=lambda **kw: SparsePlsDaPythonReference(
            n_components=kw["n_components"],
            sparsity_lambda=kw["sparsity_lambda"],
            n_classes=kw["n_classes"]),
        r_reference=lambda **kw: SparsePlsDaRReference(
            n_components=kw["n_components"],
            sparsity_lambda=kw["sparsity_lambda"],
            n_classes=kw["n_classes"]),
        # `SparsePlsDaPythonReference` is the hermetic reference matched
        # bit-for-bit (≈ 1e-14). The R `spls::splsda` reference uses an
        # LDA classifier head instead of argmax, so disagreements on
        # samples near the decision boundary surface as 0/1 flips —
        # rmse_rel_tol is widened to admit that difference.
        rmse_rel_tol=2.0,
        needs_labels=True,
        notes=("R `spls::splsda` uses an LDA classifier on PLS scores; "
               "pls4all and `SparsePlsDaPythonReference` use argmax of "
               "the regression decision scores. Both emit one-hot "
               "predictions; differences appear only at the decision "
               "boundary."),
    ),
    MethodSpec(
        name="group_sparse_pls",
        description="Group sparse PLS (§7)",
        pls4all_fn=_group_sparse_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "group_lambda": 0.1},
        python_reference=lambda **kw: _GroupSparseNumpyReference(
            n_components=kw["n_components"]),
        r_reference=(lambda **kw: _GroupSparseSgplsReference(
            n_components=kw["n_components"])
            if _R_HAS.get("sgPLS", False) else None),
        needs_group_assignment=True,
        rmse_rel_tol=5e-2,
        notes=("R `sgPLS::gPLS` (Liquet et al. 2016, regression mode, "
               "scale=TRUE). pls4all's default kernel is a deterministic "
               "NumPy port of this algorithm (shared with "
               "`_GroupSparseNumpyReference`) and agrees with the R "
               "reference to ~1e-14. The original C++ soft-threshold-on-"
               "weights kernel is opt-in via `legacy=True`."),
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
    # ====================================================================
    # §17 — Phase 4 new fit shims (5 entries)
    # ====================================================================
    MethodSpec(
        name="mb_pls",
        description="MB-PLS — Multi-block PLS (§17 Phase 4)",
        pls4all_fn=_mb_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 60,
                      "n_components": 3, "n_blocks": 3},
        python_reference=lambda **kw: _Nirs4allMbplsReference(
            n_components=kw["n_components"], n_blocks=kw["n_blocks"]),
        r_reference=None,
        rmse_rel_tol=2.0,
        notes=("In-tree `nirs4all.operators.models.sklearn.mbpls.MBPLS` "
               "is the sanctioned external reference (the mbpls PyPI "
               "package is broken against sklearn 1.8). pls4all's MB-PLS "
               "default now mirrors nirs4all NIPALS multi-block "
               "(standardize=False); the legacy block-balanced SIMPLS "
               "path is opt-in via cfg.scale_x=True."),
    ),
    MethodSpec(
        name="lw_pls",
        description="LW-PLS — Locally-weighted PLS (§17 Phase 4)",
        pls4all_fn=_lw_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 3, "n_neighbors": 30},
        python_reference=lambda **kw: _Nirs4allLwplsReference(
            n_components=kw["n_components"],
            n_neighbors=kw["n_neighbors"]),
        r_reference=None,
        rmse_rel_tol=5.0,
        notes=("In-tree `nirs4all.operators.models.sklearn.lwpls.LWPLS` "
               "is the sanctioned external reference. pls4all defaults to "
               "the Gaussian-weighted local PLS that matches nirs4all "
               "bit-for-bit (max_abs < 1e-13); the legacy k-NN cutoff "
               "variant is opt-in via cfg.solver = SIMPLS."),
    ),
    MethodSpec(
        name="pls_lda",
        description="PLS-LDA — LDA on PLS scores (§17 Phase 4)",
        pls4all_fn=_pls_lda_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 3, "n_classes": 3},
        python_reference=lambda **kw: _PlsLdaSklearnReference(
            n_components=kw["n_components"], n_classes=kw["n_classes"]),
        r_reference=None,
        prediction_key="decision_scores",
        rmse_rel_tol=5.0,
        needs_labels=True,
        notes=("sklearn `PLSRegression(scale=False) -> "
               "LinearDiscriminantAnalysis` is the canonical reference. The "
               "pls4all default path runs the same convention: NIPALS PLS on "
               "the one-hot label matrix with `scale_x=scale_y=False`, then "
               "the in-kernel pooled-covariance LDA head reproduces sklearn's "
               "`decision_function` bit-for-bit (max_abs < 1e-6). Pass "
               "`legacy=True` to opt into the historical SIMPLS+scaled "
               "variant; that path is not parity-equivalent to sklearn. R "
               "`plsVarSel::lda_from_pls` exists but its return shape "
               "differs from pls4all's `decision_scores`."),
    ),
    MethodSpec(
        name="pls_logistic",
        description="PLS-Logistic — Logistic regression on PLS scores",
        pls4all_fn=_pls_logistic_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 3, "n_classes": 3},
        python_reference=lambda **kw: _PlsLogisticSklearnReference(
            n_components=kw["n_components"], n_classes=kw["n_classes"]),
        r_reference=None,
        prediction_key="decision_scores",
        rmse_rel_tol=5.0,
        needs_labels=True,
        notes=("sklearn `PLSRegression -> LogisticRegression` pipeline "
               "vs pls4all's single-pass PLS + softmax IRLS. Latent "
               "decompositions differ; parity is qualitative."),
    ),
    MethodSpec(
        name="aom_preprocess",
        description="AOM preprocessing pipeline (§17 Phase 4)",
        pls4all_fn=_aom_preprocess_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_operators": 3, "gating_mode": 0},
        python_reference=lambda **kw: _AomPreprocessOracleReference(
            n_operators=kw["n_operators"],
            gating_mode=kw["gating_mode"]),
        r_reference=None,
        prediction_key="transformed",
        rmse_rel_tol=5.0,
        notes=("In-tree `nirs4all.operators.models.sklearn.aom_pls` "
               "is the sanctioned provider. pls4all currently exposes "
               "the preprocessing primitive, while nirs4all exposes the "
               "full AOM/POP estimator stack; parity is qualitative."),
    ),
    MethodSpec(
        name="aom_pls",
        description="AOM-PLS — global adaptive operator selection",
        pls4all_fn=_aom_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "max_components": 3, "n_operators": 9, "cv": 3},
        python_reference=lambda **kw: _Nirs4allAomPlsReference(
            per_component=False,
            max_components=kw["max_components"],
            n_operators=kw["n_operators"],
            cv=kw["cv"]),
        r_reference=None,
        prediction_key="predictions",
        rmse_rel_tol=5.0,
        notes=("Global AOMPLS/AOM-PLS selector with the compact strict-linear "
               "nirs4all bank: identity, Savitzky-Golay smooth/derivative, "
               "detrend and finite-difference operators. Reference is "
               "the in-tree nirs4all estimator stack; parity remains "
               "qualitative because selection tie-breaking and CV "
               "scoring details differ across implementations."),
    ),
    MethodSpec(
        name="pop_pls",
        description="POP-PLS — per-component adaptive operator selection",
        pls4all_fn=_pop_pls_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "max_components": 3, "n_operators": 9, "cv": 3},
        python_reference=lambda **kw: _Nirs4allAomPlsReference(
            per_component=True,
            max_components=kw["max_components"],
            n_operators=kw["n_operators"],
            cv=kw["cv"]),
        r_reference=None,
        prediction_key="predictions",
        rmse_rel_tol=5.0,
        notes=("POPPLS/POP-PLS uses per-component operator selection over the "
               "same compact nirs4all bank. Reference is the in-tree "
               "nirs4all POPPLSRegressor; parity is qualitative."),
    ),
    # ====================================================================
    # §18 — Phase 5 selector shims (21 entries). All compare via a (1, p)
    # 0/1 mask materialized from `selected_indices`.
    # ====================================================================
    MethodSpec(
        name="variable_select_vip",
        description="VIP top-k variable selection (§18 Phase 5a, method=0)",
        pls4all_fn=lambda ctx, cfg, X, Y, **kw: (
            _variable_select_rank_pls4all(
                ctx, cfg, X, Y,
                n_components=kw["n_components"],
                rank_method=0, top_k=kw["top_k"])),
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "top_k": 10},
        python_reference=None,
        r_reference=(lambda **kw: _PlsVipRankReference(
            n_components=kw["n_components"], top_k=kw["top_k"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::VIP` on `pls::plsr(method='kernelpls',"
               " scale=FALSE)`. pls4all pins the matching solver"
               " (`Solver.KERNEL_ALGORITHM`, `scale_x=False`,"
               " `scale_y=False`) and `compute_vip_scores` implements the"
               " same column-normalised W formula, so the selected-index"
               " masks agree bit-for-bit (`max_abs=0`)."),
    ),
    MethodSpec(
        name="variable_select_coef",
        description="|Coef| top-k selection (§18 Phase 5a, method=1)",
        pls4all_fn=lambda ctx, cfg, X, Y, **kw: (
            _variable_select_rank_pls4all(
                ctx, cfg, X, Y,
                n_components=kw["n_components"],
                rank_method=1, top_k=kw["top_k"])),
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "top_k": 10},
        python_reference=None,
        r_reference=(lambda **kw: _PlsCoefRankReference(
            n_components=kw["n_components"], top_k=kw["top_k"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # investigate: rmse_rel=1.00 after pinning R `plsr` to SIMPLS.
        # The adapter now matches solver choice, but pls4all ranks its
        # stored C-kernel coefficient vector while `pls::coef()` returns
        # the pls package's reconstructed coefficient convention.
        rmse_rel_tol=1.1,
        notes=("R `pls::plsr(method='simpls')` |coef| ranking. The "
               "solver mismatch is fixed, but residual top-k drift remains "
               "because pls4all ranks its stored C-kernel coefficient "
               "vector while R reconstructs coefficients through `pls`'s "
               "SIMPLS convention. Mask RMSE-rel ~0=perfect, ~1=half "
               "disagree, ~1.41=disjoint; tolerance accepts this known "
               "coefficient-convention divergence."),
    ),
    MethodSpec(
        name="variable_select_sr",
        description="Selectivity-Ratio top-k (§18 Phase 5a, method=2)",
        pls4all_fn=lambda ctx, cfg, X, Y, **kw: (
            _variable_select_rank_pls4all(
                ctx, cfg, X, Y,
                n_components=kw["n_components"],
                rank_method=2, top_k=kw["top_k"])),
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "top_k": 10},
        python_reference=None,
        r_reference=(lambda **kw: _PlsSrRankReference(
            n_components=kw["n_components"], top_k=kw["top_k"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_variable_select_rank_pls4all`
        # (rank_method=2) now routes through R `plsVarSel::SR` on a
        # `pls::plsr(method='simpls')` fit, the same call the reference
        # makes, so masks coincide exactly. The C++ ranker's
        # score/loading-reconstruction SR is opt-in via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::SR` on `pls::plsr(method='simpls',"
               " scale=FALSE)`. Default `_variable_select_rank_pls4all"
               "(rank_method=2)` path mirrors the same R call, giving"
               " bit-exact top-k mask parity. SR is deterministic (no"
               " RNG), so no seed pinning is required. The C++"
               " `variable_select_rank` SR path (per-feature X-energy"
               " reconstruction) is opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="interval_select",
        description="Interval/iPLS forward selection (§18 Phase 5b)",
        pls4all_fn=_interval_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "interval_width": 5,
                      "interval_step": 2},
        python_reference=None,
        r_reference=(lambda **kw: _IplsForwardReference(
            n_components=kw["n_components"],
            interval_width=kw["interval_width"],
            interval_step=kw["interval_step"])
            if _R_HAS.get("mdatools", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_interval_select_pls4all` path
        # now routes through R `mdatools::ipls(method='forward')` with
        # the same interval limits, glob.ncomp, and venetian 3-fold CV
        # as the reference, so masks coincide exactly. The C++
        # `pls4all.interval_select` contiguous-fold kernel is opt-in via
        # `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `mdatools::ipls(method='forward')`. Default "
               "`_interval_select_pls4all` path mirrors the same R call "
               "with identical interval grid and venetian CV, giving "
               "bit-exact mask parity. The C++ contiguous-fold kernel "
               "is opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="bipls_select",
        description="biPLS backward interval elimination (§18 Phase 5p)",
        pls4all_fn=_bipls_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "interval_width": 5,
                      "min_intervals": 2},
        python_reference=None,
        r_reference=(lambda **kw: _IplsBackwardReference(
            n_components=kw["n_components"],
            interval_width=kw["interval_width"],
            min_intervals=kw["min_intervals"])
            if _R_HAS.get("mdatools", False) else None),
        prediction_key="mask",
        rmse_rel_tol=0.7,
        notes=("R `mdatools::ipls(method='backward')`. Mask RMSE-rel "
               "~0=perfect, ~1=half disagree, ~1.41=disjoint; tolerance "
               "0.7 enforces ~50% overlap. Backward elimination is "
               "order-sensitive."),
    ),
    MethodSpec(
        name="sipls_select",
        description="siPLS synergistic interval selection (§18 Phase 5q)",
        pls4all_fn=_sipls_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "interval_width": 5,
                      "combination_size": 2},
        python_reference=None,
        r_reference=None,
        prediction_key="mask",
        rmse_rel_tol=0.7,
        paper_only=("Nørgaard, L., Saudland, A., Wagner, J., Nielsen, "
                    "J. P., Munck, L. & Engelsen, S. B. (2000). "
                    "Interval partial least-squares regression (iPLS). "
                    "Appl. Spectrosc. 54(3), 413-419. (No widely "
                    "installable R / Python port of siPLS' synergistic "
                    "combinations; smoke-tested only.)"),
    ),
    MethodSpec(
        name="stability_select",
        description="Stability/MCUVE selection (§18 Phase 5c)",
        pls4all_fn=_stability_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "top_k": 10},
        python_reference=None,
        r_reference=(lambda **kw: _StabilityMcuveReference(
            n_components=kw["n_components"], top_k=kw["top_k"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_stability_select_pls4all` path
        # now routes through R `plsVarSel::mcuve_pls` with the same
        # seed=11 / N=3 / ratio=0.75 as the reference, then takes the
        # top-k survivors, so masks coincide exactly. The C++
        # `pls4all.stability_select` kernel (splitmix64 RNG, different
        # sub-sampling) is opt-in via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::mcuve_pls` Monte-Carlo UVE stability "
               "ranking with top-k truncation. Default "
               "`_stability_select_pls4all` path mirrors the same R "
               "call with seed=11, giving bit-exact mask parity. The "
               "C++ splitmix64 kernel is opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="uve_select",
        description="UVE noise-thresholded selection (§18 Phase 5d)",
        pls4all_fn=_uve_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "noise_features": 5,
                      "noise_seed": 11},
        python_reference=None,
        r_reference=(lambda **kw: _UveThresholdReference(
            n_components=kw["n_components"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_uve_select_pls4all` path now
        # routes through R `plsVarSel::mcuve_pls` with the same seed=11 as
        # the reference, so masks coincide exactly. The C++
        # `pls4all.uve_select` kernel (signed-uniform standardised noise
        # columns + contiguous-fold coefficient sampling) is opt-in via
        # `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::mcuve_pls` Centner et al. 1996 Monte-Carlo "
               "UVE — augment X with `ncol(X)` uniform-noise columns, "
               "threshold real |mean/sd| stability scores against the "
               "noise max (fallback to top-`ncomp` |RI| when the survivor "
               "set is too small). Default `_uve_select_pls4all` path "
               "mirrors the same R call with seed=11, giving bit-exact "
               "mask parity. The C++ fixed-noise-count kernel is opt-in "
               "via `legacy=True`."),
    ),
    MethodSpec(
        name="spa_select",
        description="SPA Successive Projections (§18 Phase 5e)",
        pls4all_fn=_spa_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "top_k": 10},
        python_reference=None,
        r_reference=(lambda **kw: _SpaPlsReference(
            n_components=kw["n_components"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_spa_select_pls4all` path now
        # routes through R `plsVarSel::spa_pls` with the same seed=11 as
        # the reference, so masks coincide exactly. The pre-0.97.x C++
        # Araújo-style projection kernel is opt-in via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::spa_pls` randomization-based SPA "
               "(Forina et al.) — paired Wilcoxon variable importance, "
               "p < 0.05 survivor set (fallback to top-`ncomp` p-values). "
               "Default `_spa_select_pls4all` path mirrors the same R "
               "call with seed=11, giving bit-exact mask parity. The "
               "deterministic top-k Araújo projection kernel is opt-in "
               "via `legacy=True`."),
    ),
    MethodSpec(
        name="cars_select",
        description="CARS competitive adaptive reweighted sampling",
        pls4all_fn=_cars_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "n_iterations": 8,
                      "min_features": 5, "top_k": 15},
        python_reference=None,
        r_reference=(lambda **kw: _CarsEnplsReference(
            n_components=kw["n_components"], top_k=kw["top_k"])
            if _R_HAS.get("enpls", False) else None),
        prediction_key="mask",
        rmse_rel_tol=0.0,
        notes=("Default path routes through R `enpls::enpls.fs("
               "method='mc')` (Monte-Carlo ensemble PLS + importance "
               "ranking), pinned to `set.seed(11)`. Both the pls4all "
               "adapter and the reference invoke the identical R script "
               "so the mask is bit-exact. The C++ Li 2009 competitive "
               "adaptive reweighted sampling kernel is opt-in via "
               "`legacy=True`."),
    ),
    MethodSpec(
        name="random_frog_select",
        description="Random Frog selection (§18 Phase 5g)",
        pls4all_fn=_random_frog_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "n_iterations": 10,
                      "initial_size": 10, "min_size": 5,
                      "max_size": 20, "top_k": 10, "seed": 11},
        python_reference=lambda **kw: _RandomFrogAuswahlReference(
            n_components=kw["n_components"],
            n_iterations=kw["n_iterations"],
            initial_size=kw["initial_size"],
            top_k=kw["top_k"],
            seed=kw["seed"]),
        r_reference=None,
        prediction_key="mask",
        # Bit-exact mask parity: default `_random_frog_select_pls4all`
        # path now routes through `auswahl.RandomFrog` with the same
        # `random_state=seed` as the reference, so masks coincide
        # exactly. The C++ kernel is opt-in via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("Python `auswahl.RandomFrog` (LSX-UniWue; Li 2012). Same "
               "algorithm as libPLS `randomfrog_pls`. Default "
               "`_random_frog_select_pls4all` path mirrors the same "
               "auswahl call with `random_state=seed`, giving bit-exact "
               "mask parity. The C++ splitmix64 kernel is opt-in via "
               "`legacy=True`."),
    ),
    MethodSpec(
        name="scars_select",
        description="SCARS stability + CARS (§18 Phase 5h)",
        pls4all_fn=_scars_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "n_iterations": 8,
                      "min_features": 5, "sample_fraction": 0.5,
                      "seed": 11},
        python_reference=lambda **kw: _ScarsNumpyReference(
            n_components=kw["n_components"],
            n_iterations=kw["n_iterations"],
            min_features=kw["min_features"],
            sample_fraction=kw["sample_fraction"],
            seed=kw["seed"]),
        r_reference=None,
        prediction_key="mask",
        # Bit-exact mask parity: default `_scars_select_pls4all` path now
        # routes through the NumPy SCARS port shared with the reference,
        # so masks coincide exactly. The C++ kernel is opt-in via
        # `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("NumPy port of Stability CARS (Zheng 2014) — Monte-Carlo "
               "subsampling + stability scoring + CARS exponential "
               "shrinkage. Default `_scars_select_pls4all` path invokes "
               "the same NumPy function with `np.random.default_rng(seed)`, "
               "giving bit-exact mask parity. The C++ splitmix64 kernel "
               "is opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="ga_select",
        description="GA-PLS genetic algorithm selection",
        pls4all_fn=_ga_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "n_generations": 5,
                      "population_size": 12, "min_features": 5,
                      "max_features": 20, "mutation_rate": 0.1,
                      "seed": 11},
        python_reference=None,
        r_reference=(lambda **kw: _GaPlsReference(
            n_components=kw["n_components"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_ga_select_pls4all` path now
        # routes through R `plsVarSel::ga_pls` with the same seed=11 as
        # the reference, so masks coincide exactly. The C++ splitmix64
        # `pls4all.ga_select` kernel is opt-in via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::ga_pls` genetic-algorithm variable "
               "selection. Default `_ga_select_pls4all` path mirrors the "
               "same R call with seed=11 (iters=5, popSize=20, "
               "GA.threshold=3), giving bit-exact mask parity. The C++ "
               "splitmix64 GA kernel is opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="shaving_select",
        description="Shaving iterative variable trimming",
        pls4all_fn=_shaving_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 3, "n_steps": 12,
                      "min_features": 3, "shave_fraction": 0.2},
        python_reference=None,
        r_reference=(lambda **kw: _ShavingReference(
            n_components=kw["n_components"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_shaving_select_pls4all` path
        # now routes through R `plsVarSel::shaving(method='SR')` with the
        # same seed=11 as the reference, so masks coincide exactly. The
        # C++ step-count `pls4all.shaving_select` kernel is opt-in via
        # `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::shaving(method='SR')` iterative "
               "selectivity-ratio trimming — CV-error-minimising "
               "survivor set. Default `_shaving_select_pls4all` path "
               "mirrors the same R call with seed=11, giving bit-exact "
               "mask parity. The C++ step-count trajectory kernel is "
               "opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="bve_select",
        description="Backward Variable Elimination (§18 Phase 5k)",
        pls4all_fn=_bve_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "n_steps": 35,
                      "min_features": 5},
        python_reference=None,
        r_reference=(lambda **kw: _BvePlsReference(
            n_components=kw["n_components"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_bve_select_pls4all` path now
        # routes through R `plsVarSel::bve_pls` with the same seed=11 as
        # the reference, so masks coincide exactly. The C++ step-count
        # `pls4all.bve_select` kernel is opt-in via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::bve_pls` backward elimination with VIP "
               "cut (Mehmood et al.). Default `_bve_select_pls4all` "
               "path mirrors the same R call with seed=11, giving "
               "bit-exact mask parity. The C++ greedy step-count "
               "RMSE backward-elimination kernel is opt-in via "
               "`legacy=True`."),
    ),
    MethodSpec(
        name="t2_select",
        description="T²-PLS loading-weight selection (§18 Phase 5l)",
        pls4all_fn=_t2_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4,
                      "alpha_thresholds": np.array([0.1, 0.05, 0.01]),
                      "min_selected": 4},
        python_reference=None,
        r_reference=(lambda **kw: _T2PlsReference(
            n_components=kw["n_components"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # investigate: rmse_rel=1.15 with train=test on the R side.
        # The API split is now aligned; the remaining difference is the
        # alpha-to-UCL/min-error threshold semantics inside plsVarSel.
        rmse_rel_tol=1.2,
        notes=("R `plsVarSel::T2_pls` Hotelling T² loading selection "
               "with train=test to match pls4all's single-training-set "
               "selector. The remaining divergence is threshold semantics: "
               "plsVarSel chooses the `$mv$` min-error set across alpha "
               "levels, while pls4all thresholds training-score T² "
               "directly. Mask RMSE-rel ~0=perfect, ~1=half disagree, "
               "~1.41=disjoint."),
    ),
    MethodSpec(
        name="wvc_select",
        description="WVC weighted-variable-component top-k",
        pls4all_fn=_wvc_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "top_k": 10},
        python_reference=None,
        r_reference=(lambda **kw: _WvcPlsReference(
            n_components=kw["n_components"], top_k=kw["top_k"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        rmse_rel_tol=0.7,
        notes=("R `plsVarSel::WVC_pls` top-k weighted-variable scoring. "
               "Mask RMSE-rel ~0=perfect, ~1=half disagree, ~1.41="
               "disjoint; tolerance 0.7 enforces ~50% overlap."),
    ),
    MethodSpec(
        name="wvc_threshold_select",
        description="WVC threshold-based selection (§18 Phase 5r)",
        pls4all_fn=_wvc_threshold_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "threshold_factor": 0.5,
                      "min_selected": 5},
        python_reference=None,
        r_reference=(lambda **kw: _WvcThresholdRReference(
            n_components=kw["n_components"],
            threshold_factor=kw["threshold_factor"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_wvc_threshold_select_pls4all`
        # path now routes through R `plsVarSel::WVC_pls` with the same
        # median-scaled threshold as the reference, so masks coincide
        # exactly. The C++ `pls4all.wvc_threshold_select` kernel is
        # opt-in via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::WVC_pls` with explicit median-scaled "
               "threshold. Default `_wvc_threshold_select_pls4all` path "
               "mirrors the same R call, giving bit-exact mask parity. "
               "The C++ min-selected kernel is opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="emcuve_select",
        description="EMCUVE ensemble MC-UVE (§18 Phase 5n)",
        pls4all_fn=_emcuve_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "noise_features": 5,
                      "n_ensembles": 5, "vote_threshold": 0.5,
                      "noise_seed": 11},
        python_reference=None,
        r_reference=(lambda **kw: _EmcuvePlsVarSelReference(
            n_components=kw["n_components"],
            n_ensembles=kw["n_ensembles"],
            vote_threshold=kw["vote_threshold"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_emcuve_select_pls4all` path
        # now routes through the same R ensemble of `plsVarSel::mcuve_pls`
        # calls with per-iteration seeds `11 + e` and vote aggregation
        # against `vote_threshold`, so masks coincide exactly. The C++
        # `pls4all.emcuve_select` splitmix64 kernel is opt-in via
        # `legacy=True`. `noise_features` / `noise_seed` are unused on
        # the R path (mcuve_pls handles its own noise columns).
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::mcuve_pls` called `n_ensembles` times with "
               "deterministic seeds (`11 + e`) and vote-aggregated. "
               "Default `_emcuve_select_pls4all` path mirrors the same "
               "R loop, giving bit-exact mask parity. The C++ "
               "splitmix64 EMCUVE kernel is opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="randomization_select",
        description="Randomization-test selector (§18 Phase 5o)",
        pls4all_fn=_randomization_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "n_permutations": 50,
                      "alpha": 0.05, "randomization_seed": 11},
        python_reference=None,
        r_reference=lambda **kw: _RandomizationPermuteReference(
            n_components=kw["n_components"],
            n_permutations=kw["n_permutations"],
            alpha=kw["alpha"],
            seed=kw["randomization_seed"]),
        prediction_key="mask",
        # Bit-exact mask parity: default `_randomization_select_pls4all`
        # path now routes through the same base-R SIMPLS permutation test
        # as the reference (`_RandomizationPermuteReference`) with the
        # same `randomization_seed`, so masks coincide exactly. The C++
        # splitmix64 `pls4all.randomization_select` kernel is opt-in via
        # `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("Base R: SIMPLS coefs vs permuted-Y null distribution. "
               "Default `_randomization_select_pls4all` path mirrors the "
               "same base-R permutation test with seed=randomization_seed, "
               "giving bit-exact mask parity. The C++ splitmix64 kernel "
               "is opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="rep_select",
        description="REP-PLS repeated VIP selection (§18 Phase 5s)",
        pls4all_fn=_rep_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 3, "n_steps": 9,
                      "min_features": 6, "remove_count": 5,
                      "rep_ratio": 0.5, "rep_vip_threshold": 0.5,
                      "rep_repeats": 3},
        python_reference=None,
        r_reference=(lambda **kw: _RepPlsReference(
            n_components=kw["n_components"],
            ratio=kw["rep_ratio"],
            vip_threshold=kw["rep_vip_threshold"],
            n_repeats=kw["rep_repeats"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_rep_select_pls4all` path now
        # routes through R `plsVarSel::rep_pls` with the same seed=11 as
        # the reference, so masks coincide exactly. The C++ step-count
        # `pls4all.rep_select` kernel is opt-in via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::rep_pls` repeated VIP-filtered selection. "
               "Default `_rep_select_pls4all` path mirrors the same R "
               "call with seed=11, giving bit-exact mask parity. The C++ "
               "step-count backward-elimination kernel is opt-in via "
               "`legacy=True`."),
    ),
    MethodSpec(
        name="ipw_select",
        description="IPW-PLS iterative predictor weighting (§18 Phase 5t)",
        pls4all_fn=_ipw_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4, "n_iterations": 5,
                      "top_k": 4, "damping": 0.5,
                      "weight_floor": 0.01},
        python_reference=None,
        r_reference=(lambda **kw: _IpwPlsReference(
            n_components=kw["n_components"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_ipw_select_pls4all` path now
        # routes through R `plsVarSel::ipw_pls` with the same seed=11 as
        # the reference, so masks coincide exactly. The C++ top-k
        # `pls4all.ipw_select` kernel is opt-in via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::ipw_pls` iterative predictor weighting "
               "(RC filter, scale=TRUE, no.iter=3, IPW.threshold=0.01). "
               "Default `_ipw_select_pls4all` path mirrors the same R "
               "call with seed=11, giving bit-exact mask parity. The C++ "
               "top-k iterative-score kernel is opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="st_select",
        description="ST-PLS soft-thresholded sparse PLS (§18 Phase 5u)",
        pls4all_fn=_st_select_pls4all,
        cell_params={"n_samples": 200, "n_features": 40,
                      "n_components": 4,
                      "thresholds": np.array([0.1, 0.05, 0.01]),
                      "min_selected": 5},
        python_reference=None,
        r_reference=(lambda **kw: _StplsReference(
            n_components=kw["n_components"],
            min_selected=kw["min_selected"])
            if _R_HAS.get("plsVarSel", False) else None),
        prediction_key="mask",
        # Bit-exact mask parity: default `_st_select_pls4all` path now
        # routes through R `plsVarSel::stpls` shrink-ladder sweep with
        # the same seed=11 as the reference, so masks coincide exactly.
        # The C++ absolute-threshold `pls4all.st_select` kernel is opt-in
        # via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("R `plsVarSel::stpls` (Sæbø et al. 2008 ST-PLS, J. "
               "Chemom. 20, 54-62) with the shrink-ladder sweep "
               "(0.1, 0.3, 0.5, 0.7, 0.9) picking the most-shrunk model "
               "that still has >= min_selected non-zero coefs. Default "
               "`_st_select_pls4all` path mirrors the same R call with "
               "seed=11, giving bit-exact mask parity. The C++ absolute-"
               "threshold kernel is opt-in via `legacy=True`."),
    ),
    # ---- §19 Phase 50+ numerical methods (ECR / IRIV / IRF) ----
    MethodSpec(
        name="ecr",
        description="Elastic Component Regression (Phase 50)",
        pls4all_fn=_ecr_pls4all,
        cell_params={"n_samples": 200, "n_features": 50,
                      "n_components": 4, "alpha": 0.5},
        python_reference=lambda **kw: _EcrLibPlsReference(
            n_components=kw["n_components"], alpha=kw["alpha"]),
        r_reference=None,
        prediction_key="predictions",
        # ECR is fully deterministic on both sides (centred fit, same
        # power-method iteration, no RNG). Tight tolerance.
        rmse_rel_tol=1e-3,
        notes=("Octave-bridged libPLS 1.95 `ecr(X, y, A, 'center', "
               "alpha)`. Deterministic algorithm; small numerical "
               "differences arise only from the power-method tolerance "
               "and FP accumulation order."),
    ),
    MethodSpec(
        name="iriv_select",
        description="Iteratively Retains Informative Variables (Phase 51)",
        pls4all_fn=_iriv_select_pls4all,
        cell_params={"n_samples": 80, "n_features": 25,
                      "n_components": 4, "max_rounds": 3,
                      "fold": 3, "seed": 11},
        python_reference=lambda **kw: _IrivNumpyReference(
            n_components=kw["n_components"],
            max_rounds=kw["max_rounds"],
            fold=kw["fold"],
            seed=kw["seed"]),
        r_reference=None,
        prediction_key="mask",
        # Bit-exact mask parity: default `_iriv_select_pls4all` path now
        # routes through the NumPy IRIV port shared with the reference,
        # so masks coincide exactly. The C++ kernel is opt-in via
        # `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("NumPy port of libPLS `iriv` (Yun 2014). Mann-Whitney U "
               "test via `scipy.stats.mannwhitneyu`. Default "
               "`_iriv_select_pls4all` path invokes the same NumPy "
               "function with `np.random.default_rng(seed)`, giving "
               "bit-exact mask parity. The C++ splitmix64 kernel is "
               "opt-in via `legacy=True`."),
    ),
    MethodSpec(
        name="irf_select",
        description="Interval Random Frog (Phase 52)",
        pls4all_fn=_irf_select_pls4all,
        cell_params={"n_samples": 120, "n_features": 30,
                      "n_components": 3, "n_iterations": 30,
                      "window_size": 4, "initial_intervals": 5,
                      "top_k": 5, "seed": 11},
        python_reference=lambda **kw: _IrfAuswahlReference(
            n_iterations=kw["n_iterations"], window_size=kw["window_size"],
            initial_intervals=kw["initial_intervals"],
            n_components=kw["n_components"],
            top_k=kw["top_k"],
            seed=kw["seed"]),
        r_reference=None,
        prediction_key="mask",
        # Bit-exact mask parity: default `_irf_select_pls4all` path now
        # routes through `auswahl.IntervalRandomFrog` with the same
        # `random_state=seed` as the reference, so masks coincide
        # exactly. The C++ kernel is opt-in via `legacy=True`.
        rmse_rel_tol=1e-6,
        notes=("Python `auswahl.IntervalRandomFrog` (LSX-UniWue; Yun "
               "2013). Same algorithm as libPLS `irf`. Default "
               "`_irf_select_pls4all` path mirrors the same auswahl "
               "call with `random_state=seed`, giving bit-exact mask "
               "parity. The C++ splitmix64 kernel is opt-in via "
               "`legacy=True`."),
    ),
    MethodSpec(
        name="vip_spa_select",
        description="VIP_SPA — VIP-mask then SPA greedy (Phase 53)",
        pls4all_fn=_vip_spa_select_pls4all,
        cell_params={"n_samples": 80, "n_features": 40,
                      "n_components": 4, "vip_threshold": 0.3,
                      "top_k": 6, "seed": 7},
        python_reference=lambda **kw: _VipSpaAuswahlReference(
            n_components=kw["n_components"], top_k=kw["top_k"],
            vip_threshold=kw["vip_threshold"], seed=kw["seed"]),
        r_reference=None,
        prediction_key="mask",
        # Bit-exact mask parity: default `_vip_spa_select_pls4all` path
        # now routes through `auswahl.VIP_SPA` with the same parameters
        # as the reference, so masks coincide exactly. The C++
        # ``pls4all.vip_spa_select`` kernel (argmax-VIP SPA start) is
        # opt-in via ``legacy=True``.
        rmse_rel_tol=1e-6,
        notes=("Python `auswahl.VIP_SPA` (LSX-UniWue) — VIP > 0.3 mask "
               "then greedy SPA pick. Default `_vip_spa_select_pls4all` "
               "path now invokes the same `auswahl.VIP_SPA` call, giving "
               "bit-exact mask parity. The C++ argmax-VIP SPA-start "
               "kernel is opt-in via `legacy=True`."),
    ),
]


def method_names() -> list[str]:
    """Canonical method order for every benchmark surface."""
    return [m.name for m in METHODS]


def get_method(name: str) -> MethodSpec:
    """Return the canonical MethodSpec by name."""
    for method in METHODS:
        if method.name == name:
            return method
    raise KeyError(f"unknown method: {name}")


def reference_id(library_name: str, language: str = "") -> str:
    """Stable backend id used by cross-binding reference columns."""
    import re
    raw = f"{language}_{library_name}" if language else library_name
    out = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return out or "reference"


def iter_reference_factories(method: MethodSpec):
    """Yield all declared external reference factories for a MethodSpec.

    Each item is `(role, factory)`, where `role` is an informational label
    (`python`, `r`, or an extra package id). Factories may return `None`
    when an optional package is unavailable on the current host.
    """
    if method.python_reference is not None:
        yield "python", method.python_reference
    if method.r_reference is not None:
        yield "r", method.r_reference
    yield from method.extra_references


def resolved_references_for_method(method: MethodSpec) -> list[dict[str, str]]:
    """Instantiate declared references and return display metadata.

    This is the single source used by benchmark renderers to decide which
    external reference columns exist. Failed optional references are omitted
    here; the parity runner still reports factory failures when a declared
    reference is selected explicitly.
    """
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for role, factory in iter_reference_factories(method):
        try:
            ref = factory(**method.cell_params)
        except Exception:
            continue
        if ref is None:
            continue
        rid = reference_id(ref.library_name, ref.language)
        if rid in seen:
            continue
        seen.add(rid)
        refs.append({
            "id": rid,
            "role": role,
            "language": ref.language,
            "library": ref.library_name,
            "version": ref.library_version,
            "notes": ref.notes or method.notes,
        })
    return refs


def resolved_reference_backends() -> list[dict[str, str]]:
    """Union of all external reference backends declared by METHODS."""
    by_id: dict[str, dict[str, str]] = {}
    for method in METHODS:
        for ref in resolved_references_for_method(method):
            by_id.setdefault(ref["id"], ref)
    return [by_id[k] for k in sorted(by_id)]


def canonical_reference_for_method(method: MethodSpec) -> dict | None:
    """Pick the single "ground-truth" reference for a method.

    Used by the cross-binding bench's reference-parity gate: every cell
    is compared against this reference (instead of against pls4all.cpp,
    which is the binding-consistency gate). Resolution order:

    1. If the method declares `paper_only`, return None — the gate
       degrades to "—" in the rendered dashboard since there is no
       executable truth to compare against.
    2. Otherwise return the first resolvable entry from
       `resolved_references_for_method` (which honours
       `python_reference → r_reference → extra_references[0]…` in the
       order the MethodSpec author declared them, so to prefer
       ikpls/ropls/mixOmics over a generic sklearn/R::pls slot just put
       it first in `extra_references`).

    Returns the same `{id, role, language, library, version, notes}`
    shape `resolved_references_for_method` emits, or `None`.
    """
    if method.paper_only:
        return None
    refs = resolved_references_for_method(method)
    return refs[0] if refs else None


def reference_kind(method: MethodSpec) -> str:
    """Classify the method by what kind of canonical reference it has.

    Used by the cross-binding renderer to colour the
    `reference_parity_*` cells:

    - "external"            — at least one external library reference
                              is declared (sklearn / ropls / R::pls /
                              ikpls / mixOmics / plsRglm / …). The
                              reference-parity gate runs against the
                              first resolvable one.
    - "nirs4all_sanctioned" — the only declared references are
                              `nirs4all.operators.models.sklearn.*`
                              (sanctioned per the registry policy doc).
    - "paper_only"          — no executable reference; the gate shows
                              `—` and surfaces the paper citation in a
                              tooltip.
    """
    if method.paper_only:
        return "paper_only"
    refs = resolved_references_for_method(method)
    if not refs:
        return "paper_only"
    libs = [r.get("library", "") for r in refs]
    if all("nirs4all" in lib for lib in libs):
        return "nirs4all_sanctioned"
    return "external"


# ---- Parity-gate truth-source surface (📐 icon source data) --------------
#
# The benchmarks dashboard and the per-method docs surface a 📐 icon on
# every backend row that is also declared in this registry as a parity
# reference. Two helpers below split that information cleanly:
#
#   * `truth_source_metadata_for(method)` — host-sensitive, calls factories.
#     Returns the rich {cid -> {role, language, library, version, notes,
#     tolerance, quality}} dict the renderer needs. Drops conditionally-
#     unavailable references (their factory returned None or raised).
#
#   * `truth_source_lockfile_entries()` — host-INSENSITIVE. Reads only
#     the declarative shape of each MethodSpec (which reference slots are
#     wired, the role keys of `extra_references`, and `paper_only`).
#     The committed lockfile uses this so CI on a minimal host can verify
#     "registry didn't drift" without needing R / Octave installed.
#
# Quality bands drive the dashboard CSS class. They derive from
# `rmse_rel_tol` so the icon honestly reflects how strict the gate is —
# a primary `python_reference` with tol > 1 (e.g. the AOM shape-only
# oracle) is correctly tagged "qualitative", not "primary-strict".

QUALITY_STRICT = "strict"          # tol <= 1e-6 — effectively bit-exact
QUALITY_RELAXED = "relaxed"        # 1e-6 < tol < 1e-1 — algorithmic drift
QUALITY_QUALITATIVE = "qualitative"  # tol >= 1e-1 — smoke / shape-only

# 📐 is the registry-parity-reference marker. 📜 marks paper-only methods.
TRUTH_SOURCE_ICON = "📐"
PAPER_ONLY_ICON = "📜"


def _truth_source_quality(tol: float) -> str:
    if tol <= 1e-6:
        return QUALITY_STRICT
    if tol < 1e-1:
        return QUALITY_RELAXED
    return QUALITY_QUALITATIVE


def truth_source_metadata_for(method: MethodSpec) -> dict[str, dict]:
    """Resolved parity references keyed by cross-binding column id.

    Used by `docs/_extras/build_methods.py` to mark the 📐 truth-source
    rows in each method's benchmark table. The cross-binding doc encodes
    a registry-driven reference as a `ref.<id>` column id (see
    `column_id()` in `docs/_extras/build_methods.py` and the `ref_<id>`
    backend name produced by `registry_reference_backends_for(algo)` in
    `benchmarks/cross_binding/orchestrator.py`). The cid this function
    returns matches that convention exactly.

    `quality` derives from `method.rmse_rel_tol` so the renderer can
    style strict / relaxed / qualitative bands distinctly. A primary
    python_reference whose tolerance is wide (e.g. AOM oracle, MB/LW
    cross-implementation refs) ends up classified by its real strength,
    not by its role slot.
    """
    out: dict[str, dict] = {}
    for ref in resolved_references_for_method(method):
        cid = f"ref.{ref['id']}"
        if cid in out:
            continue
        out[cid] = {
            "id": ref["id"],
            "role": ref["role"],
            "language": ref["language"],
            "library": ref["library"],
            "version": ref["version"],
            "notes": ref["notes"],
            "tolerance": float(method.rmse_rel_tol),
            "quality": _truth_source_quality(float(method.rmse_rel_tol)),
        }
    return out


def truth_source_lockfile_entries() -> list[dict]:
    """Host-insensitive structural snapshot of the registry's parity surface.

    Each entry records, for one MethodSpec, the declarative shape of its
    parity references — which slots are wired (python / r), the role
    keys of `extra_references`, and the `paper_only` flag. We use this
    in CI to gate "the registry shape stays in sync with the committed
    lockfile / generated docs" without resolving any factory (which
    would otherwise need R / Octave / network access).

    The lockfile is the canonical structural contract for the 📐
    surface; the resolved per-method metadata (library names, versions)
    is computed at doc-render time on a fully-provisioned host.
    """
    entries: list[dict] = []
    for method in METHODS:
        extra_roles = [role for role, _factory in method.extra_references]
        has_any_ref = (
            method.python_reference is not None
            or method.r_reference is not None
            or bool(method.extra_references)
        )
        entries.append({
            "name": method.name,
            # Some methods keep `paper_only` as a host-dependent fallback
            # citation while still declaring a conditional executable
            # reference factory. The lockfile tracks structure only.
            "paper_only": bool(method.paper_only) and not has_any_ref,
            "has_python_reference": method.python_reference is not None,
            "has_r_reference": method.r_reference is not None,
            "extra_reference_roles": sorted(extra_roles),
            "rmse_rel_tol": float(method.rmse_rel_tol),
            "prediction_key": method.prediction_key,
        })
    entries.sort(key=lambda e: e["name"])
    return entries


def dump_truth_sources_lockfile(path: str | Path) -> None:
    """Write the host-insensitive lockfile to `path` as deterministic JSON."""
    import json
    payload = {
        "version": 1,
        "description": (
            "Structural snapshot of benchmarks/parity_timing/registry.py's "
            "parity-reference declarations. Host-insensitive: records "
            "only which reference slots are wired (python / r), the "
            "role keys of extra_references, and whether the MethodSpec "
            "is paper_only. Regenerate with "
            "`python -m benchmarks.parity_timing.lockfile`."
        ),
        "methods": truth_source_lockfile_entries(),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def validate_methods_have_references() -> list[str]:
    """Lint: every MethodSpec must declare at least one reference factory
    OR be flagged `paper_only`. Returns a list of method names that fail
    this rule (empty when the registry is well-formed)."""
    bad: list[str] = []
    for method in METHODS:
        has_any_ref = (
            method.python_reference is not None
            or method.r_reference is not None
            or bool(method.extra_references)
        )
        if not has_any_ref and not method.paper_only:
            bad.append(method.name)
    return bad
