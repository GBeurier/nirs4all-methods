# pls4all

> A portable PLS / NIRS engine in C++17 with a stable C ABI and thin
> first-class bindings for **Python, R, MATLAB / Octave**, plus JS /
> WebAssembly, Go, Rust, Julia, Ruby, .NET, Lua, Nim, and Android.

The same numerical core powers every binding: a model trained in Python
predicts byte-for-byte the same way in R, MATLAB, a browser, or on
Android. Cross-checked against scikit-learn, R `pls`, `ropls`,
`mixOmics`, `ikpls`, MATLAB `plsregress`.

---

## Headline — PLS SIMPLS, 1 thread, median ms

Pulled from the
[**cross-binding benchmark matrix**](docs/benchmarks/cross_binding.md)
(generated from the canonical `benchmarks.parity_timing.registry`
method catalog; committed timing snapshots are regenerated, not hand
maintained).

| n × p | pls4all C++ | pls4all Python | sklearn | pls4all R | R `pls` | pls4all MATLAB | Octave `plsregress` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 × 50    | **0.92** | 0.92 | 1.40 | **2.50** | 6.50 | **1.59** | 2.38 |
| 500 × 500   | **39.7** | 38.4 | 40.6 | **201**  | 245  | **63.4** | 65.6 |
| 2500 × 500  | **219**  | 213  | 190  | **1 300**| 1 400| **335**  | 336  |
| 10000 × 500 | **890**  | 896  | 833  | **6 300**| 6 500| **1 400**| 1 600 |

At small data (`n × p ≤ 250k`), pls4all leads externals **1.5 – 8 ×**.
At BLAS-bound sizes everyone converges. Detailed methodology, parity
gates and per-algorithm tables in
[`docs/benchmarks/`](docs/benchmarks/).

---

## Quick start

### Python

```bash
git clone https://github.com/GBeurier/pls4all.git && cd pls4all
cmake --preset blas-omp && cmake --build --preset blas-omp -j
pip install -e bindings/python
```

```python
from pls4all.sklearn import PLSRegression
import numpy as np

X = np.random.randn(500, 200)
y = 2 * X[:, 3] - X[:, 6] + 0.5 * np.random.randn(500)

m = PLSRegression(n_components=5).fit(X, y)
print(m.score(X, y))            # sklearn-compatible
# 67 sklearn-style classes available: SparseSIMPLS, CPPLS, ECRegression,
# RidgePLS, BaggingPLS, AOMPLSRegressor, …
```

### R

```bash
PLS4ALL_INCLUDE_DIR=$PWD/cpp/include \
  PLS4ALL_LIB_DIR=$PWD/build/blas-omp/cpp/src \
  R CMD INSTALL bindings/r/pls4all
```

```r
library(pls4all)

# Base R formula+S3
fit <- pls(y ~ ., data = df, ncomp = 5)
predict(fit, newdata = df)

# Tidymodels engine
spec <- pls_pls4all_reg(num_comp = 5) %>%
  set_engine("pls4all", algorithm = "sparse_simpls", sparsity_lambda = 0.05)

# mlr3 learner
lrn("regr.pls4all", ncomp = 5, algorithm = "cppls", gamma = 0.5)
```

### MATLAB / Octave

```matlab
addpath(genpath('bindings/matlab'))
mdl = pls4all.fit("sparse_simpls", X, y, "NumComponents", 5, "Lambda", 0.05);
predict(mdl, Xnew)
% 18 classdefs available: Regression, SparsePlsRegression, EcrRegression,
% MbPlsRegression, GlmRegression, MirRegression, …
```

---

## Tier 1 + Tier 2 binding coverage

| Binding | Tier 1 surface | Tier 2 idiomatic form |
|---|---|---|
| **Python** (`pls4all.sklearn`) | 64 / 68 ABI methods reachable | **67 sklearn classes + 6 fns** (`BaseEstimator` mixins, `.n4a` pickling, GridSearchCV-ready) |
| **R** (`pls4all` package) | **COMPLETE** — `pls4all_method()` dispatcher: 33 fits + 24 selectors + 4 diagnostics | **3 idioms** — base R formula+S3 (16 wrappers) · parsnip meta-engine · mlr3 R6 learner (each dispatching 16 algos via `algorithm` arg) |
| **MATLAB / Octave** (`+pls4all`) | **COMPLETE** — single MEX dispatcher: 33 fits + 24 selectors + 4 diagnostics | **18 classdefs** + unified `pls4all.fit(algo, X, y, ...)` factory |
| Julia, JS, Go, Rust, Ruby, .NET, Lua, Nim | SIMPLS via native FFI | 1 idiomatic class per language (PoC) |
| JNI / JVM | SIMPLS via JNI | (deferred) |

Cross-binding parity: every binding's SIMPLS predictions match the
shared fixture within `rmse_rel < 1e-12` (bit-exact for several). See
[`docs/parity/`](docs/parity/) and the
[cross-binding benchmark](docs/benchmarks/cross_binding.md).

---

## Documentation

- 🌐 **Sphinx site** : (configurer GitHub Pages — voir
  [`.github/workflows/docs.yml`](.github/workflows/docs.yml))
- 📊 **[Cross-binding benchmarks](docs/benchmarks/cross_binding.md)** — parity + timing matrix generated from the canonical method registry
- 🔬 **[Methodology](docs/benchmarks/methodology.md)** — reference policy, tolerances, threading, hardware
- 🏗️ **[Architecture](docs/architecture/overview.md)** — memory model · error model · threading · serialization
- 📜 **[ABI reference](docs/abi/reference.md)** — 96 `p4a_*` symbols, stability policy, changes log
- 🔌 **Bindings** — [Python](docs/bindings/python.md) · [R](docs/bindings/r.md) · [MATLAB](docs/bindings/matlab.md) · [JS / WASM](docs/bindings/js.md) · [Android](docs/bindings/android.md)
- ✅ **[Parity methodology](docs/parity/methodology.md)** — every algorithm cross-checked against an external library
- 🛠️ **[Dev workflow](docs/dev/workflow.md)** · [build](docs/dev/build.md) · [testing](docs/dev/testing.md)

---

## What's in the box

- **Algorithms** (~60) — PLS regression family (NIPALS, SIMPLS, kernel, wide-kernel, SVD, power, randomized SVD, PCR), OPLS / OPLS-DA, PLSCanonical, PLSSVD, sparse SIMPLS, CPPLS, ECR, MIR-PLS, ridge PLS, robust PLS (Huber IRLS), continuum regression, MB-PLS, LW-PLS, N-PLS, O2-PLS, missing-aware NIPALS, bagging / boosting / random-subspace ensembles, GPR-on-PLS, PLS-GLM (Gaussian / Poisson), PLS-DA / LDA / QDA / logistic / Cox, PDS / DS calibration transfer, **AOM-PLS** & **POP-PLS** (the scientific differentiator).
- **Variable selection** (24) — VIP / coefficient / SR rankers, SPA, CARS, GA-PLS, PSO, VISSA, IRIV, IRF, shaving, BVE, REP, IPW, ST-PLS, T2, WVC, EMCUVE, randomization, biPLS, siPLS, interval, stability, UVE, random frog, SCARS, VIP-SPA.
- **Diagnostics** — Hotelling T² · Q residuals · DModX · process monitoring with alarms · approximate PRESS · one-SE rule.
- **Preprocessing pipeline** — identity · center · autoscale · Pareto · SNV · MSC · EMSC · detrend · SG · ASLS · Norris-Williams · Haar wavelet · OSC · EPO.
- **Validation** — k-fold · LOO · holdout · external-fold · repeated k-fold · Monte-Carlo · Kennard-Stone · SPXY.
- **Metrics** — regression (RMSE/MAE/bias/R²/Q²/slope/intercept/RPD/RPIQ) · binary classification (sensitivity/specificity/balanced accuracy/precision/F1/MCC/AUC) · multiclass (macro/micro AUC) · calibration.

Full list at [`Overview.md`](Overview.md) ; canonical spec at
[`Direction_Technique.md`](Direction_Technique.md).

---

## Build

```bash
git clone https://github.com/GBeurier/pls4all.git && cd pls4all

# Reference build (single-thread, no BLAS — useful for parity)
cmake --preset dev-release
cmake --build --preset dev-release -j
ctest --preset dev-release --output-on-failure

# Production build (BLAS + OpenMP)
cmake --preset blas-omp
cmake --build --preset blas-omp -j

./build/blas-omp/cpp/cli/pls4all_cli --version
./build/blas-omp/cpp/cli/pls4all_cli --abi-info
```

Zero mandatory dependencies — no Eigen, no Boost, no BLAS, no pybind11,
no Rcpp on the install path. Accelerated backends (BLAS / OpenMP /
CUDA) plug in via CMake presets without changing the ABI.

---

## Run the cross-binding benchmark

```bash
# Complete overnight canonical method/reference matrix.
# Skips cells already present in results/full_matrix.csv.
# Env overrides: FORCE=1 RERUN_FAILED=1 THREADS="1 3 10" N_RUNS=5.
benchmarks/cross_binding/run_overnight.sh

# Also commit/push the refreshed web data and trigger GitHub Pages.
PUBLISH_WEB=1 benchmarks/cross_binding/run_overnight.sh

# Render an existing CSV only
python benchmarks/cross_binding/combine_and_render.py \
  --csvs benchmarks/cross_binding/results/full_matrix.csv \
  --out docs/benchmarks/cross_binding.md
```

Full re-run commands in
[`docs/benchmarks/methodology.md`](docs/benchmarks/methodology.md).

---

## License & citation

CeCILL-2.1 ([`LICENSE`](LICENSE), compatible with the GPL family,
recognised by French law).

```bibtex
@software{pls4all,
  author  = {Beurier, Grégory and contributors},
  title   = {pls4all: A portable Partial Least Squares engine with a stable C ABI},
  year    = {2026},
  url     = {https://github.com/GBeurier/pls4all},
  version = {0.94.0}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is provided.

---

## Roadmap

`pls4all` is built in deliberate phases tracked in
[`ROADMAP.md`](ROADMAP.md). Per-phase plans live under
[`roadmap/`](roadmap/) and codex-reviewed implementation logs under
[`docs/reviews/`](docs/reviews/).

| Phase | Status |
|---|---|
| 0 — ABI & build foundation | ✅ shipped |
| 1 — PLS CPU reference (NIPALS PLS1/PLS2, predict, transform, binary I/O) | ✅ shipped |
| 3 — NIRS preprocessing & validation core | ✅ live |
| 4 — Advanced PLS variants (SIMPLS / SVD / PCR / kernel / OPLS / MB-PLS / LW-PLS / …) | ✅ live |
| 5 — Variable selection (24 methods) | ✅ live |
| 6 — AOM-PLS & POP-PLS | ✅ live |
| 2 — Language bindings (real wheels / CRAN / AAR) | 🟡 R + MATLAB COMPLETE · others SIMPLS-PoC |
| 7 — Accelerated backends | 🟡 BLAS + OpenMP live · CUDA planned · batch APIs planned |
| 8 — PyPI / CRAN / npm publication | ⬜ planned (API stabilised first) |

---

`pls4all` builds on lessons learned from the experimental
[`aompls`](https://github.com/GBeurier/aompls) library and on the rich
PLS ecosystem in scikit-learn, R `pls` / `ropls` / `mixOmics` /
`plsVarSel`, and `nirs4all`.
