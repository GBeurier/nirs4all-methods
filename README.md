# pls4all

> A portable PLS / NIRS engine in C++17 with a stable C ABI and thin
> first-class bindings for **Python, R, MATLAB / Octave**, plus JS /
> WebAssembly, Go, Rust, Julia, Ruby, .NET, Lua, Nim, and Android.

The same numerical core powers every binding: a model trained in Python,
R, MATLAB, a browser, or Android is checked against the same C++ parity
contract and tolerance policy. Cross-checked against scikit-learn, R
`pls`, `ropls`, `mixOmics`, `ikpls`, MATLAB `plsregress`.

---

## Headline — 100 × 50, 1 thread, median ms

Pulled from the
[**cross-binding benchmark matrix**](docs/benchmarks/cross_binding.md)
(generated from the canonical `benchmarks.parity_timing.registry`
method catalog; committed timing snapshots are regenerated, not hand
maintained).

| Method | pls4all C++ | pls4all Python | pls4all R | R formula | R `pls` compat | R `mdatools` compat | pls4all MATLAB | Reference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PLS SIMPLS | **1.03 ms** | 1.11 ms | 2.50 ms | 6.00 ms | 6.63 ms | 4.17 ms | 1.58 ms | sklearn 1.33 ms |
| AOM-PLS | 2.64 ms | **2.60 ms** | 3.66 ms | 4.92 ms | 5.13 ms | 4.86 ms | 2.78 ms | nirs4all 10.8 ms |
| POP-PLS | 3.05 ms | **3.03 ms** | 3.78 ms | 4.90 ms | 4.97 ms | 4.92 ms | 3.08 ms | nirs4all 26.1 ms |

PLS external baselines in the same snapshot: `ikpls` 1.08 ms, R `pls`
5.22 ms, R `mixOmics` 7.77 ms, Octave `plsregress` 2.18 ms. Detailed methodology, parity
gates and per-algorithm tables are in
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
# 68 sklearn-style classes available: SparseSIMPLS, CPPLS, ECRegression,
# RidgePLS, BaggingPLS, GPRPLSRegression, …
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

# pls package style
fit <- plsr(y ~ ., data = df, ncomp = 10, method = "simpls", validation = "CV")
RMSEP(fit)
selectNcomp(fit)

# mdatools style
fit <- pls(X, y, ncomp = 10, center = TRUE, scale = FALSE, cv = 10)
predict(fit, x = Xnew)
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
| **Python** (`pls4all.sklearn`) | Registry-driven tier-1 API + AOM/POP low-level ABI | **68 sklearn classes + 8 fns** (`BaseEstimator` mixins, `.n4a` pickling, GridSearchCV-ready) |
| **R** (`pls4all` package) | **COMPLETE** — 73 registry methods via `pls4all_method()` and wrappers | NIRS-first idioms — base R formula+S3 (16 wrappers) · `pls`-compatible `plsr()` / `pcr()` · `mdatools`-compatible matrix `pls(x, y, ...)` |
| **MATLAB / Octave** (`+pls4all`) | **COMPLETE** — 73 registry methods via the single MEX dispatcher | **18 classdefs** + unified `pls4all.fit(algo, X, y, ...)` factory |
| Julia, JS, Go, Rust, Ruby, .NET, Lua, Nim | SIMPLS via native FFI | 1 idiomatic class per language (PoC) |
| JNI / JVM | SIMPLS via JNI | (deferred) |

Cross-binding parity: every binding's SIMPLS predictions match the
shared fixture within `rmse_rel < 1e-12` for the committed smoke matrix. See
[`docs/parity/`](docs/parity/) and the
[cross-binding benchmark](docs/benchmarks/cross_binding.md).

---

## Method availability map

This table condenses the cross-binding benchmark matrix into a "where do I
find this method?" view. `Python`, `R` and `MATLAB` list external reference
libraries observed in the benchmark; `nirs4all` is the sanctioned in-tree
Python reference provider for methods that do not have a packaged external
implementation.

| Method | Backend C++ | pls4all R | pls4all MATLAB | Python | R | MATLAB |
|---|---:|---|---|---|---|---|
| `pls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | `scikit-learn`<br>`ikpls` | `pls`<br>`mixOmics` | `Octave plsregress` |
| `pcr` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | `scikit-learn` | `pls` | — |
| `opls` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `ropls` | — |
| `sparse_simpls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | — | `mixOmics`<br>`spls` | — |
| `di_pls` | ✓ | — | — | `diPLSlib` | — | — |
| `recursive_pls` | ✓ | `dispatcher` | `MEX/dispatcher`<br>`classdef/factory` | `scikit-learn` | `pls` | — |
| `cppls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | — | `pls` | — |
| `weighted_pls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | `scikit-learn` | — | — |
| `robust_pls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | — | `chemometrics` | — |
| `ridge_pls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | `scikit-learn` | — | — |
| `continuum_regression` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | — | `JICO` | — |
| `n_pls` | ✓ | `dispatcher` | `MEX/dispatcher`<br>`classdef/factory` | `tensorly` | — | — |
| `kernel_pls_rbf` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `kernlab+pls` | — |
| `o2pls` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `OmicsPLS` | — |
| `approximate_press` | ✓ | — | — | — | `pls` | — |
| `pls_diagnostic_t2` | ✓ | — | — | — | `mdatools` | — |
| `pls_diagnostic_q` | ✓ | — | — | — | `mdatools` | — |
| `pls_monitoring` | ✓ | — | — | — | `mdatools` | — |
| `one_se_rule` | ✓ | — | — | — | `pls` | — |
| `so_pls` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `multiblock` | — |
| `on_pls` | ✓ | — | — | `OnPLS` | — | — |
| `rosa` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `multiblock` | — |
| `vissa_select` | ✓ | `dispatcher` | `MEX/dispatcher` | `auswahl` | — | — |
| `pso_select` | ✓ | `dispatcher` | `MEX/dispatcher` | `pyswarms` | — | — |
| `gpr_pls` | ✓ | `dispatcher` | `MEX/dispatcher` | `scikit-learn` | — | — |
| `bagging_pls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | `scikit-learn` | — | — |
| `boosting_pls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | — | `mboost` | — |
| `random_subspace_pls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher` | `scikit-learn` | — | — |
| `pls_glm` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsRglm` | — |
| `pls_qda` | ✓ | `dispatcher` | `MEX/dispatcher` | `scikit-learn` | — | — |
| `pls_cox` | ✓ | — | — | — | `plsRcox` | — |
| `pds` | ✓ | — | — | — | `base` | — |
| `ds` | ✓ | — | — | — | `base` | — |
| `mir_pls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | — | — | — |
| `missing_aware_nipals` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | — | `softImpute` | — |
| `sparse_pls_da` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `mixOmics`<br>`spls` | — |
| `group_sparse_pls` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `sgPLS` | — |
| `fused_sparse_pls` | ✓ | `dispatcher` | `MEX/dispatcher` | — | — | — |
| `pls_diagnostic_dmodx` | ✓ | — | — | — | `mdatools` | — |
| `mb_pls` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | `nirs4all` | — | — |
| `lw_pls` | ✓ | `dispatcher` | `MEX/dispatcher` | `nirs4all` | — | — |
| `pls_lda` | ✓ | `dispatcher` | `MEX/dispatcher` | `scikit-learn` | — | — |
| `pls_logistic` | ✓ | `dispatcher` | `MEX/dispatcher` | `scikit-learn` | — | — |
| `aom_preprocess` | ✓ | `dispatcher` | `MEX/dispatcher` | `nirs4all` | — | — |
| `aom_pls` / `aompls` | ✓ | `dispatcher` | `MEX/dispatcher` | `nirs4all` `AOMPLSRegressor` | — | — |
| `pop_pls` / `poppls` | ✓ | `dispatcher` | `MEX/dispatcher` | `nirs4all` `POPPLSRegressor` | — | — |
| `variable_select_vip` | ✓ | — | — | — | `plsVarSel` | — |
| `variable_select_coef` | ✓ | — | — | — | `pls` | — |
| `variable_select_sr` | ✓ | — | — | — | `plsVarSel` | — |
| `interval_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `mdatools` | — |
| `bipls_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `mdatools` | — |
| `sipls_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | — | — |
| `stability_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `uve_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `spa_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `cars_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `enpls` | — |
| `random_frog_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | — | `libPLS` |
| `scars_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | — | — |
| `ga_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `shaving_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `bve_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `t2_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `wvc_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `wvc_threshold_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `emcuve_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `randomization_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `pls+stats` | — |
| `rep_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `ipw_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `st_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | `plsVarSel` | — |
| `ecr` | ✓ | `dispatcher`<br>`formula/S3` | `MEX/dispatcher`<br>`classdef/factory` | — | — | `libPLS` |
| `iriv_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | — | `libPLS` |
| `irf_select` | ✓ | `dispatcher` | `MEX/dispatcher` | — | — | `libPLS` |
| `vip_spa_select` | ✓ | `dispatcher` | `MEX/dispatcher` | `auswahl` | — | — |

---

## Documentation

- 🌐 **Sphinx site** : (configurer GitHub Pages — voir
  [`.github/workflows/docs.yml`](.github/workflows/docs.yml))
- 📊 **[Cross-binding benchmarks](docs/benchmarks/cross_binding.md)** — parity + timing matrix generated from the canonical method registry
- 🧪 **[Cross-binding runbook](benchmarks/cross_binding/README.md)** — registry refs, clean sweeps, git-pinned refs, and legacy fixed-reference modes
- 🔬 **[Methodology](docs/benchmarks/methodology.md)** — reference policy, tolerances, threading, hardware
- 🏗️ **[Architecture](docs/architecture/overview.md)** — memory model · error model · threading · serialization
- 📜 **[ABI reference](docs/abi/reference.md)** — 96 `p4a_*` symbols, stability policy, changes log
- 🔌 **Bindings** — [Python](docs/bindings/python.md) · [R](docs/bindings/r.md) · [MATLAB](docs/bindings/matlab.md) · [JS / WASM](docs/bindings/js.md) · [Android](docs/bindings/android.md)
- ✅ **[Parity methodology](docs/parity/methodology.md)** — every algorithm cross-checked against an external library
- 🛠️ **[Dev workflow](docs/dev/workflow.md)** · [build](docs/dev/build.md) · [testing](docs/dev/testing.md)

---

## What's in the box

- **Methods** (73) — PLS regression family (NIPALS, SIMPLS, kernel, wide-kernel, SVD, power, randomized SVD, PCR), OPLS / OPLS-DA, PLSCanonical, PLSSVD, sparse SIMPLS, CPPLS, ECR, MIR-PLS, ridge PLS, robust PLS (Huber IRLS), continuum regression, MB-PLS, LW-PLS, N-PLS, O2-PLS, missing-aware NIPALS, bagging / boosting / random-subspace ensembles, GPR-on-PLS, PLS-GLM (Gaussian / Poisson), PLS-DA / LDA / QDA / logistic / Cox, PDS / DS calibration transfer, **AOM-PLS** & **POP-PLS** (the scientific differentiator).
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
# Install git-only external refs not released on PyPI yet.
python -m pip install -r benchmarks/cross_binding/requirements-git.txt

# Complete overnight canonical method/reference matrix.
# Default: registry cells + registry-declared external references only.
# Skips cells already present in results/full_matrix.csv.
# Env overrides: FORCE=1 RERUN_FAILED=1 THREADS="1 3 10" N_RUNS=40.
# N_RUNS is the adaptive total-run cap, including the warmstart.
benchmarks/cross_binding/run_overnight.sh

# Exhaustive stress matrix: all pls4all bindings/tiers, all CPU libp4a
# backend variants, the default 11-size dataset sweep, threads 1/3/10,
# and the registry-declared external references.
FULL_MATRIX=1 benchmarks/cross_binding/run_overnight.sh

# Legacy fixed-reference cross-product too. This is useful for auditing
# external-library coverage, and may emit NOT_IMPLEMENTED for algorithms
# that sklearn / ikpls / ropls / mixOmics / plsregress do not support.
FULL_MATRIX=1 REFERENCE_BACKENDS=all benchmarks/cross_binding/run_overnight.sh

# Include the CUDA build too when CUDA is available.
FULL_MATRIX=1 LIBP4A_BUILD=all benchmarks/cross_binding/run_overnight.sh

# On the Pages branch (main), commit/push web data and trigger GitHub Pages.
PUBLISH_WEB=1 benchmarks/cross_binding/run_overnight.sh

# Exhaustive run, then publish the refreshed dashboard from main.
FULL_MATRIX=1 PUBLISH_WEB=1 benchmarks/cross_binding/run_overnight.sh

# From a work branch, publish the source files without deploying live Pages.
PUBLISH_WEB=1 DEPLOY_PAGES=0 benchmarks/cross_binding/run_overnight.sh

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
  version = {0.97.3}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is provided.

---

## Roadmap

`pls4all` keeps the detailed historical phase log in
[`ROADMAP.md`](ROADMAP.md). The active roadmap is now focused on making the
same C ABI catalog available through more language bindings.

| Target binding | Goal |
|---|---|
| JS / WebAssembly | npm + browser package over the full method catalog |
| Julia | native package wrapper over the stable C ABI |
| JVM / Android | JNI / AAR binding beyond the current SIMPLS smoke |
| Rust / Go | typed bindings promoted from SIMPLS PoC to broader catalog coverage |
| .NET / Ruby / Lua / Nim | secondary bindings once the tier-2 catalog is stable |
| Swift | future mobile / desktop binding target |

### Method requests

Users can request a missing method through
[GitHub Issues](https://github.com/GBeurier/pls4all/issues). Please include
the reference implementation or library, the paper / DOI, expected inputs and
outputs, and any small reproducible dataset that can become a parity fixture.

---

`pls4all` builds on lessons learned from the experimental
[`aompls`](https://github.com/GBeurier/aompls) library and on the rich
PLS ecosystem in scikit-learn, R `pls` / `ropls` / `mixOmics` /
`plsVarSel`, and `nirs4all`.
