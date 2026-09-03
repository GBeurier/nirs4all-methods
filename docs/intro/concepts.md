# Core concepts

A 5-minute mental model of pls4all. If you are coming from `sklearn`,
`pls::plsr` or MATLAB `plsregress`, the diagram in
[the overview](overview.md#the-three-layers) is the shortest path —
this page expands the moving parts.

## 1. Context, Config, Model

Every call goes through three handles:

| Handle | Role | Lifetime |
|---|---|---|
| `Context` | RNG seed, thread count, backend (CPU / CUDA), error string | usually one per program |
| `Config`  | Algorithm choice, hyperparameters, centring / scaling flags | one per fit |
| `Model`   | The fitted artifact — coefficients, x_mean, y_mean, scores | one per `(X, y)` pair |

In every binding the three are wrapped idiomatically. In **Python**:

```python
import pls4all

with pls4all.Context() as ctx, pls4all.Config() as cfg:
    ctx.seed = 42
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver    = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.n_components = 5
    model = pls4all.Model.fit(ctx, cfg, X, y)
    y_hat = model.predict(ctx, X_new)
    model.close()
```

In **R**:

```r
fit <- pls(y ~ ., data = df, ncomp = 5, algo = "pls_simpls")
y_hat <- predict(fit, newdata = df_new)
```

In **MATLAB / Octave**:

```matlab
mdl  = n4m.fitrpls(X, y, "NumComponents", 5);
yhat = predict(mdl, Xnew);
```

The C ABI handles `n4m_context_t*`, `n4m_config_t*`, `n4m_model_t*`
are the same objects underneath.

## 2. Two API tiers

Each binding exposes the same surface in two flavours:

- **Tier-1 — raw / canonical.** A 1:1 mapping to the C ABI. Direct,
  minimum overhead, accepts NumPy / R matrix / MATLAB double exactly
  as the C layer wants it. This is what the parity benchmark drives.
- **Tier-2 — idiomatic.** Wraps tier-1 in the host language's expected
  estimator shape (sklearn `BaseEstimator`, R formula+S3 with
  `predict()` / `summary()` / `coef()`, MATLAB classdef with
  `predict` / `loss` / `score`).

Model-based Python estimators persist their fitted native state through the
same C ABI used by tier 1. Their pickle state embeds a raw N4MM payload; this is
not a general tier-2 interchange format or a nirs4all `.n4a` pipeline bundle.

## 3. Five C++ acceleration builds

`libn4m` is built five ways. The build is a property of the dynamic
library you load, not of the API — every binding sees the same
function names regardless.

| Build      | What it enables                          | When to use |
|------------|------------------------------------------|---|
| `ref`      | Scalar reference loops, no BLAS, no OMP  | Parity anchor; debugging |
| `blas`     | OpenBLAS GEMM only                       | Single-thread tight loops |
| `omp`      | OpenMP in kernel loops, no BLAS          | Many small cells |
| `blas+omp` | OpenBLAS + OpenMP (the production combo) | **Default** |
| `cuda`     | cuBLAS GEMM offload                      | Very large `n × p` |

The build axis appears explicitly in the
[benchmark dashboard](../landing/dashboard.md) under `pls4all.cpp.<suffix>`.

## 4. Determinism and reference policy

Every algorithm has a *parity reference* — the external library whose
implementation pls4all reproduces within the method's numerical
tolerance:

| Algorithm family | Reference library | Language |
|---|---|---|
| Plain PLS / PCR  | `sklearn.cross_decomposition.PLSRegression`, `pls::plsr` | Python / R |
| OPLS             | `ropls::opls` | R |
| Sparse PLS       | `spls::spls` | R |
| PLS-DA / sPLS-DA | `mixOmics`   | R |
| Kernel PLS       | `kernlab` + paper port | R / paper |
| AOM-PLS, POP-PLS | `nirs4all.operators.models.sklearn.aom_pls` from the git-pinned `nirs4all` dependency (oracle, sanctioned) | Python |
| MB-PLS, LW-PLS   | `nirs4all.operators.models.sklearn.*` (sanctioned paper port) | Python |
| Calibration      | `prospectr`, `chemometrics::stdize` | R |
| Selectors        | `plsVarSel`, `enpls`, `auswahl` | R / Python |

The reference is declared per-method in
`benchmarks/parity_timing/registry.py`. The orchestrator generates
predictions on the *same* CSV bytes for every backend in a cell so
that R's `set.seed`, NumPy's RNG and Octave's `randn("state",…)` do
not introduce cross-language drift.

Tolerances are per-algorithm; see
[benchmark methodology](../benchmarks/methodology.md#parity-tolerance).

## 5. The N4MM fitted-model payload

N4MM format version 1 stores the state needed to predict or transform with a
fitted native model without a pipeline:

- algorithm, solver, deflation and fitted-model configuration fields;
- centring and scaling statistics;
- coefficients, latent matrices and optional training scores;
- the writer ABI triple as provenance;
- a trailing FNV-1a integrity checksum.

Format version 2 is emitted only for the bounded native
`SNV -> Savitzky-Golay smooth -> PLS regression` slice. It retains the complete
format-1 model state and adds the ordered operator tags plus a versioned
semantic profile: row-wise SNV with mean/std and `ddof=0`, followed by SG with
explicit window/polynomial degree, derivative `0`, delta `1`, interpolation at
the edges and cval `+0`. The importer supports both versions; a restored v2
model reapplies the pipeline to raw X before prediction.

It does **not** contain the project version, a training-data fingerprint or a
content address. A different writer ABI is currently accepted with a warning
on the context; corrupt bytes and unsupported N4MM format versions fail.

`n4m_serialization_inspect_model_v1` reports the encoded recipe, dimensions and
prediction, transform and pipeline capability bits. The `_v1` suffix identifies
the fixed descriptor schema; it can inspect both N4MM wire versions.
ABI 2.5 adds `n4m_serialization_inspect_pipeline_v1`: format 1 reports no
pipeline, while format 2 attests the exact operator order, canonical SG
parameters, raw/model widths and stable canonical-plan fingerprint. Rust,
Python and JavaScript expose this as optional pipeline metadata rather than
inferring a plan from the capability bit.

The format is exposed by the C ABI functions
`n4m_model_export_to_buffer` / `n4m_model_import_from_buffer`. The Python slim
binding wraps them as:

```python
payload = model.to_bytes()
restored = pls4all.Model.from_bytes(ctx, payload)
restored.close()
```

R, MATLAB and JS do not currently expose fitted-model N4MM import/export
wrappers. The raw payload has no canonical filename extension. The `.n4a`
extension remains the nirs4all full-pipeline bundle namespace; a possible
`.n4am` envelope is deferred and is not part of either N4MM wire version.

## 6. The algorithm taxonomy

Algorithms are grouped for the dashboard and the
[methods index](../methods/index.md):

| Group | Representative methods |
|---|---|
| **Core PLS**            | `pls`, `cppls`, `pcr`, `opls` |
| **Sparse**              | `sparse_simpls`, `fused_sparse_pls`, `sparse_pls_da`, `group_sparse_pls` |
| **Ensemble**            | `bagging_pls`, `boosting_pls`, `random_subspace_pls` |
| **Robust / weighted**   | `robust_pls`, `weighted_pls` |
| **Nonlinear / local**   | `kernel_pls_rbf`, `lw_pls`, `gpr_pls`, `continuum_regression` |
| **Multi-block**         | `mb_pls`, `mir_pls`, `so_pls`, `on_pls`, `o2pls`, `rosa`, `n_pls` |
| **Calibration transfer**| `ds`, `pds`, `ecr` |
| **Classification**      | `pls_lda`, `pls_logistic`, `pls_qda`, `pls_glm`, `pls_cox`, `sparse_pls_da` |
| **Missing data**        | `missing_aware_nipals` |
| **Regularised**         | `ridge_pls` |
| **Selectors (28)**      | VIP, coefficient, selectivity ratio, SPA, stability, UVE, CARS, random-frog, SCARS, GA, PSO, VISSA, shaving, BVE, REP, IPW, ST, interval, BiPLS, SiPLS, T², WVC, WVC-threshold, EMCUVE, randomization, IRIV, IRF, VIP-SPA |
| **Diagnostics**         | T², Q, DModX, PRESS approx, one-SE rule, monitoring, AOM bank |

Each method has a dedicated page documenting its parameters,
bibliographic source, math principle, every binding's signature and
its parity + timing rows.
