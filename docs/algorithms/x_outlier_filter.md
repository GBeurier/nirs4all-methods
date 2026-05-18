# XOutlierFilter — six-method spectral outlier detector

## Overview

`XOutlierFilter` identifies rows of a feature matrix `X` whose spectra are
statistical outliers and returns a boolean keep mask. Six detection
methods are supported (Phase 13):

| Method | Algorithm | Threshold (default) |
|--------|-----------|---------------------|
| `mahalanobis` | Empirical covariance, `(x - mu)^T cov^{-1} (x - mu)` | `sqrt(chi2_inv(0.975, p))` |
| `robust_mahalanobis` | MinCovDet (simplified FAST-MCD) covariance | Same chi-squared threshold |
| `pca_residual` | Sum-squared residual of PCA reconstruction (Q-statistic) | 95th percentile of training Q |
| `pca_leverage` | Hotelling's T² in PCA score space | 95th percentile of training T² |
| `isolation_forest` | Tree-ensemble isolation depth (vendored mini-IF) | Contamination quantile of training scores |
| `lof` | Local Outlier Factor with `k = min(20, n-1)` | Contamination quantile of training scores |

All six methods reuse the shared `c4a_filter_stats_t` struct and the
"fit + apply" pattern that the HighLeverage filter introduced in Phase 14.

## C ABI surface

```c
typedef struct c4a_filter_x_outlier_handle_t c4a_filter_x_outlier_handle_t;

c4a_status_t c4a_filter_x_outlier_create(
    c4a_filter_x_outlier_handle_t** out,
    int32_t  method,                // 0..5
    int      use_threshold,         // 0 = method default, 1 = override
    double   threshold,
    int32_t  n_components,          // <= 0 selects default cap
    double   contamination,         // (0, 0.5]
    uint64_t seed,                  // PCG64 seed (iforest / robust_mahalanobis)
    int32_t  n_estimators,          // iforest tree count (default 100)
    int64_t  max_samples);          // iforest sub-sample size (default 256)
void         c4a_filter_x_outlier_destroy(c4a_filter_x_outlier_handle_t*);
c4a_status_t c4a_filter_x_outlier_fit(
    c4a_filter_x_outlier_handle_t* h, c4a_matrix_view_t X);
c4a_status_t c4a_filter_x_outlier_is_fitted(
    const c4a_filter_x_outlier_handle_t* h, int* out_fitted);
c4a_status_t c4a_filter_x_outlier_apply(
    const c4a_filter_x_outlier_handle_t* h,
    c4a_matrix_view_t X,
    uint8_t* mask_out, c4a_filter_stats_t* stats_out);
```

`mask_out[i] == 1` keeps sample `i`; `0` excludes it. The mask buffer is
caller-owned and must be at least `X.rows` bytes long.

## Engine layout

```
cpp/src/core/filters/x_outlier.{c,h}        # 6-method orchestrator
cpp/src/core/filters/_vendored/              # algorithm-specific implementations
   isolation_forest.{c,h}                    # ~400 LOC PCG64-seeded isolation tree ensemble
   lof.{c,h}                                 # ~250 LOC brute-force k-NN LOF
   min_cov_det.{c,h}                         # ~300 LOC simplified FAST-MCD
cpp/src/core/common/pca_helper.{c,h}        # shared PCA wrapper over Jacobi SVD
```

Shared dependencies:

- **`core/common/svd.c`** — Jacobi SVD used by `pca_helper`.
- **`core/common/linalg.c`** — Householder QR used to invert the Mahalanobis
  covariance and the MinCovDet precision matrix.
- **`core/common/rng_pcg64.c`** — PCG64 RNG seeded once per ensemble.

## Method details

### `mahalanobis` (1e-10 abs / 1e-11 rel parity)

Builds the empirical covariance from the training matrix (centred on the
column means), Tikhonov-regularises (`reg = 1e-10 * trace / p`), factors
in place with Householder QR, then computes per-row Mahalanobis distance
on apply. Threshold = `sqrt(chi2_inv(0.975, p))` (Wilson-Hilferty
approximation; matches scipy.stats.chi2 within ~1e-3 over `p in [1, 30]`).

Optional PCA reduction: when `cols > min(50, max(rows/3, 2))` the engine
fits a shared PCA on the input first (top-k components), then the
Mahalanobis covariance lives in score space.

### `robust_mahalanobis` (count-only parity, see DEFERRALS)

Replaces the empirical covariance with the FAST-MCD output of the
vendored `min_cov_det.c`. The simplified implementation runs a single
seeded random sub-sample (`h = ceil((n + p + 1) / 2)` rows) followed by
30 concentration steps; sklearn's full ensemble of `n_trials = 500`
starts is deferred to a follow-up phase.

### `pca_residual` (1e-10 abs / 1e-11 rel parity)

Top-k PCA reconstruction `X_hat = scores @ components + mean`; per-row
Q-statistic = `sum (x - x_hat)^2`. Threshold = 95th percentile of training
Q.

### `pca_leverage` (1e-10 abs / 1e-11 rel parity)

Hotelling T² in PCA score space: per-row
`T2 = sum_i (score_i^2 / explained_var_i)`. Threshold = 95th percentile
of training T².

### `isolation_forest` (count-only parity, see DEFERRALS)

Vendored ensemble of `n_estimators` trees, each built on a random
sub-sample of `max_samples` rows. Each split picks a random feature and
a random split value in `[min, max]` of the current partition. Path
length is averaged across trees, normalised by `c(max_samples)`.
Threshold = contamination quantile of training mean-path lengths.

sklearn's `IsolationForest` uses `np.random.RandomState`; c4a uses
PCG64, so masks differ by 5-8 samples per fixture even on identical
seeds (the bit-stream differs). The mask shape and contamination rate
match within tolerance.

### `lof` (count-only parity with 3-sample tolerance)

Brute-force k-NN with `k = min(20, n - 1)`. Local reachability density
`lrd_i = 1 / mean reach_dist(i, j)`; `LOF_i = mean lrd_j / lrd_i`.
Threshold = `(1 - contamination)` quantile of training LOF.

`LOF` is fully deterministic; the small parity tolerance comes from the
percentile interpolation difference between NumPy and the c4a `linear`
implementation at the contamination boundary.

## Parity tolerances summary

| Method | Tolerance | Notes |
|--------|-----------|-------|
| `mahalanobis` | 1e-10 abs / 1e-11 rel | Bit-equivalent mask |
| `robust_mahalanobis` | 8 mask flips | Simplified FAST-MCD (single start) |
| `pca_residual` | 1e-10 abs / 1e-11 rel | Bit-equivalent mask |
| `pca_leverage` | 1e-10 abs / 1e-11 rel | Bit-equivalent mask |
| `isolation_forest` | 8 mask flips | PCG64 vs RandomState divergence |
| `lof` | 3 mask flips | Percentile interpolation edge |

## DEFERRALS

* **Full FAST-MCD ensemble.** Sklearn runs `n_trials = 500` random
  starts and keeps the run with the smallest covariance determinant; the
  simplified implementation uses a single start. Robustness is
  asymptotically equivalent but the per-run mask can flip by O(8)
  samples. Tracked for a follow-up phase that adds the ensemble loop.
* **Bit-exact sklearn IsolationForest.** The c4a implementation uses
  PCG64 with the same algorithm; sklearn uses NumPy RandomState. Bit
  equivalence would require re-seeding through RandomState's tree-style
  hash, which conflicts with the project-wide PCG64 mandate. Counts
  match within tolerance.
* **kd-tree for LOF.** Brute-force k-NN is O(n^2 d) — acceptable for
  NIRS-typical `n < 1000`; a kd-tree backend lands in Phase 22+ alongside
  the splitter optimisations.

## Reference

The frozen Python reference lives at
`parity/python_generator/src/c4a_parity_filters_ref/x_outlier.py`. It
imports sklearn's `EmpiricalCovariance`, `MinCovDet`, `PCA`,
`IsolationForest`, and `LocalOutlierFactor` directly and is validated
once against nirs4all 0.8.x — subsequent nirs4all releases can drift
without breaking the parity gate.
