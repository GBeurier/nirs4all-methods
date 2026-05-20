# TransferMetricsComputer — 9-metric dataset alignment vector

Phase 20 introduces a single utility function that quantifies how well a *source* and a *target* dataset align in a shared low-dimensional space — useful for transfer-learning workflows where preprocessing aims to bring the two distributions closer before refitting a downstream model.

The utility mirrors `nirs4all.analysis.transfer_metrics.TransferMetricsComputer` (Python). One C ABI symbol — `n4m_transfer_metrics_compute` — returns a 9-field POD struct.

## Pipeline

1. **Per-dataset centering** — each matrix is shifted to have zero per-column mean.
2. **PCA** — the centered covariance `X_c^T X_c` is diagonalised via a cyclic Jacobi sweep (robust for any p; converges cubically once off-diagonals fall below `1e-13 · ||C||_F`). The top `n_components` eigenpairs are kept, then truncated to `r_use = min(r_src, r_tgt)` so both datasets occupy the same rank.
3. **Sign convention** — sklearn's deterministic `svd_flip(u_based_decision=False)`: each loading column's max-absolute entry is non-negative. Makes the output bitwise reproducible across runs and matches the Python reference.
4. **Metrics** — one struct field per metric:

| Field | Formula | Domain |
|---|---|---|
| `centroid_distance`    | \|\|mean(Z_s) − mean(Z_t)\|\|_2                             | `[0, ∞)` |
| `cka_similarity`       | \|\|C_s C_t\|\|_F² / (\|\|C_s²\|\|_F · \|\|C_t²\|\|_F)        | `[0, 1]` typically |
| `grassmann_distance`   | √(Σ θ_i²), θ = arccos(σ_i(U_s^T U_t))                       | `[0, π/2 · √r]` |
| `rv_coefficient`       | trace(C_s C_t) / √(trace(C_s²) · trace(C_t²))                | `[0, 1]` typically |
| `procrustes_disparity` | 1 − (Σ σ_i)², SVD of standardised pair (first 2 components) | `[0, 1]` |
| `trustworthiness`      | sklearn formula, `k = min(k_neighbors, n − 2)`                | `[0, 1]` |
| `spread_distance`      | \|\|cov(Z_s) − cov(Z_t)\|\|_F + ½·(mean of min pairwise dists) | `[0, ∞)` |
| `evr_source`           | Σλ_kept / Σλ_all                                              | `[0, 1]` |
| `evr_target`           | Σλ_kept / Σλ_all                                              | `[0, 1]` |

`C` here means the `r×r` covariance matrix of the score blocks `Z` (with `ddof=1`). `λ` are the full eigenvalue spectrum of `X_c^T X_c`.

`grassmann_distance` is set to NaN when the two datasets have different feature counts (`p_src != p_tgt`) — principal angles are only defined in a common ambient space. The Python reference and the C engine produce the same NaN in that case.

## Determinism + RNG choice

The `spread_distance` subsampling step uses a deterministic Fisher-Yates permutation driven by SplitMix64 keyed on `seed`. This is **not** bit-identical to `numpy.random.RandomState(seed).choice` — we mirror SplitMix64 instead because:

1. The C ABI already exposes PCG64 (Phase 1); replicating NumPy's legacy Mersenne-Twister stream would add a maintenance liability without parity benefit.
2. The internal parity fixture under `parity/python_generator/src/n4m_parity_transfer_ref/` uses the same SplitMix64 helper, so the parity fixtures encode the SplitMix64 behaviour rather than NumPy's MT19937.
3. The downstream metric (a mean of two min-distance scalars over a 100-sample subsample) is statistically robust to the choice of seed regardless.

## Parameters

`n4m_transfer_metrics_compute` takes three scalars + a seed:

| Argument | Meaning | Constraint |
|---|---|---|
| `n_components` | PCA truncation rank | `>= 1`. Effective rank = `min(n_components, p_src, p_tgt, n_src, n_tgt, rank(X_s), rank(X_t))`. |
| `k_neighbors`  | Trustworthiness neighbour count | `>= 2`. Internally capped to `min(n_src, n_tgt) - 2`. |
| `seed`         | Spread-distance subsample seed | Any `uint64`. |

The two matrix views must be row-major contiguous F64. The wrapper returns:

- `N4M_OK` on success (every field of `*out` set).
- `N4M_ERR_NULL_POINTER` if `out` is NULL or a view has NULL data with `rows*cols > 0`.
- `N4M_ERR_INVALID_ARGUMENT` if either dataset has fewer than 2 rows or a scalar fails the constraint above.
- `N4M_ERR_DTYPE_MISMATCH` / `N4M_ERR_STRIDE_INVALID` on non-conforming views.
- `N4M_ERR_OUT_OF_MEMORY` on scratch allocation failure.
- `N4M_ERR_NUMERICAL_FAILURE` if the Jacobi sweep does not converge in the budgeted 100 sweeps (extremely unlikely on well-scaled NIR data; conditions producing this are typically a sign of an upstream bug).

## ABI

```c
typedef struct n4m_transfer_metrics_t {
    double centroid_distance;
    double cka_similarity;
    double grassmann_distance;
    double rv_coefficient;
    double procrustes_disparity;
    double trustworthiness;
    double spread_distance;
    double evr_source;
    double evr_target;
} n4m_transfer_metrics_t;

N4M_API n4m_status_t n4m_transfer_metrics_compute(
    n4m_matrix_view_t X_source,
    n4m_matrix_view_t X_target,
    int32_t n_components,
    int32_t k_neighbors,
    uint64_t seed,
    n4m_transfer_metrics_t* out);
```

One ABI symbol total. Naming follows the `n4m_transfer_*` prefix reserved in `n4m.h §5` for Phase 20.

## Parity tolerance

Iterative Jacobi diagonalisation combined with multiple matrix compositions (CKA / RV / Procrustes / Grassmann) places a hard floor on the bit-exact-with-NumPy parity bar. The Phase 20 contract is:

- **Absolute tolerance**: `1e-9`
- **Relative tolerance**: `1e-10`

These are consistent with the LDLT-iterative baseline operators (Phase 5) and reflect the deepest numerical pipeline in the nirs4all-methods C ABI so far.

## References

- nirs4all 0.8.11 — `nirs4all/analysis/transfer_metrics.py` (canonical Python implementation).
- Lin et al. 2019 — Centred Kernel Alignment for representation similarity.
- Robert & Escoufier 1976 — RV coefficient.
- Wong et al. 1967 — Grassmann distance / principal angles.
- scipy.spatial.procrustes documentation — Procrustes superposition disparity formula.
- Venna & Kaski 2001 — Trustworthiness formulation (also in `sklearn.manifold.trustworthiness`).

## Files

- `cpp/src/core/utilities/transfer_metrics.{c,h}` — reference C engine (Jacobi PCA + metric kernels).
- `cpp/src/c_api/c_api_transfer_metrics.cpp` — extern "C" wrapper.
- `cpp/src/c_api/c_api_transfer_metrics_decl.h` — worktree-local declaration (moves to `n4m.h §13` at merge time).
- `cpp/tests/test_transfer_metrics.cpp` — smoke + invalid-args + parity tests.
- `parity/python_generator/src/n4m_parity_transfer_ref/transfer_metrics.py` — internal parity fixture.
- `parity/python_generator/scripts/generate_phase20_fixtures.py` — fixture generator.
- `parity/fixtures/transfer_metrics_v1.json` — 5 parity cases.
