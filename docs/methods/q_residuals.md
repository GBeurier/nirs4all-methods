# `q_residuals` — Q-residuals

_Group_: **Diagnostic** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/q_residuals.md`](../algorithms/q_residuals.md)

## Description

Per-sample residual energy outside the first `n_components` PCA
directions. Complementary to Hotelling T²: T² measures variation inside
the model, Q-residuals measure variation outside.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- Jackson, J. E., Mudholkar, G. S. (1979). "Control procedures for residuals
  associated with principal component analysis."
- Wichura, M. J. (1988). "Algorithm AS 241: The percentage points of the
  normal distribution." Applied Statistics 37, 477-484.
- Fixture: `parity/fixtures/q_residuals_v1.json`.

### Mathematical principle

For a `(n_samples × n_features)` matrix `X`:

1. Centre `X` by column means: `Xc = X - mean(X, axis=0)`.
2. Compact SVD: `Xc = U Σ V^T`.
3. Reconstruct using only the first `k = n_components` directions:
   `Xc_hat[i, :] = Σ_{j<k} U[i, j] · σ_j · V[:, j]^T`.
4. Per-sample residual:
   `Q[i] = Σ_d (Xc[i, d] − Xc_hat[i, d])²`.

### Implementation

- Same Jacobi SVD as Hotelling T² (see
  [hotelling_t2.md](hotelling_t2.md) for the SVD numerics).
- Per-sample Q parity vs LAPACK reference: **1e-10 abs / 1e-11 rel**.
- UCL parity vs `scipy.stats.norm.ppf`-based reference: **1e-6 abs /
  1e-5 rel**. The Wichura AS241 inverse-normal is accurate to ~7 digits
  in the central region (Φ⁻¹(0.95)) — sufficient for a statistical
  control limit but not bit-exact against the scipy LAPACK-quality
  inverter.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_util_q_residuals(c4a_matrix_view_t X, int32_t n_components, double alpha, double* q_per_sample, int64_t n_samples, double* ucl_out);
```

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_util_q_residuals` | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_util_q_residuals(c4a_matrix_view_t X, int32_t n_components, double alpha, double* q_per_sample, int64_t n_samples, double* ucl_out);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
