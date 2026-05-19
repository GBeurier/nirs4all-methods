# `hotelling_t2` — Hotelling T²

_Group_: **Diagnostic** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/hotelling_t2.md`](../algorithms/hotelling_t2.md)

## Description

Multivariate outlier statistic on a PCA-reduced subspace.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- Hotelling, H. (1931). "The Generalization of Student's Ratio".
- Fixture: `parity/fixtures/hotelling_t2_v1.json`.

### Mathematical principle

For a `(n_samples × n_features)` matrix `X`:

1. Centre `X` by column means: `Xc = X - mean(X, axis=0)`.
2. Compact SVD: `Xc = U Σ V^T` via the one-sided Jacobi solver in
   `core/common/svd.{c,h}`.
3. Take the first `k = n_components` columns of the score matrix
   `T = U Σ`.
4. The corresponding sample-covariance eigenvalues are
   `λ_j = σ_j² / (n_samples − 1)`.
5. Per-sample statistic:
   `T²[i] = Σ_{j<k} (T[i, j]² / λ_j)`.

### Implementation

- The PCA uses a Jacobi SVD with default tolerance `1e-14` and at most 50
  sweeps. For the typical NIRS PCA sizes (`p` up to a few hundred) this
  converges in 5–10 sweeps.
- F-quantile inverter: bisection to relative tolerance `1e-12`.
- Per-sample T² parity vs LAPACK reference (`numpy.linalg.svd`):
  **1e-10 abs / 1e-11 rel** — the structural gap between Jacobi rotations
  and Householder bidiagonal SVD.
- UCL parity vs `scipy.stats.f.ppf`: **1e-6 abs / 1e-6 rel** — the
  bisection inverter is intentionally simple and falls below the LAPACK
  / scipy approximation level.

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_util_hotelling_t2(c4a_matrix_view_t X, int32_t n_components, double alpha, double* t2_per_sample, int64_t n_samples, double* ucl_out);
```

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_util_hotelling_t2` | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_util_hotelling_t2(c4a_matrix_view_t X, int32_t n_components, double alpha, double* t2_per_sample, int64_t n_samples, double* ucl_out);
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
