# `wavelet_svd` — WaveletSVD

_Group_: **Wavelet** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/wavelet_svd.md`](../algorithms/wavelet_svd.md)

## Description

`wavelet_svd` is a chemometrics4all wavelet method exposed through the C ABI and the generated language bindings.

### Parameters

| Parameter      | Default       | Meaning                                  |
| -------------- | ------------- | ---------------------------------------- |
| `family`       | (db4)         | Wavelet family                           |
| `mode`         | (per.)        | Boundary mode                            |
| `max_level`    | 4             | Maximum decomposition depth             |
| `n_components` | —             | Flexible count or variance-ratio specifier |

## Explanations

### Bibliographic source

`nirs4all.operators.transforms.nirs.WaveletSVD` (Trygg & Wold 1998).
Implementation simplifies the nirs4all per-level Truncated SVD into a
single SVD on the flattened coefficient stack.

### Mathematical principle

Two-stage stateful dimensionality reduction matching `WaveletPCA`,
but the projection step uses Truncated SVD (no mean subtraction).

Given a training matrix $X \in \mathbb{R}^{m \times n}$ and an
effective level $K$ (clamped to `dwt_max_level(n, family)`):

1. Build the flat coefficient matrix $F \in \mathbb{R}^{m \times D}$
   exactly as for `WaveletPCA`.
2. Compact SVD $F = U S V^\top$ with the same sign-canonicalisation as
   `FlexibleSVD` (`u_based_decision = True`).
3. Energy ratio $\rho_j = S_j^2 / \sum_i S_i^2$.
4. Component count $k'$ via the flexible specifier (integer
   $\geq 1$ for count, float in $(0, 1)$ for ratio).
5. Components $= V^\top_{[:k', :]}$.  No mean is stored.

`_transform` re-applies the same DWT-flatten step on incoming rows and
projects them through $F_{\text{new}} V^{\top \top}_{[:k', :]}$.

### Implementation

Stateful: `_create / _fit / _transform / _destroy` plus `_is_fitted`
and `_output_cols`.  Same lifecycle and shape rules as `WaveletPCA`.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | — | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
/* C ABI prefix */
c4a_*
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
