# `spectral_quality` — SpectralQualityFilter

_Group_: **Filter** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/spectral_quality.md`](../algorithms/spectral_quality.md)

## Description

Identifies samples whose spectra fail at least one of six quality checks. The filter is stateless: thresholds are constructor parameters and the same checks run on every `apply` call.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- nirs4all 0.8.x — `nirs4all/operators/filters/spectral_quality.py`
- `parity/python_generator/src/c4a_parity_filters_ref/spectral_quality.py`

### Mathematical principle

`spectral_quality` follows the standard filter operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- Per-row evaluation in a single pass over the data (cache-friendly).
- Variance uses the `np.nanvar` semantics: NaN entries are skipped from the mean/sum-of-squares but Inf entries are included so that any row containing Inf has a NaN variance (which fails the check regardless of `min_variance`).
- Zero-cols input: rejects every row (consistent with the reference's behaviour when `n_features == 0`).

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


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
