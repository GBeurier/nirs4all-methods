# `aug_dead_band` — DeadBandAugmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_dead_band.md`](../algorithms/aug_dead_band.md)

## Description

`aug_dead_band` is a chemometrics4all augmentation method exposed through the C ABI and the generated language bindings.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.synthesis.DeadBandAugmenter`.

### Mathematical principle

Simulates detector dead bands by replacing random contiguous spectral
regions with noise.

Per-sample mode (`variation_scope = 0`):

```
for each row i in 0..n-1:
    u ~ Uniform(0, 1)
    if u < probability:
        for b in 0..n_bands:
            width ~ UniformInt[width_low, width_high]   # inclusive
            start ~ UniformInt[0, max(1, cols - width))
            end   = min(start + width, cols)
            row[start:end] ~ Normal(0, noise_std)
```

Batch mode (`variation_scope = 1`):

```
u ~ Uniform(0, 1)
if u < probability:
    for b in 0..n_bands:
        width, start, end = ... (as above)
        # all rows get the same band positions, but
        # the noise samples are still drawn independently
        # with size = (rows, end - start)
        X[:, start:end] ~ Normal(0, noise_std)
```

### Implementation

When `probability = 0` the augmenter is the identity. The parity fixture
exercises this case (`dead_band_zero_probability`).

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
