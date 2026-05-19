# `aug_batch_effect` — BatchEffectAugmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_batch_effect.md`](../algorithms/aug_batch_effect.md)

## Description

`aug_batch_effect` is a chemometrics4all augmentation method exposed through the C ABI and the generated language bindings.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.synthesis.BatchEffectAugmenter`.

### Mathematical principle

Simulates batch/session measurement drift via a wavelength-normalised
additive offset, a slope, and a multiplicative gain.

Let $\tilde\lambda_j$ be the centred and range-normalised wavelength axis:
$$\tilde\lambda_j = (\lambda_j - \bar\lambda) / (\lambda_{\max} - \lambda_{\min} + 10^{-10}).$$
If `wavelengths == NULL`, the integer index $0, 1, \dots, p-1$ is used in
its place.

Two modes:

**Per-sample** (`variation_scope = 0`):
$$o_i \sim \mathrm{Normal}(0, \sigma_o),\ m_i \sim \mathrm{Normal}(0, \sigma_m),\ g_i \sim \mathrm{Normal}(1, \sigma_g)$$
$$X^{\mathrm{aug}}_{i,j} = X_{i,j} \cdot g_i + o_i + m_i \tilde\lambda_j$$

**Batch** (`variation_scope = 1`):
$$o \sim \mathrm{Normal}(0, \sigma_o),\ m \sim \mathrm{Normal}(0, \sigma_m),\ g \sim \mathrm{Normal}(1, \sigma_g)$$
$$X^{\mathrm{aug}}_{i,j} = X_{i,j} \cdot g + o + m \tilde\lambda_j$$

In per-sample mode the reference draws the parameters as three separate
length-$n$ normal vectors (offsets, then slopes, then gains). The C engine
mirrors this draw order so the PCG64 state evolution matches NumPy.

### Implementation

When $\sigma_o = \sigma_m = \sigma_g = 0$ the augmenter is the identity
transform (gain ~ N(1, 0) = 1, offset/slope ~ N(0, 0) = 0). The parity
fixture exercises this case (`batch_effect_zero_batch`).

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
