# `aug_scatter_sim` — ScatterSimulationMSC

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_scatter_sim.md`](../algorithms/aug_scatter_sim.md)

## Description

`aug_scatter_sim` is a chemometrics4all augmentation method exposed through the C ABI and the generated language bindings.

### Parameters

| Parameter | Constraint | Meaning |
|-----------|------------|---------|
| `a_low`, `a_high` | `a_high >= a_low` | Additive-offset uniform range. |
| `b_low`, `b_high` | `b_high >= b_low` | Multiplicative-gain uniform range. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.spectral.ScatterSimulationMSC`.

### Mathematical principle

Simulates Multiplicative Scatter Correction (MSC) reversibility by drawing
a random additive offset and multiplicative gain per sample, then
applying

$$X^{\mathrm{aug}}_{i,:} = a_i + b_i \cdot X_{i,:}$$

with $a_i \sim \mathrm{Uniform}(a_{\mathrm{low}}, a_{\mathrm{high}})$ and
$b_i \sim \mathrm{Uniform}(b_{\mathrm{low}}, b_{\mathrm{high}})$.

The reference draws $n$ uniform `a` values followed by $n$ uniform `b`
values — the C engine mirrors this exact order so the PCG64 stream
consumption matches NumPy's vectorised `rng.uniform(low, high, size=n)`
calls.

### Implementation

When `a_low == a_high` and `b_low == b_high`, the augmenter is a
deterministic affine transform regardless of the RNG state. The parity
fixture exercises this case (`scatter_sim_constant` in
`aug_phase17_v1.json`).

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
