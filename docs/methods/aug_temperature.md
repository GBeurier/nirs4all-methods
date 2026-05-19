# `aug_temperature` — TemperatureAugmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_temperature.md`](../algorithms/aug_temperature.md)

## Description

`aug_temperature` is a chemometrics4all augmentation method exposed through the C ABI and the generated language bindings.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.environmental.TemperatureAugmenter`.

### Mathematical principle

Wavelength-aware NIR temperature effect simulator. Models three effects
across six O-H / C-H / N-H / water spectral regions:

1. **Peak position shift**: `wl_aug = wl + shift_per_°C * Δt`
2. **Intensity change**: `factor = 1 + intensity_per_°C * Δt`
3. **Band broadening**: `factor = 1 + broadening_per_°C * Δt`
   (applied via `gaussian_filter1d` with sigma = `(factor - 1) * 3`).

Each region has its own coefficients, baked into
`TEMPERATURE_REGION_PARAMS`:

| Region | range (nm) | shift/°C | intensity/°C | broadening/°C |
|--------|-----------|----------|--------------|---------------|
| OH 1st overtone | 1400–1520 | −0.30 | −0.002  | 0.001  |
| OH combination  | 1900–2000 | −0.40 | −0.003  | 0.0012 |
| CH 1st overtone | 1650–1780 | −0.05 | −0.0005 | 0.0008 |
| NH 1st overtone | 1490–1560 | −0.20 | −0.0015 | 0.001  |
| Water free      | 1380–1420 | −0.10 |  0.003  | 0.0008 |
| Water bound     | 1440–1500 | −0.35 | −0.004  | 0.0015 |

Effects are localised via a smooth sigmoid weight:

$$w_j = \frac{1}{1 + e^{-10(\lambda_j - \lambda_{\min})/20}} \cdot \frac{1}{1 + e^{10(\lambda_j - \lambda_{\max})/20}}$$

In **region-specific** mode (default), every region's effects are layered.
In **uniform mode** (`region_specific = 0`), the average coefficients are
applied across all wavelengths.

If `temperature_range` is provided, $\Delta t_i \sim \mathrm{Uniform}(\mathrm{low}, \mathrm{high})$
per row; otherwise the fixed `temperature_delta` is used for every row.

### Implementation

When `temperature_delta = 0` and no range is set, the augmenter is the
identity (early exit when `|Δt| < 0.01`). The parity fixture exercises
this case (`temperature_zero_delta`).

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
