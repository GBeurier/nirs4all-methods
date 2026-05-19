# `aug_moisture` — MoistureAugmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_moisture.md`](../algorithms/aug_moisture.md)

## Description

`aug_moisture` is a chemometrics4all augmentation method exposed through the C ABI and the generated language bindings.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.environmental.MoistureAugmenter`,
Büning-Pfaue (2003), Luck (1998).

### Mathematical principle

Models water-band shifts due to changes in water activity ($a_w$) and
moisture content. Water exists in two states (free vs. hydrogen-bonded);
the relative fraction is governed by a sigmoid in $a_w$.

For each row:

1. $a_{w,i} = \mathrm{clip}(a_{w,\mathrm{ref}} + \Delta a_{w,i}, 0, 1)$
   where $\Delta a_{w,i}$ is uniform-sampled if a range is provided, else
   the fixed scalar.
2. $f_{\mathrm{free}} = \mathrm{free\_water\_fraction} \cdot \mathrm{sigmoid}(8(a_w - 0.5))$.
3. $f_{\mathrm{bound}} = 1 - f_{\mathrm{free}}$.
4. If `enable_shift`:
   - shift the water band centred at 1435 nm by $\mathrm{bound\_shift} \cdot f_{\mathrm{bound}}$.
   - shift the water band centred at 1930 nm by $0.8 \cdot \mathrm{bound\_shift} \cdot f_{\mathrm{bound}}$.
   Each shift is applied via `np.interp` with a Gaussian-weighted shift
   window centred at the band centre.
5. If `enable_intensity`:
   $$\mathrm{enhance} = \mathrm{moisture\_content} / 0.10 - 1$$
   $$X_{i,:} \mathrel{+}= \mathrm{enhance} \cdot 0.1 \cdot (g_{1435} + g_{1930}) \cdot |\overline{X_i}|$$
   where $g_c$ is a Gaussian region centred at wavelength $c$.

### Implementation

Implementation details are defined by the public C ABI and the generated binding wrappers.

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
