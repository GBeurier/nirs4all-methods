# `aug_particle_size` — ParticleSizeAugmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_particle_size.md`](../algorithms/aug_particle_size.md)

## Description

`aug_particle_size` is a chemometrics4all augmentation method exposed through the C ABI and the generated language bindings.

### Parameters

See `c4a.h` for the full list. Key flags:

| Parameter | Meaning |
|-----------|---------|
| `use_size_range`  | 0 = clipped normal, 1 = uniform from `[low, high]`. |
| `include_path_length` | 0 = skip step 6, 1 = apply. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.scattering.ParticleSizeAugmenter`.

### Mathematical principle

Wavelength-aware augmenter modelling particle-size induced scattering.
For each row:

1. Draw particle size $s_i$:
   - `use_size_range == 1`: $s_i \sim \mathrm{Uniform}(s_{\mathrm{lo}}, s_{\mathrm{hi}})$.
   - else: $s_i \sim \mathrm{Normal}(\mu_s, \sigma_s)$ clipped to $[5, 500]$ μm.
2. Compute size ratio $r_i = s_i / s_{\mathrm{ref}}$, clipped to $[0.1, 10]$.
3. Wavelength factor (per wavelength $\lambda_j$):
$$f_j = \mathrm{clip}(\lambda_j / 1500, 0.1, 10)^{-\text{exponent}}$$
4. Per-sample baseline:
$$b_{i,j} = \mathrm{strength} \cdot (r_i^{-1/2} - 1) \cdot f_j$$
and centre $b_{i,j} \leftarrow b_{i,j} - \mathrm{mean}_j b_{i,j}$.
5. $X^{\mathrm{aug}}_{i,j} = X_{i,j} + b_{i,j}$.
6. If `include_path_length`:
$$X^{\mathrm{aug}}_{i,:} \leftarrow X^{\mathrm{aug}}_{i,:} \cdot \mathrm{clip}(1 + \mathrm{sens} \cdot \log r_i, 0.7, 1.5)$$
7. Add Gaussian-smoothed noise:
$$\eta_i \sim \mathrm{Normal}(0, 0.005 \cdot \mathrm{strength})^p,\quad
X^{\mathrm{aug}}_{i,:} \leftarrow X^{\mathrm{aug}}_{i,:} + G_{\sigma=3}(\eta_i)$$
where $G_{\sigma=3}$ is `scipy.ndimage.gaussian_filter1d` with
default settings (mode `reflect`, `truncate=4.0`).

The noise convolution reuses the existing `c4a_pp_gaussian` engine to
preserve bit-exact kernel construction and convolution semantics.

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
