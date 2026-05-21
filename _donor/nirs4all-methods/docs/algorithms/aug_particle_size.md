# ParticleSizeAugmenter

## Algorithm

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

The noise convolution reuses the existing `n4m_pp_gaussian` engine to
preserve bit-exact kernel construction and convolution semantics.

## Parameters

See `n4m.h` for the full list. Key flags:

| Parameter | Meaning |
|-----------|---------|
| `use_size_range`  | 0 = clipped normal, 1 = uniform from `[low, high]`. |
| `include_path_length` | 0 = skip step 6, 1 = apply. |

## Reference

`nirs4all.operators.augmentation.scattering.ParticleSizeAugmenter`.
