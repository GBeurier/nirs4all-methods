# WaveletDenoise

## Algorithm

Multi-level wavelet denoising with the Donoho universal threshold and
soft / hard thresholding.  Implements the recipe from
`nirs4all.operators.transforms.wavelet_denoise.WaveletDenoise`.

For each row $x \in \mathbb{R}^n$:

1. Multi-level DWT decomposition produces
   $\{c_A^{(K)}, c_D^{(K)}, c_D^{(K-1)}, \ldots, c_D^{(1)}\}$
   where $K$ is the effective level (clamped to
   `pywt.dwt_max_level(n, family)`).
2. Per-row noise estimate from the finest detail $c_D^{(1)}$:

   $$
   \sigma_{\text{MAD}} = \frac{\mathrm{median}\bigl(|c_D^{(1)}|\bigr)}{0.6745},
   \qquad
   \sigma_{\text{std}} = \mathrm{std}\bigl(c_D^{(1)}\bigr).
   $$
3. Universal threshold:
   $\theta = \sigma \cdot \sqrt{2 \log n}$.
4. Soft / hard thresholding applied to every detail band:

   $$
   T_{\text{soft}}(y; \theta) = \mathrm{sign}(y) \cdot \max(|y| - \theta, 0),
   \qquad
   T_{\text{hard}}(y; \theta) = \begin{cases} y & |y| > \theta \\ 0 & \text{otherwise} \end{cases}.
   $$
   The approximation band $c_A^{(K)}$ is kept untouched.
5. Multi-level reconstruction yields the denoised row, truncated to the
   original length $n$.

If $K$ clamps to 0 (filter longer than input) the operator passes the
input through unchanged.

## Parameters

| Parameter         | Default        | Meaning                                  |
| ----------------- | -------------- | ---------------------------------------- |
| `family`          | (db4)          | Wavelet family                          |
| `mode`            | (periodization)| Boundary mode                           |
| `level`           | 5              | Maximum decomposition level             |
| `threshold_mode`  | soft           | Thresholding mode (`soft` or `hard`)    |
| `noise_estimator` | median (MAD)   | Noise estimator (`median` or `std`)     |

## Lifecycle

Stateless: `_create / _transform / _destroy`.  Output shape matches the
input shape.

## Numerical contract

- `_transform` requires the output matrix to have the same shape as
  the input; otherwise returns `N4M_ERR_SHAPE_MISMATCH`.
- Parity tolerance against the internal parity fixture: `1e-9` absolute /
  `1e-10` relative (multi-level DWT + sort-based median + soft / hard
  thresholding accumulate slightly more floating-point noise than a
  single-level transform).

## Reference

`nirs4all.operators.transforms.wavelet_denoise.WaveletDenoise` and the
Donoho & Johnstone (1994) ideal-spatial-adaptation paper.
