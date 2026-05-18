# LinearBaselineDrift — `c4a_aug_linear_drift_*`

Adds a per-sample affine baseline drift (constant + linear-in-wavelength) to spectra.

Algorithm:

1. **Offsets** (`n_samples` uniforms in `[offset_min, offset_max)`):
   `offsets[i] = offset_min + (offset_max - offset_min) * u_off_i`.
2. **Slopes**  (`n_samples` uniforms in `[slope_min, slope_max)`):
   `slopes[i]  = slope_min  + (slope_max  - slope_min)  * u_slope_i`.
3. **Apply**: with `lambdas = arange(n_features)` centered at the mean,
   `out[i, j] = X[i, j] + offsets[i] + slopes[i] * (j - (n_features - 1)/2)`.

Mirrors `nirs4all.operators.augmentation.spectral.LinearBaselineDrift` with the implicit-wavelengths branch (no wavelength array supplied — the implicit branch uses `arange(n_features)`).

## Parameters
- `offset_min, offset_max: double` (`min <= max`) — uniform range for the per-sample offset term.
- `slope_min, slope_max: double` (`min <= max`) — uniform range for the per-sample slope.

## ABI
```c
c4a_status_t c4a_aug_linear_drift_create(
    c4a_aug_linear_drift_handle_t** out,
    c4a_rng_pcg64_state_t* rng,
    double offset_min, double offset_max,
    double slope_min,  double slope_max);
void         c4a_aug_linear_drift_destroy(c4a_aug_linear_drift_handle_t* h);
c4a_status_t c4a_aug_linear_drift_apply(
    const c4a_aug_linear_drift_handle_t* h,
    c4a_matrix_view_t X, c4a_matrix_view_t out);
```

## Reference
- Frozen Python: `parity/python_generator/src/c4a_parity_augmenters_ref/linear_drift.py`.
- Parity tolerance: 1e-15 abs.
