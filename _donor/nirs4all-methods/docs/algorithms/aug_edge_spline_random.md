# Edge, spline, and random augmenters

Twelve data-augmentation operators in the unified `n4m_aug_*` ABI category.

All twelve operators share the universal ABI shape:

```c
typedef struct n4m_aug_<NAME>_handle_t n4m_aug_<NAME>_handle_t;

n4m_status_t n4m_aug_<NAME>_create(
    n4m_aug_<NAME>_handle_t** out,
    n4m_rng_pcg64_state_t* rng,   /* MUST be non-NULL */
    /* operator-specific parameters */);

n4m_status_t n4m_aug_<NAME>_apply(
    const n4m_aug_<NAME>_handle_t* handle,
    n4m_matrix_view_t X,
    [n4m_matrix_view_t wavelengths,]   /* only for wavelength-aware ops */
    n4m_matrix_view_t out);

void n4m_aug_<NAME>_destroy(n4m_aug_<NAME>_handle_t* handle);
```

The RNG handle is stored by reference; the augmenter does NOT own it. The
caller MUST keep the RNG alive for the augmenter's lifetime. With a fixed
RNG state at `_apply` time the output is bit-exact reproducible.

## Edge artifacts (5)

### DetectorRollOff — `n4m_aug_detector_rolloff_*`

Models a wavelength-dependent detector sensitivity roll-off curve and
applies noise amplification + slight baseline distortion at the edges.
Five built-in detector presets (selected via `detector_model`):

| ID | Detector            | Optimal range (nm) | Roll-off rate |
|----|---------------------|--------------------|---------------|
| 0  | InGaAs standard     | 1000-1600          | 0.008         |
| 1  | InGaAs extended     | 1100-2200          | 0.005         |
| 2  | PbS                 | 1000-2800          | 0.004         |
| 3  | Silicon CCD         | 400-900            | 0.015         |
| 4  | Generic NIR         | 900-1700           | 0.006         |

### StrayLight — `n4m_aug_stray_light_*`

Applies the stray-light transmittance equation `T_obs = (T_true + s) / (1 + s)`
on the absorbance representation, with a wavelength-dependent stray
fraction enhanced at spectral edges via a sigmoid.

### EdgeCurvature — `n4m_aug_edge_curve_*`

Adds smile / frown / asymmetric curvature patterns on the normalised
wavelength axis. The `curvature_type` parameter is:

| ID | Type        | Behaviour                                  |
|----|-------------|--------------------------------------------|
| 0  | random      | per-sample: pick smile/frown/asym at random |
| 1  | smile       | upward curvature at edges                  |
| 2  | frown       | downward curvature at edges                |
| 3  | asymmetric  | different curvature per side               |

### TruncatedPeak — `n4m_aug_truncated_peak_*`

Per-sample, with `peak_probability`, adds a tail of a Gaussian peak whose
centre lies outside the measured wavelength range. Models absorption
bands clipped at the spectral boundary.

### EdgeArtifacts — `n4m_aug_edge_artifacts_*`

Combined wrapper. The `enabled_flags` integer selects sub-augmenters:

| Flag | Value | Sub-augmenter        |
|------|-------|----------------------|
| `N4M_AUG_EDGE_ARTIFACTS_DETECTOR_ROLL_OFF` | `0x1` | DetectorRollOff |
| `N4M_AUG_EDGE_ARTIFACTS_STRAY_LIGHT`       | `0x2` | StrayLight      |
| `N4M_AUG_EDGE_ARTIFACTS_EDGE_CURVATURE`    | `0x4` | EdgeCurvature   |
| `N4M_AUG_EDGE_ARTIFACTS_TRUNCATED_PEAKS`   | `0x8` | TruncatedPeak   |

Sub-augmenters are applied in fixed order: truncated peaks -> edge
curvature -> stray light -> detector roll-off (matches the Python
reference). `overall_strength` scales every sub-augmenter's defaults.

## Splines (5)

Built on the internal natural-cubic-spline helper (`core/common/bspline.{c,h}`)
which exposes `n4m_bspline_build_natural` + `n4m_bspline_eval[_array]`. The
helper is private to the library.

### Spline_Smoothing — `n4m_aug_spline_smooth_*`

Stateless: fits a natural cubic spline through `(j, X[i, j])` and
re-evaluates on the same grid. The Python reference uses
`scipy.interpolate.UnivariateSpline(s=1/n)`; with a natural cubic spline
the C version interpolates through every input point (smoothing factor
zero), which is the limiting behaviour as `s -> 0`.

### Spline_X_Perturbations — `n4m_aug_spline_x_perturb_*`

Fits a natural cubic spline through `(j, X[i, j])` then evaluates at
perturbed query points `xq = j + delta(j)`. `delta` is a piecewise-linear
field anchored at `delta_size` points along the x-axis; per-anchor values
draw uniformly from `[perturbation_range_min, perturbation_range_max]`.

Only `spline_degree=3` is supported by the vendored helper; passing any
other value returns `N4M_ERR_INVALID_ARGUMENT` from `_create`.

### Spline_Y_Perturbations — `n4m_aug_spline_y_perturb_*`

Generates a smooth y-axis distortion field from a natural cubic spline
through `(linspace(0, n, nb_pts), uniform(-v, v))` samples and adds it to
the spectrum. The `spline_points` parameter accepts `<= 0` to mean
"use `n_features / 2`".

### Spline_X_Simplification — `n4m_aug_spline_x_simplify_*` (v2-deferred)

Stub: `_apply` returns `N4M_ERR_NOT_IMPLEMENTED`. Implementation gated on
the v2 `rng.choice(replace=False)` primitive. See `DEFERRALS.md`.

### Spline_Curve_Simplification — `n4m_aug_spline_curve_simplify_*` (v2-deferred)

Same gap as `Spline_X_Simplification`. Stub returns
`N4M_ERR_NOT_IMPLEMENTED`. See `DEFERRALS.md`.

## Random (2)

### Rotate_Translate — `n4m_aug_rotate_translate_*`

Adds a piecewise-linear "rotation + translation" pattern (two slopes
meeting at a hinge point on the normalised x-axis) scaled by per-sample
std. Mirrors `nirs4all.operators.augmentation.random.Rotate_Translate`.

### Random_X_Operation — `n4m_aug_random_x_op_*`

Per-cell elementwise operator. `op_kind` selects:

| ID | Operation              |
|----|------------------------|
| 0  | multiply (`X * u`)     |
| 1  | add      (`X + u`)     |
| 2  | subtract (`X - u`)     |

Operands `u` draw uniformly from `[operator_range_min, operator_range_max]`.
Output is clipped to the float32 representable range (matches the Python
reference's `np.clip(..., -float32.max, float32.max)`).

## Numerical contract

- All 10 fully-implemented operators are deterministic: re-seeding the
  RNG handle before two consecutive `_apply` calls yields bit-equal
  outputs. The smoke fixtures under `parity/fixtures/aug_*_v1.json`
  exercise the shape + finite-value path; bit-exact NumPy parity of
  `rng.uniform`, `rng.choice` is gated behind v2 of the augmenter ABI.
- The 2 v2-deferred operators (`Spline_X_Simplification`,
  `Spline_Curve_Simplification`) return `N4M_ERR_NOT_IMPLEMENTED` from
  `_apply` until v2; their `_create` / `_destroy` symbols are stable.

## Reference

- nirs4all 0.8.x —
  `nirs4all.operators.augmentation.edge_artifacts`,
  `nirs4all.operators.augmentation.splines`,
  `nirs4all.operators.augmentation.random`.
- `roadmap/phase-15-18-augmenters-abi-contract.md`
- `DEFERRALS.md`
