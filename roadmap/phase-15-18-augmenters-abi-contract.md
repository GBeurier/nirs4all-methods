# Phase 15-18 — Augmenters ABI contract (locked upfront)

Per Codex Phases 5a+5b+post-batch reviews: all 39 augmenters share a uniform ABI shape locked BEFORE any of the 4 phases starts.

## Category & naming

- Category prefix: `n4m_aug_*` (NEW; not yet used).
- Per-augmenter: `n4m_aug_<snake_case_name>_handle_t` (with `_handle_t` suffix per Codex M2 convention).
- 3 symbols per augmenter: `_create`, `_apply`, `_destroy`.

## Universal ABI shape

```c
typedef struct n4m_aug_<NAME>_handle_t n4m_aug_<NAME>_handle_t;

/* _create takes the RNG handle FIRST, then operator-specific params.
 * The RNG is stored by reference; the augmenter does NOT own it.
 * Lifetime: caller must keep the RNG alive for the augmenter's lifetime. */
N4M_API n4m_status_t n4m_aug_<NAME>_create(
    n4m_aug_<NAME>_handle_t** out,
    n4m_rng_pcg64_state_t* rng,  /* MUST be non-NULL */
    /* operator-specific double / int32_t params */
);

/* _apply: stateless w.r.t. fit (RNG state advances per call).
 * In-place when X == out is allowed; out must be same shape as X. */
N4M_API n4m_status_t n4m_aug_<NAME>_apply(
    const n4m_aug_<NAME>_handle_t* handle,
    n4m_matrix_view_t X,
    n4m_matrix_view_t out);

N4M_API void n4m_aug_<NAME>_destroy(n4m_aug_<NAME>_handle_t* handle);
```

## Determinism

Every augmenter consumes random numbers from the supplied PCG64 handle. With a fixed seed and a fixed RNG state at apply time, output is bit-exact reproducible. Tests use `n4m_rng_pcg64_set_seed(rng, S)` before each `_apply` to lock the stream.

## Out-of-scope behaviours

- No `_fit` step — augmenters are stateless w.r.t. data.
- No `_inverse_transform` — augmentation is irreversible by design.
- No `_is_fitted` — there's nothing to fit.

## Per-phase scope

### Phase 15 — Noise + drift (7 augmenters)
GaussianAdditiveNoise, MultiplicativeNoise, SpikeNoise, HeteroscedasticNoiseAugmenter, LinearBaselineDrift, PolynomialBaselineDrift, PathLengthAugmenter

### Phase 16 — Wavelength + spectral (10)
WavelengthShift, WavelengthStretch, LocalWavelengthWarp, BandPerturbation, BandMasking, ChannelDropout, GaussianSmoothingJitter, UnsharpSpectralMask, SmoothMagnitudeWarp, LocalClipping

### Phase 17 — Mixup + physical + environmental (10)
MixupAugmenter, LocalMixupAugmenter, ScatterSimulationMSC, ParticleSizeAugmenter, EMSCDistortionAugmenter, BatchEffectAugmenter, InstrumentalBroadeningAugmenter, DeadBandAugmenter, TemperatureAugmenter, MoistureAugmenter

### Phase 18 — Edge artifacts + splines + random (12)
DetectorRollOffAugmenter, StrayLightAugmenter, EdgeCurvatureAugmenter, TruncatedPeakAugmenter, EdgeArtifactsAugmenter, Spline_Smoothing, Spline_X_Perturbations, Spline_Y_Perturbations, Spline_X_Simplification, Spline_Curve_Simplification, Rotate_Translate, Random_X_Operation

Total: **39 augmenters × 3 symbols = 117 new ABI symbols**, target ABI 1.8.0.

## Reference

- `/home/delete/nirs4all/nirs4all/nirs4all/operators/augmentation/{spectral,synthesis,scattering,edge_artifacts,splines,environmental,random}.py`
- Each augmenter docs page in nirs4all + the n4m algorithm doc.

## Parity tolerance

- Bit-exact (1e-15) with the frozen NumPy reference when PCG64 produces the same stream.
- Tolerance class: stochastic-with-seeded-PCG64 = byte-equal uint64 stream + 1e-15 abs on `standard_normal` draws (per Phase 1).
