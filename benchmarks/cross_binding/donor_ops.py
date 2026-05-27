# SPDX-License-Identifier: CECILL-2.1
"""Donor-operator binding specs — the source of truth for Layer-2 timing.

The C++ donor-operator timing tool (`bench/cpp/n4m_donor_bench`) times every
donor op (augmentation / preprocessing / filters / splitters) through the
public C ABI at the dashboard sizes (Layer 1). This module is the binding-side
mirror: for each timeable donor op it declares how to invoke the SAME operation
through the two Python binding tiers —

  * raw      → `n4m.python.<fn>(...)`        (dashboard backend `python_tier1`)
  * idiomatic→ `n4m.sklearn.<Class>(...)`    (dashboard backend `python_tier2`)

`bench_donor_binding_timing.py` consumes these specs to time both tiers and
assert raw↔idiomatic parity. Because both tiers wrap the identical ABI seeded
the same way, their outputs are bit-exact whenever the params match, so the
raw↔idiomatic parity gate is a sharp detector of any per-tier param drift.

Param sourcing (the load-bearing invariant — see the Codex review):
  * Where the raw fn EXPOSES a param, we pass the curated C++ donor-bench value
    (so the binding tier times the same configuration as the C++ tier, and the
    output is dump-validatable against the C++ `out` buffer).
  * Where the raw fn HARDCODES a param (several physical augmenters do), the
    raw fn's effective value is the binding tier's truth; we construct the
    idiomatic estimator with kwargs that reproduce it, so the gate still holds.
    Those ops are not dump-validatable against C++ (the binding API simply does
    not expose the C++ configuration) and are flagged accordingly.

The `reference` slot is reserved for Layer 3 (external real-library refs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Deterministic input generation — byte-for-byte mirror of n4m_donor_bench.cpp
# (splitmix64 spectra-like X in [0.2, 1.0), y = splitmix(seed ^ 0xABCDEF),
#  wavelength axis = linspace(1000, 2500, p), groups = i % 8).
# ---------------------------------------------------------------------------
_U64 = 0xFFFFFFFFFFFFFFFF


def _splitmix64_stream(seed: int, count: int) -> np.ndarray:
    """Return `count` doubles in [0.2, 1.0) from a splitmix64 stream.

    Mirrors `splitmix64` + `fill_matrix` in n4m_donor_bench.cpp exactly so the
    Python binding times the identical input bits the C++ tool used.
    """
    state = seed if seed else 0x1234567890
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        state = (state + 0x9E3779B97F4A7C15) & _U64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
        z = z ^ (z >> 31)
        u = float(z >> 11) * (1.0 / 9007199254740992.0)
        out[i] = 0.2 + 0.8 * u
    return out


def make_X(n: int, p: int, seed: int) -> np.ndarray:
    return _splitmix64_stream(seed, n * p).reshape(n, p)


def make_y(n: int, seed: int) -> np.ndarray:
    return _splitmix64_stream(seed ^ 0xABCDEF, n)


def wavelength_axis(p: int) -> np.ndarray:
    if p == 1:
        return np.array([1000.0], dtype=np.float64)
    return np.linspace(1000.0, 2500.0, p, dtype=np.float64)


def make_groups(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64) % 8


# ---------------------------------------------------------------------------
# Spec model
# ---------------------------------------------------------------------------
# Reasons a binding tier is not mapped (rendered honestly as "—").
NO_RAW = "no_raw_fn"            # raw n4m.python has no public fn for this op
NO_ESTIMATOR = "no_estimator"  # n4m.sklearn has no idiomatic class
SEMANTIC_MISMATCH = "semantic_mismatch"  # an estimator exists but is not comparable
BINDING_DEFERRED = "binding_deferred"    # intentionally out of scope


@dataclass(frozen=True)
class DonorOpSpec:
    """How to run one donor op through the two Python binding tiers.

    `raw_fn`/`raw_kwargs` build the `n4m.python` call; `idiom_cls`/`idiom_module`
    /`idiom_kwargs` build the `n4m.sklearn` call. `inputs` declares which arrays
    the op consumes. `parity` selects how an output is reduced to a comparable
    vector. `dump_validatable` is True when the raw call reproduces the C++
    donor-bench `out` buffer (param-exposing augmenters + dim-preserving pp),
    so a one-off `--dump-output` check can prove the C++→raw transcription.
    """

    bench_id: str
    dashboard_id: str
    category: str  # "aug" | "pp" | "filter" | "split"
    inputs: tuple[str, ...]  # subset of ("X", "y", "wl", "groups")
    parity: str  # "values" | "mask" | "indices"

    raw_fn: Optional[str] = None
    raw_kwargs: dict[str, Any] = field(default_factory=dict)
    raw_reason: Optional[str] = None

    idiom_module: Optional[str] = None
    idiom_cls: Optional[str] = None
    idiom_kwargs: dict[str, Any] = field(default_factory=dict)
    idiom_reason: Optional[str] = None

    dump_validatable: bool = False
    on_dashboard: bool = True  # False = no individual dashboard row (bundled)
    reference: Any = None  # Layer 3: external real-library reference spec

    def __post_init__(self) -> None:
        if (self.raw_fn is None) == (self.raw_reason is None):
            raise ValueError(f"{self.bench_id}: exactly one of raw_fn/raw_reason")
        if (self.idiom_cls is None) == (self.idiom_reason is None):
            raise ValueError(f"{self.bench_id}: exactly one of idiom_cls/idiom_reason")


# ---------------------------------------------------------------------------
# Dashboard naming. A few bench-op ids differ from the dashboard method names
# the fixture rows use (the n4m_tests group names dropped the category prefix
# for the spectral augmenters, kept the long form for others). This is the
# reverse of `bench_donor_timing.DASHBOARD_TO_BENCH`, kept here so the binding
# rows key onto the SAME dashboard algorithm as the C++ tier (no new algos).
# ---------------------------------------------------------------------------
_DASHBOARD_ALIAS: dict[str, str] = {
    "aug_edge_curve": "aug_edge_curvature",
    "aug_edge_artifacts": "aug_edge_artifacts_combined",
    "aug_random_x_op": "aug_random_x_operation",
    "aug_spline_smooth": "aug_spline_smoothing",
    "aug_spline_x_perturb": "aug_spline_x_perturbations",
    "aug_spline_y_perturb": "aug_spline_y_perturbations",
    "aug_band_mask": "band_mask",
    "aug_band_perturb": "band_perturb",
    "aug_channel_dropout": "channel_dropout",
    "aug_gauss_jitter": "gauss_jitter",
    "aug_local_clip": "local_clip",
    "aug_local_warp": "local_warp",
    "aug_magnitude_warp": "magnitude_warp",
    "aug_unsharp_mask": "unsharp_mask",
    "aug_wavelength_shift": "wavelength_shift",
    "aug_wavelength_stretch": "wavelength_stretch",
    "pp_fck_static": "fck",
}


def _dash(bench_id: str) -> str:
    return _DASHBOARD_ALIAS.get(bench_id, bench_id)


# ---------------------------------------------------------------------------
# Builders — keep the registry declarations compact and uniform.
# ---------------------------------------------------------------------------
def _pp(bench_id, raw_fn, idiom_module, idiom_cls, *, raw=None, idiom=None,
        inputs=("X",), dump=True, parity="values") -> DonorOpSpec:
    """Preprocessing op: raw `n4m.python.<fn>(X, **raw)`, idiom
    `n4m.sklearn.<module>.<Cls>(**idiom).fit_transform(X)`. Defaults to
    dump-validatable (dim-preserving deterministic transforms)."""
    return DonorOpSpec(
        bench_id=bench_id, dashboard_id=_dash(bench_id), category="pp", inputs=inputs,
        parity=parity, raw_fn=raw_fn, raw_kwargs=dict(raw or {}),
        idiom_module=idiom_module, idiom_cls=idiom_cls,
        idiom_kwargs=dict(idiom if idiom is not None else (raw or {})),
        dump_validatable=dump,
    )


def _aug(bench_id, raw_fn, idiom_cls, *, raw=None, idiom=None, wl=False,
         dump=True, on_dashboard=True) -> DonorOpSpec:
    """Augmenter: raw `n4m.python.<fn>(X[, wavelengths], **raw, seed=S)`, idiom
    `n4m.sklearn.augmentation.<Cls>(**idiom, seed=S).fit_transform(X)`."""
    inputs = ("X", "wl") if wl else ("X",)
    return DonorOpSpec(
        bench_id=bench_id, dashboard_id=_dash(bench_id), category="aug",
        inputs=inputs, parity="values", raw_fn=raw_fn, raw_kwargs=dict(raw or {}),
        idiom_module="augmentation", idiom_cls=idiom_cls,
        idiom_kwargs=dict(idiom if idiom is not None else (raw or {})),
        dump_validatable=dump, on_dashboard=on_dashboard,
    )


# ===========================================================================
# Preprocessing (46 ops). C++ params are transcribed from the PP_MAKE(...)
# rows in n4m_donor_bench.cpp; passed to BOTH tiers so the binding times the
# same configuration as the C++ tier.
# ===========================================================================
PP_SPECS: list[DonorOpSpec] = [
    # dim-preserving, stateless/stateful (dump-validatable: write c.out)
    _pp("pp_snv", "snv", "preprocessing", "SNV",
        raw=dict(with_mean=True, with_std=True, ddof=0)),
    _pp("pp_rnv", "rnv", "preprocessing", "RNV",
        raw=dict(with_center=True, with_scale=True, k=0.25)),
    _pp("pp_lsnv", "lsnv", "preprocessing", "LSNV",
        raw=dict(window=11, pad_mode="reflect", constant_value=0.0)),
    _pp("pp_area", "area_norm", "preprocessing", "AreaNormalization",
        raw=dict(method="sum")),
    _pp("pp_normalize", "normalize", "preprocessing", "Normalize",
        raw=dict(feature_min=0.0, feature_max=1.0)),
    _pp("pp_simple_scale", "simple_scale", "preprocessing", "SimpleScale"),
    _pp("pp_detrend", "detrend", "baseline", "Detrend", raw=dict(polyorder=1)),
    _pp("pp_gaussian", "gaussian", "preprocessing", "Gaussian",
        raw=dict(sigma=2.0, order=0, mode="reflect", cval=0.0, truncate=4.0)),
    _pp("pp_savgol", "savitzky_golay", "preprocessing", "SavitzkyGolay",
        raw=dict(window_length=11, polyorder=2, deriv=0, delta=1.0, mode="mirror", cval=0.0)),
    _pp("pp_first_derivative", "first_derivative", "preprocessing", "FirstDerivative",
        raw=dict(delta=1.0, edge_order=2)),
    _pp("pp_second_derivative", "second_derivative", "preprocessing", "SecondDerivative",
        raw=dict(delta=1.0, edge_order=2)),
    _pp("pp_norris_williams", "norris_williams", "preprocessing", "NorrisWilliams",
        raw=dict(segment=5, gap=3, derivative_order=1, delta=1.0)),
    _pp("pp_frac_to_pct", "frac_to_pct", "preprocessing", "FractionToPercent"),
    _pp("pp_pct_to_frac", "pct_to_frac", "preprocessing", "PercentToFraction"),
    # The raw fns key the signal type as an int (source_type/target_type);
    # the idiomatic estimators expose it as `is_percent` (0 ↔ False).
    _pp("pp_from_absorbance", "from_absorbance", "preprocessing", "FromAbsorbance",
        raw=dict(target_type=0), idiom=dict(is_percent=False)),
    _pp("pp_to_absorbance", "to_absorbance", "preprocessing", "ToAbsorbance",
        raw=dict(source_type=0, epsilon=1e-6, clip_negative=True),
        idiom=dict(is_percent=False, epsilon=1e-6, clip_negative=True)),
    _pp("pp_kubelka_munk", "kubelka_munk", "preprocessing", "KubelkaMunk",
        raw=dict(source_type=0, epsilon=1e-6), idiom=dict(is_percent=False, epsilon=1e-6)),
    # baseline-correction family (iterative, dim-preserving)
    _pp("pp_airpls", "airpls", "baseline", "AirPLS", raw=dict(lam=1e5, max_iter=15, tol=1e-3)),
    _pp("pp_arpls", "arpls", "baseline", "ArPLS", raw=dict(lam=1e5, max_iter=50, tol=1e-3)),
    _pp("pp_asls", "asls", "baseline", "AsLS", raw=dict(lam=1e5, p=0.01, max_iter=10, tol=1e-3)),
    _pp("pp_iasls", "iasls", "baseline", "IAsLS",
        raw=dict(lam=1e5, p=0.01, polyorder=2, max_iter=10, tol=1e-3)),
    _pp("pp_modpoly", "modpoly", "baseline", "ModPoly", raw=dict(polyorder=2, max_iter=100, tol=1e-3)),
    _pp("pp_imodpoly", "imodpoly", "baseline", "IModPoly", raw=dict(polyorder=2, max_iter=100, tol=1e-3)),
    _pp("pp_beads", "beads", "baseline", "BEADS",
        raw=dict(lam_0=0.5, lam_1=5.0, lam_2=4.0, max_iter=20, tol=1e-2)),
    _pp("pp_snip", "snip", "baseline", "SNIP", raw=dict(max_half_window=20)),
    _pp("pp_rolling_ball", "rolling_ball", "baseline", "RollingBall",
        raw=dict(half_window=10, smooth_half_window=5)),
    _pp("pp_wavelet_denoise", "wavelet_denoise", "augmentation", "WaveletDenoise",
        raw=dict(family="haar", mode="periodization", level=3,
                 threshold_mode="soft", noise_estimator="median")),
    # stateful, X-only fit (dump-validatable)
    _pp("pp_msc", "msc", "preprocessing", "MSC"),
    _pp("pp_emsc", "emsc", "preprocessing", "EMSC", raw=dict(degree=2)),
    _pp("pp_baseline", "baseline_center", "preprocessing", "BaselineCenter"),
    _pp("pp_log", "log_transform", "preprocessing", "LogTransform",
        raw=dict(base=10.0, offset=0.0, auto_offset=True, min_value=1e-6)),
    # OSC needs a target vector for fit().
    _pp("pp_osc", "osc", "feature_extraction", "OSC",
        raw=dict(n_components=2, scale=True), inputs=("X", "y")),
    # discretizers → int32 output; dim-preserving but compare bin indices.
    _pp("pp_kbins_disc", "kbins_discretizer", "resampling", "IntegerKBinsDiscretizer",
        raw=dict(n_bins=5, strategy="uniform"), dump=False),
    _pp("pp_range_disc", "range_discretizer", "resampling", "RangeDiscretizer",
        raw=dict(edges=[0.4, 0.6, 0.8]), idiom=dict(edges=[0.4, 0.6, 0.8]), dump=False),
    # dim-changing: output is n × out_cols (not c.out) → not dump-validatable,
    # but the raw↔idiomatic gate still applies.
    _pp("pp_crop", "crop", "resampling", "CropTransformer",
        raw=dict(start=1, end=-1), idiom=dict(start=1, end=-1), dump=False),
    _pp("pp_derivate", "derivate", "preprocessing", "Derivate",
        raw=dict(order=1, delta=1.0), dump=False),
    _pp("pp_haar", "haar", "augmentation", "Haar", dump=False),
    _pp("pp_resample", "resample", "resampling", "ResampleTransformer",
        raw=dict(num_samples=128), idiom=dict(num_samples=128), dump=False),
    _pp("pp_resampler", "resampler", "resampling", "Resampler",
        raw={}, idiom={}, inputs=("X", "wl"), dump=False),
    _pp("pp_wavelet", "wavelet", "augmentation", "Wavelet",
        raw=dict(family="haar", mode="periodization"), dump=False),
    _pp("pp_wavelet_features", "wavelet_features", "augmentation", "WaveletFeatures",
        raw=dict(family="haar", mode="periodization", max_level=3), dump=False),
    _pp("pp_wavelet_pca", "wavelet_pca", "augmentation", "WaveletPCA",
        raw=dict(family="haar", mode="periodization", max_level=3, n_components=5.0), dump=False),
    _pp("pp_wavelet_svd", "wavelet_svd", "augmentation", "WaveletSVD",
        raw=dict(family="haar", mode="periodization", max_level=3, n_components=5.0), dump=False),
    _pp("pp_flex_pca", "flexible_pca", "feature_extraction", "FlexiblePCA",
        raw=dict(n_components=5.0), dump=False),
    _pp("pp_flex_svd", "flexible_svd", "feature_extraction", "FlexibleSVD",
        raw=dict(n_components=5.0), dump=False),
    # FCK static: raw + idiom both take the kernel orders/scales as
    # `alphas`/`sigmas` (kernel_size 5, orders [0,1,2], scales [1,2] = C++).
    _pp("pp_fck_static", "fck_static", "feature_extraction", "FCKStaticTransformer",
        raw=dict(kernel_size=5, alphas=[0.0, 1.0, 2.0], sigmas=[1.0, 2.0]),
        idiom=dict(kernel_size=5, alphas=[0.0, 1.0, 2.0], sigmas=[1.0, 2.0]),
        dump=False),
]

# ===========================================================================
# Augmenters (37 ops). Two param regimes (see module docstring):
#   * param-exposing raw fns → pass the C++ donor-bench value to both tiers,
#     dump-validatable against the C++ `out` buffer.
#   * param-fixed raw fns (several physical augmenters bake their params in) →
#     the raw fn's effective value is the binding truth; the idiomatic ctor is
#     given kwargs that reproduce it (so the gate holds). Not C++-comparable,
#     so dump=False.
# The dashboard names diverge from the bench-op ids for a few augmenters; the
# `dashboard_id` keeps the row keyed as the fixture CSV expects.
# ===========================================================================
AUG_SPECS: list[DonorOpSpec] = [
    # --- param-exposing (dump-validatable) ---------------------------------
    _aug("aug_gaussian_noise", "aug_gaussian_noise", "GaussianAdditiveNoise",
         raw=dict(sigma=0.01)),
    _aug("aug_multiplicative_noise", "aug_multiplicative_noise", "MultiplicativeNoise",
         raw=dict(sigma_gain=0.05)),
    _aug("aug_spike_noise", "aug_spike_noise", "SpikeNoise",
         raw=dict(n_spikes_min=1, n_spikes_max=3, amplitude_min=-0.5, amplitude_max=0.5)),
    _aug("aug_hetero_noise", "aug_hetero_noise", "HeteroscedasticNoiseAugmenter",
         raw=dict(noise_base=1e-3, noise_signal_dep=5e-3)),
    _aug("aug_linear_drift", "aug_linear_drift", "LinearBaselineDrift",
         raw=dict(offset_min=-0.1, offset_max=0.1, slope_min=-0.001, slope_max=0.001)),
    _aug("aug_poly_drift", "aug_poly_drift", "PolynomialBaselineDrift",
         raw=dict(degree=2, coeff_min=[-0.1, -0.05, -0.033], coeff_max=[0.1, 0.05, 0.033])),
    _aug("aug_path_length", "aug_path_length", "PathLengthAugmenter",
         raw=dict(path_length_std=0.05, min_path_length=0.5)),
    _aug("aug_wavelength_shift", "aug_wavelength_shift", "WavelengthShift",
         raw=dict(shift_lo=-2.0, shift_hi=2.0), wl=True),
    _aug("aug_wavelength_stretch", "aug_wavelength_stretch", "WavelengthStretch",
         raw=dict(stretch_lo=0.98, stretch_hi=1.02), wl=True),
    _aug("aug_local_warp", "aug_local_warp", "LocalWarpAugmenter",
         raw=dict(n_control_points=5, max_shift=2.0), wl=True),
    _aug("aug_band_perturb", "aug_band_perturb", "BandPerturbationAugmenter",
         raw=dict(n_bands=3, bw_lo=5, bw_hi=15, gain_lo=0.9, gain_hi=1.1,
                  offset_lo=-0.05, offset_hi=0.05)),
    _aug("aug_band_mask", "aug_band_mask", "BandMasking",
         raw=dict(n_bands_lo=1, n_bands_hi=3, bw_lo=5, bw_hi=15, mode="zero")),
    _aug("aug_channel_dropout", "aug_channel_dropout", "ChannelDropout",
         raw=dict(dropout_prob=0.1, mode="zero")),
    _aug("aug_gauss_jitter", "aug_gauss_jitter", "GaussianJitter",
         raw=dict(sigma_lo=0.5, sigma_hi=2.0, kernel_width=5)),
    _aug("aug_unsharp_mask", "aug_unsharp_mask", "UnsharpMask",
         raw=dict(amount_lo=0.5, amount_hi=1.5, sigma=1.0, kernel_width=5)),
    _aug("aug_magnitude_warp", "aug_magnitude_warp", "MagnitudeWarp",
         raw=dict(n_control_points=5, gain_lo=0.9, gain_hi=1.1), wl=True),
    _aug("aug_local_clip", "aug_local_clip", "LocalClip",
         raw=dict(n_regions=2, width_lo=3, width_hi=10)),
    # The Phase-17 augmenters (mixup … moisture) are rolled up into the
    # fixture's single `aug_phase17` aggregate, but get INDIVIDUAL dashboard
    # rows here (on_dashboard default True): C++/binding/nirs4all timing is
    # per-op, raw↔idiom parity is bit-exact, nirs4all is timing-only, and
    # correctness provenance is the passing aug_phase17 fixture. `build_landing`
    # hides the aug_phase17 aggregate so the individual rows are the surface.
    _aug("aug_mixup", "aug_mixup", "MixupAugmenter", raw=dict(alpha=0.4)),
    _aug("aug_local_mixup", "aug_local_mixup", "LocalMixupAugmenter",
         raw=dict(alpha=0.4, k_neighbors=5)),
    # scatter_sim: the four ctor positions map straight onto the ABI create
    # args, so the C++ positional values (0.8, 1.2, -0.05, 0.05) land on
    # (a_low, a_high, b_low, b_high). The dump check proves this mapping.
    _aug("aug_scatter_sim", "aug_scatter_sim", "ScatterSimulationMSC",
         raw=dict(a_low=0.8, a_high=1.2, b_low=-0.05, b_high=0.05)),
    # --- param-fixed raw fns: idiom reproduces the raw effective values. The
    # C++ donor bench was ALIGNED to these wrapper params (so the C++ tier and
    # the binding tier time one identical config — Codex review), which makes
    # them dump-validatable (raw ≡ C++); the dump test asserts it. ----------
    _aug("aug_particle_size", "aug_particle_size", "ParticleSizeAugmenter", wl=True),
    _aug("aug_emsc_distort", "aug_emsc_distort", "EMSCDistortionAugmenter", wl=True),
    _aug("aug_detector_rolloff", "aug_detector_rolloff", "DetectorRollOffAugmenter", wl=True),
    _aug("aug_stray_light", "aug_stray_light", "StrayLightAugmenter", wl=True),
    _aug("aug_edge_curve", "aug_edge_curve", "EdgeCurvatureAugmenter", wl=True),
    _aug("aug_truncated_peak", "aug_truncated_peak", "TruncatedPeakAugmenter", wl=True),
    _aug("aug_edge_artifacts", "aug_edge_artifacts", "EdgeArtifactsAugmenter", wl=True),
    _aug("aug_spline_smooth", "aug_spline_smooth", "SplineSmoothingAugmenter"),
    _aug("aug_spline_x_perturb", "aug_spline_x_perturb", "SplineXPerturbationAugmenter"),
    _aug("aug_spline_y_perturb", "aug_spline_y_perturb", "SplineYPerturbationAugmenter"),
    _aug("aug_rotate_translate", "aug_rotate_translate", "RotateTranslateAugmenter"),
    _aug("aug_random_x_op", "aug_random_x_op", "RandomXOperation"),
    # param-fixed (idiom kwargs reproduce the raw fixed configuration):
    _aug("aug_batch_effect", "aug_batch_effect", "BatchEffectAugmenter", wl=True,
         idiom=dict(offset_std=0.02, slope_std=0.01, gain_std=0.03, variation_scope=0)),
    _aug("aug_instrument_broaden", "aug_instrument_broaden", "InstrumentalBroadeningAugmenter",
         wl=True,
         idiom=dict(fwhm=3.0, use_fwhm_range=False, fwhm_low=3.0, fwhm_high=8.0, variation_scope=0)),
    _aug("aug_dead_band", "aug_dead_band", "DeadBandAugmenter",
         idiom=dict(n_bands=1, width_low=5, width_high=10, noise_std=0.05,
                    probability=1.0, variation_scope=0)),
    _aug("aug_temperature", "aug_temperature", "TemperatureAugmenter", wl=True,
         idiom=dict(temperature_delta=5.0, use_temp_range=False, temp_low=-5.0, temp_high=5.0,
                    enable_shift=True, enable_intensity=True, enable_broadening=True,
                    region_specific=True)),
    _aug("aug_moisture", "aug_moisture", "MoistureAugmenter", wl=True,
         idiom=dict(water_activity_delta=0.1, use_aw_range=False, aw_low=0.0, aw_high=1.0,
                    reference_water_activity=0.5, free_water_fraction=0.3, bound_water_shift=25.0,
                    moisture_content=0.10, enable_shift=True, enable_intensity=True)),
]

# ===========================================================================
# Filters (12 ops). Timed op is fit+apply; the binding returns a keep-mask.
# X-outlier variants share one ABI op selected by `method`; y-outlier operate
# on the target vector. C++ params from the X_OUTLIER / Y_OUTLIER macros.
# ===========================================================================
def _xout(bench_id, method, n_components) -> DonorOpSpec:
    kw = dict(method=method, use_threshold=False, threshold=0.0,
              n_components=n_components, contamination=0.1, seed=42,
              n_estimators=100, max_samples=256)
    return DonorOpSpec(
        bench_id=bench_id, dashboard_id=bench_id, category="filter", inputs=("X",),
        parity="mask", raw_fn="x_outlier_filter", raw_kwargs=kw,
        idiom_module="filters", idiom_cls="XOutlierFilter", idiom_kwargs=dict(kw),
    )


def _yout(bench_id, method) -> DonorOpSpec:
    kw = dict(method=method, threshold=3.0, lower_percentile=5.0, upper_percentile=95.0)
    return DonorOpSpec(
        bench_id=bench_id, dashboard_id=bench_id, category="filter", inputs=("y",),
        parity="mask", raw_fn="y_outlier_filter", raw_kwargs=kw,
        idiom_module="filters", idiom_cls="YOutlierFilter", idiom_kwargs=dict(kw),
    )


FILTER_SPECS: list[DonorOpSpec] = [
    _xout("filter_x_outlier_mahalanobis", "mahalanobis", 0),
    _xout("filter_x_outlier_robust_mahalanobis", "robust_mahalanobis", 0),
    _xout("filter_x_outlier_pca_residual", "pca_residual", 5),
    _xout("filter_x_outlier_pca_leverage", "pca_leverage", 5),
    _xout("filter_x_outlier_isolation_forest", "isolation_forest", 0),
    _xout("filter_x_outlier_lof", "lof", 0),
    _yout("filter_y_outlier_iqr", "iqr"),
    _yout("filter_y_outlier_zscore", "zscore"),
    _yout("filter_y_outlier_percentile", "percentile"),
    _yout("filter_y_outlier_mad", "mad"),
    DonorOpSpec(
        bench_id="filter_leverage", dashboard_id="filter_leverage", category="filter",
        inputs=("X",), parity="mask", raw_fn="high_leverage_filter",
        raw_kwargs=dict(method="hat", threshold_multiplier=3.0, absolute_threshold=None,
                        n_components=0, center=True),
        idiom_module="filters", idiom_cls="HighLeverageFilter",
        idiom_kwargs=dict(method="hat", threshold_multiplier=3.0, absolute_threshold=None,
                          n_components=0, center=True),
    ),
    DonorOpSpec(
        bench_id="filter_quality", dashboard_id="filter_quality", category="filter",
        inputs=("X",), parity="mask", raw_fn="spectral_quality_filter",
        raw_kwargs=dict(max_nan_ratio=0.1, max_zero_ratio=0.5, min_variance=1e-6,
                        max_value=None, min_value=None, check_inf=True),
        idiom_module="filters", idiom_cls="SpectralQualityFilter",
        idiom_kwargs=dict(max_nan_ratio=0.1, max_zero_ratio=0.5, min_variance=1e-6,
                          max_value=None, min_value=None, check_inf=True),
    ),
]

# ===========================================================================
# Splitters (9 ops). Timed op is one split()/split_fold(0). The raw binding
# exposes only 3 splitters; the other 6 are idiomatic-only (python_tier1 → —).
# C++ params from the split_* create() rows.
# ===========================================================================
SPLIT_SPECS: list[DonorOpSpec] = [
    DonorOpSpec(
        bench_id="split_kennard_stone", dashboard_id="split_kennard_stone", category="split",
        inputs=("X",), parity="indices", raw_fn="kennard_stone", raw_kwargs=dict(test_size=0.25),
        idiom_module="splitters", idiom_cls="KennardStoneSplitter", idiom_kwargs=dict(test_size=0.25),
    ),
    DonorOpSpec(
        bench_id="split_spxy", dashboard_id="split_spxy", category="split",
        inputs=("X", "y"), parity="indices", raw_fn="spxy", raw_kwargs=dict(test_size=0.25),
        idiom_module="splitters", idiom_cls="SPXYSplitter", idiom_kwargs=dict(test_size=0.25),
    ),
    DonorOpSpec(
        bench_id="split_kbins_stratified", dashboard_id="split_kbins_stratified", category="split",
        inputs=("y",), parity="indices", raw_fn="kbins_stratified",
        raw_kwargs=dict(test_size=0.25, seed=42, n_bins=5, strategy="uniform"),
        idiom_module="splitters", idiom_cls="KBinsStratifiedSplitter",
        idiom_kwargs=dict(test_size=0.25, seed=42, n_bins=5, strategy="uniform"),
    ),
    DonorOpSpec(
        bench_id="split_spxy_fold", dashboard_id="split_spxy_fold", category="split",
        inputs=("X", "y"), parity="indices", raw_reason=NO_RAW,
        idiom_module="splitters", idiom_cls="SPXYFoldSplitter",
        idiom_kwargs=dict(n_splits=5, y_metric="euclidean"),
    ),
    DonorOpSpec(
        bench_id="split_spxy_g_fold", dashboard_id="split_spxy_g_fold", category="split",
        inputs=("X", "y", "groups"), parity="indices", raw_reason=NO_RAW,
        idiom_module="splitters", idiom_cls="SPXYGroupFoldSplitter",
        idiom_kwargs=dict(n_splits=5, y_metric="euclidean", aggregation="mean"),
    ),
    DonorOpSpec(
        bench_id="split_kmeans", dashboard_id="split_kmeans", category="split",
        inputs=("X",), parity="indices", raw_reason=NO_RAW,
        idiom_module="splitters", idiom_cls="KMeansSplitter",
        idiom_kwargs=dict(test_size=0.25, seed=42, max_iter=100),
    ),
    DonorOpSpec(
        bench_id="split_bsgk", dashboard_id="split_bsgk", category="split",
        inputs=("y", "groups"), parity="indices", raw_reason=NO_RAW,
        idiom_module="splitters", idiom_cls="BinnedStratifiedGroupKFoldSplitter",
        idiom_kwargs=dict(n_splits=5, n_bins=5, strategy="uniform", shuffle=True, seed=42),
    ),
    DonorOpSpec(
        bench_id="split_systematic_circular", dashboard_id="split_systematic_circular",
        category="split", inputs=("y",), parity="indices", raw_reason=NO_RAW,
        idiom_module="splitters", idiom_cls="SystematicCircularSplitter",
        idiom_kwargs=dict(test_size=0.25, seed=42),
    ),
    DonorOpSpec(
        bench_id="split_split_splitter", dashboard_id="split_split_splitter", category="split",
        inputs=("X",), parity="indices", raw_reason=NO_RAW,
        idiom_module="splitters", idiom_cls="SPlitSplitter",
        idiom_kwargs=dict(test_size=0.25, seed=42),
    ),
]

REGISTRY: dict[str, DonorOpSpec] = {
    s.bench_id: s for s in (*PP_SPECS, *AUG_SPECS, *FILTER_SPECS, *SPLIT_SPECS)
}


def spec_for(bench_id: str) -> DonorOpSpec:
    return REGISTRY[bench_id]


def all_specs() -> list[DonorOpSpec]:
    return list(REGISTRY.values())


# ---------------------------------------------------------------------------
# Execution — build inputs, invoke each tier, reduce output for parity.
# n4m is imported lazily so the registry can be introspected without libn4m.
# ---------------------------------------------------------------------------
def build_inputs(spec: DonorOpSpec, n: int, p: int, seed: int) -> dict[str, Any]:
    """Build exactly the inputs the C++ donor bench feeds this op."""
    inp: dict[str, Any] = {"X": make_X(n, p, seed)}
    if "y" in spec.inputs:
        inp["y"] = make_y(n, seed)
    if "wl" in spec.inputs:
        inp["wl"] = wavelength_axis(p)
    if "groups" in spec.inputs:
        inp["groups"] = make_groups(n)
    return inp


def _reduce(spec: DonorOpSpec, out: Any) -> np.ndarray:
    """Flatten an op output to a 1-D vector comparable across tiers."""
    if spec.parity == "indices":
        train, test = out
        return np.concatenate([np.asarray(train, dtype=np.int64),
                               np.asarray(test, dtype=np.int64)])
    return np.asarray(out, dtype=np.float64).ravel()


def run_raw(spec: DonorOpSpec, inp: dict[str, Any], seed: int) -> np.ndarray:
    """Invoke the raw `n4m.python` tier and reduce its output."""
    if spec.raw_fn is None:
        raise RuntimeError(spec.raw_reason or NO_RAW)
    from n4m import python as npy
    fn = getattr(npy, spec.raw_fn)
    X = inp["X"]
    kw = dict(spec.raw_kwargs)
    if spec.category == "aug":
        if "wl" in spec.inputs:
            kw["wavelengths"] = inp["wl"]
        out = fn(X, seed=seed, **kw)
    elif spec.category == "pp":
        if spec.bench_id == "pp_osc":
            out = fn(X, inp["y"], **kw)
        elif spec.bench_id == "pp_resampler":
            wl = inp["wl"]
            out = fn(X, wl, wl, **kw)
        else:
            out = fn(X, **kw)
    elif spec.category == "filter":
        res = fn(inp["y"] if "y" in spec.inputs else X, **kw)
        out = res[0] if isinstance(res, tuple) else res
    elif spec.category == "split":
        # Splitters take the target as an n×1 matrix (as_f64_2d turns a 1-D
        # vector into 1×n, which the ABI rejects as a shape mismatch).
        y_col = inp["y"].reshape(-1, 1) if "y" in inp else None
        if spec.bench_id == "split_spxy":
            out = fn(X, y_col, **kw)
        elif spec.bench_id == "split_kbins_stratified":
            out = fn(y_col, **kw)
        else:
            out = fn(X, **kw)
    else:  # pragma: no cover - guarded by __post_init__
        raise ValueError(spec.category)
    return _reduce(spec, out)


def run_idiom(spec: DonorOpSpec, inp: dict[str, Any], seed: int) -> np.ndarray:
    """Invoke the idiomatic `n4m.sklearn` tier and reduce its output."""
    if spec.idiom_cls is None:
        raise RuntimeError(spec.idiom_reason or NO_ESTIMATOR)
    import importlib
    mod = importlib.import_module(f"n4m.sklearn.{spec.idiom_module}")
    cls = getattr(mod, spec.idiom_cls)
    X = inp["X"]
    kw = dict(spec.idiom_kwargs)
    if spec.category == "aug":
        if "wl" in spec.inputs:
            kw["wavelengths"] = inp["wl"]
        est = cls(seed=seed, **kw)
        out = est.fit_transform(X)
    elif spec.category == "pp":
        if spec.bench_id == "pp_resampler":
            # Identity resample (source axis == target axis), matching the raw
            # tier; Resampler must be built with a target grid up front.
            est = cls(target_wavelengths=inp["wl"], **kw)
            out = est.fit(source_wavelengths=inp["wl"]).transform(X)
        elif spec.bench_id == "pp_osc":
            out = cls(**kw).fit(X, inp["y"]).transform(X)
        else:
            out = cls(**kw).fit_transform(X)
    elif spec.category == "filter":
        est = cls(**kw)
        res = est.fit_apply(inp["y"] if "y" in spec.inputs else X)
        out = res[0] if isinstance(res, tuple) else res  # (mask, FilterStats)
    elif spec.category == "split":
        est = cls(**kw)
        y_col = inp["y"].reshape(-1, 1) if "y" in inp else None
        if "groups" in spec.inputs:
            res = est.split(X, y_col, inp["groups"]) if "X" in spec.inputs \
                else est.split(y_col, inp["groups"])
        elif "y" in spec.inputs and "X" in spec.inputs:
            res = est.split(X, y_col)
        elif "y" in spec.inputs:
            res = est.split(y_col)
        else:
            res = est.split(X)
        out = _first_fold(res)
    else:  # pragma: no cover
        raise ValueError(spec.category)
    return _reduce(spec, out)


def _first_fold(res: Any) -> tuple:
    """Normalise a splitter's return to a single (train, test) pair.

    Fold splitters may return a list/iterator of folds; we take fold 0 to match
    the C++ donor bench (`split_fold(0)`). Single splitters return the pair.
    """
    if isinstance(res, tuple) and len(res) == 2 and np.ndim(res[0]) == 1:
        return res
    folds = list(res)
    return folds[0]


# ===========================================================================
# Layer 3 — external real-library reference (nirs4all, the universal donor
# baseline). One `ref_python_nirs4all` reference per on-dashboard op; the
# `nirs4all` operators map ~1:1 onto the donor methods (the n4m idiomatic
# classes were modelled on them). Per Codex this is a THIRD independent API
# mapping — nirs4all ctor kwargs / invoke / output shape are NOT assumed from
# the class name. Stochastic augmenters get timing-only (`value_parity=False`)
# because nirs4all's numpy RNG ≠ n4m's PCG64.
# ===========================================================================
@dataclass(frozen=True)
class ReferenceSpec:
    library: str          # "nirs4all"
    module: str           # nirs4all.operators submodule
    cls: str              # nirs4all class name
    kwargs: dict[str, Any]
    invoke: str           # "transform" | "split" | "filter_mask" | "augment"
    value_parity: bool    # False ⇒ timing-only (stochastic; RNG differs)
    tol_rel: float = 1e-6
    note: str = ""


# idiom class name → nirs4all class name (only where they differ).
_N4_CLASS: dict[str, str] = {
    "SNV": "StandardNormalVariate", "RNV": "RobustStandardNormalVariate",
    "LSNV": "LocalStandardNormalVariate", "MSC": "MultiplicativeScatterCorrection",
    "EMSC": "ExtendedMultiplicativeScatterCorrection", "AsLS": "ASLSBaseline",
    "IAsLS": "IASLS", "BaselineCenter": "Baseline",
    "SPXYFoldSplitter": "SPXYFold", "SPXYGroupFoldSplitter": "SPXYGFold",
    "BinnedStratifiedGroupKFoldSplitter": "BinnedStratifiedGroupKFold",
    # augmenters (timing-only, but use the real op for representative timing)
    "LocalWarpAugmenter": "LocalWavelengthWarp",
    "BandPerturbationAugmenter": "BandPerturbation", "GaussianJitter": "GaussianSmoothingJitter",
    "UnsharpMask": "UnsharpSpectralMask", "MagnitudeWarp": "SmoothMagnitudeWarp",
    "LocalClip": "LocalClipping", "SplineSmoothingAugmenter": "Spline_Smoothing",
    "SplineXPerturbationAugmenter": "Spline_X_Perturbations",
    "SplineYPerturbationAugmenter": "Spline_Y_Perturbations",
    "RotateTranslateAugmenter": "Rotate_Translate", "RandomXOperation": "Random_X_Operation",
}
_N4_MODULE = {"pp": "transforms", "filter": "filters", "split": "splitters",
              "aug": "augmentation"}
# Per-op nirs4all ctor kwargs where the idiom kwarg names/values don't carry
# over (signature-filtering would silently drop them — Codex's #1 risk).
_N4_KWARGS: dict[str, dict[str, Any]] = {
    "filter_x_outlier_mahalanobis": dict(method="mahalanobis", contamination=0.1),
    "filter_x_outlier_robust_mahalanobis": dict(method="robust_mahalanobis", contamination=0.1),
    "filter_x_outlier_pca_residual": dict(method="pca_residual", n_components=5, contamination=0.1),
    "filter_x_outlier_pca_leverage": dict(method="pca_leverage", n_components=5, contamination=0.1),
    "filter_x_outlier_isolation_forest": dict(method="isolation_forest", contamination=0.1, random_state=42),
    "filter_x_outlier_lof": dict(method="lof", contamination=0.1),
    "filter_y_outlier_iqr": dict(method="iqr", threshold=3.0, lower_percentile=5.0, upper_percentile=95.0),
    "filter_y_outlier_zscore": dict(method="zscore", threshold=3.0, lower_percentile=5.0, upper_percentile=95.0),
    "filter_y_outlier_percentile": dict(method="percentile", threshold=3.0, lower_percentile=5.0, upper_percentile=95.0),
    "filter_y_outlier_mad": dict(method="mad", threshold=3.0, lower_percentile=5.0, upper_percentile=95.0),
}
# Ops whose value-parity vs live nirs4all is not meaningful (seeded RNG that
# differs from n4m, or a known scipy edge-handling divergence). Timing still
# runs; the cross-parity verdict is recorded as not-applicable with a reason.
_N4_TIMING_ONLY: dict[str, str] = {
    "filter_x_outlier_isolation_forest": "seeded RNG differs from n4m",
    "filter_x_outlier_lof": "approximate-NN ordering differs from n4m",
    "split_kmeans": "seeded k-means init differs from n4m",
    "split_bsgk": "shuffle RNG differs from n4m",
    "split_systematic_circular": "seeded start differs from n4m",
    "split_split_splitter": "seeded twinning differs from n4m",
    "split_kbins_stratified": "seeded bin shuffle differs from n4m",
    # Different operation / API / output-dimensionality than n4m — a value
    # comparison would be apples-to-oranges, so time only.
    "pp_normalize": "nirs4all Normalize is per-sample feature_range; n4m is per-feature min-max",
    "pp_range_disc": "nirs4all RangeDiscretizer takes a bin count, n4m takes explicit edges",
    "pp_crop": "crop end-index convention differs (output width differs)",
    "pp_derivate": "derivative edge convention differs (output width differs)",
    "pp_wavelet_features": "wavelet feature layout differs (output width differs)",
    "pp_wavelet_pca": "wavelet-PCA component count differs (output width differs)",
    "pp_wavelet_svd": "wavelet-SVD component count differs (output width differs)",
    "pp_fck_static": "FCK kernel-set convention differs (output width differs)",
}

# Deterministic ops that DO compute the same operation but diverge from
# upstream nirs4all at the bench config (different defaults / edge handling /
# numerics). These record an honest `divergent` reference-parity verdict; the
# allowlist documents WHY so a divergence here is not mistaken for a regression
# (the integrity test fails on any UNEXPECTED divergence).
_N4_EXPECTED_DIVERGENCE: dict[str, str] = {
    "pp_savgol": "Savitzky-Golay edge mode differs (nirs4all uses the scipy default)",
    "pp_airpls": "airPLS iteration/lambda schedule differs from n4m",
    "pp_modpoly": "ModPoly iteration schedule differs from n4m",
    "pp_imodpoly": "I-ModPoly iteration schedule differs from n4m",
    "pp_snip": "SNIP clipping iteration differs from n4m",
    "pp_rolling_ball": "rolling-ball window handling differs from n4m",
    "pp_wavelet_denoise": "wavelet denoise thresholding differs from n4m",
    "pp_msc": "MSC reference/scale formulation differs from n4m",
    "pp_emsc": "EMSC polynomial basis formulation differs from n4m",
    "pp_haar": "Haar coefficient layout/normalisation differs from n4m",
    "pp_wavelet": "wavelet coefficient layout/normalisation differs from n4m",
    "pp_flex_pca": "PCA sign/component convention differs from n4m",
    "pp_flex_svd": "SVD sign/component convention differs from n4m",
    "filter_x_outlier_robust_mahalanobis": "robust covariance estimator differs from n4m",
    "split_spxy_fold": "fold-0 assignment differs from n4m",
    "split_spxy_g_fold": "group fold-0 assignment differs from n4m",
}

# Donor methods deliberately NOT timed at any layer (parity-only). They are not
# in `n4m_donor_bench --list` (the registry == --list), so they never reach the
# binding or reference pipelines; the dashboard keeps their fixture parity row:
#   - filter_composite  — needs externally-owned sub-filters (no standalone op)
#   - pp_epo            — a synthetic spectrum yields a degenerate EPO projection
#   - filter_y_outlier_constant_input / _nan_exclusion — validation-only scenarios
#
# Second references (scipy / sklearn / R / Julia) are intentionally NOT added:
# nirs4all is the universal donor baseline and already wraps scipy/sklearn for
# the common ops (savgol→scipy, pca/isolation_forest/lof→sklearn), so a separate
# scipy/sklearn column would mostly measure "without the nirs4all wrapper", not
# an independent algorithm. The only future exception worth adding is a TRUE
# independent implementation (e.g. R `prospectr` Kennard-Stone/SPXY).
PARITY_ONLY_DONOR_METHODS = (
    "filter_composite", "pp_epo",
    "filter_y_outlier_constant_input", "filter_y_outlier_nan_exclusion",
)


def reference_for(spec: DonorOpSpec) -> Optional[ReferenceSpec]:
    """Build the nirs4all ReferenceSpec for an on-dashboard donor op."""
    if not spec.on_dashboard:
        return None
    if spec.idiom_cls is None:
        return None
    n4_cls = _N4_CLASS.get(spec.idiom_cls, spec.idiom_cls)
    kwargs = _N4_KWARGS.get(spec.bench_id, dict(spec.idiom_kwargs))
    invoke = {"pp": "transform", "filter": "filter_mask",
              "split": "split", "aug": "augment"}[spec.category]
    if spec.category == "aug":
        value_parity, note = False, "stochastic — nirs4all RNG differs from n4m PCG64"
    elif spec.bench_id in _N4_TIMING_ONLY:
        value_parity, note = False, _N4_TIMING_ONLY[spec.bench_id]
    else:
        value_parity, note = True, ""
    return ReferenceSpec(library="nirs4all", module=_N4_MODULE[spec.category],
                         cls=n4_cls, kwargs=kwargs, invoke=invoke,
                         value_parity=value_parity, note=note)


def _filtered_kwargs(cls_obj, kwargs: dict) -> dict:
    """Keep only kwargs the nirs4all ctor accepts (drop n4m-only params)."""
    import inspect
    try:
        params = inspect.signature(cls_obj.__init__).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def run_reference(spec: DonorOpSpec, ref: ReferenceSpec, inp: dict[str, Any],
                  seed: int) -> np.ndarray:
    """Invoke the nirs4all reference and reduce its output (lazy import)."""
    import importlib
    mod = importlib.import_module(f"nirs4all.operators.{ref.module}")
    cls = getattr(mod, ref.cls)
    X = inp["X"]
    kw = _filtered_kwargs(cls, ref.kwargs)
    if ref.invoke == "transform":
        if spec.bench_id == "pp_osc":
            est = cls(**kw)
            out = est.fit(X, inp["y"]).transform(X) if "y" in inp else est.fit_transform(X)
        else:
            out = cls(**kw).fit_transform(X)
    elif ref.invoke == "augment":
        for seed_kw in ("random_state", "seed"):
            if seed_kw in _filtered_kwargs(cls, {seed_kw: seed}):
                kw[seed_kw] = seed
                break
        est = cls(**kw)
        # Spectral/edge augmenters take the wavelength axis as a fit param
        # (no ctor slot); others reject the kwarg, so fall back.
        if "wl" in spec.inputs:
            try:
                out = est.fit_transform(X, wavelengths=inp["wl"])
            except TypeError:
                out = est.fit_transform(X)
        else:
            out = est.fit_transform(X)
    elif ref.invoke == "filter_mask":
        # nirs4all filters always take X for fit/get_mask; y-outlier ones also
        # take the target. get_mask returns a boolean keep-mask over samples.
        y = inp["y"] if "y" in spec.inputs else None
        est = cls(**kw)
        est.fit(X, y)
        out = np.asarray(est.get_mask(X, y)).astype(np.float64)
    elif ref.invoke == "split":
        # sklearn-style: split(X, y, groups), y as 1-D; take fold 0.
        est = cls(**kw)
        y = inp.get("y")
        groups = inp.get("groups")
        folds = list(est.split(X, y, groups))
        out = folds[0]
    else:  # pragma: no cover
        raise ValueError(ref.invoke)
    return _reduce(spec, out)

