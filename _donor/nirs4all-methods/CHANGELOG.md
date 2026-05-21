# Changelog

All notable changes to nirs4all-methods will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to dual semver: **project version** in `N4M_PROJECT_VERSION_*` and **ABI version** in `N4M_ABI_VERSION_*` (both in `cpp/include/n4m/n4m_version.h`).

## [Unreleased]

## [0.1.0+abi.1.9.0] — Phase 21.5-A pre-Phase-22 ABI cleanup (M2 splitter rename + M3 Y-outlier fit/apply split)

### Changed (source-breaking)

Two ABI corrections required before Phase 22 (Python binding) freezes the
surface for downstream consumers. Source-breaking but zero external
consumers exist today (no bindings shipped yet), so this minor-version
bump 1.8.0 → 1.9.0 is judged adequate.

- **M2 (Splitter opaque-handle naming).** Renamed nine Phase 11 splitter
  typedefs to use the project-wide `_handle_t` suffix:
  - `n4m_split_kennard_stone_t`            → `n4m_split_kennard_stone_handle_t`
  - `n4m_split_spxy_t`                     → `n4m_split_spxy_handle_t`
  - `n4m_split_spxy_fold_t`                → `n4m_split_spxy_fold_handle_t`
  - `n4m_split_spxy_g_fold_t`              → `n4m_split_spxy_g_fold_handle_t`
  - `n4m_split_kmeans_t`                   → `n4m_split_kmeans_handle_t`
  - `n4m_split_kbins_stratified_t`         → `n4m_split_kbins_stratified_handle_t`
  - `n4m_split_binned_strat_group_kfold_t` → `n4m_split_binned_strat_group_kfold_handle_t`
  - `n4m_split_systematic_circular_t`      → `n4m_split_systematic_circular_handle_t`
  - `n4m_split_split_splitter_t`           → `n4m_split_split_splitter_handle_t`

  `n4m_split_result_t` is intentionally retained: it is a value-typed
  struct (heap-owned index arrays), not an opaque handle.

  Also renamed two splitter enums for consistency:
  - `n4m_split_bin_strategy_t` (`N4M_SPLIT_BIN_*`) →
    `n4m_split_kbins_strategy_t` (`N4M_SPLIT_KBINS_*` — already used by
    the engine internals)
  - `n4m_split_group_agg_t`  (`N4M_SPLIT_GROUP_AGG_*`) →
    `n4m_split_aggregation_t` (`N4M_SPLIT_AGGREGATION_*`)

  `n4m_split_y_metric_t` was already correctly named — kept as-is.

- **M3 (Y-outlier filter fit/apply split).** Replaced the single-call
  `n4m_filter_y_outlier_fit_get_mask` with the classical sklearn split,
  matching the Phase 14 leverage / quality / composite filter contracts:
  - `n4m_filter_y_outlier_fit(handle, y, n)` — learns the bounds.
  - `n4m_filter_y_outlier_apply(handle, y, n, mask, stats)` — writes
    the keep-mask and stats using the previously fitted bounds. Returns
    `N4M_ERR_NOT_FITTED` when called on an unfitted handle.
  - `n4m_filter_y_outlier_is_fitted(handle, out)` — reports 0/1.

  Net symbol delta: `-1, +3` (one rename, two additions). The frozen
  NumPy reference `n4m_parity_filters_ref.y_outlier.y_outlier_fit_get_mask`
  is unchanged — it stays a single-call API on the reference side.

### Build / version

- ABI 1.8.0 → 1.9.0.
- `cpp/abi/expected_symbols_*.txt` regenerated.

## [0.1.0+abi.1.8.0] — Phases 6, 13, 15-18 parallel batch (wavelets, X-outlier filter, augmenters)

### Added

This release lands six worktrees developed in parallel under the Phase
15-18 augmenter ABI contract. All additions are surface-additive — no
existing symbol changes signature, no behavioural regression on the 171
prior tests. ABI minor bumps from 1.7.0 to 1.8.0.

- **Phase 6 — Wavelets & denoising (6 ops, 27 symbols + 4 enums)**:
  Wavelet (single-level DWT), Haar, WaveletDenoise (multi-level VisuShrink),
  WaveletFeatures (per-level statistical features), WaveletPCA and
  WaveletSVD (stateful DWT-flatten + PCA / SVD). Shared filter banks for
  haar / db4 / sym4 / coif1 under `core/common/wavelet_kernels.{c,h}`. 12
  parity tests, bit-exact match against the frozen
  `n4m_parity_wavelets_ref` (pywt-based) reference.
- **Phase 13 — XOutlierFilter (1 op, 6 methods, 5 symbols + 1 enum)**:
  Mahalanobis, RobustMahalanobis (MinCovDet shrinkage), PCA-Residual
  (Q-statistic), PCA-Leverage (Hotelling T²), IsolationForest, LOF. Adds
  vendored implementations under `core/filters/_vendored/` and a shared
  `core/common/pca_helper.{c,h}` (delegates to the existing `svd.{c,h}`
  from Phase 9). 12 parity tests.
- **Phase 15 — Augmenters: noise + drift (7 ops, 21 symbols)**: opens the
  new `n4m_aug_*` ABI category. GaussianNoise, MultiplicativeNoise,
  SpikeNoise, HeteroscedasticNoise, LinearDrift, PolynomialDrift,
  PathLength. Bit-exact 1e-15 PCG64 parity against the frozen
  `n4m_parity_augmenters_ref` (NumPy Generator) reference. 14 tests.
- **Phase 16 — Augmenters: wavelength + spectral (10 ops, 30 symbols)**:
  WavelengthShift, WavelengthStretch, LocalWarp, BandPerturbation,
  BandMasking, ChannelDropout, GaussJitter, UnsharpMask, MagnitudeWarp,
  LocalClip. Shared `spectral_common.{c,h}` helper. 20 tests.
- **Phase 17 — Augmenters: mixup + physical + environmental (10 ops, 30
  symbols)**: Mixup, LocalMixup, ScatterSimulationMSC, ParticleSize,
  EMSCDistortion, BatchEffect, InstrumentalBroadening, DeadBand,
  Temperature, Moisture. Shared `aug_rng_utils.{c,h}` (NumPy-style uniform
  / normal / integers / beta / permutation helpers) and `aug_interp`
  (linear interpolation). 25 tests.
- **Phase 18 — Augmenters: edge + splines + random (12 ops, 36 symbols)**:
  DetectorRollOff, StrayLight, EdgeCurvature, TruncatedPeak,
  EdgeArtifacts (combined wrapper with 4-bit flag selector), SplineSmooth,
  SplineXPerturbations, SplineYPerturbations, SplineXSimplification
  (v2-deferred — returns N4M_ERR_NOT_IMPLEMENTED), SplineCurveSimplification
  (v2-deferred), RotateTranslate, RandomXOperation. Shared B-spline core
  under `core/common/bspline.{c,h}`. Adds the internal-only
  `c_api/rng_state_internal.hpp` (shared by `c_api_rng.cpp` and the four
  augmenter wrappers). 12 tests.

### Changed

- `cpp/include/n4m/n4m.h` gains six new sections (§23-§28,
  one per integrated phase) plus extended ABI guard-rail
  `N4M_STATIC_ASSERT`s for the five new public enums. Previous §23 (ABI
  guard rails) renumbered to §29.
- `c_api_rng.cpp` refactored to include the new `rng_state_internal.hpp`
  rather than defining the public RNG handle struct inline. The four
  augmenter wrappers (`c_api_augmenters*.cpp`) now share that header
  instead of re-declaring the struct bit-identically per TU.
- `cpp/abi/expected_symbols_{linux,macos,windows}.txt` snapshot bumped
  from 251 to 400 entries. `.github/workflows/abi-check.yml` comment
  updated accordingly.

### Surface delta (per `nm -D --defined-only`)

- Symbols 251 → 400 (+149).
- Tests 171 → 266 (+95).
- ABI 1.7.0 → 1.8.0.

## [0.1.0+abi.1.7.0] — Phases 7-21 parallel batch (signal conversion, OSC/EPO, FlexiblePCA/SVD, resampling, splitters, filters, metrics, transfer metrics, FCK)

### Added

This release lands the ten Phase 7-21 worktrees that were developed in parallel
under the same ABI base. All additions are surface-additive — no existing
symbol changes signature, no behavioural regression on the 82 prior tests.
ABI minor bumps from 1.6.0 to 1.7.0.

- **Phase 7 — Signal conversion (5 stateless ops, 15 symbols)**: ToAbsorbance,
  FromAbsorbance, PercentToFraction, FractionToPercent, KubelkaMunk.
- **Phase 8 — Orthogonalization (2 stateful ops, 12 symbols)**: OSC (Wold
  1998) and EPO (Roger 2003) with fit/transform/inverse_transform contracts.
- **Phase 9 — Feature selection (2 stateful ops + shared Jacobi SVD, 12
  symbols)**: FlexiblePCA, FlexibleSVD, plus internal `n4m_svd_compact`
  (one-sided Jacobi with wide-matrix transpose fallback) used by Phases 9, 19.
- **Phase 10 — Resampling / cropping / target discretization (5 ops)**:
  Resampler (interpolation), CropTransformer, ResampleTransformer,
  IntegerKBinsDiscretizer, RangeDiscretizer (int32 output).
- **Phase 11 — Splitters (9 ops, NEW `n4m_split_*` category, 31 symbols)**:
  KennardStone, SPXY, SPXYFold, SPXYGFold (group-aware), KMeans, KBinsStratified,
  BinnedStratifiedGroupKFold, SystematicCircular, SPlit (data twinning), plus
  the shared `n4m_split_result_t` struct + `n4m_split_result_destroy`.
- **Phase 12 — YOutlierFilter (NEW `n4m_filter_*` category, 3 symbols)**:
  IQR / Z-score / Percentile / MAD detection methods, plus the shared
  `n4m_filter_stats_t` struct used by every filter operator.
- **Phase 14 — Filter meta operators (3 ops, 14 symbols)**: HighLeverageFilter
  (hat-matrix / PCA leverage), SpectralQualityFilter (6 row-level thresholds),
  CompositeFilter (ANY / ALL combinator over sub-filters).
- **Phase 19 — Signal type detection + NIRS metrics + T² / Q (11 symbols)**:
  - `n4m_signal_detect` auto-detects spectrum type (absorbance, reflectance,
    transmittance, Kubelka-Munk, log(1/R), log(1/T), and percent variants).
  - Eight regression metrics: RMSE, MAE, bias, SEP, RPD, RPIQ, R², NRMSE.
  - Two multivariate outlier statistics: Hotelling T², Q-residuals
    (Jackson-Mudholkar UCL).
- **Phase 20 — Transfer metrics (NEW `n4m_transfer_*` category, 1 symbol)**:
  `n4m_transfer_metrics_compute` returns nine dataset-alignment metrics
  (centroid_distance, CKA, Grassmann, RV, Procrustes, trustworthiness,
  spread_distance, EVR source/target).
- **Phase 21 — FCK static transformer (4 symbols, `n4m_pp_fck_static_*` +
  `n4m_fck_kernel_1d`)**: bank of fractional convolutional kernels with
  reflect-mode 1D convolution.

### Changed

- ABI minor bumps from 1.6.0 to 1.7.0. The public surface gains ~126 new
  symbols (126 → 252).
- `n4m.h` grows ten new sections (§13-§22) and renumbers the trailing ABI
  guard-rail block to §23. All additions are inside the same `extern "C"`
  block as the existing surface; no enum-size or struct-layout changes.
- Internal helper `n4m_svd_compact` (`cpp/src/core/common/svd.{c,h}`) is the
  single shared SVD used by Phases 9, 19 (Hotelling T² / Q-residuals).
  Phase 20's `transfer_metrics.c` keeps its private Jacobi-PCA — its API
  surface is incompatible with the shared helper and the parity reference
  was generated against it.

### Notes

- All ten worktrees declared their tests through `register_*_tests` entry
  points; the standalone test mains (`main_phase19.cpp`, `main_splitters.cpp`,
  `test_transfer_metrics_main.cpp`) are dropped in favour of a single
  `n4m_tests` executable (no auxiliary CTest targets).
- Test suite grew from 82/82 to 171/171 passing (~89 new tests).
- Zero new build warnings; sanitizer-clean.

## [0.1.0+abi.1.6.0] — Phase 5b baseline correction (rest) + LogTransform split

### Added

- **Six new stateless baseline correction operators** completing the family
  started in Phase 5a (`_create / _transform / _destroy` ABI contract):
  - **ModPoly** (`n4m_pp_modpoly_*`, 3 symbols) — Lieber & Mahadevan-Jansen
    2003 iterative polynomial baseline. Per-row Householder QR (shared with
    Detrend) factored once and replayed against the clipped y at each
    iteration.
  - **IModPoly** (`n4m_pp_imodpoly_*`, 3 symbols) — Gan, Ruan, Mo 2006
    σ-stopping variant of ModPoly. Convergence on the relative change of
    the residual stdev rather than the baseline.
  - **SNIP** (`n4m_pp_snip_*`, 3 symbols) — Ryan 1988 / Morháč 1997 LLS-
    transformed peak-clipping. Pure arithmetic, no linear algebra.
  - **RollingBall** (`n4m_pp_rolling_ball_*`, 3 symbols) — Kneen & Annegarn
    1996 min-then-max morphological baseline with optional moving-average
    smoothing.
  - **IAsLS** (`n4m_pp_iasls_*`, 3 symbols) — He 2014 AsLS variant with
    polynomial-prefit weight initialisation. Shares the pentadiagonal
    LDLT solver and the shared Vandermonde QR.
  - **BEADS** (`n4m_pp_beads_*`, 3 symbols, simplified variant) — Ning &
    Selesnick 2014 simplified to a pentadiagonal reweighted-L2 surrogate.
    See `docs/algorithms/beads.md` for the deferred 7-diagonal Chebyshev
    variant.
- **LogTransform `_fit / _transform` split** (Phase 2 carry-forward,
  finally landed): `auto_offset != 0` now captures the safety offset via
  `n4m_pp_log_fit` and reuses the cached `_fitted_offset` on every
  subsequent `_transform` call. `auto_offset == 0` is unchanged
  (stateless, no fit required). Adds 2 new symbols
  (`n4m_pp_log_fit`, `n4m_pp_log_is_fitted`).
- **Frozen NumPy reference** for the 6 new operators under
  `parity/python_generator/src/n4m_parity_pybaselines_ref/`. The Phase 5a
  validation script extends to Phase 5b (10 ops total against
  `pybaselines==1.1.4`).
- **Parity fixtures** under `parity/fixtures/`:
  `modpoly_v1.json`, `imodpoly_v1.json`, `iasls_v1.json`, `beads_v1.json`
  (4 cases each) and `snip_v1.json`, `rolling_ball_v1.json` (3 cases each).
- **13 new tests** in `cpp/tests/test_preprocessing_baselines.cpp`
  (6 smoke + 6 parity) and `cpp/tests/test_preprocessing_stateless.cpp`
  (1 LogTransform fit/transform split semantic check, +1 strengthened
  AirPLS smoke on constant input). Total: 69 → **82 tests**.

### Changed

- **Banded LDLT solver enhancements** in
  `cpp/src/core/common/banded_solver.{c,h}`:
  - Added `n4m_banded5_factor_into(L_buf, D_buf, ...)` — same algorithm
    as `n4m_banded5_factor` but writes into caller-owned scratch buffers,
    so AsLS / AirPLS / ArPLS / IAsLS / BEADS no longer allocate inside
    the iteration loop.
  - Added `n4m_banded5_solve_into(L_buf, D_buf, ...)` — companion solve
    against the in-place factor.
  - Added `n4m_second_diff_penalty_pent5(n, lam, ...)` — shared
    `lam * D_2^T D_2` penalty builder; replaces three inline copies
    (AsLS / AirPLS / ArPLS).
  - Added `n4m_relative_l2_diff(w, w_new, n)` — shared convergence
    helper; replaces two inline copies (AsLS / ArPLS).
- **AsLS / AirPLS / ArPLS refactored** to use the new helpers and the
  in-place LDLT factor. Per-row L / D scratch buffers replace per-iteration
  mallocs (saves up to `max_iter` mallocs per row in the hot path).
  Phase 5a parity tests still pass at their original tolerances
  (1e-7 abs / 1e-8 rel for AsLS / ArPLS; 1e-7 / 1e-8 default and
  1e-6 / 1e-5 for the high_lam / tight_tol_short AirPLS cases).
- **Strengthened AirPLS smoke test** (Phase 5a #7 carry-forward): now
  asserts `|out[i]| < 1e-6` on a constant input, not just `isfinite`.

### ABI

- ABI 1.5.0 → **1.6.0** (additive: 20 new symbols, no breakages).
- Symbol count 106 → **126**.
  - +18 from the 6 baseline ops (3 symbols each).
  - +2 from the LogTransform split (`n4m_pp_log_fit`,
    `n4m_pp_log_is_fitted`).
- All three platform expected-symbols lists updated:
  `cpp/abi/expected_symbols_{linux,macos,windows}.txt`.
- `n4m.h` adds §12 (Phase 5b baseline correction rest), updates §8
  (LogTransform lifecycle banner).

## [0.1.0+abi.1.5.0] — Phase 5a baseline correction core

### Added

- **Four stateless baseline correction operators** with the `_create /
  _transform / _destroy` ABI contract:
  - **Detrend** (`n4m_pp_detrend_*`, 3 symbols) — polynomial baseline
    subtraction via Householder QR on a (n, polyorder+1) Vandermonde
    matrix.  Matches `np.polyfit + np.polyval` per row.  `polyorder >= 0`.
  - **AsLS** (`n4m_pp_asls_*`, 3 symbols) — Asymmetric Least Squares
    (Eilers & Boelens 2005).  Iteratively reweighted penalised least
    squares with a 2nd-order penalty `lam * D_2^T D_2`.
  - **AirPLS** (`n4m_pp_airpls_*`, 3 symbols) — Adaptive iteratively
    reweighted PLS (Zhang 2010).  Weight update via exponential of
    clipped negative residuals; convergence on `|sum(neg)| / |y|_1`.
  - **ArPLS** (`n4m_pp_arpls_*`, 3 symbols) — Asymmetrically reweighted
    PLS (Baek 2015).  Logistic weight update based on the distribution
    of negative residuals.
- **Pentadiagonal symmetric LDLT solver** at
  `cpp/src/core/common/banded_solver.{c,h}` — O(n) factorisation +
  solve for `(diag(w) + lam D_2^T D_2)` systems.  Validated against a
  dense Gauss-Jordan reference (residual ~9e-12 on lam=1e3 systems).
  Internal-only; not exposed in the public ABI.
- **Frozen NumPy reference** under
  `parity/python_generator/src/n4m_parity_pybaselines_ref/`: pure
  NumPy + scipy.sparse implementations of the four operators, validated
  once against `pybaselines==1.1.4` and now the canonical parity floor.
  Future pybaselines releases can drift without breaking the n4m
  parity gates.
- **Parity fixtures** under `parity/fixtures/`: `detrend_v1.json` (4
  cases sweeping polyorder), `asls_v1.json` / `airpls_v1.json` /
  `arpls_v1.json` (4 cases each sweeping lam / p / tol / max_iter).
- **8 new tests** in `cpp/tests/test_preprocessing_baselines.cpp` (4
  smoke + 4 parity).  Total: 61 → **69 tests**.

### Changed (Phase 5 opener cleanup — per Codex Phase 3+4 review)

- **Extracted shared Householder QR** to
  `cpp/src/core/common/linalg.{c,h}` (`n4m_householder_qr`,
  `n4m_apply_qt`, `n4m_back_solve_R`).  Replaces three in-line copies
  previously living in `emsc.c`, `savitzky_golay.c` (main coef build),
  and `savitzky_golay.c` (`sg_fit_edge_left/right`).  Bit-exact: all
  61 prior tests still pass.
- **Extracted shared JSON fixture parser** to
  `cpp/tests/fixture_parser.hpp` (header-only, ~400 LOC).  Replaces
  three in-line copies previously living in
  `test_preprocessing_{stateless,stateful,smoothing}.cpp`.  Net
  deletion: ~1,000 LOC of duplicated parser code across the three
  test files.
- **Fixed `n4m_pp_derivate_*` documentation** in `n4m.h` §9: the
  banner now honestly states that `_fit` memoises the input column
  count for shape validation at transform time (previously claimed
  `_fit` was a pure no-op).

### ABI

- ABI 1.4.0 → **1.5.0** (additive: 12 new symbols, no breakages).
- Symbol count 94 → **106**.
- All three platform expected-symbols lists updated:
  `cpp/abi/expected_symbols_{linux,macos,windows}.txt`.

## [0.1.0+abi.1.4.0] — Phase 4 derivatives & smoothing

### Added

- **Five stateless preprocessing operators** with the `_create / _transform /
  _destroy` ABI contract:
  - **SavitzkyGolay** (`n4m_pp_savgol_*`, 3 symbols) — matches
    `scipy.signal.savgol_filter`.  Vandermonde-QR pseudo-inverse for the
    coefficient build (mathematically equivalent to scipy's SVD-based
    lstsq path).  Five boundary modes — `mirror`, `constant`, `nearest`,
    `wrap`, `interp` — with the same `interp` polynomial-fit-at-edges
    behaviour as `_savitzky_golay._fit_edge`.
  - **FirstDerivative** (`n4m_pp_first_derivative_*`, 3 symbols) — matches
    `np.gradient(X, delta, axis=1, edge_order)` for `edge_order` in
    {1, 2}.  Shape-preserving.
  - **SecondDerivative** (`n4m_pp_second_derivative_*`, 3 symbols) — two
    passes of `np.gradient`.
  - **NorrisWilliams** (`n4m_pp_norris_williams_*`, 3 symbols) — centred
    moving-average smoother followed by a gap finite difference, applied
    `derivative_order` times.  Matches `nirs4all.operators.transforms.
    norris_williams.norris_williams`.
  - **Gaussian** (`n4m_pp_gaussian_*`, 3 symbols) — matches
    `scipy.ndimage.gaussian_filter1d`.  Hermite-polynomial kernel
    recursion that produces the exact same kernel bytes as
    `_gaussian_kernel1d`.  Five boundary modes — `reflect`, `constant`,
    `nearest`, `mirror`, `wrap`.
- **Two new enum types** in the public header: `n4m_pp_savgol_mode_t`
  (5 values) and `n4m_pp_gaussian_mode_t` (5 values).
- **Boundary-padding helper** (`cpp/src/core/common/padding.{c,h}`) —
  shared index resolution for the five SciPy modes.  Used by SG and
  Gaussian; reused unchanged by Phases 5+.
- **C engines** under `cpp/src/core/preprocessing/derivatives/` and
  `cpp/src/core/preprocessing/smoothing/`.  Pure ISO C11; no
  third-party dependencies.  SG uses an in-place Householder QR
  factorisation (~100 LOC).
- **C ABI wrappers** in `cpp/src/c_api/c_api_preprocessing.cpp` — 15 new
  symbols (3 × 5).  Same try/catch / null-safe / matrix-view validation
  conventions as the Phase 2 / 3 wrappers.
- **Parity fixtures** under `parity/fixtures/`: `savgol_v1.json`
  (22 cases sweeping window / polyorder / deriv / 5 modes / delta-scaling
  / deriv > polyorder short-circuit), `first_derivative_v1.json` (4),
  `second_derivative_v1.json` (4), `norris_williams_v1.json` (5),
  `gaussian_v1.json` (26 — 5 (sigma, order) tuples × 5 modes + a non-default
  `truncate` check).
- **Fixture generator** `parity/python_generator/scripts/
  generate_phase4_fixtures.py`.  Loads `norris_williams.py` directly from
  the nirs4all source via `importlib`, mirroring the Phase 2 / 3 generator
  pattern.
- **10 new tests** in `cpp/tests/test_preprocessing_smoothing.cpp`: 5
  smoke + 5 parity.  Total: 51 → **61 / 61**.
- **Documentation** under `docs/algorithms/`: one Markdown page per
  Phase 4 operator (`savitzky_golay.md`, `first_derivative.md`,
  `second_derivative.md`, `norris_williams.md`, `gaussian.md`).

### Changed

- **ABI bump**: 1.3.0 → 1.4.0 (additive).  Library SOVERSION stays at 1.
- **Total exported symbols**: 79 → **94** (15 new wrappers, no removals).
- **Public header** gains §10 "Phase 4 — Stateless derivatives & smoothing
  preprocessings"; the previous §10 (ABI guard rails) is renumbered to
  §11.  Two new static-size assertions cover `n4m_pp_savgol_mode_t` and
  `n4m_pp_gaussian_mode_t`.

### Numerical parity findings

- The Vandermonde-QR pseudo-inverse path agrees with scipy's SVD-based
  `lstsq` to within ~1e-11 on the supported `(window_length, polyorder,
  deriv)` sweep.  The 1e-9 / 1e-10 SG / Gaussian tolerance reflects this
  structural gap, not an algorithmic drift.
- `np.gradient(X, delta, axis=1, edge_order=2)` evaluates the boundary
  formula `(-3 X[0] + 4 X[1] - X[2]) / (2 \delta)` and the central
  difference `(X[j+1] - X[j-1]) / (2 \delta)` per element; the C engine
  uses the same arithmetic order and matches byte-for-byte (1e-12 / 1e-13).
- The Hermite recursion in scipy's `_gaussian_kernel1d` reproduces
  bit-exactly in our C engine because both apply the same `(D + P)`
  matrix in the same order; the only divergence comes from the
  correlation accumulator order which is bounded by the kernel length.

### Deferred (Phase 5+)

- A shared JSON parser for the test harness — the parity-fixture parser
  is currently triplicated across `test_preprocessing_{stateless,stateful,
  smoothing}.cpp`.  Will get pulled out at Phase 6 once we have enough
  fixtures to justify the indirection.
- A shared QR helper — the Householder code is now copied in three places
  (`emsc.c`, `savitzky_golay.c`, and the SG edge fit).  We hold off until
  Phase 5 needs a fourth copy, then promote to `cpp/src/core/common/qr.{c,h}`.

## [0.1.0+abi.1.3.0] — Phase 3 stateful scatter preprocessings

### Added

- **Four stateful preprocessing operators** with the `_create / _fit /
  _transform / _destroy` ABI contract (and `_inverse_transform` / `_is_fitted`
  where applicable):
  - **MSC** (`n4m_pp_msc_*`, 6 symbols) — Multiplicative Scatter Correction
    with per-column linear regression against the per-row mean. Invertible.
    Matches `nirs4all.operators.transforms.nirs.MultiplicativeScatterCorrection(scale=False)`.
  - **EMSC** (`n4m_pp_emsc_*`, 5 symbols) — Extended MSC with polynomial
    wavelength terms (degree configurable). Per-row least-squares solve
    against `[reference, w^1, …, w^degree]` via Householder QR. Not
    invertible.
  - **Baseline** (`n4m_pp_baseline_*`, 6 symbols) — Per-column mean
    centering (equivalent to `StandardScaler(with_std=False)`). Invertible.
    Matches `nirs4all.operators.transforms.signal.Baseline`. Class name is
    historical; this is NOT a baseline correction in the spectroscopic
    sense (the polynomial / AsLS family lives in Phase 5).
  - **Derivate** (`n4m_pp_derivate_*`, 5 symbols) — `np.diff(X, n=order,
    axis=1) / delta**order`. Output column count is
    `cols - order`; a helper `n4m_pp_derivate_output_cols(order, input_cols)`
    is exposed for callers to size their output buffers. Lifecycle is
    stateful-shaped (with a no-op `_fit`) for ABI uniformity with the
    other Phase 3 operators.
- **Stateful contract enforcement**: every stateful operator tracks a
  `fitted` flag on its internal state. `_transform` (and
  `_inverse_transform`) return `N4M_ERR_NOT_FITTED` when called before a
  successful `_fit`. Calling `_fit` again replaces the prior state
  (sklearn-style refit semantics).
- **C engines** under `cpp/src/core/preprocessing/{scatter,scaling,
  derivatives}/`: 4 `.c` + `.h` pairs. Pure ISO C11, no third-party
  dependencies. EMSC uses an in-place Householder QR factorisation (~100
  LOC) rather than normal-equations + Cholesky to stay numerically close
  to LAPACK gelsd on the ill-conditioned raw-integer wavelength basis.
- **C ABI wrappers** in `cpp/src/c_api/c_api_preprocessing.cpp` — 22 new
  symbols (6 + 5 + 6 + 5). Same try/catch / null-safe / matrix-view
  validation conventions as the Phase 2 stateless wrappers.
- **Parity fixtures** (`parity/fixtures/{msc,emsc,baseline_center,
  derivate}_v1.json`) — every fixture exercises the fit/transform split
  with a *different* matrix for fit and transform (seeds 20260518 +
  20260519) so the test verifies that fitted parameters are correctly
  reused.
- **Fixture generator** `parity/python_generator/scripts/generate_phase3_fixtures.py`
  — loads nirs4all directly from source via `importlib`, mirroring the
  Phase 2 generator pattern.
- **14 new tests** in `cpp/tests/test_preprocessing_stateful.cpp`: 4
  smoke + 4 parity + 4 not-fitted + 1 baseline inverse + 1 derivate
  output-cols helper. Tolerance: 1e-12 abs / 1e-13 rel for Baseline and
  Derivate (closed-form ops); 1e-10 / 1e-11 for MSC; 5e-10 / 5e-10 for
  EMSC (Householder QR vs LAPACK gelsd on the raw-integer wavelength
  basis costs ~8 decimal digits, which is structural to the conditioning
  of $B$).
- **Documentation** under `docs/algorithms/`: one Markdown page per Phase
  3 operator (`msc.md`, `emsc.md`, `baseline_center.md`, `derivate.md`).
- **Cross-phase review artefacts**:
  - `docs/reviews/PHASES.md` — aggregate verdict table for Phases 0-3 and
    placeholders for Phases 4-26.
  - `docs/reviews/DEFERRALS.md` — cross-phase deferral tracker
    pre-populated with all open items from Phases 0-2 + the closed
    Phase-3 items (nirs4all pinning, PLS-orphan cleanup, etc.).

### Removed (Codex R3 + R4)

- **n4m.h orphan PLS declarations** — config, pipeline, model,
  operator_bank, gating_strategy, AOM (global + per-component),
  validation_plan, method_result, component_coefficients,
  n4m_array_t opaque type + 4 accessors. None had implementations in
  `cpp/src/`; all were inherited from pls4all and never wired. n4m.h
  shrunk from **2208** lines to **680** lines (-69%).
- **Static asserts** in the ABI guard-rail section that referenced
  removed enums (`n4m_algorithm_t`, `n4m_solver_t`, `n4m_deflation_t`,
  `n4m_operator_kind_t`, `n4m_gating_mode_t`, `n4m_model_array_t`).
- **Reintroduce-n4m_array_t-at-Phase-9** deferral logged at
  `docs/reviews/DEFERRALS.md`.

### Numerical parity findings

- nirs4all's `MultiplicativeScatterCorrection` uses `np.polyfit(ref,
  col, 1)` per column where `ref = mean(X, axis=1)` is the per-row mean.
  The slope/intercept come back as `[a, b] = polyfit(...)`, then
  transform is `(X[:, j] - b) / a`. The n4m closed-form `(a, b)`
  computation matches `np.polyfit`'s Vandermonde + lstsq path to within
  ~2 ULPs.
- For EMSC the gold standard is `np.linalg.lstsq(B, x_i, rcond=None)`
  which dispatches to LAPACK gelsd (SVD-based). The n4m Householder QR
  implementation agrees to ~8 decimal digits across the supported degree
  range. The gap is structural to the conditioning of the raw-integer
  wavelength basis and would not improve by switching solver families
  short of a full SVD implementation.
- `np.diff(X, n=order, axis=1) / delta**order` evaluates `delta**order`
  as repeated multiplication when `order` is an integer (Python's
  `int.__pow__`). The n4m engine matches this exactly and then divides
  each output element by the precomputed scalar (true element-wise
  division, not multiply-by-reciprocal).

### ABI

- Project version stays 0.1.0; **ABI bumps to 1.3.0** (additive: 22 new
  symbols; the n4m.h trim removed only orphan declarations with no
  exported counterparts, so it is not an ABI-breaking change). Total
  exported symbols: 57 → **79**.

### Tests

- 51/51 passing: 7 phase-0 smoke + 13 phase-1 RNG + 14 phase-2
  preprocessing + 17 phase-3 stateful preprocessing (14 original + 3
  refit-replaces-state tests added per Opus review).

### Review

- **Opus post-review** at `docs/reviews/phase-3/opus-post.md`. Verdict:
  ACCEPT with 3 recommended follow-ups, all applied:
  1. Refit tests added for MSC, EMSC, Baseline (`pp_<op>_refit`).
  2. Derivate vs nirs4all `Derivate` class semantic divergence logged
     in `docs/reviews/DEFERRALS.md`.
  3. EMSC switched from `* (1/c[0])` to `/ c[0]` to preserve byte-exact
     NumPy rounding discipline.

## [0.1.0+abi.1.2.0] — Phase 2 stateless preprocessings

### Added

- **Seven stateless preprocessing operators** with bit-exact `nirs4all` 0.8.x / `numpy` 1.26.4 parity:
  - **SNV** (`n4m_pp_snv_*`) — row-wise mean / std normalisation with configurable `with_mean`, `with_std`, `ddof`.
  - **LocalSNV** (`n4m_pp_lsnv_*`) — sliding-window SNV with `reflect` / `edge` / `constant` padding and cumsum-based moving statistics matching `nirs4all.LocalStandardNormalVariate`.
  - **RobustSNV** (`n4m_pp_rnv_*`) — median + `k * MAD` normalisation with optional center / scale, in-place quickselect median.
  - **AreaNormalization** (`n4m_pp_area_*`) — per-row division by `sum` / `abs_sum` / `trapz` area (trapezoidal integration with unit spacing, pair-sum form matching `np.trapz`).
  - **Normalize** (`n4m_pp_normalize_*`) — column-wise L2-norm scaling or user-defined-range scaling.
  - **SimpleScale** (`n4m_pp_simple_scale_*`) — column-wise min-max to `[0, 1]`.
  - **LogTransform** (`n4m_pp_log_*`) — element-wise `log(X + δ)` with optional fit-time auto-offset and arbitrary base.
- **C engines** under `cpp/src/core/preprocessing/{scatter,scaling}/` (7 `.c` + `.h` pairs). Pure ISO C11, no third-party dependencies, all accumulations preserve NumPy's left-to-right rounding order.
- **C ABI wrappers** in `cpp/src/c_api/c_api_preprocessing.cpp` — 21 new symbols (3 × 7 operators: `create / destroy / transform`), each wrapped in `try` / `catch` so no C++ exception ever crosses the boundary. Input validation enforces row-major contiguous F64 matrix views and matching output shape.
- **Parity fixtures** (`parity/fixtures/{snv,lsnv,rnv,area_norm,normalize,simple_scale,log_transform}_v1.json`) generated from a deterministic 50 × 200 NIR-shaped spectrum block (PCG64 seed=20260518); doubles encoded as big-endian hex to avoid JSON float-formatting loss.
- **Fixture generator** `parity/python_generator/scripts/generate_phase2_fixtures.py` — loads nirs4all's reference operators directly from source via `importlib` to avoid heavyweight package imports.
- **14 new tests** in `cpp/tests/test_preprocessing_stateless.cpp` (2 per operator: a lifecycle smoke test + a parity sweep over every fixture case). Tolerance: bit-equality OR `1e-12` absolute OR `1e-13` relative.
- **Documentation** under `docs/algorithms/` — one Markdown page per operator with algorithm, parameters, numerical contract, and reference.

### Numerical parity findings

- `numpy.mean / numpy.std` and `nirs4all`'s `X / std` evaluate divisions per-element. Replacing them with `X * (1 / std)` introduces single-ULP drift on certain inputs — all kernels use true division to preserve byte-for-byte parity.
- `numpy.trapz` / `scipy.integrate.trapezoid` with unit spacing computes `0.5 * sum(x[:-1] + x[1:])`, not `sum(x) - (x[0] + x[-1]) / 2`. The two forms are algebraically equal but FP-distinct; we use the pair-sum form.
- `nirs4all.RobustStandardNormalVariate` evaluates the two `if` blocks (center, scale) sequentially: when `with_center=False`, MAD is computed on `|X|` (raw), not on `|X - median|`. Reproducing this exact branching was the only RNV bug found during parity tuning.

### ABI

- Project version stays 0.1.0; **ABI bumps to 1.2.0** (additive: 21 new symbols, no breaking changes). Total exported symbols: 36 → **57**.

### Tests

- 34/34 passing: 7 phase-0 smoke + 13 phase-1 RNG + 14 phase-2 preprocessing.

### Reviews

- **Opus post-review** at `docs/reviews/phase-2/opus-post.md`. Verdict: REVISE → 3 high-confidence fixes applied: (1) SNV flat-row threshold tightened from `>= 1e-15` to `!= 0` to match nirs4all exactly, (2) `n4m_pp_lsnv_pad_mode_t` and `n4m_pp_area_method_t` enums hoisted from internal headers into public `n4m.h`, (3) LogTransform per-call recomputation behaviour documented in n4m.h.
- **Codex Phases 1+2 review** at `docs/reviews/phase-2/codex-post.md`. Verdict: ACCEPT with Phase 3 prerequisites. Applied: test Runner renamed to `nirs4all-methods`, `parity/python_generator/pyproject.toml` added with numpy 1.26.4 pin, `cpp/src/c_api/n4m_linux.map` version script added (defense-in-depth on top of `-fvisibility=hidden`). Confirmed all 57 exports are `n4m_*`.
- Per user direction, **Codex review runs every two phases** in addition to per-phase Opus reviews.

## [0.1.0+abi.1.1.0] — Phase 1 PCG64 RNG

### Added

- **PCG64 random number generator** with bit-exact NumPy parity:
  - `cpp/src/core/common/rng_pcg64.{h,c}` — pure-C PCG-XSL-RR 128/64 implementation with native `__uint128_t` (gcc/clang) and MSVC two-uint64 fallback. Full NumPy `SeedSequence(seed)` expansion via SplitMix64-style 4-stage pool mixer.
  - `cpp/src/core/common/ziggurat.c` + `ziggurat_constants.h` — vendored NumPy 1.26.4 Ziggurat tables (BSD-3-Clause, license-compatible with CeCILL-2.1) with 256-region Marsaglia–Tsang algorithm for `standard_normal_fill`.
  - `cpp/src/c_api/c_api_rng.cpp` — 7 C ABI wrappers (`n4m_rng_pcg64_{create, destroy, set_seed, uint64, uint64_fill, standard_normal_fill, advance}`) behind opaque handle `n4m_rng_pcg64_state_t`. Try/catch wrapped, null-pointer-safe, `destroy(NULL)` no-op.
- **Parity fixture** `parity/fixtures/_rng_pcg64_stream_v1.json` (848 kB) — 5 seeds × (4096 uint64 + 4096 standard_normal) generated with `numpy==1.26.4` (pinned per CONTRIBUTING.md). Doubles encoded as big-endian hex to avoid JSON float-precision loss.
- **13 new tests** in `cpp/tests/test_rng_pcg64.cpp`: 3 functional (null-safety, set_seed-resets-stream, advance-matches-repeats) + 5 uint64 byte-exact parity (one per seed) + 5 standard_normal IEEE-754 bit-exact parity (one per seed). Hand-rolled JSON parser keeps the zero-dep policy.
- **Documentation** `docs/algorithms/rng_pcg64.md` — algorithm, API, vendoring provenance, parity policy.

### Fixed

- `cmake/CompilerWarnings.cmake` — split C/C++ warning flags. C++-only flags (`-Wnon-virtual-dtor`, `-Wold-style-cast`, `-Woverloaded-virtual`, `-Wzero-as-null-pointer-constant`) now wrapped in `$<COMPILE_LANGUAGE:CXX>` generator expression so they no longer trigger "valid for C++/ObjC++ but not for C" warnings on `.c` TUs.

### ABI

- Project version stays 0.1.0; **ABI bumps to 1.1.0** (additive: 7 new symbols, no breaking changes). Total exported symbols: 29 → **36** (`n4m_rng_pcg64_*` × 7 added).

### Review

- Opus post-review at `docs/reviews/phase-1/opus-post.md`. Verdict: ACCEPT. Two pre-commit nits fixed (fixture provenance, C-vs-C++ warnings). One medium issue fixed (algorithm doc). Five low-priority items tracked for Phase 2+.

### Tests

- 20/20 passing: 7 phase-0 smoke + 13 phase-1 RNG (3 functional + 10 NumPy parity).


## [0.1.0+abi.1.0.0] — Phase 0 bootstrap

### Added

- C++17 core skeleton mirroring `pls4all`'s architecture: `cpp/src/core/{config,context,matrix_view,status,version}.cpp` + `cpp/src/c_api/{c_api_context,c_api_matrix,c_api_version}.cpp`.
- Public C ABI header `cpp/include/n4m/n4m.h` (Phase 0 surface: status codes, dtype/backend enums, matrix view, context lifecycle, version queries). Full operator surface to be added phase by phase.
- CMake build system inherited from pls4all with 20+ presets in `CMakePresets.json` (dev-debug, dev-release, ci-*, blas-on, sanitizer-*). Zero mandatory dependencies in the reference build.
- Tiny CLI `n4m_cli` with `--version`, `--abi-info`, `--selfcheck`.
- Hand-rolled zero-dependency test harness in `cpp/tests/main.cpp` + `cpp/tests/harness.hpp`, with 7 Phase 0 smoke tests (all passing).
- GitHub Actions workflows inherited from pls4all (ci, parity-gate, docs, abi-check, coverage, sanitizers, version-sync) — to be re-enabled after Phase 1 once the new C ABI surface is wired.
- `ROADMAP.md` with the 27-phase plan, `ARCHITECTURE.md` with the load-bearing decisions, `CONTRIBUTING.md` with the Codex+Opus review workflow.

### Stripped from the pls4all template

- All PLS algorithm code in `cpp/src/core/` (50+ files: `pls_*`, `simpls*`, `opls*`, `aom_*`, `*_selection.cpp`, `model.cpp`, `pipeline.cpp`, `config.{cpp,hpp}` …).
- All PLS-specific C ABI files in `cpp/src/c_api/` (`c_api_{config,model,pipeline,operator_bank,gating_strategy,validation_plan,aom_selection,method_result,component_coefficients}.cpp`).
- All PLS test fixtures and ABI test suites in `cpp/tests/{abi,fixtures}/`.
- `bindings/` entirely (rebuilt in Phases 22-25 for Python / R / MATLAB / JS-WASM).
- `benchmarks/` entirely (rebuilt in Phase 26).
- `parity/python_generator/` and `parity/r_generator/` (rebuilt in Phase 1+ with n4m adapters; `parity/schema/` and `parity/tolerances.md` retained as inherited contract).
- `docs/methods/` (90 PLS algorithm docs), `docs/parity_audit_2026_05/`, `docs/reviews/phase-0/` (old pls4all reviews), `docs/benchmarks/cross_binding*.md`.
- Pls4all-specific top-level docs (`Overview.md`, `Direction_Technique.md`, `DISTRIBUTION.md`, `Backlog.md`) — replaced by `ARCHITECTURE.md`, `ROADMAP.md`.
- 97+ pls4all `roadmap/phase-*.md` files — replaced by n4m-scoped phases.
- `CI workflows except `ci.yml` moved to `.github/workflows.disabled/` (`parity-gate`, `release-python`, `abi-check`, `version-sync`, `coverage`, `sanitizers`, `docs`). Each is re-enabled at the phase that wires its inputs.

### ABI

- Initial ABI version `1.0.0`. `cpp/abi/expected_symbols_{linux,macos,windows}.txt` populated with the 29 actually-exported `n4m_*` symbols from `libn4m.so.1.0.0` (status / backend / dtype / matrix_view / context lifecycle / version queries).

### Opus post-review

- Independent Opus review completed (transcript at `docs/reviews/phase-0/opus-post.md`). 9 high-confidence issues identified and fixed before commit. 5 medium/low-priority items deferred to Phase 1+ with explicit tracking in the roadmap.

[Unreleased]: https://github.com/GBeurier/nirs4all-methods/compare/v0.1.0...HEAD
[0.1.0+abi.1.0.0]: https://github.com/GBeurier/nirs4all-methods/releases/tag/v0.1.0
