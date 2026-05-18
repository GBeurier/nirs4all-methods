# Parity gate

`chemometrics4all` is validated against multiple reference PLS implementations. The parity gate exists so that any numerical change in the C core surfaces immediately as a test failure, with a documented tolerance per pair of implementations.

## Two-gate parity architecture

chemometrics4all mirrors the pls4all parity model: there are *two* distinct notions of parity, each with its own comparator and CI stage.

| Gate | Comparator | Tolerance | What it verifies | Where it runs |
|---|---|---|---|---|
| **Gate 1 — binding parity** | `parity/binding_parity.py` :: `binding_parity(pred, ref, tolerance=1e-6)` | `max|pred - ref| <= 1e-6` (default) | Every language binding (python, R, MATLAB, JS/WASM) produces bit-identical outputs to libc4a, the C kernel reference. | Stage 4 of `.github/workflows/parity-gate.yml`. **No-op until Phase 22 ships the first binding.** |
| **Gate 2 — reference parity** | `parity/reference_parity.py` :: `reference_parity(pred, ref, tolerance)` | `||pred - ref||_RMS / ||ref||_RMS <= tol` (per-op in `parity/tolerances.md` + the `TOLERANCE_TABLE` in `parity/run_reference_parity.py`) | libc4a's algorithms agree with the canonical external reference (numpy / scipy / sklearn / pywt / pybaselines / nirs4all). | Stage 2 of the unified parity gate (`parity/python_generator/scripts/run_parity_gate.py` → `parity/run_reference_parity.py`). |

The unified parity-gate runner orchestrates four stages:

1. **Stage 1 — fixture determinism.** Every fixture under `parity/fixtures/` must regenerate bit-identically from the pinned generator in `parity/python_generator/`. Run by `run_parity_gate.py::stage1_fixture_determinism`.
2. **Stage 2 — reference parity (Gate 2).** Run by `parity/run_reference_parity.py`, which loads each canonical implementation and compares to the committed fixtures via `reference_parity`.
3. **Stage 3 — C++ parity tests.** Asserts that libc4a matches the frozen fixtures within `parity/tolerances.md` per-row tolerances. Run by `chemometrics4all_tests`.
4. **Stage 4 — binding parity (Gate 1).** Per-binding bit-exact smoke against libc4a using `binding_parity`. Skipped today; activates when at least one binding ships.

The two comparators live in **pure-numpy modules** (`parity/binding_parity.py`, `parity/reference_parity.py`) so they can be shared by the gate runner and the per-binding orchestrators without pulling in the rest of the gate plumbing. The contract mirrors `pls4all/benchmarks/parity_timing/_comparator.py` so both projects use the same comparator semantics.


## Layout

```
parity/
├── schema/
│   ├── fixture_schema_v1.json     JSON Schema for the fixture file format.
│   └── fixture_schema_v1.md       Human description of every field.
├── fixtures/
│   ├── manifest.json              Per-fixture SHA-256 + generator git rev.
│   ├── synthetic_small_pls1_v1.json
│   ├── synthetic_small_pls2_v1.json
│   ├── synthetic_tiny_centered_v1.json
│   ├── synthetic_simpls_tiny_pls1_v1.json
│   ├── synthetic_simpls_small_pls2_v1.json
│   ├── synthetic_svd_tiny_pls1_v1.json
│   ├── synthetic_svd_small_pls2_v1.json
│   ├── synthetic_kernel_tiny_pls1_v1.json
│   ├── synthetic_kernel_wide_pls2_v1.json
│   ├── synthetic_wide_kernel_pls1_v1.json
│   ├── synthetic_wide_kernel_pls2_v1.json
│   ├── synthetic_oscores_tiny_pls1_v1.json
│   ├── synthetic_oscores_small_pls2_v1.json
│   ├── synthetic_power_tiny_pls1_v1.json
│   ├── synthetic_power_small_pls2_v1.json
│   ├── synthetic_randomized_svd_tiny_pls1_v1.json
│   ├── synthetic_randomized_svd_small_pls2_v1.json
│   ├── synthetic_canonical_tiny_pls1_v1.json
│   ├── synthetic_canonical_small_pls2_v1.json
│   ├── synthetic_canonical_svd_tiny_pls1_v1.json
│   ├── synthetic_canonical_svd_small_pls2_v1.json
│   ├── synthetic_pls_da_binary_v1.json
│   ├── synthetic_pls_da_multiclass_v1.json
│   ├── synthetic_opls_tiny_pls1_v1.json
│   ├── synthetic_opls_small_pls1_v1.json
│   ├── synthetic_opls_small_pls2_v1.json
│   ├── synthetic_opls_da_binary_v1.json
│   ├── synthetic_opls_da_multiclass_v1.json
│   ├── synthetic_pls_svd_tiny_v1.json
│   ├── synthetic_pls_svd_small_v1.json
│   ├── synthetic_pipeline_identity_v1.json
│   ├── synthetic_pipeline_center_v1.json
│   ├── synthetic_pipeline_autoscale_v1.json
│   ├── synthetic_pipeline_pareto_v1.json
│   ├── synthetic_pipeline_snv_v1.json
│   ├── synthetic_pipeline_msc_v1.json
│   ├── synthetic_pipeline_emsc_v1.json
│   ├── synthetic_pipeline_detrend_poly_v1.json
│   ├── synthetic_pipeline_savgol_smooth_v1.json
│   ├── synthetic_pipeline_savgol_derivative_v1.json
│   ├── synthetic_pipeline_asls_v1.json
│   ├── synthetic_pipeline_norris_williams_v1.json
│   ├── synthetic_pipeline_wavelet_haar_v1.json
│   ├── synthetic_pipeline_osc_v1.json
│   ├── synthetic_pipeline_epo_v1.json
│   ├── synthetic_metrics_regression_v1.json
│   ├── synthetic_classification_binary_metrics_v1.json
│   ├── synthetic_classification_multiclass_metrics_v1.json
│   ├── synthetic_classification_calibration_curve_v1.json
│   ├── synthetic_pls_lda_multiclass_v1.json
│   ├── synthetic_pls_logistic_multiclass_v1.json
│   ├── synthetic_mb_pls_block_weighted_v1.json
│   ├── synthetic_lw_pls_local_window_v1.json
│   ├── synthetic_variable_importance_pls2_v1.json
│   ├── synthetic_variable_selection_rankers_v1.json
│   ├── synthetic_interval_selection_moving_window_v1.json
│   ├── synthetic_coefficient_stability_mcuve_v1.json
│   ├── synthetic_uve_artificial_variables_v1.json
│   ├── synthetic_spa_pls_projection_v1.json
│   ├── synthetic_cars_pls_competitive_v1.json
│   ├── synthetic_random_frog_pls_v1.json
│   ├── synthetic_scars_pls_stability_v1.json
│   ├── synthetic_ga_pls_wrapper_v1.json
│   ├── synthetic_shaving_pls_recursive_v1.json
│   ├── synthetic_bve_pls_backward_v1.json
│   ├── synthetic_t2_pls_hotelling_v1.json
│   ├── synthetic_wvc_pls_numeric_v1.json
│   ├── synthetic_component_coefficients_pls2_v1.json
│   ├── synthetic_validation_kfold_balanced_v1.json
│   ├── synthetic_validation_leave_one_out_v1.json
│   ├── synthetic_validation_holdout_v1.json
│   ├── synthetic_validation_external_folds_v1.json
│   ├── synthetic_validation_repeated_kfold_v1.json
│   ├── synthetic_validation_monte_carlo_v1.json
│   ├── synthetic_validation_kennard_stone_v1.json
│   ├── synthetic_validation_spxy_v1.json
│   ├── synthetic_cv_kfold_nipals_pls1_v1.json
│   ├── synthetic_cv_kfold_nipals_pls2_v1.json
│   ├── synthetic_component_cv_simpls_pls2_v1.json
│   ├── synthetic_pcr_tiny_pls1_v1.json
│   └── synthetic_pcr_small_pls2_v1.json
├── tolerances.md                  Pair-wise abs / rel tolerance table.
├── python_generator/              Pinned scikit-learn + NumPy/SciPy preprocessing/SIMPLS/kernel/wide/oscores/power/randomized/canonical/PLSSVD/PLS-DA/PLS-LDA/PLS-logistic/MB-PLS/LW-PLS/OPLS/OPLS-DA/SVD/PCR/validation/CV/variable-selection/interval-selection/stability-selection/UVE/SPA/CARS/RandomFrog/SCARS/GA/Shaving/BVE/T2/WVC adapters.
└── r_generator/                   Pinned pls / ropls / mixOmics adapters.
```

## Regenerating the fixtures

```bash
cd parity/python_generator
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
generate-fixtures --check       # asserts bit-identity with checked-in fixtures
generate-fixtures --regenerate  # rewrites the JSONs from scratch
```

R-side regeneration uses `renv` and produces the canonical cross-implementation
fixtures once the R generator lands:

```bash
cd parity/r_generator
Rscript -e 'renv::restore()'
Rscript generate_fixtures.R --check       # asserts bit-identity
```

## Current status

The checked-in fixtures are green in CI. NIPALS fixtures are generated from
scikit-learn `PLSRegression`; SIMPLS fixtures are generated from a deterministic
NumPy port of the local `nirs4all`/AOM SIMPLS reference; SVD fixtures are
generated from NumPy covariance-SVD directions with regression deflation.
Kernel and wide-kernel fixtures are generated from a NumPy linear-kernel PLS
reference. Orthogonal-scores fixtures are generated from a NumPy port of the
R `pls` `oscorespls.fit` recurrence. Power fixtures are generated from NumPy
singular-pair power iteration. Randomized-SVD fixtures mirror the C++
SplitMix64-seeded singular-vector iteration. PLSCanonical fixtures mirror
scikit-learn canonical deflation for NIPALS and SVD. PLSSVD fixtures mirror
scikit-learn's direct cross-covariance SVD score model and gate the chemometrics4all
latent projection prediction convention. PLS-DA fixtures use dummy-coded class
targets fitted through scikit-learn `PLSRegression`. PLS-LDA and PLS-logistic
fixtures reuse those score spaces with deterministic NumPy classifier heads.
MB-PLS fixtures use deterministic block autoscaling and weighting before
fitting sklearn `PLSRegression(scale=False)`. LW-PLS fixtures use deterministic
global-autoscaled kNN windows and local sklearn `PLSRegression` refits. OPLS and
OPLS-DA fixtures are generated from a deterministic NumPy OPLS NIPALS recurrence
with one shared predictive score for multi-response targets.
PCR fixtures are generated from NumPy PCA/SVD with score-space
least squares. Preprocessing pipeline fixtures are generated from deterministic
NumPy identity, center, autoscale, Pareto, SNV, MSC, EMSC, polynomial detrending,
Savitzky-Golay, ASLS, Norris-Williams, Haar wavelet, supervised OSC and EPO transforms.
Regression metric fixtures are generated from deterministic NumPy formulas for
RMSE, MAE, bias, R2/Q2, observed-vs-predicted slope/intercept, RPD and RPIQ.
Binary classification metric fixtures are generated from deterministic NumPy
formulas for sensitivity, specificity, balanced accuracy, precision/F1, MCC and
average-rank AUC.
Variable-importance fixtures are generated from sklearn `PLSRegression` scores,
weights and loadings for VIP and selectivity-ratio parity.
Variable-selection fixtures rank those VIP scores, original-scale coefficient
magnitudes and selectivity-ratio scores with deterministic score-descending /
index-ascending tie handling.
Interval-selection fixtures scan contiguous feature windows with deterministic
k-fold sklearn `PLSRegression` refits and rank windows by RMSE.
Stability-selection fixtures refit sklearn `PLSRegression` over deterministic
Monte-Carlo subsets and rank features by coefficient mean/std ratios.
UVE fixtures threshold real-variable stability scores against deterministic
artificial-variable scores, SPA fixtures use a coefficient seed plus projection
residual selection, and CARS fixtures use deterministic exponential-retention
competitive adaptive reweighted sampling with k-fold RMSE scoring. Random Frog
fixtures use deterministic add/remove/swap subset walks and inclusion
frequencies. SCARS fixtures use deterministic calibration subsampling,
stability-weighted CARS retention and k-fold RMSE scoring. GA-PLS fixtures use
deterministic population initialization, elitism, crossover, mutation and k-fold
RMSE fitness. Shaving fixtures recursively eliminate low-score PLS variables and
score every retained subset by k-fold RMSE. BVE fixtures greedily evaluate every
one-variable removal by k-fold RMSE and keep the best backward path/subset.
T2 fixtures compute Hotelling T2 over PLS loading weights, apply alpha-specific
UCL thresholds with top-k fallback and score each subset by k-fold RMSE.
WVC fixtures mirror `plsVarSel` WVC2 for numeric regression with standardized X,
SVD component weights, normalized WVC scores and deterministic top-k selection.
Component-coefficient fixtures are generated from sklearn `PLSRegression`
weights/loadings and gate original-scale coefficient blocks for every latent
prefix.
Validation split fixtures are generated from deterministic Python references
for balanced contiguous k-fold, leave-one-out and holdout plans. Cross-validation
fixtures use scikit-learn `PLSRegression` over the same deterministic k-fold
plans and gate out-of-sample predictions plus aggregate metrics.
C++ parity tests assert predictions, coefficients, preprocessing statistics,
transforms, CV predictions, regression/classification metrics, variable
importance, variable-selection rankers, component coefficients and latent arrays
plus interval/stability/UVE/SPA/CARS/RandomFrog/SCARS/GA/Shaving/BVE/T2/WVC-selection scans within `tolerances.md`.
