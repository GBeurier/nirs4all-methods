# Parity gate

`pls4all` is validated against multiple reference PLS implementations. The parity gate exists so that any numerical change in the C core surfaces immediately as a test failure, with a documented tolerance per pair of implementations.

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
│   ├── synthetic_pcr_tiny_pls1_v1.json
│   └── synthetic_pcr_small_pls2_v1.json
├── tolerances.md                  Pair-wise abs / rel tolerance table.
├── python_generator/              Pinned scikit-learn + NumPy/SciPy preprocessing/SIMPLS/kernel/wide/oscores/power/randomized/canonical/PLSSVD/PLS-DA/OPLS/OPLS-DA/SVD/PCR adapters.
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
scikit-learn's direct cross-covariance SVD score model and gate the pls4all
latent projection prediction convention. PLS-DA fixtures use dummy-coded class
targets fitted through scikit-learn `PLSRegression`. OPLS and
OPLS-DA fixtures are generated from a deterministic NumPy OPLS NIPALS recurrence
with one shared predictive score for multi-response targets.
PCR fixtures are generated from NumPy PCA/SVD with score-space
least squares. Preprocessing pipeline fixtures are generated from deterministic
NumPy identity, center, autoscale, Pareto, SNV, MSC and polynomial detrending transforms. C++ parity tests
assert predictions, coefficients, preprocessing statistics, transforms and
latent arrays within `tolerances.md`.
