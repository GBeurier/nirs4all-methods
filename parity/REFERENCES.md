# Parity references — donors, dependencies, and how to avoid matrix holes

This document records **which external library is the reference (donor) for each
method family**, the **dependencies** needed to run those references, and the
**setup + doctor** scripts that guarantee the divergence/benchmark matrix is
never left with a hole *because a reference library was not installed*.

See [`../PRODUCTION_AUDIT.md`](../PRODUCTION_AUDIT.md) §1 for the full parity model
(donor → snapshot → oracle; C++ native + external libs + backends vs oracle at
**1e-8**; bindings vs C++ native at **1e-12**).

## The rule

> Every method has **one documented donor** chosen as the most credible
> implementation for the community. Methods that originate in **nirs4all**
> (augmentations, scatter/baseline preprocessing, splitters, native AOM/moment
> orchestration, and direct native heads such as Ridge or the multibranch neural
> regressor) use **nirs4all** as the donor. The only exception is a genuine *paper-only* method, which must be
> explicitly marked as such.

A "missing value" in the matrix must always mean *paper-only / not-applicable*,
**never** "the reference library was not installed". The doctor below enforces that.

## Row L2 normalization (project 1.3.16)

`preprocessing.scaling.row_l2_normalize` preserves the per-row Euclidean operation
from the Web donor `web-app/src/engine/algo/preprocessing.ts` (`normalize`, L2
branch). Its native analytic cases and real-WASM role cases cover `[3,4]`, zero
rows, batch/subset independence and scaled finite extremes; NaN and infinity are
explicitly refused. This is distinct from the column-wise `Normalize` method.
The named witnesses are `cpp/tests/test_estimator.cpp::test_row_l2_normalize` and
`bindings/js/test/run_row_l2_normalize.mjs`. These targeted checks do not claim
shared-fixture coverage or bitwise equivalence for the donor's naive extreme-value
arithmetic. No existing numerical fixture, registry lockfile or frozen historical
bibliography is rewritten for this repair.

## Multibranch neural regressor and block RMS scaling (project 1.3.17)

`models.neural.multibranch_regressor` is an **original native Methods
composite** (dense / conv1d / conv2d branches, concatenated learned embeddings,
common head, joint MSE backpropagation with Adam and decoupled weight decay,
PCG64-seeded initialization and minibatch order). It is not copied from a
nirs4all SDK implementation, PyTorch, Keras or any other framework, so it has
**no external donor and no external-parity claim**; it is not a paper-only
method either, because its owning implementation is executable and qualified.
Like the direct native Ridge head, it is listed explicitly in
`NIRS4ALL_DONOR_METHODS` (`catalog/scripts/validate.py`) as a nirs4all-origin
method. Owning source: `cpp/src/core/neural/multibranch_network.{hpp,cpp}` and
the estimator adapter `cpp/src/core/estimator/neural_adapters.cpp`.

Its evidence is analytic and cross-native, not reference parity:

- `cpp/tests/test_multibranch_regressor.cpp`: central finite differences against
  the backpropagated gradient for every parameter tensor of a three-branch
  (conv1d `[16]`, dense `[3]`, conv2d `[5,5,2]`) network; supervised training
  lowers the loss and changes every convolution/embedding tensor, `epochs = 0`
  keeps the seeded initialization; N4ME export/import predicts and transforms
  bit-identically without refitting and re-exports the same bytes; column-major
  and padded row-stride inputs equal row-major results; identical seeds refit
  identically; non-finite data and undeclared conv2d shapes are refused.
- `cpp/tests/test_multibranch_gradients.cpp`: an independent integrator probe of
  the same kernel (finite differences with and without a hidden head layer,
  noncontiguous minibatch rows, joint learning of both convolution branches).
- `bindings/js/test/run_multimodal_teaching.mjs`: the real WASM build reproduces
  the native public-ABI prediction checksum of the same deterministic fixture
  within `1e-9` relative, replays the N4ME state exactly and transports a
  high-bit u64 seed.

`preprocessing.scaling.block_rms` stays in the blanket nirs4all preprocessing
category. Its analytic witness is the training-fitted scalar
`s = sqrt(sum(X_train^2) / n_train)` without centering: for training rows
`[[3, 4], [0, 0]]`, `s = sqrt(12.5)`; unseen rows are divided by that fitted
scalar without changing the state, and an all-zero training block keeps scale
one (`bindings/python/tests/test_estimator_roles.py::test_block_rms_uses_training_scalar_and_replays`
and the BlockRMS section of `bindings/js/test/run_multimodal_teaching.mjs`).

No registry entry, truth-source lockfile, numerical fixture or frozen
historical bibliography is added or rewritten for these methods.

## Spectral conversions (project 1.5.2)

`parity/spectral_conversions/generate.py` runs every signal and axis conversion
used to prepare spectra for other software against an independent external
reference and freezes the reference outputs in
`parity/fixtures/spectral_conversions_external_v1.json`, replayed without the
references by `bindings/python/tests/test_spectral_conversions.py`.

| Method | External reference | Agreement |
|---|---|---|
| `to_absorbance` (fraction, percent) | photobiology `T2A` (R) | 3e-15 relative |
| `from_absorbance` (fraction, percent) | photobiology `A2T` (R) | 2e-15 relative |
| `kubelka_munk` (fraction, percent) | mdatools `prep.ref2km` (R) | 7e-15 absolute (see the SSE2 note) |
| `from_kubelka_munk` | `stats::uniroot` over `prep.ref2km` (R) | 4e-16 relative |
| `percent_to_fraction`, `fraction_to_percent` | astropy.units | 2e-16 relative |
| `reflectance_from_references` | prospectr `readASD`, asdreader `get_spectra` on three MIT-licensed ASD files | identical |
| `splice_correction` | specdal `jump_correct` (additive); spectrolab `match_sensors` (multiplicative, tapered) | identical; 4e-16 relative |
| `resampler` (linear, nearest, cubic) | scipy `interp1d`, `CubicSpline` not-a-knot | identical; 3e-15 relative |
| `axis_unit_conversion` (six unit pairs, values and spectral densities) | astropy `spectral()`, `spectral_density()` | 3e-16 axis, 7e-16 density |

`KubelkaMunk`'s SSE2 path (x86-64) evaluates the equivalent `0.5/R - 1 + 0.5R`:
its relative error near R = 1 reaches a few 1e-12 while the absolute error stays
below 1e-14; other targets evaluate `(1 - R)^2 / (2R)`.

`parity/spectral_conversions/vendor_check.py` compares
`reflectance_from_references` with the reflectance or transmittance written by
vendor software in the public sample files of nirs4all-formats (not copied
here): identical to AvaSoft 6/7 binary files (.TRM) and OceanView, within the
two-decimal rounding of SVC and of the AvaSoft 6/7 long text export. The
AvaSoft 8 text export (`avasoft8.txt`) differs by up to 0.6 % relative where
the reference has signal, beyond its printed precision: AvaSoft 8 applies
processing the file does not describe. Spectral Evolution DARWin applies its own detector
splice/overlap processing, which `splice_correction` approaches (median within
2.3e-4 reflectance) but does not reproduce. References needed to regenerate:
R photobiology 0.14.3, mdatools 0.15.0, prospectr 0.2.8, asdreader 0.1.3,
spectrolab 0.0.19; Python astropy, scipy, pandas, specdal 0.2.1.

## Explainability (ABI 2.23)

`diagnostics.explain.{model_agnostic,linear_shap,tree_shap}` take the nirs4all
SDK SHAP analyzer, i.e. the Python `shap` package, as donor. Exact Shapley
values, permutation/kernel convergence and path-dependent TreeSHAP (including a
feature repeated along one path) are checked against brute-force oracles in
`cpp/tests/test_explain.cpp`; `bindings/python/tests/test_explain.py` pins
parity with `shap` (`LinearExplainer`, `TreeExplainer(tree_path_dependent)` on
scikit-learn trees/forests/boosting, `PermutationExplainer`, `ExactExplainer`).
Sampled explainers (permutation, kernel, LIME) are statistical: parity is a
tolerance on the estimate, not bitwise agreement.

ABI 2.24 additions are pinned in `cpp/tests/test_importance.cpp` against direct
formulas: the local gradient explainer equals linear SHAP (mean reference) on
affine models and `2x` on a quadratic; VIP, the target-projection selectivity
ratio (Rajalahti et al. 2009) and standardized coefficients are recomputed from
the exported latent structure; raw-input pipeline coefficients reproduce the
pipeline predictions (1e-10 relative); profile Spearman matrices equal
`scipy.stats.spearmanr` 1.15.3 on tied data.

## Setup (one command)

```bash
N4M_R_ENV=/home/delete/miniconda3/envs/pls4all_r scripts/setup_parity_references.sh
```

Installs all reference dependencies into the parity venv + R env and finishes by
running the doctor with `--strict`. Idempotent.

## Doctor (verify before any sweep / in CI)

```bash
N4M_R_ENV=/home/delete/miniconda3/envs/pls4all_r \
  parity/python_generator/.venv/bin/python parity/scripts/check_references.py --strict
```

Prints, per dependency layer, what is installed; exits non-zero (with `--strict`)
if any **required** reference dependency is missing. `--json` for machine output.

## Dependency layers

| Layer | Where | What |
|---|---|---|
| **Python references** | parity venv (`parity/python_generator/.venv`) | `scikit-learn`, `numpy`, `scipy`, `auswahl` (RandomFrog/IRF/VIP_SPA/VISSA), `pyswarms` (BinaryPSO), `tensorly` (N-PLS), `ikpls`, `diPLSlib==2.5.0` (di_pls — pinned; the registry shims target this API) |
| **R references** | `$N4M_R_ENV` (R 4.3.x) | `plsVarSel`, `enpls`, `multiblock`, `plsRglm`, `plsRcox`, `chemometrics`, `JICO`, `ropls`, `mixOmics`, `sgPLS`, `mdatools`, `mboost`, `softImpute`, `survival`, `survAUC`, `rms`, `permute`, `prospectr`, `pls`, `spls`, `OmicsPLS`, `kernlab`, `corpcor`, `genalg`, `baseline`, `EMSC`, `wavelets`, `plsdepot`, `mvdalab`, `HDANOVA`, `multiway` |
| **Octave / libPLS** | Octave (from `$N4M_R_ENV` or system) + vendored `bindings/octave/libPLS_1.95` (called via a **one-shot `octave-cli --eval`**, not a persistent oct2py session) | `ecr` (Elastic Component Regression), `cars` (CARS selector) |

### Notes / footguns

- **`OCTAVE_HOME` is required for a conda/relocatable Octave.** Without it, `octave-cli`
  cannot find its standard m-file library and `version`/`pkg` are undefined, which makes
  the bridge fail. `_resolve_octave()` (registry) and the doctor derive `OCTAVE_HOME` as the
  prefix above `bin/octave-cli` automatically; override with `OCTAVE_EXECUTABLE` / `OCTAVE_HOME`.
  The libPLS bridge uses a **one-shot `octave-cli --eval`** (`_octave_run_libpls`); a persistent
  `oct2py` session stalls on Octave 10, so `oct2py` is now **optional/unused**.
- **R `mbpls` is intentionally NOT installed.** The PyPI/CRAN `mbpls` is broken against
  sklearn ≥1.8 and unused — `mb_pls` uses the in-tree `nirs4all...sklearn.mbpls` reference.
- **`rpy2` is not used** — the registry shells out to `Rscript`, not in-process R.
- The donor↔method assignment is currently authoritative in
  `benchmarks/parity_timing/registry.py` (`MethodSpec.python_reference` / `r_reference` /
  `extra_references`). `PRODUCTION_AUDIT.md` §7 tracks migrating this into the catalog
  (`catalog/methods/<id>.yaml`) as the single source of truth, with a `validate.py` gate
  that fails on any method lacking a documented reference that is not marked `paper_only`.
- The ≈104 non-PLS donor methods use **nirs4all** as the donor, frozen as IEEE-754 hex
  fixtures under `parity/fixtures/` and asserted bit-exact by the C++ doctests.

## Per-method donor summary (PLS family)

| Donor (`reference_kind`) | Count | Examples |
|---|---:|---|
| external lib | 65 | sklearn (`pls`, `pcr`, `pls_lda`), R `pls` (`cppls`, solver variants), R `plsVarSel` (selectors), R `multiblock` (`so_pls`), matlab/libPLS (`ecr`, `cars`), `diPLSlib` (`di_pls`), `ikpls` (cross-check) |
| `nirs4all_sanctioned` | 5 | `aom_pls`, `aom_preprocess`, `lw_pls`, `mb_pls`, `pop_pls` |
| `paper_only` (no reference) | 3 | `fused_sparse_pls`, `mir_pls`, `sipls_select` |
| nirs4all (donor methods) | ≈135 | augmentation, preprocessing, filters, splitters, native `aom_pop` orchestration, direct Ridge, native multibranch regressor |
