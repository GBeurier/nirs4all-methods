# AOM Methods Gap Analysis

Date: 2026-06-05

This audit separates what already exists in `nirs4all-methods` from what still
had to be added from the AOM / moment exploration.

Hard invariant: deployable methods must not route by dataset name, source name,
database name, dataset id, or equivalent identity labels.

## Already In nirs4all-methods

Native/catalogued primitives:

- AOM preprocessing and AOM global PLS selection:
  - `aom_pop.aom_preprocessing`;
  - `aom_pop.aom_pls`;
  - `aom_pop.pop_pls`.
- Model heads or neighboring model families:
  - PLS, CPPLS, Continuum Regression, ECR, Ridge / Ridge-PLS, robust and
    weighted PLS, PCR via the existing method set.
  - ABI-close Python helper functions now expose reusable moment-model heads:
    `n4m.ridge`, `n4m.cppls`, `n4m.continuum_regression`, and `n4m.ecr`.
- Preprocessing families needed by the lab evidence:
  - Savitzky-Golay;
  - finite differences / derivatives;
  - Norris-Williams;
  - detrend;
  - Gaussian smoothing;
  - SNV, MSC, EMSC, local/weighted/piecewise variants;
  - baseline correction families;
  - FCK static transformer.

Python/reference Phase 7f objects already present before this pass:

- `AOMStructuralPolicy`;
- `AOMStructuralPolicyWithPgt1200Admissions`;
- `AOMStructuralPolicyWithP700ProtocolUnified`;
- `AOMStructuralPolicyWithP700BlockLocalAdmission`;
- `AOMControlSelector`;
- `AOMTrueBankEndpointPortfolio`;
- `AOMMidPEndpointStack`;
- `AOMEndpointMarginStabilityGate`;
- `AOMFallbackBlendGate`.
- `AOMRidgeBlender`;
- `AOMOperatorPLSStack`.

## Missing Before This Pass

The target repo had primitives and selector skeletons, but it did not yet have
the product-facing method layer needed by an end user:

- no built-in AOM preprocessing chain bank reproducing the useful diversity of
  the lab campaigns;
- `AOMControlSelector` was Ridge-only;
- no direct `fit/predict` robust-HPO estimator that builds the bank, screens
  Ridge/PLS heads, and reports the selected chain;
- no compact end-user preset;
- no wide/lab-style preset for controlled reproduction of larger preprocessing
  diversity.

## Added In This Pass

Python/reference methods:

- `AOMPreprocessingChain`:
  - fold-safe sequential preprocessing transformer;
  - deep-copies and fits each step inside CV;
  - used as the standard chain object for generated banks.
- `build_aom_control_chain_bank(profile=...)`:
  - `compact`: fast end-user bank;
  - `robust`: broader default bank;
  - `wide` / `lab` / `moment_lab`: larger lab-style bank;
  - includes the recurring families: raw, row center, SNV, MSC, EMSC,
    detrend, SavGol variants, Norris-Williams, derivatives, Gaussian,
    robust/area normalization, and selected short compositions.
- `AOMControlSelector` now supports:
  - Ridge heads;
  - PLS1 heads;
  - fold-local chain fitting;
  - mean/median/quantile/max criteria;
  - rank-mean or simple ensembles over top candidates.
- `AOMRobustHPORegressor`:
  - configurable end-user estimator;
  - builds or accepts a chain bank;
  - screens Ridge and/or PLS heads;
  - exposes `selection_report_`;
  - accepts metadata for audit only.
- `AOMRobustHPOCompact`:
  - fast compact preset for users.
- `AOMRobustHPOWide`:
  - larger lab-style preset for reproducing broad preprocessing diversity;
  - not recommended as the default until benchmarked from the integrated
    package.
- `NativeAOMRobustHPORegressor`:
  - sklearn-style wrapper over the native strict-linear robust-HPO ABI;
  - uses `input_coefficients` folded into the original feature space;
  - predicts on new spectra without replaying the selected preprocessing chain
    in Python.
- `AOMRidgeBlender`:
  - fold-safe OOF simplex blend ported from the AOM-Ridge headline family;
  - supports explicit estimator/factory candidates or a default chain+Ridge
    pool;
  - Python reference layer for explicit custom candidates; the strict-linear
    compact/wide Ridge pool is also available natively as
    `n4m.aom_ridge_blender` / `n4m_aom_ridge_blender_fit`.
- `AOMOperatorPLSStack`:
  - guarded fixed-operator PLS score stack with Ridge head;
  - optional OOF baseline gate for false-positive control;
  - default operator bank is restricted to strict-linear fixed matrices;
  - Python reference layer for custom operators and baseline gates; the
    compact/wide strict-linear PLS1 score stack is also available natively as
    `n4m.aom_operator_pls_stack` / `n4m_aom_operator_pls_stack_fit`.
- Native `n4m.aom_sweep_run` / `n4m_aom_sweep_run`:
  - configurable compact/wide strict-linear AOM chain sweep;
  - screens custom Ridge lambda grids and PLS component grids;
  - accepts explicit fold ids;
  - returns a candidate table with `candidate_id`, `chain_id`, `head_id`,
    `param`, `cv_rmse`, plus selected OOF/final predictions.
- Native `n4m.aom_chain_sweep_run` / `n4m_aom_chain_sweep_run`:
  - accepts caller-provided strict-linear preprocessing chains through flat
    chain/operator/parameter offsets;
  - supports identity, detrend, Savitzky-Golay, Norris-Williams, finite
    difference, Whittaker and FCK in this fold-safe path;
  - rejects stateful or non-strict operators such as SNV/MSC/EMSC/OSC/EPO.
- AOM route policy switch:
  - `moment_policy="auto"` uses guarded exact operator-moment routes;
  - `moment_policy="materialized"` or `"legacy"` forces the old
    materialized-chain screen for route timing comparisons and small-cell
    production guards;
  - `moment_policy="force_moments"` / `"moments_only"` rejects any
    candidate-screen route that would leave operator moments.
- AOM score-only output mode:
  - `score_only=True` keeps candidate scores, selected ids, route counters and
    fold ids while skipping selected-model outputs for broad ranking passes;
  - it avoids selected-model output refits and OOF/model buffers in
    operator-moment and materialized candidate-screen routes, but does not yet
    fuse the fold-local materialized scoring fits.
- ABI-close Python wrappers for the historical native AOM selectors:
  - `n4m.aom_pls` / `n4m.aom_global_select` call `n4m_aom_global_select`;
  - `n4m.pop_pls` / `n4m.aom_per_component_select` call
    `n4m_aom_per_component_select`;
  - both build the compact strict-linear operator bank by default, accept
    caller-provided strict operators, and build fold-safe validation plans from
    explicit `fold_ids` or contiguous CV folds.
- Reusable native AOM/POP selected models:
  - both historical selectors now export `input_coefficients` and `intercept`;
  - `n4m.sklearn.NativeAOMPLSRegressor` and
    `n4m.sklearn.NativePOPPLSRegressor` predict on new `X` from those
    input-space linear states.
- Native `n4m.aom_ridge_blender` / `n4m_aom_ridge_blender_fit`:
  - blends the native compact/wide strict-linear AOM Ridge candidate pool from
    fold-local OOF predictions;
  - returns candidate predictions and simplex weights for audit;
  - returns final `input_coefficients` and `intercept` for replay in original
    feature space;
  - is wrapped by `NativeAOMRidgeBlenderRegressor` for sklearn-style reuse;
  - requires strictly positive Ridge lambdas and does not cover custom Python
    estimator candidates.
- Native `n4m.aom_operator_pls_stack` / `n4m_aom_operator_pls_stack_fit`:
  - fits fold-local PLS1 score projectors over compact/wide strict-linear AOM
    operator views;
  - selects `(n_components, alpha)` by train-only CV criterion;
  - returns final stack features and Ridge head for audit;
  - returns `input_coefficients` and `input_intercept` for direct replay in
    original feature space;
  - is wrapped by `NativeAOMOperatorPLSStackRegressor` for sklearn-style reuse.

## Still Not In nirs4all-methods

Still intentionally absent or deferred:

- broad GPU/moment grinder:
  - remains a lab capacity-audit tool, not an end-user product method;
- source-route or dataset-name lookup:
  - explicitly forbidden;
- test-tuned property routes:
  - require held-out validation before becoming a deployable selector;
- native ABI endpoints for every historical AOM reference object:
  - the Python portfolio objects are now available for the main source-free
    route/policy/gate/blend/operator-stack patterns, but most are still
    pre-ABI and outside the strict native catalog; `n4m.aom_ridge_blender` is
    the native exception for the strict-linear AOM Ridge blend, and
    `n4m.aom_operator_pls_stack` is the native exception for the strict-linear
    PLS1 score stack.
- custom batched GPU screen for hundreds of thousands of preprocessing chains:
  - not shipped; the current product method builds in CUDA-enabled `libn4m`
    and uses the existing native model path, but it is not a fused GPU grinder.
- batched sweep engine ABI:
  - `n4m_moments_*` is now shipped as a raw/centered sufficient-statistics
    layer with exact row-subset subtraction and train-only recentering;
  - `n4m_sweep_run` now ships Ridge CV plus compatible single-target
    NIPALS/regression PLS1 component screening from train/held-out moments,
    with materialized prefix fallback for other PLS regimes;
  - `n4m_aom_sweep_run` now ships a configurable AOM strict-linear compact/wide
    sweep over those Ridge/PLS grids; Ridge candidate rows use exact
    operator-moment scoring when `p <= n_train`, including in mixed Ridge+PLS
    sweeps, compatible single-target NIPALS PLS1 rows score from moments, and
    identity/SavGol/Norris/finite/Gaussian/FCK chains now have a banded moment route
    outside the old dense `p <= 48` guard; `detrend_poly` chains use an exact
    structured low-rank projection moment route and Whittaker chains use an
    exact structured pentadiagonal solve route; both can compose with those
    banded local operators; CPU `auto` routes Ridge `p > n_train` rows through
    the exact materialized dual-Ridge scorer and PLS `min_train < 4p` rows
    through the exact materialized prefix scorer while CUDA `auto` keeps the
    moment route;
  - `n4m_aom_chain_sweep_run` now ships strict-linear user-defined chain
    descriptors with the same hybrid Ridge/PLS1-moment behavior and
    materialized fallbacks, or fail-fast strict behavior with
    `moment_policy="force_moments"`;
  - batched IKPLS and fused operator-moment updates are still design/backlog
    items, so the full 200k-chain AOM moment sweep remains outside the product
    ABI.

## Current Product Recommendation

Expose three levels to users:

- `n4m.aom_robust_hpo(..., profile="compact")` for the ABI-stable fast product
  screen; use `NativeAOMRobustHPORegressor` when a fitted sklearn-style
  predictor is needed from the same native result;
- `n4m.aom_sweep_run(...)` when the user wants to control Ridge lambdas, PLS
  component grids or explicit CV folds over the same native strict-linear bank;
- `n4m.aom_chain_sweep_run(...)` when the user wants to define the
  strict-linear preprocessing chains directly; use `moment_policy` to compare
  `auto` moment routes against the legacy materialized route, or
  `force_moments` to assert that the candidate screen did not leave moments;
  use `score_only=True` for first-pass chain ranking;
- `n4m.aom_pls(...)` and `n4m.pop_pls(...)` when the user wants the older
  global/per-component AOM-PLS selectors over a compact strict operator bank;
  use `NativeAOMPLSRegressor` or `NativePOPPLSRegressor` for sklearn-style
  reuse on new spectra;
- `n4m.aom_ridge_blender(...)` when the user wants a native Ridge-only OOF
  simplex blend over the compact/wide strict-linear AOM bank; use
  `NativeAOMRidgeBlenderRegressor` for fitted reuse;
- `n4m.aom_operator_pls_stack(...)` when the user wants a native PLS1
  operator-score stack with a Ridge head over the compact/wide strict-linear
  AOM bank; use `NativeAOMOperatorPLSStackRegressor` for fitted reuse;
- `n4m.aom_robust_hpo(..., profile="wide")` for a broader native strict-linear
  screen;
- `AOMRobustHPORegressor(profile="wide")` or `AOMRobustHPOWide` when fold-local
  stateful chains are required in Python/reference experiments.

The canonical catalog now contains `aom_pop.robust_hpo`,
`aom_pop.ridge_blender`, and `aom_pop.operator_pls_stack` through public native
ABI symbols and timing rows. The broader Python portfolio/gating objects remain
reference-layer APIs until each gets a native ABI contract and timing row.
`aom_pop.robust_hpo` also exposes `input_coefficients`, `intercept`,
dimension diagnostics, and the native sklearn replay wrapper.
`aom_pop.ridge_blender` now does the same for the weighted Ridge blend.
`aom_pop.operator_pls_stack` exposes `input_coefficients` and
`input_intercept` for the folded single-target stack.
