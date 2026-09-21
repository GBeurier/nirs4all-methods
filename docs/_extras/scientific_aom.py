"""Reviewed scientific records for the current AOM and moment surfaces.

These records are separate from the preserved legacy bibliography because the
ABI-2 AOM/moment APIs are product-specific compositions.  Where no paper
defines that exact surface, the record says so and points to its implementation
instead of implying that a related AOM publication validates it.
"""
from __future__ import annotations


_REPO = "https://github.com/GBeurier/nirs4all-methods/blob/main/"
_AOM_PAPER = (
    "Beurier, G. et al. (2026). *AOM-PLS / POP-PLS* paper companion, "
    "arXiv:2605.13587, https://arxiv.org/abs/2605.13587. The product "
    "variants below are implementation-specific extensions; the paper does "
    "not by itself specify their ABI-2 orchestration."
)


def _record(
    title: str, paper: str, principle: str, use_cases: str, limitations: str,
    implementation: str, provenance: str,
) -> dict[str, str]:
    return {
        "title": title,
        "paper": paper,
        "principle": principle,
        "use_cases": use_cases,
        "limitations": limitations,
        "implementation": implementation,
        "provenance": provenance,
    }


SCIENTIFIC_CONTENT: dict[str, dict[str, str]] = {
    "aom_pop_calibration": _record(
        "Versioned AOM branch calibration",
        "No canonical publication defines this complete ABI-2 calibration protocol. "
        + _AOM_PAPER,
        "Evaluate the declared raw, SNV, and MSC branches fold-locally with the "
        "strict10 operator bank, select the branch, chain, and PLS component or "
        "Ridge penalty by training-fold RMSE, then refit the selected configuration "
        "on every calibration row.",
        "Reproducible global or screened AOM calibration when branch state, candidate "
        "identity, and the final refit must share one versioned native contract.",
        "The fast protocol screens candidates and does not prove that discarded "
        "candidates are inferior. Selection scores require an outer assessment, and "
        "sample-adaptive SNV or MSC branches cannot be reduced to one raw-input affine map.",
        "`n4m.model_selection.aom_calibration` exposes the global and Fast AOM PLS/Ridge "
        "estimators; C ABI `n4m_model_selection_aom_calibration_fit` and "
        "`n4m_model_selection_aom_calibration_predict`.",
        _REPO
        + "cpp/src/core/aom_calibration.cpp; "
        + _REPO
        + "cpp/src/c_api/c_api_aom_calibration.cpp; "
        + _REPO
        + "bindings/python/src/n4m/model_selection/aom_calibration.py",
    ),
    "aom_pop_linear_stack_compress": _record(
        "Affine linear-stack compression",
        "No separate publication defines this ABI-2 deployment operation; it is the "
        "exact affine composition used by the versioned AOM stack contract. Source "
        "contract: "
        + _REPO
        + "cpp/include/n4m/ensemble.h.",
        "Given base coefficient columns B, base intercepts b, meta weights w, and a "
        "meta intercept c, compute the deployment predictor with coefficients B w and "
        "intercept b w + c, preserving every output column without refitting.",
        "Collapsing an already fitted fully affine stack into one portable coefficient "
        "matrix and intercept vector for lower-cost replay.",
        "Every preprocessing and base predictor must already be affine in the supplied "
        "input coordinates. The operation performs no fitting, validation, or leakage check "
        "and cannot compress nonlinear or sample-adaptive transformations.",
        "`n4m.ensemble.compress_linear_stack`; C ABI "
        "`n4m_ensemble_linear_stack_compress`.",
        _REPO
        + "cpp/src/core/linear_stack.cpp; "
        + _REPO
        + "cpp/src/c_api/c_api_linear_stack.cpp; "
        + _REPO
        + "bindings/python/src/n4m/_impl/linear_ridge_stack.py",
    ),
    "aom_pop_linear_ridge_stack": _record(
        "Nested-OOF linear Ridge stack",
        "No canonical publication defines this exact AOM candidate bank and affine-export "
        "surface; it is a product-specific stacked generalization protocol. "
        + _AOM_PAPER,
        "Tune every affine base route inside each outer training fold, assemble only its "
        "out-of-fold predictions, fit a Ridge meta-head on that OOF design, refit the bases "
        "on all calibration rows, and compose the fitted stack into one affine predictor.",
        "Combining complementary strict-linear AOM views while retaining an auditable OOF "
        "training path and a compact deployment representation.",
        "The inner and outer fold identities are part of the estimator contract. Meta-model "
        "selection is not an unbiased test estimate, correlated bases can destabilize weights, "
        "and nonlinear or sample-adaptive bases cannot be exported as one affine predictor.",
        "`n4m.ensemble.LinearRidgeStackRegressor` orchestrates native Ridge sweeps and "
        "`n4m_ensemble_linear_stack_compress`; it has no standalone fit C ABI symbol.",
        _REPO
        + "bindings/python/src/n4m/_impl/linear_ridge_stack.py; "
        + _REPO
        + "cpp/src/core/linear_stack.cpp",
    ),
    "aom_robust_hpo": _record(
        "Strict-linear AOM robust-HPO screen",
        "No single canonical paper defines this ABI-2 compact/wide screen. " + _AOM_PAPER,
        "Evaluate a finite bank of strict-linear spectral chains with Ridge and/or PLS heads by "
        "contiguous-fold CV RMSE, select the minimum-score tuple, refit on all calibration rows, "
        "and fold its linear coefficients back to the original feature space.",
        "Fast, reproducible comparison of a declared preprocessing bank when the final model must "
        "remain a replayable linear predictor.",
        "Candidate selection must be nested inside external validation. Native v1 excludes stateful, "
        "sample-fitted, nonlinear, and source-routed transformations; a CUDA build does not make "
        "the complete candidate bank a fused GPU search.",
        "`n4m.model_selection.aom_search.aom_robust_hpo` and `AOMRobustHPOSweepRegressor`; "
        "C ABI `n4m_model_selection_robust_hpo_fit`.",
        _REPO + "cpp/src/core/aom_robust_hpo.cpp; " + _REPO + "cpp/src/c_api/c_api_method_result.cpp",
    ),
    "aom_chain_ridge_pls": _record(
        "Strict-chain AOM Ridge-PLS selector",
        "No canonical publication defines this strict-chain Ridge-PLS product surface. " + _AOM_PAPER,
        "For every declared strict-linear chain, PLS component count, and Ridge penalty, fit within "
        "each CV fold and select the lowest validation RMSE. Composition of linear chain maps permits "
        "the final Ridge-PLS coefficients to be expressed in raw input space.",
        "Selecting among a small, auditable set of linear preprocessing chains while retaining a "
        "single coefficient vector for deployment.",
        "Only chains that are genuinely linear and sample-independent can be folded back exactly. "
        "Cross-validation chooses among many candidates and therefore needs an outer assessment; it "
        "does not cover MSC/SNV/EMSC or nonlinear candidate families.",
        "`n4m.model_selection.aom_search.aom_chain_ridge_pls` and "
        "`AOMChainRidgePLSRegressor`; there is no standalone C ABI entry point.",
        _REPO + "cpp/src/core/aom_chain_ridge_pls.cpp; " + _REPO + "bindings/python/src/n4m/model_selection/aom_search.py",
    ),
    "aom_pop_aom_chain_fixed_fit": _record(
        "AOM fixed-chain refit",
        "No single canonical paper defines this fixed-candidate ABI wrapper. " + _AOM_PAPER,
        "Fit one already-selected strict AOM chain and head on the supplied calibration data. The "
        "operation separates candidate choice from the final refit, preserving the selected chain's "
        "configuration and prediction surface.",
        "Materializing a model selected by a prior, recorded search or by an external validation protocol.",
        "This operation performs no independent model selection and cannot repair leakage in the "
        "upstream choice. Its result is valid only for the recorded chain/head parameters and compatible "
        "input wavelength layout.",
        "`n4m.model_selection.aom_search.aom_chain_fixed_fit_run`; "
        "C ABI `n4m_model_selection_aom_chain_fixed_fit_run`.",
        _REPO + "cpp/src/c_api/c_api_method_result.cpp",
    ),
    "aom_pop_aom_chain_screen_refit": _record(
        "AOM chain screen and refit",
        "No single canonical paper defines this screen/refit orchestration surface. " + _AOM_PAPER,
        "Screen a declared strict-linear candidate pool using the configured inexpensive criterion, retain "
        "the candidates prescribed by the policy, then refit those candidates with exact CV before choosing "
        "the final model.",
        "Large but finite strict-chain searches where an auditable two-stage reduction is needed before "
        "costly exact validation.",
        "A screening score is not the final selection metric. The retained-pool policy changes the chance "
        "of losing the optimum and must be reported; outer validation remains necessary.",
        "`n4m.model_selection.aom_campaign.aom_chain_screen_refit_campaign`; no standalone C ABI symbol.",
        _REPO + "bindings/python/src/n4m/model_selection/aom_campaign.py",
    ),
    "aom_chain_sweep_run": _record(
        "AOM strict-chain sweep",
        "No single canonical paper defines this strict-chain ABI sweep. " + _AOM_PAPER,
        "Enumerate a declared bank of strict-linear preprocessing chains and head hyperparameters, score "
        "each by fold-local CV, and return a ranked result table with a refitted winner.",
        "Reproducible comparison of transparent, finite AOM candidate sets where ranking diagnostics are "
        "as important as the selected model.",
        "The sweep is a model-selection procedure, not a test-set estimate. It excludes stateful and "
        "nonlinear transformations from the fold-back guarantee and can be expensive as chains grow.",
        "`n4m.model_selection.aom_search.aom_chain_sweep_run`; "
        "C ABI `n4m_model_selection_aom_chain_sweep_run`.",
        _REPO + "cpp/src/core/aom_chain_sweep.cpp; " + _REPO + "bindings/python/src/n4m/model_selection/aom_search.py",
    ),
    "aom_pls_superblock": _record(
        "AOM PLS superblock",
        "No publication specifies this ABI-2 superblock wrapper; it composes the AOM-PLS family. " + _AOM_PAPER,
        "For $B$ declared operators, form outputs $Z_b\\in\\mathbb{R}^{n\\times p}$ and concatenate "
        "$Z=[Z_1|\\cdots|Z_B]\\in\\mathbb{R}^{n\\times Bp}$. Columns are centered and each block can be "
        "RMS-scaled before PLS is fitted on $Z$; the component-count grid is evaluated fold-locally and the "
        "selected head is refit on the full training rows.",
        "Multi-block spectroscopy where distinct preprocessing branches are intentionally retained as model inputs.",
        "Block construction changes the feature geometry and can overweight blocks with more columns. Centering, "
        "RMS scaling, and component selection must be fold-local; raw-space fold-back is only valid for the "
        "declared strict-linear branches.",
        "`n4m.compose.aom_superblock.aom_pls_superblock` and `AOMPLSSuperblock`; no standalone C ABI symbol.",
        _REPO + "bindings/python/src/n4m/_impl/native.py; " + _REPO + "bindings/python/src/n4m/compose/aom_superblock.py",
    ),
    "aom_ridge_pls_superblock": _record(
        "AOM Ridge-PLS superblock",
        "No canonical publication defines this combined superblock head. " + _AOM_PAPER,
        "Build $Z=[Z_1|\\cdots|Z_B]\\in\\mathbb{R}^{n\\times Bp}$ from strict operator outputs, center columns, "
        "and optionally RMS-scale each block. Ridge-PLS is then fitted on $Z$ while CV evaluates the declared "
        "component-count and $\\lambda$ grids; the winning preprocessing and head are refit on all training rows.",
        "Correlated multi-branch spectral representations where a linear, coefficient-exporting head is required.",
        "The additional Ridge penalty and number of components must be selected fold-locally without using a test set. "
        "Only the declared strict-linear branches support raw-space replay; block scaling changes the penalty geometry.",
        "`n4m.compose.aom_superblock.aom_ridge_pls_superblock` and `AOMRidgePLSSuperblock`; no standalone C ABI symbol.",
        _REPO + "bindings/python/src/n4m/_impl/native.py; " + _REPO + "bindings/python/src/n4m/compose/aom_superblock.py",
    ),
    "aom_staged_chain_campaign": _record(
        "Staged strict-chain AOM campaign",
        "No single canonical paper defines this staged orchestration layer. " + _AOM_PAPER,
        "Run several declared score-only strict-chain screens, merge and deduplicate their retained candidates, "
        "then exact-CV-refit the retained union. Optional impact and rank diagnostics compare screen and refit evidence.",
        "Campaigns that need resumable, auditable search across compact, wide, or focused strict-linear banks.",
        "Stage labels and external audits must not choose the winner; only train-side exact-CV refit may do so. "
        "A partial screen is not evidence that unvisited candidates are inferior.",
        "`n4m.model_selection.aom_campaign.aom_staged_chain_campaign`; Python orchestration over libn4m, no new C ABI.",
        _REPO + "bindings/python/src/n4m/model_selection/aom_campaign.py",
    ),
    "aom_sweep_run": _record(
        "AOM strict-chain sweep",
        "The global AOM-PLS paper motivates operator selection; this ABI-2 sweep has no separate canonical paper. " + _AOM_PAPER,
        "Build the internal compact or wide bank of supported strict-linear preprocessing chains, including selected "
        "two-operator chains, score each chain/head/parameter configuration by fold-local CV, and refit the selected "
        "configuration on all training rows while returning its descriptor and diagnostics.",
        "A bounded, reproducible comparison of the built-in strict-linear AOM chain bank with Ridge and compatible PLS heads.",
        "The bank is finite and profile-defined rather than an arbitrary chain search. Candidate selection requires an "
        "outer evaluation, and results are comparable only under the same profile, folds, head mask, and preprocessing scope.",
        "`n4m.model_selection.aom_search.aom_sweep_run`; C ABI `n4m_model_selection_aom_sweep_run`.",
        _REPO + "cpp/src/core/aom_sweep.cpp; " + _REPO + "bindings/python/src/n4m/model_selection/aom_search.py",
    ),
    "aom_operator_pls_stack": _record(
        "AOM operator latent-score Ridge stack",
        "No single paper defines this ABI-2 composition. " + _AOM_PAPER,
        "For each declared strict AOM view, standardize its $n\\times p$ matrix, fit a PLS1 projector with "
        "$k_{eff}$ retained components, and concatenate the resulting latent scores into an "
        "$n\\times\\sum_b k_{eff,b}$ design. A Ridge head is fitted on that design. Cross-validation scores and "
        "selects the component count and Ridge penalty; it does not use base-model prediction stacking.",
        "Combining complementary strict AOM views through a compact supervised latent design for a univariate response.",
        "The implementation supports one response and relies on each view's PLS1 score projection. OOF predictions "
        "are used to score candidate component/penalty settings, not as the columns of the final Ridge design. "
        "Correlated views can still make the Ridge weights unstable and selection needs an outer assessment.",
        "`n4m.ensemble.aom_operator_pls_stack` and `AOMOperatorPLSStackRegressor`; "
        "C ABI `n4m_ensemble_aom_operator_pls_stack_fit`.",
        _REPO + "cpp/src/core/aom_operator_pls_stack.cpp; " + _REPO + "bindings/python/src/n4m/ensemble.py",
    ),
    "aom_ridge_active_superblock": _record(
        "Active AOM Ridge superblock",
        "No canonical paper defines this active-superblock variant. " + _AOM_PAPER,
        "Form centered (optionally RMS-scaled) blocks $Z_b$. With centered response $Y_c$, each candidate has signature "
        "$s_b=c_bZ_b^\\top Y_c$ and default activity score $\\lVert s_b\\rVert_F^2$ (KTA and blend scores are optional). "
        "The selector keeps top-$M$ candidates while pruning highly correlated signatures, enforcing family caps/identity, "
        "then fits Ridge on the selected concatenated superblock.",
        "Sparse or constrained superblock workflows that need a linear Ridge prediction surface and branch diagnostics.",
        "Activity, signature-correlation pruning, family caps, and Ridge penalty are refit inside each fold. This is not "
        "a substitute for external validation and cannot fold non-linear or stateful branches into raw-space coefficients.",
        "`n4m.compose.aom_superblock.aom_ridge_active_superblock` and `AOMRidgeActiveSuperblock`; no standalone C ABI symbol.",
        _REPO + "bindings/python/src/n4m/_impl/native.py; " + _REPO + "bindings/python/src/n4m/compose/aom_superblock.py",
    ),
    "aom_ridge_blender": _record(
        "AOM Ridge out-of-fold simplex blender",
        "No canonical paper defines this exact n4m candidate bank; it uses non-negative stacking. " + _AOM_PAPER,
        "Generate OOF predictions for every strict-linear AOM Ridge candidate, solve non-negative weights constrained to "
        "sum to one, refit each candidate on all rows, and blend their raw-space linear coefficients.",
        "Stable combination of several plausible strict AOM Ridge candidates while retaining a replayable linear predictor.",
        "Weights must be learned from OOF rather than in-sample predictions. Highly correlated candidates can yield unstable "
        "weights; the native bank is finite and is not a generic arbitrary-estimator blender.",
        "`n4m.ensemble.aom_ridge_blender` and `AOMRidgeBlenderRegressor`; C ABI `n4m_ensemble_aom_ridge_blender_fit`.",
        _REPO + "cpp/src/core/aom_ridge_blender.cpp; " + _REPO + "bindings/python/src/n4m/ensemble.py",
    ),
    "aom_ridge_global": _record(
        "AOM Ridge global selector",
        "No canonical paper defines this strict-linear Ridge route. " + _AOM_PAPER,
        "Treat each declared strict AOM operator as a one-step chain, choose the operator and positive Ridge penalty by CV "
        "RMSE, then export the selected linear predictor in original input coordinates.",
        "A concise, auditable global preprocessing choice for Ridge calibration models.",
        "It cannot model interactions among sequential operators. A low CV score after selection is optimistic unless evaluated "
        "in an outer split; only strict linear transformations can be folded back.",
        "`n4m.model_selection.aom_search.aom_ridge_global` and `AOMRidgeGlobalRegressor`; no standalone C ABI symbol.",
        _REPO + "bindings/python/src/n4m/_impl/native.py; " + _REPO + "bindings/python/src/n4m/model_selection/aom_search.py",
    ),
    "aom_ridge_mkl_superblock": _record(
        "AOM Ridge MKL-light superblock",
        "No canonical publication defines this product-specific MKL-light surface. " + _AOM_PAPER,
        "For each centered block $Z_b$, compute $K_b=Z_bZ_b^\\top$ and its alignment "
        "$A_b=\\langle K_b,Y_cY_c^\\top\\rangle_F/(\\lVert K_b\\rVert_F\\lVert Y_cY_c^\\top\\rVert_F)$. The top-$k$ "
        "blocks receive simplex weights $w_b=\\max(A_b,0)/\\sum_b\\max(A_b,0)$; multiplying each retained block by "
        "$\\sqrt{w_b}$ before Ridge yields the final weighted superblock. If every retained alignment is nonpositive, "
        "the implementation uses uniform weights $1/k$ instead.",
        "Comparing a small collection of complementary strict spectral representations with a controlled linear head.",
        "The weighting is not equivalent to full multiple-kernel learning. The uniform fallback makes nonpositive alignments "
        "non-discriminative. Top-$k$, weights, and Ridge penalty are selected "
        "fold-locally and refit on all rows; stateful/nonlinear branches break simple coefficient replay.",
        "`n4m.compose.aom_superblock.aom_ridge_mkl_superblock` and `AOMRidgeMKLSuperblock`; no standalone C ABI symbol.",
        _REPO + "bindings/python/src/n4m/_impl/native.py; " + _REPO + "bindings/python/src/n4m/compose/aom_superblock.py",
    ),
    "aom_ridge_superblock": _record(
        "AOM Ridge superblock",
        "No canonical paper defines this ABI-2 superblock wrapper. " + _AOM_PAPER,
        "Create $B$ branch outputs $Z_b\\in\\mathbb{R}^{n\\times p}$, concatenate "
        "$Z=[Z_1|\\cdots|Z_B]\\in\\mathbb{R}^{n\\times Bp}$, center columns, optionally RMS-scale each block, "
        "then fit Ridge on the resulting superblock. Its penalty is selected and refit within the declared CV protocol.",
        "Linear calibration from multiple explicitly declared preprocessing branches.",
        "Column scale and block width affect Ridge regularization; centering, RMS scaling and penalty selection must be "
        "fold-local. The method is restricted to the declared strict-linear branch family for raw-space replay.",
        "`n4m.compose.aom_superblock.aom_ridge_superblock` and `AOMRidgeSuperblock`; no standalone C ABI symbol.",
        _REPO + "bindings/python/src/n4m/_impl/native.py; " + _REPO + "bindings/python/src/n4m/compose/aom_superblock.py",
    ),
    "models_ensembles_moment_stack": _record(
        "Moment-stack ensemble",
        "No canonical paper defines this n4m moment-stack product route; its implementation is a documented "
        "linear stack over moment-compatible candidate predictions.",
        "Build a candidate prediction matrix from the declared moment-compatible routes and fit a linear meta-level "
        "combination using out-of-fold predictions, so the stack learns weights without reusing a candidate's own fit rows.",
        "Combining a fixed, auditable set of linear/moment-compatible spectral predictors when complementary errors are expected.",
        "The stack requires a valid OOF plan and can overfit when many highly correlated candidates are included. It is a "
        "product composition rather than a claim of a new statistical estimator; assess it in an outer validation loop.",
        "`n4m.ensemble.moment_stack` and `MomentStack`; no standalone C ABI symbol.",
        _REPO + "bindings/python/src/n4m/ensemble.py; " + _REPO + "bindings/python/src/n4m/_impl/moment_facade.py",
    ),
    "models_regularized_ridge": _record(
        "Ridge regression",
        "Hoerl, A. E. & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. "
        "Technometrics 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634.",
        "Estimate $\\hat{\\beta}_\\lambda=(X^\\top X+\\lambda I)^{-1}X^\\top y$. The positive L2 penalty makes the "
        "normal equations invertible under collinearity and shrinks unstable coefficients toward zero without selecting features.",
        "A linear calibration baseline for strongly collinear spectra or as the head of a strict-linear AOM route.",
        "The penalty changes with feature scale, so scaling belongs inside the fitting/CV protocol. Ridge retains all variables "
        "and its selected $\\lambda$ requires outer validation for an unbiased performance estimate.",
        "`n4m.estimators.regression.regularized.ridge` and `RidgeRegressor`; C ABI `n4m_estimators_ridge_fit`.",
        _REPO + "cpp/src/core/ridge.cpp; " + _REPO + "bindings/python/src/n4m/estimators/regression/regularized.py",
    ),
    "utilities_moments": _record(
        "Moment sufficient-statistics utilities",
        "No single canonical paper defines the ABI utility surface; it implements standard first- and second-moment "
        "sufficient statistics used by linear-model updates.",
        "Accumulate quantities such as sample count, feature sums, cross-products, and response cross-products so compatible "
        "linear fits can be computed or updated without retaining every original row.",
        "Auditable aggregation and replay of moment-compatible linear calibration calculations.",
        "Moments do not retain row identities, nonlinear transformations, or arbitrary fold structure. Merging statistics is only "
        "valid for identically defined feature spaces and preprocessing; it cannot recover information discarded upstream.",
        "`n4m.lowlevel.moments`; C ABI symbols in the `n4m_lowlevel_moments_*` family.",
        _REPO + "bindings/python/src/n4m/lowlevel/moments.py; " + _REPO + "cpp/include/n4m/lowlevel.h",
    ),
    "utilities_sweep": _record(
        "Native candidate sweep utility",
        "No canonical scientific paper defines this orchestration utility; it is a reproducible enumeration and scoring surface.",
        "Evaluate every configuration in a declared finite grid under the supplied validation plan, collect scores and diagnostics, "
        "then expose a deterministic ranked table rather than silently choosing a configuration.",
        "Small, explicit hyperparameter or method grids where the complete candidate record is needed for audit and comparison.",
        "A sweep increases selection-induced optimism; use nested or held-out evaluation. It cannot make an under-specified grid "
        "represent an untested method family and should not be confused with Bayesian optimization.",
        "`n4m.model_selection.sweep`; C ABI `n4m_model_selection_sweep_run`.",
        _REPO + "bindings/python/src/n4m/model_selection/sweep.py; " + _REPO + "cpp/include/n4m/model_selection.h",
    ),
}
