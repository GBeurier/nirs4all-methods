# Phase 7f - AOM robust HPO portfolio integration

Status: proposed from `moment_sweep_proto` stop/go audit.

Initial local scaffold:

- `bindings/python/src/n4m/sklearn/aom_portfolio.py`
  contains the first source-free `AOMStructuralPolicy` reference object, the
  train-only `AOMEndpointMarginStabilityGate`, and a dependency-light
  `AOMControlSelector` with fold-local chains and a Ridge head.
  It now also contains `AOMTrueBankEndpointPortfolio` and
  `AOMMidPEndpointStack` reference objects for guarded endpoint selection and
  train-OOF pairwise blending, plus an optional `AOMFallbackBlendGate`.
- The same module exposes source-free structural presets:
  `AOMStructuralPolicyWithPgt1200Admissions`,
  `AOMStructuralPolicyWithP700ProtocolUnified`, and
  `AOMStructuralPolicyWithP700BlockLocalAdmission`.
- `bindings/python/tests/test_aom_structural_policy.py`
  validates feature-count routing, metadata invariance, stable-endpoint
  admission, stack fallback, source-free preset routes, Ridge-CV chain
  selection, true-bank endpoint selection, mid-p OOF blending, optional
  fallback blending, and the absence of the legacy Beer-admission public name.

Goal: integrate the validated AOM selector/portfolio stack as a deployable
robust-HPO layer, without claiming AOM-oracle or TabPFN-oracle parity.

## Decision context

The `moment_sweep_proto` campaign tested massive preprocessing screens,
repeated-CV verification, stable survivor selection, and PLS/Ridge/PCR/
Continuum/ECR bagging/stacking. The useful result is not a new universal
linear-moment champion. The useful result is a bounded integration decision:

- broad linear-moment grinding improves some endpoint choices but plateaus as
  a general oracle-recovery method;
- train-property routes can produce no-harm/local gains, but cannot reliably
  recover the test-oracle endpoint and fail on some capacity-wall datasets;
- the existing AOM structural/endpoint stack is strong integration material
  and should be ported as a robust default/portfolio.

Primary audit artifacts:

- `nirs4all-lab/moment_sweep_proto/artifacts/linear_moment_stop_go_decision.md`;
- `nirs4all-lab/moment_sweep_proto/artifacts/linear_moment_gain_history.csv`;
- `nirs4all-lab/moment_sweep_proto/artifacts/nirs4all_methods_aom_integration_handoff.md`;
- `nirs4all-lab/differentiable-aom/docs/DEPLOYABLE_AOM_INTEGRATION_SCORECARD.md`.

## Hard invariant

Deployable routing must not depend on dataset name, database name, source name,
dataset id, or any equivalent identity label.

Those fields may appear in audit reports only. Changing them must not change
the selected route or predictions.

Required tests for every selector/portfolio route:

- fit the same `X, y` twice with different `dataset_id` / `database_name`;
- assert the selected route is unchanged;
- assert predictions are unchanged within the method tolerance;
- allow only the audit report text/metadata fields to differ.

## Existing n4m anchors

The core AOM operator path already exists and should be reused:

- strict AOM operators: `cpp/src/core/aom_operators.cpp`;
- AOM global / POP selection: `cpp/src/core/aom_selection.cpp`;
- C ABI wrappers: `cpp/src/c_api/c_api_aom_selection.cpp`;
- public declarations: `cpp/include/n4m/pls.h`;
- benchmark runner: `benchmarks/runners/aom_global.py`;
- roadmap phases: 6a, 6b, 6c, 6d, 6e, 6f.

This phase is not a rewrite of Savitzky-Golay, Norris-Williams, detrend,
Whittaker, FCK, or the current AOM global selection kernel. It is the
selector/portfolio layer above those primitives.

## Integration order

### 1. Python reference portfolio

Add high-level Python reference estimators, backed by existing C ABI primitives
where possible:

- `AOMStructuralPolicy`;
- `AOMStructuralPolicyWithPgt1200Admissions`;
- source-free `AOMStructuralPolicyWithP700BlockLocalAdmission`;
- `AOMControlSelector`;
- `AOMTrueBankEndpointPortfolio`;
- `AOMEndpointMarginStabilityGate`;
- `AOMMidPEndpointStack`.

Do not expose the legacy public name
`AOMStructuralPolicyWithBlockLocalBeerAdmissions`. The lab implementation is
source-free internally, but the name implies a source identity route.

### 2. Method catalog and docs

The canonical method catalog requires public `n4m_*` ABI symbols. The current
Phase 7f objects are Python/reference estimators, so they must not be added to
`catalog/methods.yaml` until a native ABI route/predict/report surface exists.

Track catalog-ready future entries in:

- `docs/architecture/aom_robust_hpo_catalog_ready_manifest.csv`.

Track the pre-ABI diffusion dashboard in:

- `docs/architecture/aom_robust_hpo_phase7f_dashboard.csv`.

After native ABI support exists, add catalog entries only for methods whose
route is train-only:

- `aom.structural_policy`;
- `aom.control_selector`;
- `aom.truebank_endpoint_portfolio`;
- `aom.midp_endpoint_stack`;
- `aom.endpoint_margin_stability_gate`.

Every catalog description must say:

- robust AOM HPO / portfolio target;
- no dataset/source/database-name routing;
- not an AOM-oracle or TabPFN-oracle parity claim.

### 3. Native ABI only after reference validation

Do not rush a large C ABI surface for the whole portfolio. First validate the
Python/reference estimator semantics and fold isolation. Native C ABI additions
can then expose:

- selection reports;
- endpoint decisions;
- route reports;
- selected-chain summaries;
- prediction APIs for finalized presets.

## Candidate status

| Candidate | Status | Reason |
|-----------|--------|--------|
| `AOMStructuralPolicy` | integrate | feature-count structural policy, not source-routed |
| `AOMStructuralPolicyWithPgt1200Admissions` | integrate | high-p admission is train-only |
| `AOMStructuralPolicyWithP700BlockLocalAdmission` | integrate after rename | source-free replacement for legacy Beer-named class |
| `AOMEndpointMarginStabilityGate` | integrate | CV/OOF agreement route, mid-p evidence is strong |
| `AOMMidPEndpointStack` | Python reference added | cleanest AOM-oracle subregion; not TabPFN parity |
| `AOMControlSelector` | integrate baseline | conservative small-p/high-p backbone |
| `AOMTrueBankEndpointPortfolio` | Python reference added | useful small-p enrichment; no oracle claim |
| `AOMRidgeBlender` | reuse/audit | mature fold-safe OOF blender in `nirs4all-aom` |
| `AOMFallbackBlendGate` | Python reference added | conservative OOF threshold only |
| guarded `AOMOperatorPLSStack` | diagnostic/control | requires OOF gate and out-of-panel false-positive tests |
| source-route lookup | do not integrate | violates no-name-routing invariant |
| broad linear-moment grinding | research only | too slow/fragile for product default |

## Acceptance gates

Minimum local gates before this phase is marked shipped:

- metadata-invariance tests for all routed selectors;
- fold isolation tests for each CV/OOF selector;
- synthetic fixtures for selected route and prediction shape;
- at least one benchmark/dashboard row per integrated method;
- catalog validation passes;
- docs state robust-HPO target and no oracle-parity claim;
- no method default route depends on dataset/source identity.

## Deferred

- Full native C ABI for every portfolio object.
- GPU-backed repeated-CV moment grinder as a product method.
- ECR/Continuum top256 repeated-CV default path.
- Any TabPFN-oracle parity claim.
