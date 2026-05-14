# Phase 4o — SIMPLS Component-Count CV

Goal: add an internal model-selection kernel that scores each latent-component
prefix by deterministic cross-validation, starting with SIMPLS.

## Scope

- Run the existing leakage-safe CV engine for component counts `1..K`.
- Record regression metrics per component prefix.
- Select the best component count by minimum RMSE, with deterministic low-count
  tie breaking.
- Add NumPy SIMPLS parity for the full metrics-by-component table.

## Non-goals

- Public C ABI exposure for model selection.
- Repeated/Monte-Carlo aggregation.
- One-standard-error or nested-CV selection rules.
- Sparse / MB / LW component selection.

## Acceptance

- Generated fixture compares C++ metrics by component against the pinned Python
  SIMPLS reference.
- Existing CV and SIMPLS solver behavior remains unchanged.
- No exported `p4a_*` symbols are added.
