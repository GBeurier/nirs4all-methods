# Phase 4i - PLSCanonical

Goal: turn the reserved `P4A_ALGO_PLS_CANONICAL` ABI enum into a live fitted-model path without adding public symbols.

## Scope

- `P4A_ALGO_PLS_CANONICAL` + `P4A_SOLVER_NIPALS` + `P4A_DEFLATION_CANONICAL`.
- `P4A_ALGO_PLS_CANONICAL` + `P4A_SOLVER_SVD` + `P4A_DEFLATION_CANONICAL`.
- Shared fitted-model contract: `predict`, `transform`, model arrays and binary serialization remain available.
- Rank guard follows scikit-learn PLSCanonical: `n_components <= min(n_samples, n_features, n_targets)`.
- Python parity generator gains deterministic fixtures mirroring scikit-learn 1.4.2 PLSCanonical semantics.

## Acceptance Criteria

- Two NIPALS canonical fixtures and two SVD canonical fixtures are checked in with manifest SHA-256 entries.
- C++ ABI tests verify predictions, coefficients, preprocessing statistics, latent arrays, stored scores and serialization round-trips.
- Python `Config` exposes the `Deflation` enum so smoke tests can configure canonical deflation.
- `pls4all_cli --selfcheck`, release CTest, fixture determinism, ABI symbol diff and dependency audit pass locally.
- Release tag `phase-4i-canonical` points at the final commit.

## Deferred

- CCA / mode-B and PLS-SVD transformer semantics.
- mixOmics cross-implementation canonical parity from the R generator.
- Y-side transform output through a dedicated public accessor; the current ABI stores `P4A_MODEL_Y_SCORES_U` when `store_scores=1`.
