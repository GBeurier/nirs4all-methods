# AOM / Moment Exploration Stop-Go Handoff

Date: 2026-06-04

Source archive in the lab repo:

- `/home/delete/nirs4all/nirs4all-lab/_archive/studies/aom_moment_exploration_2026_06_04/README.md`
- `/home/delete/nirs4all/nirs4all-lab/_archive/studies/aom_moment_exploration_2026_06_04/campaign_timeline.md`
- `/home/delete/nirs4all/nirs4all-lab/_archive/studies/aom_moment_exploration_2026_06_04/artifact_manifest.csv`
- `/home/delete/nirs4all/nirs4all-lab/_archive/studies/aom_moment_exploration_2026_06_04/paper_map.md`
- `/home/delete/nirs4all/nirs4all-lab/_archive/studies/aom_moment_exploration_2026_06_04/paper_opportunities.csv`

Versioned lab documentation mirror:

- `/home/delete/nirs4all/nirs4all-lab/docs/aom_moment_exploration_2026_06_04/README.md`
- `/home/delete/nirs4all/nirs4all-lab/docs/aom_moment_exploration_2026_06_04/campaign_timeline.md`
- `/home/delete/nirs4all/nirs4all-lab/docs/aom_moment_exploration_2026_06_04/artifact_manifest.csv`
- `/home/delete/nirs4all/nirs4all-lab/docs/aom_moment_exploration_2026_06_04/paper_map.md`
- `/home/delete/nirs4all/nirs4all-lab/docs/aom_moment_exploration_2026_06_04/paper_opportunities.csv`

Target interpretation for `nirs4all-methods`:

- integrate the robust AOM HPO / portfolio pieces already evidenced in the lab;
- do not integrate broad GPU repeated-CV preprocessing grinding as a product
  method;
- do not route by dataset/source/database identity;
- catalog only the part that has a native public `n4m_*` ABI and validation
  surface; keep broader Python/reference portfolio objects outside the canonical
  catalog until they get the same treatment.

Current Phase 7f artifacts:

- `docs/architecture/aom_methods_gap_analysis.md`
- `roadmap/phase-7f-aom-robust-hpo-portfolio.md`
- `docs/architecture/aom_robust_hpo_integration_manifest.csv`
- `docs/architecture/aom_robust_hpo_catalog_ready_manifest.csv`
- `docs/architecture/aom_robust_hpo_phase7f_dashboard.csv`
- `bindings/python/src/n4m/sklearn/aom_portfolio.py`
- `bindings/python/tests/test_aom_structural_policy.py`
- `benchmarks/cross_binding/tests/test_aom_phase7f_contract.py`

The current integrated product surface includes:

- native `n4m_aom_robust_hpo_fit`;
- Python `n4m.aom_robust_hpo`;
- catalog method `aom_pop.robust_hpo`;
- compact/wide native strict-linear preprocessing banks;
- native `n4m_moments_compute`, `n4m_moments_subset_compute` and
  `n4m_moments_subtract`;
- Python `n4m.moments` and `n4m.moments_train_from_heldout`;
- native `n4m_sweep_run` Ridge-CV mode;
- Python `n4m.sweep_run`;
- CPU and CUDA-build smoke validation.

The current Python reference surface also includes:

- `AOMStructuralPolicy` and source-free presets;
- `AOMControlSelector`;
- `AOMTrueBankEndpointPortfolio`;
- `AOMMidPEndpointStack`;
- `AOMEndpointMarginStabilityGate`.
- `AOMFallbackBlendGate`.

The lab stop-go decision is final enough to stop dataset-by-dataset
over-tuning. The first native ABI surface is now integrated. Remaining work is
to decide whether the non-native portfolio/gating objects deserve the same
product hardening, and separately to extend `n4m_sweep_run` with batched IKPLS,
operator descriptors and fused batched GPU preprocessing screens beyond the
current Ridge plus materialized-PLS mode if the moment substrate proves worth
industrialising.
