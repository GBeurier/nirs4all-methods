# Handoff - AOM / moment portage

Date: 2026-06-05

## Current state

This branch has a broad AOM/moment integration in `nirs4all-methods`.

Completed and validated in the latest pass:

- Direct PCR head:
  - C ABI `n4m_pcr_fit`
  - Python `n4m.pcr`
  - Facade `n4m.moment.pcr`
  - sklearn wrapper `NativePCRRegressor`
  - catalog method `models.pls.pcr`
  - docs and timing smoke
- Moment-only OOF stack:
  - `n4m.moment_stack(...)`
  - `NativeMomentStackRegressor`
  - catalog method `models.ensembles.moment_stack`
  - uses train-only OOF predictions from Ridge / PLS sweep / PCR / continuum / ECR / CPPLS
  - no nonlinear lift, no transformed-spectrum stacking, no dataset-name routing
- Candidate preprocessing impact audit:
  - `n4m.aom_candidate_preprocessing_impact`
  - exposed through `n4m`, `n4m.aom`, and `n4m.moment`
  - groups scored candidate reports by stage, operator, option, chain position and head/stage option
  - reports best/mean/median score, rank stats and improvement vs identity baseline when present
- Existing broader surfaces in this branch:
  - `n4m.aom` and `n4m.moment` facades
  - strict-chain grid builder and streaming iterator
  - AOM chain screen/refit campaign helpers
  - fixed selected-candidate reuse
  - route summaries, rank diagnostics, candidate report save/load
  - CPU/CUDA backend recommendation knobs, including `backend_min_cuda_product`
  - timing smoke scripts for direct heads, moment stack, moment sweep, AOM sweep/screen/refit and CUDA facade smoke

## Validation just run

From `/home/delete/nirs4all/nirs4all-methods`:

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python -m py_compile \
  bindings/python/src/n4m/python.py \
  bindings/python/src/n4m/__init__.py \
  bindings/python/src/n4m/aom/__init__.py \
  bindings/python/src/n4m/moment/__init__.py \
  bindings/python/src/n4m/sklearn/native_sweeps.py \
  bindings/python/src/n4m/sklearn/__init__.py \
  bindings/python/tests/test_moment_model_wrappers.py \
  bindings/python/tests/test_aom_moment_facade.py \
  benchmarks/cross_binding/bench_moment_stack_timing.py \
  benchmarks/cross_binding/bench_direct_moment_heads_timing.py
```

```bash
PYTHONPATH=bindings/python/src N4M_LIB_PATH=build/dev-release/cpp/src/libn4m.so \
  /home/delete/.venv/bin/python -m pytest \
  bindings/python/tests/test_aom_moment_facade.py \
  bindings/python/tests/test_moment_model_wrappers.py -q
```

Result: `61 passed`.

```bash
PYTHONPATH=bindings/python/src /home/delete/.venv/bin/python catalog/scripts/validate.py --strict-abi
```

Result: `PASS`, 200 methods, ABI coverage `701/701`.

```bash
git diff --check
```

Result: pass.

Earlier in the same pass, `catalog/scripts/validate.py --check-references` also passed after adding `models.ensembles.moment_stack` as a nirs4all-donor method.

## Important files

- `bindings/python/src/n4m/python.py`
- `bindings/python/src/n4m/aom/__init__.py`
- `bindings/python/src/n4m/moment/__init__.py`
- `bindings/python/src/n4m/sklearn/native_sweeps.py`
- `bindings/python/tests/test_aom_moment_facade.py`
- `bindings/python/tests/test_moment_model_wrappers.py`
- `cpp/src/c_api/c_api_method_result.cpp`
- `cpp/include/n4m/pls.h`
- `catalog/methods/models.pls.pcr.yaml`
- `catalog/methods/models.ensembles.moment_stack.yaml`
- `docs/architecture/aom_moment_coverage_matrix.md`
- `docs/architecture/aom_moment_worklog.md`
- `docs/methods/aom_chain_sweep_run.md`
- `docs/methods/pcr.md`
- `docs/methods/moment_stack.md`

## Remaining work

The remaining gaps are engine/performance work, not simple method wiring:

1. Fused/batched IKPLS grinder.
   - Current code has an exact moment route for compatible single-target PLS1 and optional CUDA fold scheduling/many-batched knobs.
   - It is still not a fully fused many-chain/many-fold/many-candidate IKPLS engine.

2. Full arbitrary-chain moment screen.
   - Current strict-linear coverage includes the practical local/banded/structured operators used in the AOM grids.
   - Non-compatible regimes still fall back or materialize.

3. CUDA fused sweep kernels.
   - CUDA builds work and there are device routes/counters.
   - There is no final grouped fused CUDA kernel suite for the full cartesian.

4. Broad benchmark/recall campaigns.
   - Need a controlled campaign against robust AOM oracle / AOM Ridge oracle / AOM PLS oracle / TabPFN baselines.
   - The new `aom_candidate_preprocessing_impact` helper should be used to analyze which preprocessing families/options justify more budget.

5. Staged cartesian orchestration.
   - The current chain grid can stream and score strict-linear chains.
   - A full staged cartesian runner like the proto `cartesian.py` / `impact.py` is not yet a first-class benchmark workflow in this repo.

## Constraints to keep

- Do not add hors-moment nonlinear lifts in this pass.
- Do not select by dataset name, source name, dataset id or equivalent identity.
- Do not use test-set selection for production model choice. Test ranking can be an offline audit only.
- Keep `libn4m` as the single C++/binding infrastructure rather than extracting AOM into a separate runtime.

## Suggested next step

Stop expanding features and run one focused benchmark campaign:

1. Pick 10 datasets.
2. Run compact/wide/lab strict-chain screens with Ridge, PLS and mixed heads.
3. Retain top global + per-head candidates.
4. Exact-refit retained rows.
5. Use `aom_candidate_preprocessing_impact` and `aom_candidate_rank_diagnostics` to decide whether more cartesian budget is justified.
6. Compare against the real robust AOM oracle and TabPFN baselines from `n4a-lab` / `n4a-paper` / `n4a-aom`.

