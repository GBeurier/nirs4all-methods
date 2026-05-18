# Stabilisation plan — parity, dashboard and releases

Date: 2026-05-19
Scope: parity gates, cross-binding dashboard, slow methods, and
PyPI/CRAN readiness for pls4all 0.97.0 / ABI 1.16.0.

## Audit summary

The project is close to the intended architecture, but the gates are not
yet strict enough to be a release barrier. The main issue is semantic:
binding parity and reference parity are both present in parts of the
pipeline, but older fields, docs and dashboard filters still collapse them
into one verdict.

Local audit results:

| Check | Result |
|---|---|
| `ctest --preset dev-release --output-on-failure` | passed |
| `python -m benchmarks.parity_timing.lockfile --check` | passed, structural only |
| full Python binding tests | failed on `UVESelector` pipeline smoke |
| sklearn wrapper parity script | passed, but narrower than full tests |
| fixture regeneration check | blocked by missing historical `AOM_v0` oracle |
| small cross-binding PLS/PCR sample | confirmed external rows can be mislabeled as binding failures |
| slow-method pls4all smoke | confirmed selector/PCR timing and adapter issues need focused work |
| `scripts/bump_version.sh --check` | passed |
| ABI symbol diff | failed: the current library exports additional `p4a_*` symbols absent from `cpp/abi/expected_symbols_linux.txt` |

## Implementation status

Stabilization status:

- P0 gate semantics implemented in the orchestrator: external rows are no
  longer binding-parity failures, reference parity compares all successful
  rows against the canonical oracle, and `--only-pls4all` consumes stored
  oracle snapshots instead of skipping Gate 2.
- P1 dashboard/static docs updated to render the two gates and to merge
  canonical `ref_*` rows atomically. External cells render only the
  reference gate because they have no binding gate.
- P2 Python selector smoke fixed for UVE, and tier-2 selector wrappers now
  fail closed on unknown registry parameters. Python/R/MATLAB selector
  ValidationPlan defaults are aligned to the canonical 3-fold contiguous
  plan.
- P2 dashboard refresh data covers the previously red `100x50` cells for
  `continuum_regression`, PCR and the selector smoke set; unavailable
  formula/classdef selector wrappers are classified as not available rather
  than failed parity.
- P3 first performance pass landed for PCR batch projection and
  cross-validation fold-buffer reuse.
- P4 ABI snapshot refreshed for the public 1.16.0 symbols already exported
  by the current shared library.

## P0 — make gates truthful

1. In `benchmarks/cross_binding/orchestrator.py`, compute binding parity
   only for `pls4all_core` and `pls4all_binding` rows. External rows must
   get `binding_parity_ok = None` or an explicit not-applicable code.
2. Keep reference parity for every successful row, including external
   libraries, against the canonical registry reference.
3. When a run intentionally omits canonical external references
   (`--only-pls4all`), load the stored oracle snapshot. Missing snapshots
   are setup failures that must be fixed by running the canonical reference
   backend.
4. Make missing required reference oracles a hard error in release-gate mode,
   with
   allowlisted `paper_only` methods only.
5. Move workstation-specific reference paths to environment/configuration
   or pinned packages. The AOM/POP oracle must be reproducible from a clean
   clone or explicitly excluded from a strict gate.

## P1 — fix dashboard and generated docs

1. Update `docs/_extras/build_landing.py` so canonical `ref_*` rows replace
   stale legacy cells atomically: `ok`, `reason`, both parity verdicts,
   timings, reference metadata and canonical flags.
2. Update dashboard filtering to use `reference_parity` for external
   libraries and `binding_parity` for pls4all rows.
3. Propagate method tolerance into CSV/JSON so drift/divergent thresholds
   use "10x method tolerance" instead of a hardcoded `rmse_rel < 10`.
4. Render both gates in static Markdown tables, or state clearly that a
   table is binding-only. Prefer using the existing `dual_parity_label()`
   helper instead of one-icon legacy output.
5. Exclude the synthetic reference column from timed-cell statistics and
   preset matching.
6. Keep `sphinx-design` enabled and load `tab-combo.js`; otherwise the
   generated method pages lose their tabbed content.

## P2 — restore binding parity

1. Fix the UVE sklearn pipeline failure by choosing an explicit policy for
   empty selections: add a `min_features`/fallback option or use a fixture
   parameter set that cannot select zero features in pipeline smoke tests.
   **Done.**
2. Stop silently dropping registry parameters in tier-2 wrappers. Add
   adapter maps for alias names or fail closed when a registry parameter is
   unsupported by a wrapper constructor. **Done for selector smoke.**
3. Unify selector validation plans across Python registry, sklearn classes,
   R dispatcher and MATLAB MEX. The cheapest deterministic option is a
   shared 3-fold contiguous plan; the more flexible option is to serialize
   fold indices through benchmark parameters. **Done with the 3-fold
   contiguous plan.**
4. Add C++ fixture coverage for selectors currently covered only by
   registry smoke tests.

## P3 — performance work

1. PCR: replace full `p x p` Jacobi eigensolve with a deterministic
   SVD/LAPACK or partial top-component solver, and use an `n x n` path when
   `p >> n`. **Partially done:** PCR now batches component projections and
   avoids score storage when not requested.
2. R vendoring: regenerate the vendored libp4a copy instead of manually
   carrying divergent `model.cpp` code.
3. Selectors: introduce a shared fitness evaluator that reuses buffers,
   validation folds and prediction arrays instead of reallocating for every
   candidate. **Started:** cross-validation fold buffers are reused across
   candidate evaluations.
4. Parallelize independent candidate evaluations for PSO, VISSA, BVE and
   IRIV while reducing results in deterministic order to preserve tie-breaks
   and RNG behavior.
5. Replace repeated full sorts with `nth_element` where only top-k masks are
   needed.

## P4 — packaging and release gates

1. Refresh the ABI snapshot intentionally. The audit saw more exported
   `p4a_*` symbols than `cpp/abi/expected_symbols_linux.txt` records.
2. Ensure Python sdist is either a real source build with CMake inputs
   included, or do not publish sdists until that path is supported.
3. Keep Python wheels smoke-tested from the built artifact, not from the
   editable checkout.
4. Keep R CRAN checks on the built tarball, and remove non-portable flags
   such as architecture-specific `-march=*` from CRAN builds.
5. Add a vendored-core sync check for the R package.
6. Treat MATLAB packaging as separate from PyPI/CRAN readiness until
   `toolbox.prj`, `release.m`, the complete MEX build and File Exchange
   workflow exist.

## Definition of "green"

The project is ready to resume method additions when:

- C++ fixture parity is reproducible from a clean clone;
- full Python tests, including sklearn pipeline smoke, are green;
- cross-binding Gate 1 is green for every shipped pls4all binding;
- cross-binding Gate 2 is green or explicitly relaxed for every shipped
  method and scheduled external reference;
- dashboard cells display both gates without legacy alias confusion;
- `pip install pls4all` and `R CMD check --as-cran` are validated from
  built artifacts;
- slow methods have baseline benchmarks and at least one profiling-backed
  optimization plan each.
