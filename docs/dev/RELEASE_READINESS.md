# nirs4all-methods — Release Readiness & Task Plan

> **Scope.** Everything that must happen before `libn4m` / `n4m` can be released against the six stated goals:
> **G1** CRAN R packages · **G2** PyPI Python packages · **G3** drop-in integration into `nirs4all` ·
> **G4** WASM/WebGL browser lib for a web page · **G5** GPU acceleration (CUDA / WebGPU / other) ·
> **G6** a new **direct Ridge** regression.
>
> **Engine state (historical baseline, verified 2026-06-03):** project **0.98.0**, ABI **1.10.0**
> (`cpp/include/n4m/n4m_version.h`), 188 catalog methods, ABI surface reconciled 669/669, P0 parity closed
> (`PRODUCTION_AUDIT.md`).
> Branch `main` is **strictly ahead** of `origin/cuda-spline-packaging` (all 7 of its commits are already in
> `main` — verified by `git cherry`/`patch-id`), so there is **no merge work** and `PRODUCTION_AUDIT.md` is safe.
>
> **How this document was produced.** A 4-phase multi-agent review: 14 parallel reviewers (one per
> dimension/goal) reading the actual source/config/CI, then an independent **adversarial verifier** re-opened
> every `blocker`/`high` finding and tried to refute it against current `main`. 66 agents, ~1,341 file/command
> inspections. Severities below are the **post-verification** values. Of 9 raw "blockers", **2 survived**
> verification as true blockers, 2 were **refuted entirely**, and the rest were re-graded. Task IDs are kept so
> findings are traceable.

---

## Current RC status (verified 2026-07-03)

This document keeps the June multi-agent audit below as provenance, but the status
has moved since then. The current RC head is project **1.0.1**, ABI **2.0.0**
(`cpp/include/n4m/n4m_version.h`). The hardcoded Python ABI smoke asserts and the
`nirs4all-methods` wheel loader repair path have been closed; `release-npm.yml`
exists; `npm test` now includes the generic method-result smoke; and the compiled
C++ tests now cover a representative model-fixture slice through
`test_models_pls.cpp`, `test_models_extra.cpp`, `test_internal_cv.cpp`,
`test_sweep.cpp` and `test_ridge.cpp`.

The remaining RC risks are no longer the two June blockers in section 0. They are:
external distribution proof for CRAN/npm/sdist/post-publish paths, wider
representative binding parity beyond the existing R/Octave/JS smoke families, and
an explicit full-registry/multi-shape parity dashboard rather than treating the
compiled model fixture slice as exhaustive coverage.

---

## 0. Executive verdict

The engine is **production-correct and architecturally clean** — the four load-bearing rules (C ABI public /
C++ internal, zero mandatory deps, never-free-across-the-boundary, stride-aware) genuinely hold on the core
model/predict path, exception translation is uniform, memory ownership is symmetric and documented, and the
RNG ports are bit-faithful. The packaging machinery (CRAN vendored build, PyPI dual-wheel + Trusted Publishing,
WASM build, version-sync, Linux ABI gate) is **substantially built**. What blocks a clean release is **not math
or architecture** — it is a small set of release-engineering defects, a large in-tree test-coverage gap on the
model solvers, pervasive documentation drift from the `p4a`/`pls4all`→`n4m` rename, and net-new feature work
(direct Ridge, browser-GPU, batched-GPU).

**Severity distribution (post-verification, 143 findings + 10 Ridge tasks + 5 critic gaps):**

| Severity | Count | Meaning |
|---|---:|---|
| **Blocker** | **2** | Will hard-fail the release pipeline as-is. Fix first. |
| High | ~33 | Must be done before the relevant goal is "released". |
| Medium | ~60 | Quality/coherence; do before or shortly after first release. |
| Low | ~40 | Polish, hardening, documentation. |
| Refuted | 2 | Investigated and dismissed (see §10). |

### The two confirmed release blockers

| ID | What | Why it blocks | Fix |
|---|---|---|---|
| **`abi-assert-1-9-0-blocks-release`** | `release-python.yml:273,423` hardcode `assert pls4all.abi_version() == (1, 9, 0)`. Runtime now returns **(1,10,0)** (RNG ABI bump). | The smoke step runs **before publish** on every `v*.*.*` tag *and* on any PR touching `cpp/**` or `bindings/python/**`. Both asserts `AssertionError` → the slim `pls4all` wheel can never publish, and qualifying PRs red-X. | Derive the expected ABI from the wheel/header at test time (`assert v.startswith(expected + '+abi.')`), never hardcode the triple. |
| **`n4m-loader-auditwheel-libs`** | `bindings/python/src/n4m/_ffi.py` searches only `<pkg>/lib/` with exact names; it has **no** sibling-`.libs/` / content-addressed `libn4m-<hash>` handling (the slim `pls4all/_ffi.py` does). | After `auditwheel`/`delocate`/`delvewheel` repair, the lib is renamed + grafted into `nirs4all_methods.libs/`. `release-wheels.yml` runs `CIBW_TEST_COMMAND=pytest …/test_import.py` (`import n4m; n4m.abi_version()`) against the **repaired** wheel in a clean venv → loader miss → **build fails on every OS** → publish (`needs: matrix`) blocked. The full `nirs4all-methods` PyPI package cannot be produced. | Glob the correct sibling dir — **`nirs4all_methods.libs/`** (the normalized *distribution* name, **not** `n4m.libs`; package `n4m` ≠ dist `nirs4all-methods`) — plus patterns `libn4m*.so*` / `libn4m*.dylib` / `*.dll`. Iterate all `*.libs/` siblings to be safe. |

Both are small (effort **S/M**) and on the G2 critical path. Nothing else is a hard blocker.

### Verification correction worth flagging up front

The single most consequential correction the adversarial pass produced: **the "unmerged `cuda-spline-packaging`
branch that deletes `PRODUCTION_AUDIT.md`" premise is false.** That branch is fully subsumed by `main`; the CUDA
build fix, R CRAN tarball, Python dual-wheels, WASM/Octave CI, and spline augmentation are **already on `main`**.
The only action is to **delete the 6 stale branches** (`delete-subsumed-branches`) so the false "stranded work"
risk disappears. Several other "blockers" (`verify-license-token`, `gpu-stale-bundled-so`, the stale macOS/Windows
ABI snapshot files) were also down-graded or refuted because the verifier found committed check logs / gitignore
rules / "the file isn't actually CI-gated" evidence the first pass missed.

---

## How to read this

- **Part A** — engine code-review findings (gate *everything*, not a single goal).
- **Part B** — per-goal release tracks (G1–G5).
- **Part C** — the direct-Ridge (G6) implementation blueprint.
- **§9** — cross-cutting risks; **§10** — refuted findings; **§11** — sequenced roadmap; **§12** — master checklist.

Each task line is `[SEV · effort] id — what` followed by *acceptance*. `SEV` is the post-verification severity.

---

# Part A — Engine code-review findings (cross-cutting)

These gate correctness/quality for **all** goals. None is a hard blocker, but the test gaps in A7 are the
biggest threat to release *confidence* across CRAN, PyPI and the bindings.

## A1. C++ numerical core — *mostly ready*

The core (~63k LOC) is broadly sound: clean coefficient back-transform (scale folded into coefficients at fit,
`model.cpp:1034,3911`), partial-pivoted inversions, two-pass variance in metrics, bit-faithful RNG ports. The
real weaknesses are at **NIRS scale (n≪p)** and in **conditioning**, none caught by the small-fixture parity
harness.

- **[High · L] `pcr-gram-pxp-conditioning`** — `fit_pcr_svd` (`model.cpp:3470-3617`) unconditionally forms a
  dense **p×p** Gram `XᵀX` and Jacobi-diagonalizes it (≤64 sweeps, O(p³)), allocating ~216 MB at p=3000 and
  *squaring* κ(X). The dual-Gram (n×n) machinery already exists (continuum regression, `extra_pls.cpp:2282`).
  *Verifier correction:* a true fix needs an **SVD-of-X** (not just the dual Gram) to fix the accuracy half, and
  the regression test must use an **ill-conditioned** 100×2000 matrix vs an SVD oracle. *Acceptance:* n<p path
  avoids the p×p Gram; ill-conditioned test matches SVD oracle ≤1e-8.
- **[Medium · M] `householder-qr-cancellation-and-pivot`** — `n4m_householder_qr` v-norm uses a cancellation-prone
  grouping (`linalg.c:40`); `n4m_back_solve_R` rejects only an *exactly* zero pivot (`linalg.c:103`) → garbage on
  rank-deficient SG/EMSC fits. Use the non-cancelling identity; reject `|diag| ≤ tol·max|diag|`.
- **[Medium · M] `gram-svd-condition-squaring`** — `n4m_svd_truncated_tall_gram` (`svd.c:537`) recovers σ via
  `sqrt(eig(AᵀA))` → effective κ squared. Guard/fallback to one-sided Jacobi on ill-conditioned input.
- **[Medium · M] `simpls-single-pass-gram-schmidt`** — SIMPLS deflation uses single-pass classical Gram-Schmidt
  (`model.cpp:3807`); orthogonality drifts for high component counts (a=15–30). Use modified GS / reorthogonalize.
- **[Medium · M] `kernel-pls-rebuilds-kernel-per-component`** — kernel PLS rebuilds the n×n kernel inside the
  per-component loop (`model.cpp:2695`), O(a·n²·p), defeating the fast n<p path. Build once, deflate the kernel.
- **[Low · S] `q2-equals-r2-conflation`** — `metrics.cpp:192` sets `q2 = r2` (only valid for OOF predictions).
- **[Low · M] `power-iter-hard-fail-and-tol-cap`** — multi-target power iteration hard-fails `CONVERGENCE_FAILED`
  on degenerate spectra (`model.cpp:978`) where sklearn/R return a subspace direction; also silently clamps user
  `tol` to ≤1e-12 (`model.cpp:925`).
- **[Low · S] `naive-dot-no-compensation`** — `dot`/`squared_norm` use naive accumulation; pairwise sum for long
  reductions improves cross-build bit-stability (matters most on scalar WASM).

## A2. C ABI boundary — *mostly ready*

Rules 1–3 hold cleanly (extern-C headers, uniform exception→`n4m_status_t` translation, symmetric ownership,
fixed per-context error buffer). **Rule 4 is only partial**, and there is header/version drift.

- **[High · S] `abi-gate-not-enforced-macos-windows`** — only the **Linux** `abi-check` job diffs against its
  snapshot (`abi-check.yml:36`); macOS extracts+asserts only the `n4m_` prefix and **Windows asserts nothing**.
  The documented "3-platform fail-closed ABI gate" is Linux-only. *Acceptance:* macOS/Windows jobs `diff` /
  `Compare-Object` the live exports against the committed snapshot and `exit 1` on any diff.
- **[High · M] `macos-windows-abi-snapshots-stale`** *(build-area; same root cause)* — `expected_symbols_{macos,windows}.txt`
  each have **500** entries vs Linux's **671** (missing the whole selection/method-result/aom/config family), are
  byte-identical to each other, and carry a `@@N4M_1` suffix the `nm -gU`/`dumpbin` extractors never emit (proof
  they were hand-edited, never regenerated). *Acceptance:* regenerate from real builds on macOS/Windows runners in
  the correct format; CI red on an injected symbol change.
- **[Medium · M] `stride-rule4-divergence-doc-or-fix`** — Rule 4 promises "no copy at the boundary", but only the
  model/predict/method-result paths honor strides; the **entire preprocessing/augmenter/splitter/selection/metrics/
  wavelet surface** rejects non-row-major-contiguous-F64 (`require_rowmajor_f64`, `c_api_preprocessing.cpp:173`),
  forcing column-major R/MATLAB callers to copy. Either upgrade those paths to strided reads or amend Rule 4 to
  scope full stride-awareness to the model path (decision recorded in docs).
- **[Medium · S] `header-banner-version-drift`** — `n4m.h:3` says "v1.8.0", `pls.h:3` says "pls4all v1.1.0"; actual
  ABI is 1.10.0 and `pls.h` wrongly claims to be the sole include.
- **[Medium · L] `category-headers-temp-stubs`** — the 11 advertised category headers (`models.h`, …) are TEMP
  stubs that just `#include "n4m/n4m.h"`. Decide before 1.0 whether they are installed/public or removed.
- **[Medium · S] `committed-binding-so-abi-skew`** *(see also `gpu-stale-bundled-so`, refuted in §10)* — stale
  `libn4m.so.1.9.0` files sit under both Python binding trees; they are **gitignored build artifacts** (the wheel
  rebuilds fresh at 1.10.0), so this is a working-tree-hygiene nit, not skew that ships.
- **[Low · S] `predict-into-status-swallowed`** — `copy_predictions` (`c_api_method_result.cpp:115`) drops
  `predict_into`'s status (`(void)status`), so `result['predictions']`/`rmse` can reflect a half-written buffer
  while fit returns `N4M_OK`. Surface or document as best-effort.
- **[Low/Medium · M] `abi-stability-experimental-flag-for-1.0`** — both headers say "experimental until v1.0.0";
  tagging 1.0 freezes **669** symbols as a permanent contract. Consciously choose the frozen subset (`stability_policy.md`).

## A3. Memory safety — *mostly ready*

Design is solid (RAII + `make_unique`+`release()`-on-success, bounds-checked serialize/Reader with exact-count
gates, integer-overflow guards on output sizing, UBSan fatal in CI). **The gap is coverage, not correctness.**

- **[High · L] `abi-memory-test-suite`** — `cpp/tests` is numerical/parity-only; **zero** tests call
  `n4m_model_fit/predict/transform/export/import`, `n4m_array_free`, the method-result getters, or the error-buffer
  API. The untested surface is exactly Rules 3 & 4. Add `test_c_abi_memory.cpp` (null/undersized/over-strided/
  double-free/error-reuse/dtype-mismatch on every public entry), run under `ci-asan_ubsan`. *(Some per-method
  NULL/invalid-arg tests do exist, e.g. `test_filters_y.cpp:457` — the model-level surface is the hole.)*
- **[High · L] `sanitize-binding-paths`** — `sanitizers.yml` only sanitizes the C++ ctest + CLI selfcheck; the **R
  glue** (`r_dispatch.c` 59.8 KB, manual PROTECT/UNPROTECT + `longjmp`-after-destroy + external-ptr finalizers +
  `n4m_method_result_destroy` ordering) — the highest-risk memory surface — is **never** seen by a sanitizer. CRAN's
  own valgrind/UBSan farm would be first to find a bug. Add a job that runs R or Python under ASan/UBSan.
- **[Medium · S] `lsan-explicit-leak-gate`** — no `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`; relies on the
  implicit default. Make it explicit; document the sanitizer gate in `docs/dev/testing.md`.
- **[Medium · M] `document-stride-buffer-contract`** — the zero-copy stride ABI cannot know the backing buffer is
  large enough; document the exact `(rows-1)·row_stride+(cols-1)·col_stride+1` requirement for every OUT view +
  add a debug assertion (depends on `abi-memory-test-suite`).
- **[Medium · S] `wasm-legacy-shape-checks`** — `n4m_wasm_pls_fit_legacy` (`wasm_entry.c:92`) memcpys p·q/p/q out
  of array views without checking shape or `v.data != NULL`.
- **[Medium · M] `cublas-thread-safety`** *(see A4)* — process-global cuBLAS handle is not concurrency-safe.
- **[Low · M] `rpack-result-oom-leak`** — in `pack_result` (`r_dispatch.c:218`), an R allocator `longjmp` under
  memory pressure skips `n4m_method_result_destroy(mr)` → core leak. Destroy before any R allocation can longjmp.
- **[Low · S] `cuda-dim-narrowing`** — `cuda_dispatch` casts dims `size_t→int`; reject `> INT_MAX` before cuBLAS.
- **[Low/Med] `fuzz-deserialize`** *(verifier: downgraded high→low)* — the deserialize path is bounds-safe by
  inspection (exact-count gate, `checked_product`, FNV-1a checksum, trailing-byte rejection); a libFuzzer harness
  is good hygiene (a dormant `N4M_BUILD_FUZZ` option already exists, unused) but not release-blocking.

## A4. Threading & concurrency — *partial*

CPU "one context per thread" is sound (per-instance Context, reentrant RNG, 16 race-free OMP loops with no nested
BLAS). Three real gaps:

- **[High · L] `cublas-shared-handle-race`** — a `cuda-on` build routes all ~46 `linalg::gemm/gemv/ger` sites
  through a **single process-wide** cuBLAS handle (`cuda_dispatch.cpp:68`, the header itself says "Phase 45b will add
  a per-stream pool"), and dispatch is **compile-time** (no `ctx.backend()` gate), so two threads fitting concurrently
  race — violating the `n4m.h:226` "across contexts thread-safe" guarantee. Only affects opt-in `cuda-on` (so not
  G1/G2/G3), but for G5: make the handle/stream per-thread **or** amend the guarantee + `DEFERRALS.md` to declare
  cuda-on single-thread-only.
- **[Medium · M] `context-backend-runtime-dispatch`** — `n4m_context_set_backend(REFERENCE_CPU)` is a no-op for
  numerics in a cuda-on build (compile-time `#if N4M_USE_CUDA`). Gate dispatch on the runtime backend.
- **[Medium · M] `num-threads-wire-or-document`** — `ctx.num_threads` is **dead state**: nothing in core reads it,
  no `omp_set_num_threads`/`openblas_set_num_threads` anywhere. Either honor it in the OMP regions or document it as
  advisory and point users at the env vars (footgun for G3 callers expecting `num_threads=1` to serialize).
- **[Medium · M] `oversubscription-runtime-guard`** — the production `blas-omp` build enables both OpenBLAS and
  OpenMP; a wheel user gets silent N² thread oversubscription (CMake `message(WARNING)` is invisible to them). Call
  `openblas_set_num_threads(1)` when OMP is the outer layer, or document/enforce one parallelism layer.
- **[Medium · M] `multithread-test-coverage`** *(verifier: high→medium)* — the cross-context guarantee is **entirely
  unexercised** (no `std::thread`/threading test anywhere; the existing `ci-tsan` job runs single-threaded). Add a
  C++ test fitting from N≥4 threads with per-thread Context asserting bit-identical coefficients; run under TSan.
- **[Medium · L] `tsan-cover-parallel-paths`** — `ci-tsan` builds with OpenMP **OFF**, so the only fork-join path
  is never seen by TSan. Add a TSan run with an instrumented OpenMP runtime (clang libomp).
- **[Low · M] `cuda-concurrency-test`** — GPU concurrency/determinism test once the handle race is fixed.

## A5. Build / CMake / ABI gates — *mostly ready*

`n4m_version.h` is a genuine single source of truth; `bump_version.sh --check` passes and is CI-gated; SONAME
wiring tracks ABI major. Gaps centre on the cross-platform ABI gate and a missing linkage gate.

- (`abi-gate-not-enforced-macos-windows`, `macos-windows-abi-snapshots-stale` — see A2; the two together are the
  load-bearing build finding.)
- **[Medium · M] `abi-snapshot-regen-script`** — no `scripts/regen_abi_snapshots.sh`; the three extractions are
  ad-hoc shell embedded in CI/CLAUDE.md and subtly different, which is *how* the macOS/Windows files silently drifted.
  Add one canonical regen command (`--check` = no-op on clean tree).
- **[Medium · M] `soname-linkage-gate-not-in-ci`** — CLAUDE.md lists the `readelf -d … SONAME|NEEDED|RPATH|RUNPATH`
  check as a release blocker, but **no workflow runs it**. A stray conda `RUNPATH` (from the `blas-omp` preset) or
  unexpected `NEEDED` ships undetected. Assert SONAME + an allow-listed NEEDED set + no external RUNPATH.
- **[Low · S] `blas-preset-hardcoded-conda-path`** — `blas-omp`/`blas-on` hardcode `$CONDA_PREFIX/lib/libopenblas.so`
  + bake `$CONDA_PREFIX` into RPATH (non-portable for redistributables). Use `find_package(BLAS)` or error clearly.
- **[Low · S] `claude-md-def-file-claim-stale`** — CLAUDE.md/ARCHITECTURE claim MSVC exports via a `.def` file;
  none exists (it's `__declspec(dllexport)` via `N4M_API`).
- **[Low · S] `makefile-stub-targets-exit-0`** — `make dashboard-*`/`new-binding` print "not wired" and `exit 0`;
  a scripted release flow reads that as success. Exit non-zero or remove.

## A6. Security & input validation — *partial*

Deliberate "trust the caller for buffer length" contract; strong on config (typed setters, no JSON/YAML parser in
core) and iterative caps. The verifier **down-graded the WASM security findings** because the shipped JS path only
exposes `n4m_matrix_view_init_rowmajor` with an enforced `data.length === rows*cols` and a binding-sized buffer, so
the headline "untrusted browser can inject oversized strides" attack is **unreachable as shipped**. Still worth the
defense-in-depth, especially before the generic 188-method path is exposed to the web.

- **[High · L] `no-fuzz-harness`** — there is **no fuzz harness anywhere**; sanitizers run only well-formed
  fixtures. Before exposing untrusted spectra (G4) add a libFuzzer target over `validate` + a representative
  fit/transform/select with random dims/strides/dtypes/contents under ASan+UBSan; corpus of degenerate/INT32_MAX
  dims, NaN/Inf.
- **[Low · M] `view-no-buffer-length-bound`** *(verifier high→low)* — `n4m_matrix_view_t` carries no length;
  `read/write_value` dereference `ptr[row*row_stride+col*col_stride]` unbounded. Add an optional
  `n4m_matrix_view_validate_bounded(view, byte_capacity)` for untrusted callers + document the contract in `n4m.h §3`.
- **[Low · M] `core-rowscols-overflow-inconsistent`** *(verifier high→low)* — `checked_matrix_size` is copy-pasted in
  ~21 selection files and **not** used in `copy_matrix_checked` (`model.cpp:106`, raw `rows*cols`). Centralize one
  overflow-checked multiply. *(The CARS "no upper cap on n_iterations" sub-claim was refuted — it is int32-bounded.)*
- **[Low · M] `wasm-js-double-length-check`** *(verifier high→low)* — `ffi.ts` uses JS-double arithmetic for the
  length check and never checks `_malloc==0`; all impact is contained to the WASM sandbox of one tab. Add
  `Number.isSafeInteger` + element cap + `_malloc===0` guard.
- **[Medium · S] `security-md-stale`** — `SECURITY.md` uses `p4a_*`/`pls4all` naming and references a non-existent
  `p4a_model_import_from_buffer`; no WASM threat model. Rewrite for n4m + add a WASM/untrusted-input section.
- **[Medium · S] `input-driven-iteration-caps`** — CARS validates `n_iterations≥1` with no upper bound and pre-sizes
  `n_iterations*p` (`cars_selection.cpp:122,265`); cap stochastic-selector repetition counts (DoS guard).
- **[Medium · S] `wasm-memory-growth-dos`** — `ALLOW_MEMORY_GROWTH=1`/`MAXIMUM_MEMORY=2GB`; enforce a documented
  input-size budget in the binding and map OOM to a catchable JS error (not an abort/tab-crash).
- **[Medium · M] `view-validation-negative-tests`** — no test rejects a malformed view; the JS harness even bypasses
  the length guard. Add C++ + JS negative tests.
- **[Low · L] `validate-float-view-duplication`** — `validate_float_view`/`read_value`/`write_value` re-defined in
  dozens of core files; consolidate so a boundary hardening fix can't miss a sibling.

## A7. Tests & parity — *partial* (the biggest confidence gap)

The preprocessing/filter/splitter/metric/RNG coverage is genuinely deep (1184 cases, 108 fixtures, fixture
determinism is a hard 1e-12 CI gate). But the **model solvers have essentially no in-tree test**, and the docs
**claim coverage that does not exist**.

- **[High · L] `wire-cpp-model-parity-tests`** *(raw blocker → high)* — **zero** compiled C++ tests exercise any PLS/
  SIMPLS/PCR/OPLS/CV/selector fit. The 78 `synthetic_*pls*/pcr/opls/cv/selection` fixtures are loaded by nothing;
  the `cpp_header.py` generator emits to `cpp/tests/fixtures/` which **does not exist** and is never invoked by CMake.
  Worse, `parity-gate.yml:157` and `parity/README.md` **assert** ctest covers all these models — both are **false**.
  All model numerics ship validated only by the cross-binding harness at *one* input cell. *Acceptance:* generate +
  compile `test_models_pls.cpp`/`test_pcr.cpp`/`test_opls.cpp`/`test_cv.cpp`/`test_selection.cpp`; ctest fails on any
  divergence beyond `tolerances.md`; fix the false gate/README claims.
- **[High · M] `fix-or-remove-doctest-uve-test`** — `test_uve_r_exact.cpp` `#include <doctest/doctest.h>` but the
  repo uses a hand-rolled harness and **there is no doctest** anywhere; it is in **no** `add_executable`. So the
  Jaccard-1.0 R-exact MCUVE claim (and `HANDOFF_RNG_DASHBOARD.md`'s "276 doctest cases pass") is backed by a test
  that **runs nowhere**. Port it to `N4M_TEST_REQUIRE` (+ `n4m_internal_tests` static link) or delete it.
- **[High · L] `expand-binding-parity-method-coverage`** — each non-Python binding asserts parity for **one** method
  (R: `pls_simpls` only; Octave: one PLS fit; JS: SIMPLS-family only); the per-PR gate covers only `pls`+`pcr`. So
  for G1/G4 we have proof the ABI plumbing works for ~2 methods, not for the ~188 reachable through `n4m_method_result`.
  A binding-side marshalling bug (stride/transpose/column-major, selector index arrays, multi-output Y) for any other
  method passes CI. Widen to a representative set per family at 1e-12 native / 1e-9 WASM.
- **[Medium · M] `multi-regime-model-parity`** *(verifier high→medium)* — `--registry-cells` expands to exactly
  **one** `(n,p)` shape per method; the only multi-size tool (`per_method_parity.py`, 3 shapes) is wired into **no**
  CI. *(Verifier nuance: the `pls_lda/sparse_simpls/cppls` cases were harness-config artifacts, not engine bugs that
  escaped — but ≥3-shape per-method parity is still worthwhile hardening.)*
- **[Closed in RC · S] `refresh-stale-js-parity-fixture`** — `bindings/js/test/parity_fixture.json` now records the
  current `1.0.1+abi.2.0.0` provenance; `npm test` still gates the fixture numerics.
- **[Partially closed in RC · S] `fix-parity-gate-overstated-claim`** — `parity-gate.yml` + `parity/README.md` now state
  the compiled model-fixture slice and the remaining full-registry coverage gap explicitly.
- **[Medium · M] `make-fullsweep-status-visible`** — `nightly-parity.yml` always `exit 0` and uploads a CSV only; a
  pass→fail regression sits unnoticed in an artifact. Emit a compared baseline pass-rate / open an issue; exclude
  Python-only rows from the denominator.
- **[Medium · M] `selector-rng-parity-coverage-matrix`** — no per-selector table classifying each as exact-Jaccard /
  R-exact-RNG / numpy-RNG / informational; only `mcuve_pls` has a proven R-exact path (and its test doesn't compile).
- **[Low · M] `self-consistency-numeric-not-just-determinism`** — AOM/POP self-consistency checks only determinism +
  finiteness; add structural numerical invariants / golden snapshots.
- **[Low · M] `enforce-coverage-floor-on-c-api`** — `coverage.yml` is informational; the ~4k-LOC method-result
  dispatcher is likely largely uncovered. Add a floor once model tests exist.
- **[Low · M] `audit-skipped-xfail-conditional-tests`** — several gates SKIP silently when a toolchain is absent
  (R/Octave/JS legs, nirs4all `|| echo unavailable`); a release could be cut from a Python-only run. Require the
  toolchain or record which legs ran.

## A8. Docs & repo hygiene — *partial*

Current/honest: CLAUDE.md, AGENTS.md, CONTRIBUTING, DEFERRALS, bindings/SPEC. **Badly stale** (pre-rename): the
big distribution/architecture docs, README CLI commands, the Sphinx site, SECURITY, CITATION.

- **[High · L] `distribution-md-stale` / `rewrite-distribution-md`** *(3 reviewers flagged this — one canonical task)*
  — `DISTRIBUTION.md` is the named "single source of truth for where pls4all is published" but is pervasively stale:
  title "pls4all", `P4A_ABI_VERSION = 1.16.0`, version `0.97.0`, the deleted header `cpp/include/pls4all/p4a_version.h`,
  `libp4a` SONAME chains, ~40 `github.com/GBeurier/pls4all` URLs, `@pls4all/wasm` npm scope, `p4a.wasm` "~120 KB"
  (actual `n4m.wasm` 1.5 MB), and Go/Rust/.NET/Ruby/Lua/Nim channels that are frozen `_archive`. 396/1580 lines carry
  stale tokens. A packager following it verbatim misnames every artifact. *Acceptance:* n4m/libn4m naming, current
  version/ABI/header path, active-vs-frozen channel split; grep for `p4a_|libp4a|0.97.0|1.16.0` returns only intended refs.
- **[Medium · M] `rewrite-architecture-md`** *(verifier high→medium)* — `docs/ARCHITECTURE.md` is the **canonical
  spec that overrides CLAUDE.md**, yet is entirely `p4a`/`pls4all`, cites the non-existent `cpp/include/pls4all/p4a.h`
  and a dead `docs/Direction_Technique.md`, and names `pls4all_core/_c/_cli` (actual `n4m_*`). 13 stale matches.
- **[Medium · M] `fix-readme-cli-binary-and-table`** *(verifier high→medium)* — README tells users to run
  `…/pls4all_cli` (binary is `n4m_cli` → copy-paste fails) and labels the full-engine columns "pls4all C++/Python/R/
  MATLAB"; also a stale `libp4a` at `README.md:276`. *(The "ABI 1.9" sub-claim was refuted — no such string in README.)*
- **[Medium · S] `gitignore-or-remove-scratch-files` / `scratch-file-hygiene`** — `HANDOFF_RNG_DASHBOARD.md`,
  `irf_*.txt`, `r_T2pls_body.txt`, `r_t2_helpers.txt`, `finish-lib-progress-irf-probe.txt`,
  `benchmarks/cross_binding/reclassify_composites.py` are **untracked and not gitignored** → a `git add .` during
  release prep would commit them (and they pollute a `git archive` source tarball). Delete/relocate/ignore.
- **[Medium · S] `fix-sphinx-conf-and-about`** — `docs/conf.py` + `about.md` brand the published site as "pls4all".
- **[Medium · S] `fix-security-md-naming`** — public Security-tab doc uses `p4a_*` symbol names that don't exist.
- **[Low · S] `fix-changelog-header-and-unreleased-abi`**, **`fix-python-version-docstring`**,
  **`decide-citation-cff-naming`**, **`update-status-md-or-mark-legacy`** — targeted naming/ABI-string fixes;
  leave historical CHANGELOG entries intact.

---

# Part B — Release tracks

## G1 — CRAN R packages (`n4m` + slim `pls4all`) — *mostly ready*

Genuinely advanced: the self-contained vendored build (`N4M_R_VENDOR=1`) is **on main**, ships 222 vendored
C/C++/Fortran TUs + `configure`/`Makevars(.win)`, **no compiled artifacts**, registers routines, links zero
external BLAS/OpenMP (CRAN-safe), and carries full third-party attribution. The work is CRAN-incoming polish + one
real verification gap.

- **[High · M] `real-cran-farm-check`** *(raw blocker → high)* — the only check evidence is local conda-R runs with
  `_R_CHECK_CRAN_INCOMING_REMOTE_=false` + `--no-manual`, which suppress exactly the checks CRAN runs (URL/license/
  incoming feasibility, the LaTeX PDF manual). Committed scratch logs show **2 WARN/3 NOTE** (n4m) and **1 WARN/3
  NOTE** (pls4all), the WARNINGs being environment artifacts (`checkbashisms` missing, locale). *Acceptance:*
  win-builder R-release+devel **+** R-hub v2 (linux/win/macos) **+** macOS-builder return **0 ERROR / 0 WARN** for
  **both** packages with incoming checks + PDF manual; save the logs.
- **[Medium · S] `title-acronym-note`** — quote/expand "NIRS" in the n4m Title (CRAN incoming NOTE).
- **[Medium · S] `authors-ctb-malformed`** — `person('pls4all','contributors', role='ctb')` is a fabricated person;
  CRAN reviewers push back. Use an org/real-person form with `comment=`.
- **[Medium · S] `dual-package-duplication`** — `n4m` and `pls4all` ship **byte-identical** 222-TU vendored source
  (slim only at the R API level). Add a `cran-comments.md` rationale + sequence the submissions.
- **[Low · S] `stale-date-field`** *(verifier high→low)* — `DESCRIPTION Date: 2026-05-20` hardcoded; `bump_version.sh`
  doesn't sync it. Drop the field (R derives `Packaged:`) or stamp at release. *(Not a check failure today.)*
- **[Low · S] `march-nocona-doc`** — the non-portable `-march=nocona` flag comes from **conda-R's Makeconf, not the
  package** (package Makevars use only `-Ivendor … -DN4M_*` + `CXX_STD=CXX17`). Confirm absent on the CRAN farm; doc it.
- **[Low · M] `fortran-lto-types`** — verify the 11 FITPACK `.f` TUs are LTO-clean under CRAN's gfortran (intrinsic-
  shadowing dummy args `cos/sin`).
- **[Low · S] `pls-generic-masking-note`** — n4m re-exports `MSEP/RMSEP/R2/selectNcomp/mvr/plsr/pcr` (masks the `pls`
  package; legal since Suggests-only). Document the intentional shim.
- **[Low · S] `vignette-build-budget`** — confirm 222-TU compile + vignette + examples stay within CRAN's time budget.
- **[Low · S] `tarball-regen-from-current-source`** — regenerate the submission tarballs fresh from the release tag
  (the committed `*_0.98.0.tar.gz` are stale build outputs); src/vendor must byte-match `cpp/` at the tag.
- **Refuted:** `verify-license-token` — committed `R CMD check` logs show "DESCRIPTION meta-information … OK"; the
  `CeCILL (== 2.1)` token standardizes with no LICENSE pointer (see §10).

## G2 — PyPI Python packages (`nirs4all-methods` → `n4m`, `pls4all`) — *partial*

Two PyPI projects from one source tree (`make_python_package.py` emits the generated per-package dirs). The slim
`pls4all` wheel (`release-python.yml`) is mature; the full `nirs4all-methods` wheel (`release-wheels.yml`) is thinner
and carries **both blockers**.

- **[BLOCKER · M] `abi-assert-1-9-0-blocks-release`** / `abi-loader-version-drift` — see §0. Fix the hardcoded
  `(1,9,0)` asserts; derive the ABI at runtime from `n4m_get_abi_version_*`. (Also fix `_ffi.py:33` hardcoded
  candidate filename `libn4m.so.1.9.0` and add `_ffi.py` ABI constants to the `bump_version.sh` manifest so `--check`
  catches future drift — `ffi-abi-minor-hardcode-drift`.)
- **[BLOCKER · M] `n4m-loader-auditwheel-libs`** — see §0. Glob `nirs4all_methods.libs/` (not `n4m.libs`).
- **[High · L] `sdist-broken-no-lib-no-source`** *(raw blocker → high)* — the published sdist contains **neither
  libn4m nor C++ source** (`MANIFEST.in` only includes the empty `src/pls4all/lib/`; `setup.py` compiles nothing), so
  `pip install --no-binary :all:` or any off-matrix platform installs a package whose `import` raises ImportError at
  load. It can also non-deterministically bundle a stale local `.so`. *Acceptance:* either ship a from-source-buildable
  sdist (cpp + CMake + build backend hook) **or** make it deliberately metadata-only that fails loudly; verify
  `--no-binary` never yields a broken import. *(Common wheel platforms are covered, so this bites edge cases — hence
  high not blocker.)*
- **[High · L] `unify-nirs4all-methods-workflow`** — `release-wheels.yml` builds 4 cp3X wheels per platform (no retag
  → misleading cp-ABI tags for a ctypes-only pkg), has **no sdist job**, skips musllinux, uses macos-13/14 instead of
  universal2, sets no `MACOSX_DEPLOYMENT_TARGET`, and has no installed/post-publish smoke. Bring it to parity with the
  mature `release-python.yml`.
- **[Medium · S] `macos-deployment-target-sync`** *(verifier high→medium)* — `release-wheels.yml` macOS jobs use the
  `dev-release` preset (no OSX deployment pin) → the dylib's min-version can exceed the wheel's platform tag (import-
  fail on older macOS than advertised). Pin `MACOSX_DEPLOYMENT_TARGET=11.0` to match the preset.
- **[Medium · S] `pypi-readme-mislabel-full-pkg`** — confirm `twine check` passes and the generated `nirs4all-methods`
  README isn't the slim `pls4all` one + drops the stale `1.9.0` ABI example.
- **[Medium · S] `clean-stale-generated-dirs-and-debris`** — remove stale generated dirs + `dist/` debris + leaked
  `*.egg-info/` from the working tree.
- **[Low · S] `shared-libn4m-double-bundle`** — both wheels bundle the identical full 2.4 MB libn4m; document the
  duplication-by-design (or plan a shared-lib package). *(Verified safe: both load `RTLD_LOCAL`, no symbol clash.)*
- **[Low · S] `abi3-wheel-strategy-doc`** — centralize the "py3-none per platform, abi3 N/A (pure ctypes)" policy so
  per-cpX builds aren't reintroduced.
- **[Low · S] `refresh-distribution-md`** — (same DISTRIBUTION.md rewrite as A8, PyPI section).
- **[Low · M] `gpu-cuda-wheel-variant`** — document the GPU-wheel decision (CPU-only PyPI + from-source `cuda-on`, or
  a follow-up `nirs4all-methods-cu12` variant). See G5.
- **Refuted:** `gpu-stale-bundled-so` — the `libn4m.so.1.9.0` under the binding trees is **gitignored** and rebuilt
  fresh (1.10.0) in the wheel job; nothing stale ships (see §10).

## G3 — Drop-in integration into nirs4all — *partial*

n4m's Python surface is genuinely sklearn-shaped (estimators subclass `BaseEstimator`+mixins, `fit/predict/transform/
get_params/set_params`, multi-output, pickle bit-exact via the `.n4a` C-ABI wire format). nirs4all today has **zero**
n4m references (greenfield) and resolves models by class-path string/instance, so the swap is config/instance-level,
not a rewrite. **This is a downstream feature for the user's G3 goal — it does NOT gate the engine's own 1.0 release.**

- **[High · M] `map-model-classes`** — **DONE** → `docs/nirs4all_integration_map.md` (per-class table, file:line-cited).
  Verdicts over nirs4all's 42 exported estimators: **drop-in 1** (PCR), **adapter-needed 8** (PLSDA, IKPLS, OPLS,
  OPLSDA, MBPLS, SparsePLS, SIMPLS, RobustPLS), **blocked 5** (LWPLS, RecursivePLS, KernelPLS, NLPLS, KPLS — real C-ABI
  gaps), **keep-native 28**. Two corrections from the mapping pass: **`DiPLS` is a NAME COLLISION** (nirs4all `DiPLS`
  = *Dynamic* PLS with time-lags; pls4all `DIPLSRegression` = *Domain-Invariant* PLS needing `X_target`) → keep-native,
  **not** 1:1 — do not auto-map it. **RobustPLS is adapter-needed, not blocked** (`n4m_robust_pls_fit` exports
  coefficients). All model fits live only in the slim **`pls4all`** dist; the full `n4m` package has no method-result
  FFI yet, so even drop-ins target `pls4all.sklearn` until that subsystem is built. **No n4m equivalent:** FCKPLS,
  OKLMPLS, IntervalPLS(regressor), KOPLS, DiPLS(Dynamic), AOM/POP/AOM-Ridge/FastAOM families, TabPFN.
- **[High · L] `adapter-shim-or-repoint`** — build the integration shim: either re-point each mappable nirs4all class
  to delegate to the n4m estimator (preserving import path, `get_params` arg names, `_estimator_type`, `_webapp_meta`
  that 19 model files carry) or reference `pls4all.sklearn` in pipeline configs. API signatures differ (`scale/center/
  backend` vs `center_x/scale_x/solver`), so a naive import-path swap won't work.
- **[Medium · L] `in-sample-predict-coef-export`** *(raw blocker → medium)* — `_in_sample.py` raises
  `NotImplementedError` on predict-on-new-X for 15 wrappers; nirs4all always predicts on held-out folds. *Verifier
  nuance:* **RobustPLS already exports coefficients** in the C ABI (`n4m_robust_pls_fit` → coefficients/x_mean/y_mean)
  — a pure-Python wrapper fix. **LWPLS** is genuinely local (no global coef → needs the C kernel to accept a separate
  predict-X). **RecursivePLS** is a moving-window evaluator (different semantics). Fix per-model.
- **[Medium · M] `consolidate-pypi-packaging`** *(verifier high→medium)* — models live in dist `pls4all`, preprocessing
  in dist `nirs4all-methods`; a clean drop-in needs **both**, each bundling its own libn4m. Decide: one dist re-exporting
  both, or two pinned dists. *(Verified: two `RTLD_LOCAL` copies do **not** risk double-free — the worst case is doubled
  payload + an incoherent dependency surface.)*
- **[Medium · M] `missing-model-decision`** — declare keep-native vs port for FCKPLS/OKLMPLS/IntervalPLS-regressor/
  AOM/POP/FastAOM/TabPFN (relevant to G6: AOM-Ridge ↔ the new direct Ridge).
- **[Medium · M] `plsda-predict-proba`** — add `predict_proba`/`classes_` to n4m classifier wrappers (nirs4all PLSDA
  exposes it; controller tolerates absence but loses probabilities).
- **[Medium · M] `swap-acceptance-parity-gate`** *(verifier high→medium)* — the 5 nirs4all donors (aom/pop/lw/mb_pls,
  aom_preprocess) currently pass at a **loose 5× RMS** tolerance (they achieve ~1e-15). Tighten the donor tolerances
  in the n4m parity gate and define a swap-acceptance gate; gate the pipeline/operator surface, not just per-method.
- **[Medium · M] `check-estimator-conformance`** — run `sklearn.check_estimator` + clone/deepcopy/joblib round-trip
  against every n4m estimator nirs4all will use (n4m deliberately keeps the cached bundle across `set_params`).
- **[Medium · S] `seeding-reproducibility-note`** — n4m uses PCG64 + explicit seeds; nirs4all uses the numpy global
  RNG. Stochastic swaps (bagging/boosting/selectors) will **not** bit-reproduce nirs4all. Document + test statistical
  equivalence.
- **[Low · S] `fit-val-kwarg-support`** — note per mapped class whether the n4m target consumes `X_val/y_val`
  (AOMPLS uses it; n4m base `fit(X,y)` silently drops it).
- **[Low · S] `version-pin-contract`** *(verifier high→low)* — when G3 lands, pin nirs4all→n4m version range + document
  ABI/`.n4a` compatibility. *Verifier corrections:* the `.n4a` "refuses on ABI mismatch" claim is **false** (the C
  importer only hard-rejects on serialization-format-version mismatch; ABI skew → warning + `N4M_OK`), and nirs4all's
  own `.n4a` is an unrelated cloudpickle namesake. Also fix the stale `_model.py` docstring that claims a rejection
  the code doesn't perform.

## G4 — WASM / WebGL browser lib — *partial*

A working `MODULARIZE`/`EXPORT_ES6` Emscripten build of the **full** engine → **scalar** `n4m.wasm` (1.5 MB, ~497 KB
gzip; **no** SIMD/threads/GPU — verified). The raw PLS path is bit-exact (~2e-16). The generic 188-method path **is**
unlocked (the i64/BigInt fix `5107fae`) — but the docs still say it's broken, the regression test isn't in CI, and
there is **no publish workflow**.

- **[BLOCKER · M] `npm-publish-workflow`** *(raw blocker → high by verifier, but it is the hard gate for G4)* — there
  is **no npm/JSR publish job** anywhere (on main or the subsumed branch); the package can only be hand-published,
  off-version, with no provenance. Add a tag-triggered workflow: build emscripten preset → smoke+generic tests →
  `npm publish --access public --provenance`; dry-run green in PR CI. *(Graded high because the lib is hand-publishable
  in the interim and it doesn't block G1/G2/G3.)*
- **[High · S] `wire-generic-test-into-ci`** — the generic-path guard `run_generic_method.mjs` exists but `npm test`
  runs **only** `run_smoke.mjs`, so the single proof that ~143 method-result producers work from JS never runs in CI
  and can silently regress on an emsdk bump. Add it to `npm test` + the `js-wasm` job.
- **[Medium · M] `fix-stale-generic-path-docs`** *(verifier high→medium)* — `README.md`/`INPUT_CONTRACT.md` still tell
  consumers the generic path is "not yet enabled" / blame an "Emscripten codegen bug"; `wasm_entry.c` is already
  corrected. Pure docs fix.
- **[Medium · M] `wasm-size-trim-brotli`** — drop `ASSERTIONS=1` in release, scope `EXPORTED_FUNCTIONS` to what
  nirs4all-lite calls, enable `-Os`/closure, document brotli precompression (target <1 MB raw).
- **[Medium · M] `browser-html-example`** — every example is Node `.mjs`; add a self-contained `index.html`
  (`<script type=module>`) fitting PLS on a `Float64Array` — the literal "embed in a web page" deliverable and the
  nirs4all-lite seed.
- **[Medium · M] `memory-growth-untrusted-input`** — document the max input size implied by `MAXIMUM_MEMORY=2GB`,
  surface OOM as a catchable error, document the manual-`destroy()` contract (+ optional `FinalizationRegistry`).
- **[Medium · M] `decide-threads-coop-coep`** — record the SIMD/threads decision (`-msimd128` is cheap, broadly
  supported, speeds the f64 GEMM hot path; pthreads needs `SharedArrayBuffer` → COOP/COEP cross-origin-isolation —
  a real deployment constraint to document if pursued).
- **[Medium · S] `version-sync-js-package`** — confirm `bump_version.sh --check` covers `package.json` (a prior
  artifact shipped with 3 divergent version triples).
- **[Medium · S] `reconcile-cuda-spline-branch-js`** — ensure the BigInt fix + `@nirs4all/methods-wasm` rename +
  js-wasm CI survive any branch reconciliation; fix stale `p4a`-era roadmap notes. *(Largely moot now that the branch
  is subsumed — still verify the JS leg is in the release matrix.)*
- **[Medium · M] `distribution-md-js-section-rewrite`** *(verifier high→medium)* — DISTRIBUTION.md §3.4 still says
  `@pls4all/wasm` / `p4a.wasm` / "~120 KB"; reality is `@nirs4all/methods-wasm` / `n4m.wasm` / 1.5 MB.
- **[Low · S] `fix-i64-pair-read-methodresult`** *(verifier high→low)* — `methodResult.ts:79` carries a misleading
  "use HEAP32 instead" comment + an `as unknown as number` cast; the code is actually correct (every BigInt is
  `Number()`-wrapped). Cosmetic; optionally add a multi-output shape assertion.
- **[Low · M] `browser-loading-locatefile`** *(verifier high→low)* — the built Emscripten glue already emits the
  bundler-friendly `new URL("n4m.wasm", import.meta.url)` fallback, so it loads in Vite/webpack. Add an optional
  `locateFile`/`wasmUrl` pass-through + a browser/bundler example (ergonomics, not a load failure).
- **WebGL note:** WebGL is **not** a viable general-linalg path (no compute shaders / f64 GEMM). Browser-GPU = WebGPU
  (see G5).

## G5 — GPU acceleration (CUDA / WebGPU / other) — *partial*

The single-fit cuBLAS offload is **real, correct, honestly reported, and on main** (not branch-stranded). But it is
**build/link-checked only** (no GPU runner, no GPU test), it is **≈parity at NIRS sizes** by construction (fresh
malloc+H2D+kernel+D2H+free per call; decompositions stay on CPU — no cuSOLVER), there is **no GPU packaging path**,
and the genuinely valuable **batched** path + **browser WebGPU** are greenfield.

*Already shipped & correct:* row-major↔column-major bridge, exception-safe dispatch, honest `n4m_backend_is_available`,
`N4M_BACKEND_OPENCL/METAL` reserved enums.

- **[Medium · L] `gpu-ci-correctness-gate`** *(verifier high→medium — opt-in build, doesn't gate G1/G2/G3)* — the
  RTX-4090/5090 bit-parity was a one-off manual check; any edit to the transpose-bridge math passes CI green. Add a
  self-hosted/paid GPU job fitting pls/pcr/mb_pls/gpr_pls under `cuda-on` vs a CPU build, assert ≤1e-12, required on
  PRs touching `cuda_dispatch.cpp`/`linalg.hpp`.
- **[Medium · M] `gpu-doctest-dispatch-unit`** — per-kernel doctest (gemm/gemv/ger vs scalar) for all (trans_a,trans_b)
  combos, skipped without a GPU.
- **[Medium · S] `gpu-strided-buffer-contract`** — `cuda_dispatch` silently assumes contiguous `lda==cols` (the header
  admits a strided caller "would silently truncate") and `ger` copies `M*lda` inconsistently with gemm's `M*N`. Assert
  the precondition or honor strides.
- **[Medium · M] `gpu-reference-cpu-no-fallback-trap`** — in a cuda-on build `set_backend(REFERENCE_CPU)` returns OK
  but still runs on GPU. Fall back to scalar, or refuse with `N4M_ERR_BACKEND_UNAVAILABLE` (footgun for G3 determinism).
- **[Medium · L] `gpu-cuda-wheel-packaging`** — there is **no** CUDA build in any release workflow; the only GPU build
  is from-source `cuda-on`. For G5-via-G2: ship a documented `n4m-cu12` wheel (CUDA toolkit in a manylinux image) or a
  conda-forge cuda variant. CRAN stays CPU-only by design.
- **[Medium · XL] `gpu-batched-cv-abi`** — the real PLS-on-GPU win (per DEFERRALS.md) is batching **K folds × C
  components × hyperparameter grid** into one GPU job. ABI 1.22.0 now documents
  `n4m_pls_cross_validate` as an exact PLS-only reference surface that delegates
  to `n4m_sweep_run`; the grouped/fused executor still needs to replace the
  internals before this becomes the high-throughput grinder.
- **[Low · XL] `gpu-batched-execution-impl`** — implement it: device-resident data, streams, `cublasDgemmBatched`/
  strided-batched, per-stream/handle pool. *Acceptance:* >2× wall-clock vs CPU `blas-omp` on a representative large
  NIRS CV sweep.
- **[Low · L] `gpu-cusolver-decompositions`** — only cuBLAS BLAS3 is offloaded; SVD/eigen (`svd.c`) stay on CPU. For
  PCR/SVD-PLS/kernel-Gram, the decomposition dominates at large p — evaluate cuSOLVER (only worth it after batching).
- **[Low · L] `gpu-fp32-precision-path`** — the engine is double-only; consumer GPUs run f64 at ~1/32 of f32. Decide
  on an opt-in f32 path (tolerance-bounded) or declare datacenter-f64-only.
- **[Low · XL] `gpu-webgpu-browser-path`** — browser-GPU (G4+G5) is greenfield: the only linalg seam
  (`linalg.hpp` gemv/gemm/ger) is **synchronous + throwing**, fundamentally incompatible with WebGPU's async
  device/queue model — a WebGPU GEMM can't be called synchronously from inside a NIPALS/SIMPLS loop without an async
  fit-path refactor or a Worker+`Atomics.wait` blocking shim; WebGPU f64 is not guaranteed (f32-first → parity-tolerance
  impact); the win is **batched-only** (mirroring the CUDA finding). Scope it in DEFERRALS.md before any work.
- **[Low · S] `gpu-rocm-metal-sycl-alts`** — record the ROCm(HIP)/Metal/SYCL position: the `linalg.hpp` seam is the
  integration point and HIP is a near-CUDA-source mirror of `cuda_dispatch.cpp`; the `OPENCL/METAL` enums are
  placeholders, not commitments.

---

# Part C — G6: Direct Ridge regression (full implementation blueprint)

**Why it's new.** Today's "ridge" (`fit_ridge_pls`, `extra_pls.cpp:1827`) is ridge-**augmented** SIMPLS/NIPALS — it
builds `[X; √λ·I]`/`[Y;0]` and runs PLS, keeping a rank-truncated `n_components` fit. It is **not** the closed-form
`β = (XᵀX + λI)⁻¹XᵀY`. (The header comment `extra_pls.hpp:124` claiming an SVD normal-equations solve is inaccurate.)
A genuine closed-form Ridge-Gram solve **already exists privately** as `fit_ols` (`c_api_advanced.cpp:129`) — a useful
reference pattern, but single-target, no standardization, no SVD/path. The catalog has only a `ridge_pls` stub.

**Math design.** Direct multi-output Ridge, sklearn/glmnet-compatible:
- *Conventions:* center X and Y by column means; **intercept = y_mean − x_mean·β** (penalty **not** applied to the
  intercept — critical for parity). sklearn does **not** scale X (default `center_x=1, scale_x=0`); glmnet/`MASS::lm.ridge`
  standardize columns then de-scale. Reuse the existing `center_x/scale_x/center_y/scale_y` flags. Zero-variance column
  → scale 1.0 (sklearn convention) so it gets a 0 coefficient, not NaN.
- *Primal (p ≤ n):* solve `(XcᵀXc + λI_p)B = XcᵀYc`. **Default = QR on the augmented `[Xc; √λ·I]`** — the textbook
  stable Ridge solve, reusing the **already-shipped** `n4m_householder_qr`+`n4m_back_solve_R` (zero new linalg).
- *Dual / kernel (p ≫ n — the dominant NIRS regime):* `(K + λI_n)A = Yc`, `K = XcXcᵀ`, then `B = XcᵀA`. O(n³+n²p)
  instead of O(p³). **AUTO:** dual when p>n, primal otherwise. Linear kernel for v1 (RBF/poly is a follow-on; the kernel
  infra exists in `kernel_pls.cpp`).
- *λ-path + selection:* one thin SVD `Xc = USVᵀ`, then `B = V·diag(sᵢ/(sᵢ²+λ))·UᵀYc` per λ (amortized) and near-free
  **GCV**. v1 scope = fixed-λ + GCV-over-path; full k-fold RidgeCV via the existing `cross_validate_regression` as a
  follow-on. Multi-output Y native (B is p×q; factor shared across columns).

**ABI surface (additive, minor bump 1.10.0 → 1.11.0).** One new symbol in `pls.h` after `n4m_continuum_regression_fit`
(there is **no** generic string dispatcher — every method is its own `n4m_*_fit`):
```c
N4M_API n4m_status_t n4m_ridge_fit(
    n4m_context_t* ctx, const n4m_config_t* cfg,
    const n4m_matrix_view_t* X, const n4m_matrix_view_t* Y,
    const double* lambdas, int64_t n_lambdas,   /* path; NULL+0 => single λ from cfg */
    n4m_method_result_t** out_result);
```
Config additions (data-only, thread λ through the fit arg + reuse center/scale setters to keep the surface to **one**
new symbol): `ridge_lambda`, `ridge_solver` (AUTO/CHOLESKY/SVD/DUAL), `ridge_fit_intercept`, `ridge_lambda_select`
(NONE/GCV/CV). New result keys via a `pack_ridge_result`: `coefficients` (p×q), **`intercept`** (1×q, new — sklearn
returns `intercept_`), `x_mean`, **`x_scale`** (new), `y_mean`, `predictions`, `rmse`, `lambda_selected`, and for a
path `lambda_path` + `gcv_scores`/`cv_scores`. **Release gate:** add the symbol to all three `expected_symbols_*.txt`
+ a `changes_log.md` entry **in the same PR** (ABI diff fails closed), and `bump_version.sh --check` after the bump.

**Core plan.** New `cpp/src/core/ridge.cpp`/`.hpp` (parallel to `ecr.cpp`), wired via `target_sources()` in
`n4m_targets.cmake` (never a 2nd `add_library(n4m_core)`). Steps: standardize (reuse `column_means`/`subtract_means`
+ new zero-variance-aware `column_scale`) → choose solver (AUTO) → primal augmented-QR (reuse shipped primitives) /
dual Gram solve (needs a small SPD Cholesky or QR-on-K) / SVD path + GCV → recover intercept + de-scale → predictions/
rmse. Edge cases: p>n, λ=0 rank-deficient, zero-variance cols, multi-output, single-sample guard, NaN/Inf λ rejected.
C-API home: `c_api_advanced.cpp` (already owns the ridge-Gram pattern), standard try/catch→`n4m_status_t`.

**Parity references.** Primary: `sklearn.linear_model.Ridge` (`solver='cholesky'` primal / `'svd'` path), compare
`coef_` and `intercept_` separately, **1e-10** primal / 1e-8 SVD+dual (donor→oracle: n4m is the Gate-A 1e-12 oracle,
sklearn the Gate-B 1e-8 reference). Secondary: `glmnet(alpha=0, standardize=TRUE)` and `MASS::lm.ridge` — **document
the λ reparameterisation** (glmnet objective `(1/2n)‖y-Xb‖²+λ·pen` ⇒ `λ_sklearn = n·λ_glmnet`, modulo standardization)
or parity spuriously "fails"; gate 1e-6 after reparam+de-standardize. `KernelRidge(kernel='linear')` cross-checks the
dual ≡ primal to 1e-10. **Do not reuse** `RidgePlsSklearnReference` (`registry.py:601`) — that's the augmented-PLS
method.

**Catalog & bindings.** New `catalog/methods/models.regularized.ridge.yaml` (sibling to the `ridge_pls` stub; do not
overload it), `since_abi: 1.11.0`, `abi_symbols: [n4m_ridge_fit]`, filled parity block. Bindings (all call the one
core — never re-implement): **Python full n4m** (currently has **no** ridge) — ctypes decl + `n4m.python.ridge(...)`
functional wrapper + an idiomatic **sklearn `Ridge` regressor** (`coef_`/`intercept_`, `alpha`/`fit_intercept`/`solver`)
in `n4m/sklearn/` — *this is the drop-in for G3 replacing nirs4all's Ridge*; **slim pls4all** `ridge_fit`; **R** "ridge"
dispatch branch (re-vendor the new core file); **WASM/JS** TS wrapper + Emscripten export (strong WASM candidate — small,
closed-form, no threads); **Julia/MATLAB-Octave/JNI** dispatch branches; Sphinx page noting the glmnet λ reparam.

**GPU story.** Single-fit GPU is **free**: primal Gram (`XcᵀXc`), dual Gram (`XcXcᵀ`), and reconstruction (`XcᵀA`) are
all `linalg::gemm` → already dispatch to cuBLAS under `cuda-on`, bit-identical to CPU. The factorization stays CPU
(small at NIRS sizes → little single-fit win, consistent with ≈parity). **Ridge is the strongest batched-GPU candidate
in the library** — the SVD path makes a λ-grid a cheap diagonal rescale + gemm, and k-fold×λ-grid is an embarrassingly
parallel batch of SPD solves (`cusolverDnDpotrfBatched`) — exactly DEFERRALS.md's deferred "axis 2". Recommendation:
ship CPU Ridge + free single-fit GPU now; file batched-RidgeCV-on-GPU as the flagship deferred task (needs cuSOLVER;
only cuBLAS is linked today).

**Ridge task list:** `ridge-1` core (L) · `ridge-2` SPD Cholesky/QR-on-K helper (S) · `ridge-3` SVD λ-path+GCV (M) ·
`ridge-4` `n4m_ridge_fit` + `pack_ridge_result` + ABI 1.11 bump (M) · `ridge-5` regen 3 ABI snapshots + changes_log +
`bump_version --check` (**S, gate**) · `ridge-6` parity fixtures + `RidgeSklearnReference` (L) · `ridge-7` catalog entry
(S) · `ridge-8` Python wrappers (M) · `ridge-9` R/WASM/Julia/MATLAB/JNI + Sphinx (M) · `ridge-10` GPU confirm + file
batched-RidgeCV (M).

---

# §9. Cross-cutting risks (no single owner)

1. **ABI-bump fan-out.** Any new symbol (Ridge, batched-CV) must land in the header + **all three** snapshots +
   `changes_log.md` + catalog + `_ffi` + smoke asserts **in one PR**, or the fail-closed ABI diff (Linux today) / the
   smoke breaks. The hardcoded `(1,9,0)` assert is the live proof this fan-out is currently mishandled.
2. **Cross-platform ABI gate is Linux-only.** macOS/Windows — the platforms that ship CRAN/PyPI binaries — have no
   snapshot diff and wrong snapshot files. Fix before any 1.0 freeze.
3. **Model-blob cross-version policy** *(critic gap, High)*. The serializer writes the ABI triple but the **reader
   checks only the serialization-format version** — a model saved by a newer n4m silently loads into an older runtime.
   Define + enforce a load policy with boundary tests (and fix the stale `_model.py` docstring that claims otherwise).
4. **Documentation truth.** Three "single sources of truth" (DISTRIBUTION.md, ARCHITECTURE.md, CI gate messages,
   parity/README) currently **assert things that are false** (wrong artifact names, non-existent coverage). For a
   release these mislead the maintainer and reviewers as much as a code bug.
5. **Supply chain.** No SBOM / signed source archive / SLSA provenance / SHA256SUMS / reproducible-build epoch exists
   despite being fully specified in DISTRIBUTION.md §A/§B (`no-source-archive-sbom-provenance-ci`, High). The only
   Release asset is the R tarball.
6. **Namespace/registry custody** (`claim-and-secure-namespaces`, Medium). The repo rename already broke Trusted
   Publishing once (v0.98.0 invalid-publisher). Confirm PyPI publishers point at `nirs4all-methods`; reserve the npm
   `@nirs4all` scope + CRAN names; custody the tokens/GPG/2FA offline.
7. **Version cut.** `main` is ~75 commits past `v0.98.0` with an additive ABI 1.9→1.10; the next release **must** bump
   the project semver (e.g. 0.99.0) and cut the CHANGELOG `[Unreleased]` block (`changelog-cut-and-version-bump`).
8. **Direct Ridge is a vertical slice** across core → solver → generator → catalog → 6 bindings → ABI gate; sequence
   it as one coherent unit, not piecemeal.
9. **Web accessibility/i18n** *(critic gap)*: the JS binding/web example has no aria/i18n story for a public web page.

# §10. Investigated and refuted (do **not** action)

- **`verify-license-token`** → *not-an-issue.* Committed `R CMD check --as-cran` logs show "DESCRIPTION
  meta-information … OK"; `CeCILL (== 2.1)` standardizes with no LICENSE pointer required. (Minor: an `inst/COPYRIGHTS`
  line dangling-references a LICENSE file — cosmetic.)
- **`gpu-stale-bundled-so`** → *not-an-issue.* The `libn4m.so.1.9.0` under the Python binding trees are **gitignored
  build artifacts**, never tracked; `build_libn4m_in_wheel.sh` deletes + rebuilds them fresh (1.10.0) inside every
  cibuildwheel run. (Working-tree hygiene only — folded into `clean-stale-generated-dirs-and-debris`.)
- Partially-refuted sub-claims (the parent task survives, the detail was wrong): CARS "no n_iterations cap"
  (it's int32-bounded); README "ABI 1.9 string" (none exists); WASM "oversized-stride attack" (unreachable — rowmajor-
  only init + length guard); `.n4a` "refuses on ABI mismatch" (it warns + returns OK); fix-i64-read "1e32 garbage"
  (every BigInt is `Number()`-wrapped).

# §11. Sequenced roadmap

**Phase 0 — Unblock the pipeline & stop the lies (days).**
`abi-assert-1-9-0-blocks-release` · `n4m-loader-auditwheel-libs` · `delete-subsumed-branches` ·
`abi-gate-not-enforced-macos-windows` + `macos-windows-abi-snapshots-stale` + `abi-snapshot-regen-script` ·
`fix-parity-gate-overstated-claim` · `distribution-md-stale` / `rewrite-architecture-md` / `fix-readme-cli-binary-and-table`
/ `fix-security-md-naming` / Sphinx + scratch-file hygiene · `changelog-cut-and-version-bump`.

**Phase 1 — Trust the engine (1–2 weeks).**
`wire-cpp-model-parity-tests` · `fix-or-remove-doctest-uve-test` · `abi-memory-test-suite` · `sanitize-binding-paths`
+ `lsan-explicit-leak-gate` · `expand-binding-parity-method-coverage` · `multithread-test-coverage` ·
`pcr-gram-pxp-conditioning` + the A1 conditioning fixes · `no-fuzz-harness` (deserialize + matrix-view).

**Phase 2 — Ship the packages (parallel with Phase 1).**
G1: `real-cran-farm-check` + the CRAN-incoming polish (`title-acronym-note`, `authors-ctb-malformed`,
`dual-package-duplication`, `tarball-regen`). G2: `sdist-broken-no-lib-no-source` · `unify-nirs4all-methods-workflow`
· `macos-deployment-target-sync`. G4: `npm-publish-workflow` · `wire-generic-test-into-ci` · `fix-stale-generic-path-docs`
· `browser-html-example`. Supply chain: `no-source-archive-sbom-provenance-ci` · `release-r-no-tag-trigger` ·
`soname-linkage-gate-not-in-ci` · `claim-and-secure-namespaces` · model-blob version policy.

**Phase 3 — Features & downstream (weeks).**
G6 **direct Ridge** as one vertical (`ridge-1..10`, ABI 1.11 bump). G3 nirs4all: `map-model-classes` →
`adapter-shim-or-repoint` → `swap-acceptance-parity-gate` + per-model fixes (`in-sample-predict-coef-export`,
`plsda-predict-proba`, `seeding-reproducibility-note`, `consolidate-pypi-packaging`). G5 GPU: `gpu-ci-correctness-gate`
· `gpu-reference-cpu-no-fallback-trap` · `cublas-shared-handle-race` · then the deferred `gpu-batched-cv-abi` →
`gpu-batched-execution-impl` (the real "GPU possible" deliverable) and the `gpu-webgpu-browser-path` scope note.

---

# §12. Master checklist (by severity, post-verification)

### Blockers (2)
- [ ] **`abi-assert-1-9-0-blocks-release`** (G2) — derive ABI at test time in `release-python.yml:273,423`.
- [ ] **`n4m-loader-auditwheel-libs`** (G2) — glob `nirs4all_methods.libs/` in `n4m/_ffi.py`.

### High (~33)
ABI/build: `abi-gate-not-enforced-macos-windows` · `macos-windows-abi-snapshots-stale`.
Tests: `wire-cpp-model-parity-tests` · `fix-or-remove-doctest-uve-test` · `expand-binding-parity-method-coverage` ·
`abi-memory-test-suite` · `sanitize-binding-paths` · `no-fuzz-harness`.
Core: `pcr-gram-pxp-conditioning`.
Threading: `cublas-shared-handle-race`.
Docs: `distribution-md-stale`/`rewrite-distribution-md`.
G1: `real-cran-farm-check`.
G2: `sdist-broken-no-lib-no-source` · `unify-nirs4all-methods-workflow` · `abi-loader-version-drift` *(=blocker fix)*.
G3: `map-model-classes` · `adapter-shim-or-repoint`.
G4: `npm-publish-workflow` · `wire-generic-test-into-ci`.
Process/supply-chain: `no-source-archive-sbom-provenance-ci`.
Ridge: `ridge-5` (ABI snapshot gate) · the catalog+bindings tasks the critic flagged.

### Medium (~60) — see the per-section lists above (A1–A8, G1–G6). Highlights:
`stride-rule4-divergence-doc-or-fix` · `category-headers-temp-stubs` · `num-threads-wire-or-document` ·
`oversubscription-runtime-guard` · `multithread-test-coverage` · `tsan-cover-parallel-paths` ·
`abi-snapshot-regen-script` · `soname-linkage-gate-not-in-ci` · `multi-regime-model-parity` ·
`make-fullsweep-status-visible` · `selector-rng-parity-coverage-matrix` · `rewrite-architecture-md` ·
`fix-readme-cli-binary-and-table` · `fix-sphinx-conf-and-about` · `fix-security-md-naming` · `security-md-stale` ·
`input-driven-iteration-caps` · `wasm-memory-growth-dos` · `view-validation-negative-tests` ·
G1: `title-acronym-note` · `authors-ctb-malformed` · `dual-package-duplication`.
G2: `macos-deployment-target-sync` · `pypi-readme-mislabel-full-pkg` · `clean-stale-generated-dirs-and-debris`.
G3: `in-sample-predict-coef-export` · `consolidate-pypi-packaging` · `missing-model-decision` · `plsda-predict-proba` ·
`swap-acceptance-parity-gate` · `check-estimator-conformance` · `seeding-reproducibility-note`.
G4: `fix-stale-generic-path-docs` · `wasm-size-trim-brotli` · `browser-html-example` · `memory-growth-untrusted-input` ·
`decide-threads-coop-coep` · `version-sync-js-package` · `distribution-md-js-section-rewrite`.
G5: `gpu-ci-correctness-gate` · `gpu-doctest-dispatch-unit` · `gpu-strided-buffer-contract` ·
`gpu-reference-cpu-no-fallback-trap` · `gpu-cuda-wheel-packaging` · `gpu-batched-cv-abi`.
Process: `release-r-no-tag-trigger` · `delete-subsumed-branches` · `changelog-cut-and-version-bump` ·
`ffi-abi-minor-hardcode-drift` · `release-process-doc-contradiction` · `claim-and-secure-namespaces` ·
`security-md-p4a-regression`. Cross-cutting: **model-blob version policy**.

### Low (~40) — polish/hardening: the remaining A1–A8 items, the G-track `Low` items, the GPU `Low` (batched-impl,
cuSOLVER, fp32, webgpu, rocm/metal), and the scratch-file/doc-naming cleanups (`fix-i64-pair-read-methodresult`,
`browser-loading-locatefile`, `view-no-buffer-length-bound`, `core-rowscols-overflow-inconsistent`,
`wasm-js-double-length-check`, `fuzz-deserialize`, `stale-date-field`, `version-pin-contract`, etc.).

### Refuted — do not action: `verify-license-token`, `gpu-stale-bundled-so` (see §10).

---

*Generated 2026-06-03 by a 4-phase multi-agent review (14 dimension reviewers + adversarial verification of every
blocker/high finding + a direct-Ridge design pass + a completeness critic). All file:line citations were valid on
`main` @ `3cab139` at review time — re-verify before acting on any single item.*
