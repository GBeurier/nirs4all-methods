# Changelog

All notable changes to chemometrics4all will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to dual semver: **project version** in `C4A_PROJECT_VERSION_*` and **ABI version** in `C4A_ABI_VERSION_*` (both in `cpp/include/chemometrics4all/c4a_version.h`).

## [Unreleased]

## [0.1.0+abi.1.2.0] — Phase 2 stateless preprocessings

### Added

- **Seven stateless preprocessing operators** with bit-exact `nirs4all` 0.8.x / `numpy` 1.26.4 parity:
  - **SNV** (`c4a_pp_snv_*`) — row-wise mean / std normalisation with configurable `with_mean`, `with_std`, `ddof`.
  - **LocalSNV** (`c4a_pp_lsnv_*`) — sliding-window SNV with `reflect` / `edge` / `constant` padding and cumsum-based moving statistics matching `nirs4all.LocalStandardNormalVariate`.
  - **RobustSNV** (`c4a_pp_rnv_*`) — median + `k * MAD` normalisation with optional center / scale, in-place quickselect median.
  - **AreaNormalization** (`c4a_pp_area_*`) — per-row division by `sum` / `abs_sum` / `trapz` area (trapezoidal integration with unit spacing, pair-sum form matching `np.trapz`).
  - **Normalize** (`c4a_pp_normalize_*`) — column-wise L2-norm scaling or user-defined-range scaling.
  - **SimpleScale** (`c4a_pp_simple_scale_*`) — column-wise min-max to `[0, 1]`.
  - **LogTransform** (`c4a_pp_log_*`) — element-wise `log(X + δ)` with optional fit-time auto-offset and arbitrary base.
- **C engines** under `cpp/src/core/preprocessing/{scatter,scaling}/` (7 `.c` + `.h` pairs). Pure ISO C11, no third-party dependencies, all accumulations preserve NumPy's left-to-right rounding order.
- **C ABI wrappers** in `cpp/src/c_api/c_api_preprocessing.cpp` — 21 new symbols (3 × 7 operators: `create / destroy / transform`), each wrapped in `try` / `catch` so no C++ exception ever crosses the boundary. Input validation enforces row-major contiguous F64 matrix views and matching output shape.
- **Parity fixtures** (`parity/fixtures/{snv,lsnv,rnv,area_norm,normalize,simple_scale,log_transform}_v1.json`) generated from a deterministic 50 × 200 NIR-shaped spectrum block (PCG64 seed=20260518); doubles encoded as big-endian hex to avoid JSON float-formatting loss.
- **Fixture generator** `parity/python_generator/scripts/generate_phase2_fixtures.py` — loads nirs4all's reference operators directly from source via `importlib` to avoid heavyweight package imports.
- **14 new tests** in `cpp/tests/test_preprocessing_stateless.cpp` (2 per operator: a lifecycle smoke test + a parity sweep over every fixture case). Tolerance: bit-equality OR `1e-12` absolute OR `1e-13` relative.
- **Documentation** under `docs/algorithms/` — one Markdown page per operator with algorithm, parameters, numerical contract, and reference.

### Numerical parity findings

- `numpy.mean / numpy.std` and `nirs4all`'s `X / std` evaluate divisions per-element. Replacing them with `X * (1 / std)` introduces single-ULP drift on certain inputs — all kernels use true division to preserve byte-for-byte parity.
- `numpy.trapz` / `scipy.integrate.trapezoid` with unit spacing computes `0.5 * sum(x[:-1] + x[1:])`, not `sum(x) - (x[0] + x[-1]) / 2`. The two forms are algebraically equal but FP-distinct; we use the pair-sum form.
- `nirs4all.RobustStandardNormalVariate` evaluates the two `if` blocks (center, scale) sequentially: when `with_center=False`, MAD is computed on `|X|` (raw), not on `|X - median|`. Reproducing this exact branching was the only RNV bug found during parity tuning.

### ABI

- Project version stays 0.1.0; **ABI bumps to 1.2.0** (additive: 21 new symbols, no breaking changes). Total exported symbols: 36 → **57**.

### Tests

- 34/34 passing: 7 phase-0 smoke + 13 phase-1 RNG + 14 phase-2 preprocessing.

### Reviews

- **Opus post-review** at `docs/reviews/phase-2/opus-post.md`. Verdict: REVISE → 3 high-confidence fixes applied: (1) SNV flat-row threshold tightened from `>= 1e-15` to `!= 0` to match nirs4all exactly, (2) `c4a_pp_lsnv_pad_mode_t` and `c4a_pp_area_method_t` enums hoisted from internal headers into public `c4a.h`, (3) LogTransform per-call recomputation behaviour documented in c4a.h.
- **Codex Phases 1+2 review** at `docs/reviews/phase-2/codex-post.md`. Verdict: ACCEPT with Phase 3 prerequisites. Applied: test Runner renamed to `chemometrics4all`, `parity/python_generator/pyproject.toml` added with numpy 1.26.4 pin, `cpp/src/c_api/c4a_linux.map` version script added (defense-in-depth on top of `-fvisibility=hidden`). Confirmed all 57 exports are `c4a_*`.
- Per user direction, **Codex review runs every two phases** in addition to per-phase Opus reviews.

## [0.1.0+abi.1.1.0] — Phase 1 PCG64 RNG

### Added

- **PCG64 random number generator** with bit-exact NumPy parity:
  - `cpp/src/core/common/rng_pcg64.{h,c}` — pure-C PCG-XSL-RR 128/64 implementation with native `__uint128_t` (gcc/clang) and MSVC two-uint64 fallback. Full NumPy `SeedSequence(seed)` expansion via SplitMix64-style 4-stage pool mixer.
  - `cpp/src/core/common/ziggurat.c` + `ziggurat_constants.h` — vendored NumPy 1.26.4 Ziggurat tables (BSD-3-Clause, license-compatible with CeCILL-2.1) with 256-region Marsaglia–Tsang algorithm for `standard_normal_fill`.
  - `cpp/src/c_api/c_api_rng.cpp` — 7 C ABI wrappers (`c4a_rng_pcg64_{create, destroy, set_seed, uint64, uint64_fill, standard_normal_fill, advance}`) behind opaque handle `c4a_rng_pcg64_state_t`. Try/catch wrapped, null-pointer-safe, `destroy(NULL)` no-op.
- **Parity fixture** `parity/fixtures/_rng_pcg64_stream_v1.json` (848 kB) — 5 seeds × (4096 uint64 + 4096 standard_normal) generated with `numpy==1.26.4` (pinned per CONTRIBUTING.md). Doubles encoded as big-endian hex to avoid JSON float-precision loss.
- **13 new tests** in `cpp/tests/test_rng_pcg64.cpp`: 3 functional (null-safety, set_seed-resets-stream, advance-matches-repeats) + 5 uint64 byte-exact parity (one per seed) + 5 standard_normal IEEE-754 bit-exact parity (one per seed). Hand-rolled JSON parser keeps the zero-dep policy.
- **Documentation** `docs/algorithms/rng_pcg64.md` — algorithm, API, vendoring provenance, parity policy.

### Fixed

- `cmake/CompilerWarnings.cmake` — split C/C++ warning flags. C++-only flags (`-Wnon-virtual-dtor`, `-Wold-style-cast`, `-Woverloaded-virtual`, `-Wzero-as-null-pointer-constant`) now wrapped in `$<COMPILE_LANGUAGE:CXX>` generator expression so they no longer trigger "valid for C++/ObjC++ but not for C" warnings on `.c` TUs.

### ABI

- Project version stays 0.1.0; **ABI bumps to 1.1.0** (additive: 7 new symbols, no breaking changes). Total exported symbols: 29 → **36** (`c4a_rng_pcg64_*` × 7 added).

### Review

- Opus post-review at `docs/reviews/phase-1/opus-post.md`. Verdict: ACCEPT. Two pre-commit nits fixed (fixture provenance, C-vs-C++ warnings). One medium issue fixed (algorithm doc). Five low-priority items tracked for Phase 2+.

### Tests

- 20/20 passing: 7 phase-0 smoke + 13 phase-1 RNG (3 functional + 10 NumPy parity).


## [0.1.0+abi.1.0.0] — Phase 0 bootstrap

### Added

- C++17 core skeleton mirroring `pls4all`'s architecture: `cpp/src/core/{config,context,matrix_view,status,version}.cpp` + `cpp/src/c_api/{c_api_context,c_api_matrix,c_api_version}.cpp`.
- Public C ABI header `cpp/include/chemometrics4all/c4a.h` (Phase 0 surface: status codes, dtype/backend enums, matrix view, context lifecycle, version queries). Full operator surface to be added phase by phase.
- CMake build system inherited from pls4all with 20+ presets in `CMakePresets.json` (dev-debug, dev-release, ci-*, blas-on, sanitizer-*). Zero mandatory dependencies in the reference build.
- Tiny CLI `chemometrics4all_cli` with `--version`, `--abi-info`, `--selfcheck`.
- Hand-rolled zero-dependency test harness in `cpp/tests/main.cpp` + `cpp/tests/harness.hpp`, with 7 Phase 0 smoke tests (all passing).
- GitHub Actions workflows inherited from pls4all (ci, parity-gate, docs, abi-check, coverage, sanitizers, version-sync) — to be re-enabled after Phase 1 once the new C ABI surface is wired.
- `ROADMAP.md` with the 27-phase plan, `ARCHITECTURE.md` with the load-bearing decisions, `CONTRIBUTING.md` with the Codex+Opus review workflow.

### Stripped from the pls4all template

- All PLS algorithm code in `cpp/src/core/` (50+ files: `pls_*`, `simpls*`, `opls*`, `aom_*`, `*_selection.cpp`, `model.cpp`, `pipeline.cpp`, `config.{cpp,hpp}` …).
- All PLS-specific C ABI files in `cpp/src/c_api/` (`c_api_{config,model,pipeline,operator_bank,gating_strategy,validation_plan,aom_selection,method_result,component_coefficients}.cpp`).
- All PLS test fixtures and ABI test suites in `cpp/tests/{abi,fixtures}/`.
- `bindings/` entirely (rebuilt in Phases 22-25 for Python / R / MATLAB / JS-WASM).
- `benchmarks/` entirely (rebuilt in Phase 26).
- `parity/python_generator/` and `parity/r_generator/` (rebuilt in Phase 1+ with c4a adapters; `parity/schema/` and `parity/tolerances.md` retained as inherited contract).
- `docs/methods/` (90 PLS algorithm docs), `docs/parity_audit_2026_05/`, `docs/reviews/phase-0/` (old pls4all reviews), `docs/benchmarks/cross_binding*.md`.
- Pls4all-specific top-level docs (`Overview.md`, `Direction_Technique.md`, `DISTRIBUTION.md`, `Backlog.md`) — replaced by `ARCHITECTURE.md`, `ROADMAP.md`.
- 97+ pls4all `roadmap/phase-*.md` files — replaced by c4a-scoped phases.
- `CI workflows except `ci.yml` moved to `.github/workflows.disabled/` (`parity-gate`, `release-python`, `abi-check`, `version-sync`, `coverage`, `sanitizers`, `docs`). Each is re-enabled at the phase that wires its inputs.

### ABI

- Initial ABI version `1.0.0`. `cpp/abi/expected_symbols_{linux,macos,windows}.txt` populated with the 29 actually-exported `c4a_*` symbols from `libc4a.so.1.0.0` (status / backend / dtype / matrix_view / context lifecycle / version queries).

### Opus post-review

- Independent Opus review completed (transcript at `docs/reviews/phase-0/opus-post.md`). 9 high-confidence issues identified and fixed before commit. 5 medium/low-priority items deferred to Phase 1+ with explicit tracking in the roadmap.

[Unreleased]: https://github.com/GBeurier/chemometrics4all/compare/v0.1.0...HEAD
[0.1.0+abi.1.0.0]: https://github.com/GBeurier/chemometrics4all/releases/tag/v0.1.0
