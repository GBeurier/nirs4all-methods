# chemometrics4all architecture

## North star

A portable C++17 library that ships **everything around PLS**: preprocessings, splitters, filters, augmentations, signal-type detection, transfer metrics, FCK feature engineering. Every operator is callable identically from Python, R, MATLAB, JS/WASM, and 9 other languages through a single stable C ABI (`libc4a`). Zero mandatory runtime dependencies; optional accelerated backends.

`chemometrics4all` does **not** ship models. PLS regression/classification variants live in [pls4all](https://github.com/GBeurier/pls4all).

## Layout

```
chemometrics4all/
├─ cpp/
│  ├─ include/chemometrics4all/   # public C ABI: c4a.h, c4a_version.h, c4a_export.h.in
│  ├─ src/
│  │  ├─ core/                    # internal C++ (preprocessing/, augmentations/, splitters/,
│  │  │                           #               filters/, utilities/, common/)
│  │  └─ c_api/                   # extern "C" boundary, error buffer, opaque handles
│  ├─ cli/c4a_cli.cpp             # tiny CLI: --version, --abi-info, --selfcheck, --bench
│  ├─ tests/                      # hand-rolled doctest-style harness, ABI smoke + parity
│  └─ abi/expected_symbols_*.txt  # golden ABI per OS, diffed in CI
├─ bindings/
│  ├─ python/                     # ctypes + sklearn-compatible classes
│  ├─ r/                          # .Call + parsnip
│  ├─ matlab/                     # MEX + classdefs
│  ├─ js/                         # Emscripten/WASM
│  └─ {go,rust,julia,ruby,dotnet,lua,nim,jni,octave}/  # tier-1 stubs
├─ parity/
│  ├─ schema/c4a_fixture_schema_v1.json
│  ├─ tolerances.md
│  ├─ python_generator/           # nirs4all + scipy + scikit-learn + pywt + pybaselines
│  ├─ r_generator/                # prospectr + mdatools + baseline + waveslim
│  └─ fixtures/*.json             # ~250 JSON fixtures across phases
├─ benchmarks/                    # 5 suites: preprocess_matrix, augment_matrix, splitter, filter, cross_binding
├─ docs/                          # Sphinx + myst-parser, auto-deployed to GitHub Pages
├─ roadmap/phase-*.md             # per-phase plan (27 phases)
├─ scripts/bump_version.sh        # single source of truth = c4a_version.h
├─ CMakePresets.json              # dev-*, ci-*, blas-*, sanitizer-*
└─ .github/workflows/             # ci.yml, parity-gate.yml, docs.yml, abi-check.yml,
                                  # coverage.yml, sanitizers.yml, version-sync.yml, bench-nightly.yml
```

## Load-bearing decisions (mirrored from pls4all)

1. **C++ core + stable C ABI public.** Internal C++ stays in `cpp/src/`; only `cpp/include/chemometrics4all/c4a.h` is exported. ABI symbols are `c4a_*`, gated by `C4A_API` visibility, tracked in `expected_symbols_{linux,macos,windows}.txt`.

2. **Zero mandatory deps.** The reference build uses no BLAS, no OpenMP, no CUDA, no pybind11, no Eigen, no Boost, no pywt, no scipy, no sklearn. Hot kernels are vendored: PCG64 RNG (NumPy-compatible), banded LDLT solver, wavelet filter banks, randomized PCA, k-means++, Isolation Forest, LOF. Optional accelerated backends are CMake-gated.

3. **Opaque-handle C ABI.** Every stateful operator is a `c4a_*_state_t` opaque handle with `_create / _fit / _transform / _inverse_transform / _destroy` symbols. Stateless operators expose only `_transform` (or a static `_apply`). Error messages are stored in a per-context 4 KiB buffer.

4. **Parity gates.** Every operator has a `parity/fixtures/<id>_v1.json` with input + expected output + tolerance class. CI fails if any fixture diverges beyond the tolerance row in `parity/tolerances.md`. References are pinned (numpy 1.26.4, scipy 1.11.4, pywt 1.6.0, pybaselines 1.1.4, prospectr 0.2.7, …).

5. **Bit-exact stochastic parity.** Augmenters and stochastic ops (CARS, MCUVE, IsolationForest, KMeans, …) use a portable PCG64 RNG bit-equivalent to NumPy's `Generator(PCG64(seed))`. A 4096-draw sanity fixture (`parity/fixtures/_rng_pcg64_stream_v1.json`) guards drift. Tolerance class for these ops is the same as deterministic ops (1e-12 abs / 1e-13 rel).

6. **Dual versioning.** Project semver and ABI semver are independent. ABI major bumps only on signature change / symbol removal; minor on add; patch on bugfix-only. Single source of truth: `cpp/include/chemometrics4all/c4a_version.h`. `scripts/bump_version.sh` propagates to all manifests; `version-sync.yml` workflow gates any drift.

7. **Codex+Opus review workflow.** Each phase has a brief in `roadmap/phase-N-<name>.md`. Codex reviews the brief BEFORE implementation (`docs/reviews/phase-N/codex-pre.md`), reviews the diff AFTER (`codex-post.md`), and Opus runs an independent post-review (`opus-post.md`). All three transcripts are committed.

## Memory & threading model

- **Column-major internally** (Fortran convention) for optimal BLAS interop.
- **Stride-aware** matrix views: callers pass row-major (NumPy), column-major (R), transposed, or sliced views without forced copies.
- **Per-context** error buffer + RNG state — `c4a_context_t` is **not** thread-safe; use one context per thread or guard externally.
- **No global state** other than the version/build-info constants.

## Build matrix

| Target | Status |
|--------|:------:|
| Linux x86_64 (gcc-11/12/13, clang-14/16) | ✅ |
| Linux aarch64 | ✅ (cross + native) |
| macOS x86_64 / arm64 / universal2 (Apple clang) | ✅ |
| Windows x86_64 MSVC / MinGW UCRT64 | ✅ |
| Linux Android arm64 / x86_64 (NDK) | ⚪ (Phase 25+) |
| wasm32-emscripten | ⚪ (Phase 25) |

## See also

- [ROADMAP.md](ROADMAP.md) — 27-phase implementation plan
- [CONTRIBUTING.md](CONTRIBUTING.md) — Codex+Opus review workflow
- [parity/README.md](parity/README.md) — Parity strategy
- [docs/architecture/](docs/architecture/) — Detailed design documents
