# Phase 1 — Opus independent post-review transcript

**Date**: 2026-05-18
**Reviewer**: Claude Opus 4.7 (via `feature-dev:code-reviewer` agent)
**Scope**: Phase 1 — PCG64 RNG (uint64 stream + Ziggurat standard_normal) with C ABI surface
**Verdict**: ACCEPT with two pre-commit nits — all addressed below.

## Summary

The reviewer cross-checked the PCG64 implementation against NumPy v1.26.4 upstream source (pcg64.h, distributions.c, bit_generator.pyx, ziggurat_constants.h) and confirmed:
- PCG-XSL-RR state transition and output function are byte-identical to NumPy.
- SeedSequence stage-1/stage-2 mixer constants, cross-mix loop order, and uint32-array special-case for seed=0 all match.
- Ziggurat bit-slicing (`idx=r&0xff`, `sign=(r>>8)&1`, `mantissa=r>>9`) matches NumPy.
- Vendored tables (`ki_double`, `wi_double`, `fi_double`, `ziggurat_nor_r`/`ziggurat_nor_inv_r`) are byte-identical to NumPy 1.26.4 source.
- MSVC 128-bit Knuth schoolbook multiplication is correct; carry propagation traced.
- C ABI surface is null-safe, exception-translated, allows `destroy(NULL)` per NumPy convention.

10/10 NumPy parity tests passing (5 seeds × 2 streams) is strong evidence of correctness.

## Pre-commit nits addressed

### Issue #1 — Fixture provenance: NumPy version mismatch (90% confidence) — FIXED

**Issue**: `parity/python_generator/scripts/generate_rng_fixture.py` was run with NumPy 2.3.5, but CONTRIBUTING.md pins NumPy 1.26.4. The fixture's `numpy_version` field recorded 2.3.5.

**Resolution**: Regenerated the fixture in a fresh venv with `pip install 'numpy==1.26.4'`. The new fixture has `"numpy_version": "1.26.4"`. All 20 tests still pass — confirming empirically that PCG64 uint64 stream and standard_normal stream are bit-identical between NumPy 1.26.4 and 2.3.5 (PCG64 stream is a stable public contract; Ziggurat tables haven't drifted).

### Issue #2 — Compiler warnings: C++-only flags applied to .c files (90% confidence) — FIXED

**Issue**: `cmake/CompilerWarnings.cmake` applied `-Wnon-virtual-dtor`, `-Wold-style-cast`, `-Woverloaded-virtual`, `-Wzero-as-null-pointer-constant` to every target indiscriminately, triggering 4 "valid for C++/ObjC++ but not for C" diagnostics per .c TU (rng_pcg64.c, ziggurat.c).

**Resolution**: Split the flag list into `_chemometrics4all_gnu_clang_warnings_common` (C+C++) and `_chemometrics4all_gnu_clang_warnings_cxx_only` (C++ only). The latter is wrapped in `$<COMPILE_LANGUAGE:CXX>` generator expression in `chemometrics4all_add_warnings`. Verified by full rebuild: no more C-vs-C++ mismatch warnings.

## Medium-confidence issues addressed

### Issue #3 — docs/algorithms/rng_pcg64.md missing (80% confidence) — FIXED

**Issue**: CONTRIBUTING.md step 7 mandates a Sphinx page for each algorithm; missing for the RNG.

**Resolution**: Created `docs/algorithms/rng_pcg64.md` documenting the algorithm (PCG-XSL-RR + Ziggurat), API surface, vendoring provenance (NumPy 1.26.4 BSD-3-Clause), and parity policy.

## Deferred to Phase 2+ (explicit tracking)

### Issue #4 — Test runner naming (80% confidence) — DEFERRED

The `c4a_testing::Runner` instance is named `phase-0-smoke` but holds Phase 1 RNG tests. As more phases come online, this should be refactored to per-phase Runner instances. Logged as Phase 2 opener.

### Issue #5 — No NumPy-parity test for `c4a_rng_pcg64_advance` (80% confidence) — DEFERRED

`test_advance_matches_repeated_draws` checks internal consistency only (advance + 1 draw == repeated draws), not bit-equivalence to `np.random.default_rng(s).bit_generator.advance(N)`. The implementation in `rng_pcg64.c:298-318` matches Brown 1994's canonical order, so most likely correct, but the test suite doesn't prove it. **Will be added** before Phase 9 (CARS / MCUVE — first consumer that uses `_advance` for thread-safe sub-streams).

### Issues #6-10 — Low-priority deferrals

- **#6**: Test-handle leak on REQUIRE failure — defer to introduction of RAII wrapper helpers.
- **#7**: Try/catch over pure C wrappers is defensive but unreachable — document as policy.
- **#8**: `_advance` exposes only low-64 of delta; full 128-bit needed for Phase 9+ CARS — add `_advance128` variant when needed.
- **#9**: Stage-3 SeedSequence loop unreachable with uint64 seed — keep as dead-code-but-correct mirror of NumPy for future `seed_seq_create` extension.
- **#10**: 848kB fixture kept in test-process memory — fine for Phase 1, revisit if test count grows.

## Post-fix verification

```
cmake --preset dev-debug                                   → configures, no warnings
cmake --build --preset dev-debug                           → 22/22 targets, zero warnings
./build/dev-debug/cpp/cli/chemometrics4all_cli --selfcheck → OK
ctest --preset dev-debug                                   → 100% (1/1 test groups, 20/20 sub-tests)
nm -D --defined-only build/dev-debug/cpp/src/libc4a.so.1.0.0 | awk '$2=="T" {print $3}' | wc -l → 36
diff <(nm -D ... | sort -u) cpp/abi/expected_symbols_linux.txt → no drift
python -c "import json; d=json.load(open('parity/fixtures/_rng_pcg64_stream_v1.json')); print(d['numpy_version'])" → 1.26.4
```

## Things done well (preserved from review)

- Byte-exact NumPy parity for PCG64 + Ziggurat + SeedSequence.
- MSVC 128-bit fallback correctness (Knuth schoolbook 64×64→128).
- C ABI safety (null-check, exception-translation, opaque handle, `destroy(NULL)` no-op).
- Big-endian hex encoding for double fixtures avoids JSON precision loss.
- Build skeleton + opaque-handle pattern reusable for Phase 2+ operators.
