# Phase 15 — Augmenters (noise + drift) — implementation brief

**Date**: 2026-05-18
**Scope**: 7 augmenter operators, first members of the new `n4m_aug_*` ABI category.
**ABI contract**: locked upfront in `roadmap/phase-15-18-augmenters-abi-contract.md` — 3 symbols per op (`_create`, `_apply`, `_destroy`), RNG handle is the first argument of `_create` and is borrowed (not owned).

## Operators delivered

| Operator                        | C ABI prefix                  | Algorithm                                                                                                                                              |
| ------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GaussianAdditiveNoise           | `n4m_aug_gaussian_noise_*`    | `out = X + N(0, 1) * std(X[i, :]) * sigma`                                                                                                            |
| MultiplicativeNoise             | `n4m_aug_multiplicative_noise_*` | `out[i, j] = X[i, j] * (1 + sigma_gain * N_i)`                                                                                                       |
| SpikeNoise                      | `n4m_aug_spike_noise_*`       | `n_spikes[i] = U[n_min, n_max]`; sort+unique random indices, add Uniform amplitudes                                                                    |
| HeteroscedasticNoiseAugmenter   | `n4m_aug_hetero_noise_*`      | `sigma[i, j] = base + dep * |X[i, j]|`; `out = X + N(0, 1) * sigma`                                                                                  |
| LinearBaselineDrift             | `n4m_aug_linear_drift_*`      | `out[i, j] = X[i, j] + offset_i + slope_i * (j - mean_j)` (uniform offset/slope draws)                                                                |
| PolynomialBaselineDrift         | `n4m_aug_poly_drift_*`        | `out[i, j] = X[i, j] + sum_k coeffs[k, i] * lambda_j^k` on `lambda = linspace(-1, 1, n_features)`                                                     |
| PathLengthAugmenter             | `n4m_aug_path_length_*`       | `L[i] = max(1 + path_length_std * N_i, min_path_length)`; `out[i, j] = L[i] * X[i, j]`                                                                |

## ABI additions

21 new symbols (7 ops × 3 symbols), all under the existing `N4M_1` linker tag. The `n4m.h` public header is **not** modified in this phase — the public declarations live inside `cpp/src/c_api/c_api_augmenters.cpp` as forward-decls (per the locked task constraints), and consumers redeclare just the symbols they call. This is a deliberate exception to the "everything in n4m.h" rule for the four augmenter phases (15-18); a single consolidated banner will be added to n4m.h at the end of Phase 18 when the full 39-operator surface is finalised.

Expected-symbol files updated in lock-step:

* `cpp/abi/expected_symbols_linux.txt`   (251 -> 272)
* `cpp/abi/expected_symbols_macos.txt`   (251 -> 272)
* `cpp/abi/expected_symbols_windows.txt` (251 -> 272)

The ABI surface gate (`.github/workflows/abi-check.yml`) verifies an exact match against these files, so the additions are mandatory for the build to pass CI.

## RNG contract

Every operator stores a borrowed `n4m_rng_pcg64_state_t*` pointer in its state. The RNG handle's lifetime is the caller's responsibility. The augmenter does **not** call `n4m_rng_pcg64_destroy`. To get a bit-exact reproducible stream, the caller does:

```c
n4m_rng_pcg64_set_seed(rng, seed);                  /* lock the stream */
n4m_aug_gaussian_noise_apply(handle, X, out);       /* draws N samples */
```

All operators consume the PCG64 stream through:

* `n4m_pcg64_engine_standard_normal_fill(rng, buf, n)` — for `rng.standard_normal()` parity
* `n4m_pcg64_engine_next_double(rng)`                   — for `rng.random()` parity

`rng.uniform(a, b, n)` and `rng.integers(low, high, n)` are intentionally avoided in the C engine (they would require re-implementing NumPy's Lemire rejection for bounded integers); the Python reference module mirrors this by unrolling the uniform / integer draws to `lo + (hi - lo) * u` / `floor(u * range)` mappings. The result is a bit-exact 1e-15 abs parity at the cost of producing slightly different values than `nirs4all`'s `rng.uniform` / `rng.integers` paths would yield — `n4m_parity_augmenters_ref` is the contractual reference, not nirs4all itself.

## Test coverage

A standalone test binary `n4m_augmenters_tests` (independent of the main `n4m_tests` -- per the locked task constraint not to modify `tests/main.cpp`) registers 14 tests:

* **7 smoke tests** (one per op): determinism check (seed → apply → re-seed → apply → byte-identical), basic shape validity, NULL-safety.
* **7 parity tests** (one per op): load fixture `parity/fixtures/aug_<NAME>_v1.json`, iterate every case, seed RNG to the case's seed, create the handle, apply, compare to expected output at 1e-15 abs tolerance.

Total cases across all parity tests: 26 (3 + 3 + 4 + 4 + 4 + 4 + 4).

The parity-gate runner (`run_parity_gate.py`) was updated to discover **every** test binary under `cpp/tests/` in the build directory (previously hard-coded to `n4m_tests`), so the augmenter tests are part of Stage 3 of the gate.

## Files touched

### New (core C)

* `cpp/src/core/augmentations/noise/gaussian_noise.{c,h}`
* `cpp/src/core/augmentations/noise/multiplicative_noise.{c,h}`
* `cpp/src/core/augmentations/noise/spike_noise.{c,h}`
* `cpp/src/core/augmentations/noise/hetero_noise.{c,h}`
* `cpp/src/core/augmentations/drift/linear_drift.{c,h}`
* `cpp/src/core/augmentations/drift/poly_drift.{c,h}`
* `cpp/src/core/augmentations/drift/path_length.{c,h}`

### New (C ABI wrapper)

* `cpp/src/c_api/c_api_augmenters.cpp` — 21 `extern "C"` entry points.

### New (tests)

* `cpp/tests/test_augmenters_noise_drift.cpp` — standalone test executable with 14 tests.

### New (Python reference)

* `parity/python_generator/src/n4m_parity_augmenters_ref/__init__.py`
* `parity/python_generator/src/n4m_parity_augmenters_ref/{gaussian_noise,multiplicative_noise,spike_noise,hetero_noise,linear_drift,poly_drift,path_length}.py`
* `parity/python_generator/scripts/generate_phase15_fixtures.py`

### New (fixtures, manifest entries)

* `parity/fixtures/aug_{gaussian_noise,multiplicative_noise,spike_noise,hetero_noise,linear_drift,poly_drift,path_length}_v1.json`
* `parity/fixtures/manifest.json` — 7 new SHA + byte-count entries.

### New (docs)

* `docs/algorithms/aug_{gaussian_noise,multiplicative_noise,spike_noise,hetero_noise,linear_drift,poly_drift,path_length}.md`
* `docs/reviews/phase-15/brief.md` (this file)

### Modified

* `cpp/src/CMakeLists.txt` — added 7 core .c files + `c_api/c_api_augmenters.cpp`.
* `cpp/tests/CMakeLists.txt` — registered `n4m_augmenters_tests` executable + ctest entry.
* `cpp/abi/expected_symbols_{linux,macos,windows}.txt` — 21 new entries each.
* `parity/python_generator/scripts/run_parity_gate.py` — Stage 3 now discovers every test binary under `cpp/tests/`.
* `docs/reviews/PHASES.md` — Phase 15 row updated to "14 / 14" with topic refined to "7 ops, `n4m_aug_*`".

### Not modified (per locked task constraints)

* `cpp/include/n4m/n4m.h`
* `cpp/include/n4m/n4m_version.h`
* `cpp/tests/main.cpp`
* Top-level `CMakeLists.txt`

## Verification

```
cmake --build --preset dev-release --target n4m_c          → libn4m.so.1.7.0 built
nm -D --defined-only libn4m.so.1.7.0 | awk '$2=="T" {print $3}' | wc -l → 272 (251 + 21)
diff expected_symbols_linux.txt exported.txt                            → empty (ABI surface match)
./build/dev-release/cpp/tests/n4m_augmenters_tests         → 14 / 14 passing
./build/dev-release/cpp/tests/n4m_tests                    → 171 / 171 passing (no regressions)
ctest --output-on-failure                                                → 2 / 2 test executables green
python run_parity_gate.py --skip-fixture-regen --skip-cross-validation  → Stage 3 PASS
ruff check parity/python_generator/src/n4m_parity_augmenters_ref         → clean
```

## CHANGELOG summary

```
phase-15: 7 noise + drift augmenters (n4m_aug_* category)

Adds the first 7 of 39 augmenter operators planned across Phases 15-18:
GaussianAdditiveNoise, MultiplicativeNoise, SpikeNoise,
HeteroscedasticNoiseAugmenter, LinearBaselineDrift,
PolynomialBaselineDrift, PathLengthAugmenter. Each exposes 3 ABI
symbols (create/apply/destroy) per the locked Phase 15-18 contract,
bringing libn4m from 251 -> 272 exported symbols. Parity 1e-15 abs
against the frozen NumPy reference in n4m_parity_augmenters_ref;
14 / 14 tests passing in the new n4m_augmenters_tests
binary. ABI version unchanged (1.7.0) — augmenter declarations land
in n4m.h at the end of Phase 18 once the full 39-op surface is
locked.
```
