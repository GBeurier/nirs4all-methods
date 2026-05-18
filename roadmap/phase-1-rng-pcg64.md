# Phase 1 — PCG64 RNG (load-bearing infrastructure)

## Goal

Implement a portable PCG64 random number generator in C, bit-exact against NumPy's `Generator(PCG64(seed))` uint64 stream, with a `standard_normal` generator via Ziggurat. Expose via C ABI as `c4a_rng_pcg64_t`. This unlocks bit-exact parity for **all 39 augmenters** (Phases 15-18), stochastic feature selectors (CARS, MCUVE — Phase 9), randomized X-outlier filters (IsolationForest, LOF — Phase 13), and stochastic splitters (KMeans — Phase 11).

## Scope decision

The Phase 1 plan originally bundled PCG64 + PCA helper + banded LDLT solver + wavelet filter banks + B-spline. To keep phases shippable, only **PCG64** lands in Phase 1. The other helpers are inlined at their first-consuming phase:

- **PCA helper** (randomized SVD) → inlined at **Phase 9** (FlexiblePCA / FlexibleSVD) and reused by Phase 11 (KennardStone PCA mode), Phase 13 (XOutlierFilter pca_residual / pca_leverage), Phase 19 (Hotelling T² / Q-residuals), Phase 20 (transfer metrics).
- **Banded LDLT solver** → inlined at **Phase 5b** (AsLS / AirPLS / ArPLS / IAsLS / BEADS).
- **Wavelet filter banks** (Haar / db4 / sym4 / coif1) → inlined at **Phase 6**.
- **B-spline** (cubic) → inlined at **Phase 18** (Spline_* augmenters).
- **wavelength_grid.hpp** → inlined at **Phase 10** (Resampler / CropTransformer).

This re-scoping is recorded in ROADMAP.md.

## Files to create

| Path | Purpose |
|------|---------|
| `cpp/src/core/common/rng_pcg64.h` | C header for the PCG64 state struct + functions |
| `cpp/src/core/common/rng_pcg64.c` | C implementation (uint64 stream + jump-ahead + reseed) |
| `cpp/src/core/common/ziggurat_constants.h` | Vendored NumPy Ziggurat tables (BSD-3-Clause); `wi_double`, `ki_double`, `fi_double` |
| `cpp/src/core/common/ziggurat.c` | Ziggurat `standard_normal_fill` algorithm |
| `cpp/src/c_api/c_api_rng.cpp` | extern "C" wrappers: `c4a_rng_pcg64_create / destroy / uint64 / standard_normal_fill / set_seed / advance` |
| `cpp/tests/test_rng_pcg64.cpp` | Bit-exact parity tests vs `_rng_pcg64_stream_v1.json` |
| `parity/python_generator/scripts/generate_rng_fixture.py` | Generate the parity fixture from NumPy |
| `parity/fixtures/_rng_pcg64_stream_v1.json` | 4096 uint64 + 4096 standard_normal draws for seeds {0, 1, 42, 12345, 0xDEADBEEF} |

## C ABI surface added (Phase 1 deltas)

```c
/* Added to cpp/include/chemometrics4all/c4a.h: */
typedef struct c4a_rng_pcg64_state_t c4a_rng_pcg64_state_t;
C4A_API c4a_status_t c4a_rng_pcg64_create(uint64_t seed, c4a_rng_pcg64_state_t** out);
C4A_API void         c4a_rng_pcg64_destroy(c4a_rng_pcg64_state_t* rng);
C4A_API c4a_status_t c4a_rng_pcg64_set_seed(c4a_rng_pcg64_state_t* rng, uint64_t seed);
C4A_API c4a_status_t c4a_rng_pcg64_uint64(c4a_rng_pcg64_state_t* rng, uint64_t* out);
C4A_API c4a_status_t c4a_rng_pcg64_uint64_fill(c4a_rng_pcg64_state_t* rng, uint64_t* out, size_t n);
C4A_API c4a_status_t c4a_rng_pcg64_standard_normal_fill(c4a_rng_pcg64_state_t* rng, double* out, size_t n);
C4A_API c4a_status_t c4a_rng_pcg64_advance(c4a_rng_pcg64_state_t* rng, uint64_t delta);
```

7 new exported symbols. Total ABI surface: 36 symbols.

## Implementation notes

### PCG64 algorithm

Mirror NumPy 1.26.4's `numpy/random/src/pcg64/pcg64.c`:
- State is a 128-bit unsigned integer (`__uint128_t` on gcc/clang; manual high/low pair on MSVC).
- Output function: PCG-XSL-RR (XOR-shift low + rotation right) producing uint64.
- Default multiplier: `2549297995355413924ull`, default increment seeded from the user seed via SplitMix64 (matching NumPy's `pcg64_set_seed`).

The seed-to-state mapping uses NumPy's `SeedSequence` rules: a single uint64 seed gets expanded to a 128-bit state via SplitMix64 ratcheted 4 times. We replicate that exactly for bit-equivalence.

### Ziggurat for `standard_normal`

NumPy uses a 256-region Ziggurat with vendored tables `ki_double[256]`, `wi_double[256]`, `fi_double[256]`. The algorithm:

```
do {
    r = pcg64.uint64()
    idx = r & 0xFF                          // 8 bits → region
    sign_bit = (r >> 8) & 1
    u = (int64_t)(r >> 9) * wi_double[idx]  // mantissa → fast accept
    if (abs(u) < ki_double[idx]) {
        return sign_bit ? -u : u
    }
    if (idx == 0) {
        // Tail: rejection sampling via -log(unif) / r
        // ...
    } else {
        // Wedge: fi_double[idx] vs fi_double[idx-1]
        // ...
    }
} while (true)
```

Tables are vendored from NumPy's `numpy/random/src/distributions/distributions.c` (BSD-3-Clause). Header declares them as static const so they're internal-only.

### Bit-exact parity verification

`parity/python_generator/scripts/generate_rng_fixture.py`:
```python
import json
import numpy as np

seeds = [0, 1, 42, 12345, 0xDEADBEEF]
fixture = {"seeds": {}}
for s in seeds:
    rng = np.random.default_rng(seed=s)
    fixture["seeds"][str(s)] = {
        "uint64": [int(rng.bit_generator.random_raw()) for _ in range(4096)],
        "standard_normal": list(np.random.default_rng(seed=s).standard_normal(4096)),
    }

with open("parity/fixtures/_rng_pcg64_stream_v1.json", "w") as f:
    json.dump(fixture, f)
```

`cpp/tests/test_rng_pcg64.cpp`:
- Loads the JSON.
- For each seed, calls `c4a_rng_pcg64_create(seed, &rng)`, then 4096 uint64 draws + 4096 standard_normal draws.
- Asserts byte-equality on uint64 (1e-15 tolerance on double for standard_normal).

## Parity references

| Operator | Reference | Tolerance |
|----------|-----------|-----------|
| `c4a_rng_pcg64_uint64_fill` | `np.random.default_rng(seed).bit_generator.random_raw(n)` | **exact** (byte equality) |
| `c4a_rng_pcg64_standard_normal_fill` | `np.random.default_rng(seed).standard_normal(n)` | **1e-15 abs / 0 rel** (bit-exact double) |

## Acceptance criteria

- ✅ `cmake --build` builds `libc4a.so.1.0.0` with 7 new symbols.
- ✅ `cpp/abi/expected_symbols_linux.txt` updated to 36 symbols.
- ✅ `cpp/tests/test_rng_pcg64.cpp` passes byte-equality vs NumPy fixture for all 5 seeds × 4096 uint64 draws.
- ✅ standard_normal parity within 1e-15 for all 5 seeds × 4096 draws.
- ✅ Build passes on `dev-debug`, `ci-linux-gcc12-release` presets.
- ✅ Phase 1 commit pushed to `main`, GitHub Actions CI green.
- ✅ Opus post-review transcript at `docs/reviews/phase-1/opus-post.md`.

## Verification

```bash
cd /home/delete/nirs4all/chemometrics4all
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/chemometrics4all_tests   # phase-0 smoke + phase-1 rng tests
```

Expected: all phase-0 (7) + phase-1 (5 seeds × 2 fields = 10) test names pass.

## Next phase

[Phase 2 — Stateless scatter](phase-2-stateless-scatter.md): SNV, LocalSNV, RobustSNV, AreaNormalization, Normalize, SimpleScale, LogTransform.
