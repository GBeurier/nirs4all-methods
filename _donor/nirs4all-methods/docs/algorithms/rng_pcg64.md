# PCG64 Random Number Generator

## Overview

`nirs4all-methods` ships a portable C implementation of the [PCG64](https://www.pcg-random.org/) random number generator that is **bit-exact compatible with NumPy's** `numpy.random.Generator(PCG64(seed))`. This unlocks deterministic, reproducible, cross-platform parity for every stochastic operator the library exposes — augmentations, randomized feature selectors (CARS, MCUVE), isolation-forest-based outlier filters, k-means splitters, and any future Monte-Carlo procedure.

## Why PCG64

- **Reproducibility across language bindings.** A Python user, an R user, and a C++ user can seed identically and get the same draws — no scipy/sklearn dependency required in the C path.
- **Statistical quality.** PCG passes BigCrush, TestU01, PractRand. Period 2¹²⁸. No known statistical defects.
- **Performance.** ~1 ns per uint64 on modern CPUs. Roughly twice as fast as MT19937, with 4× the period and far better small-state collision properties.
- **Portability.** Pure C, no SIMD intrinsics, no platform dependencies. Compiles unchanged on Linux/macOS/Windows/Android/WASM.

## API surface

```c
typedef struct n4m_rng_pcg64_state_t n4m_rng_pcg64_state_t;

N4M_API n4m_status_t n4m_rng_pcg64_create(uint64_t seed,
                                          n4m_rng_pcg64_state_t** out);
N4M_API void         n4m_rng_pcg64_destroy(n4m_rng_pcg64_state_t* rng);
N4M_API n4m_status_t n4m_rng_pcg64_set_seed(n4m_rng_pcg64_state_t* rng,
                                            uint64_t seed);
N4M_API n4m_status_t n4m_rng_pcg64_uint64(n4m_rng_pcg64_state_t* rng,
                                          uint64_t* out);
N4M_API n4m_status_t n4m_rng_pcg64_uint64_fill(n4m_rng_pcg64_state_t* rng,
                                               uint64_t* out, size_t n);
N4M_API n4m_status_t n4m_rng_pcg64_standard_normal_fill(
    n4m_rng_pcg64_state_t* rng, double* out, size_t n);
N4M_API n4m_status_t n4m_rng_pcg64_advance(n4m_rng_pcg64_state_t* rng,
                                            uint64_t delta);
```

7 ABI symbols. Opaque-handle pattern — internals are not exposed.

## Algorithm details

### Seed expansion (SeedSequence)

A single uint64 user seed expands to the 256-bit PCG64 state (state + increment, each 128 bits) via NumPy's `SeedSequence` algorithm:

1. **Stage 1 — pool initialization.** SplitMix64 mixer with constants `INIT_A = 0x43b0d7e5`, `MULT_A = 0x931e8875`, `INIT_B = 0x8b51f9dd`, `MULT_B = 0x58f38ded` initializes a 4×uint32 pool.
2. **Stage 2 — cross-mix.** Pairwise mixer with `MIX_MULT_L = 0xca01f9dd`, `MIX_MULT_R = 0x4973f715`, `XSHIFT = 16` cross-mixes the pool.
3. **Stage 3 — state generation.** The first 4 uint32s of the pool become 2 uint64s for the PCG64 initial state and increment.

This matches NumPy's `_seed_sequence.pyx` byte-exactly. See `cpp/src/core/common/rng_pcg64.c::n4m_pcg64_engine_seed`.

### Output function (PCG-XSL-RR)

For each draw:

```
state = state * MULT_128 + INC_128         // 128-bit LCG step
hi = state >> 64
lo = state & 0xFFFFFFFFFFFFFFFF
output = rotr64(hi ^ lo, hi >> 58)
```

Multiplier `MULT_128 = 0x2360ed051fc65da4_4385df649fccf645`. Increment is derived from the seed (always odd via `| 1`). The PCG-XSL-RR scheme XOR-folds the upper and lower halves of the state, then applies a data-dependent rotate (top 6 bits of state).

### Ziggurat for standard_normal

The Gaussian generator uses the [Marsaglia–Tsang 256-region Ziggurat](https://en.wikipedia.org/wiki/Ziggurat_algorithm) with NumPy's precomputed tables (`ki_double[256]`, `wi_double[256]`, `fi_double[256]`). For each draw:

1. Pull one uint64 from PCG64.
2. Use 8 bits as a region index, 1 bit as sign, 55 bits as mantissa.
3. Fast accept if `|u| < ki_double[idx]` (~97% of the time).
4. Slow path: tail rejection (region 0) or wedge interpolation (regions 1-255) using `fi_double`.

The bit-slicing is `idx = r & 0xFF; sign = (r >> 8) & 0x1; mantissa = (r >> 9)` — identical to NumPy's `random_standard_normal_fill`.

### Jump-ahead

`n4m_rng_pcg64_advance(rng, delta)` jumps the state forward by `delta` LCG steps in O(log delta) time using Brown's algorithm (binary fast-exponentiation of the LCG recurrence). Useful for parallel stream subdivision: thread *i* of *N* uses `advance(i * N_per_thread)` to start at its own stream offset.

## Vendored content

The Ziggurat tables (`ki_double`, `wi_double`, `fi_double`, `ziggurat_nor_r`, `ziggurat_nor_inv_r`) and the SeedSequence + PCG64 algorithm are vendored from [NumPy 1.26.4](https://github.com/numpy/numpy/tree/v1.26.4/numpy/random/src) under the [BSD-3-Clause license](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt). The license is compatible with nirs4all-methods's CeCILL-2.1.

Files vendored:
- `cpp/src/core/common/ziggurat_constants.h` ← NumPy `numpy/random/src/distributions/ziggurat_constants.h`
- Algorithmic structure of `cpp/src/core/common/rng_pcg64.c` ← NumPy `numpy/random/src/pcg64/pcg64.{c,h}` + `numpy/random/src/distributions/distributions.c`

The tables are byte-identical between NumPy 1.26.4 and 2.3.5; the PCG64 raw uint64 stream is a documented stable contract across NumPy versions ([NumPy random compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html)).

## Parity policy

Fixture: `parity/fixtures/_rng_pcg64_stream_v1.json` (848 kB, generated by `parity/python_generator/scripts/generate_rng_fixture.py`).

Contents: for each of 5 seeds (0, 1, 42, 12345, 0xDEADBEEF):
- 4096 consecutive uint64 draws via `rng.bit_generator.random_raw(4096)`.
- 4096 consecutive `standard_normal` draws via a fresh `default_rng(seed).standard_normal(4096)`.

Doubles are encoded as big-endian hex (8 bytes per double) to avoid JSON float-precision loss.

Test (`cpp/tests/test_rng_pcg64.cpp`):
- uint64 stream: **byte-equal** (exact integer comparison).
- standard_normal: **1e-15 abs / 0 rel** (IEEE-754 bit-equal double, allowing one-ulp slack which the implementation never actually consumes).

## Verification

```bash
cd /home/delete/nirs4all/nirs4all-methods
cmake --preset dev-debug
cmake --build --preset dev-debug
./build/dev-debug/cpp/tests/n4m_tests
```

Expected: all phase-1 RNG tests pass (10 parity + 3 functional) byte-exactly.

## References

1. Melissa O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation" (2014). [pcg-random.org](https://www.pcg-random.org/paper.html)
2. Marsaglia, G. & Tsang, W. W., "The Ziggurat Method for Generating Random Variables", *Journal of Statistical Software* 5(8) (2000).
3. F. R. Brown, "Random Number Generation with Arbitrary Strides", *Transactions of the American Nuclear Society* 71 (1994). (Jump-ahead algorithm.)
4. NumPy random module source: [github.com/numpy/numpy/tree/v1.26.4/numpy/random](https://github.com/numpy/numpy/tree/v1.26.4/numpy/random)
