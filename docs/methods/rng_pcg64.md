# `rng_pcg64` — PCG64 Random Number Generator

_Group_: **Utility** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/rng_pcg64.md`](../algorithms/rng_pcg64.md)

## Description

`rng_pcg64` is a chemometrics4all utility method exposed through the C ABI and the generated language bindings.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

1. Melissa O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation" (2014). [pcg-random.org](https://www.pcg-random.org/paper.html)
2. Marsaglia, G. & Tsang, W. W., "The Ziggurat Method for Generating Random Variables", *Journal of Statistical Software* 5(8) (2000).
3. F. R. Brown, "Random Number Generation with Arbitrary Strides", *Transactions of the American Nuclear Society* 71 (1994). (Jump-ahead algorithm.)
4. NumPy random module source: [github.com/numpy/numpy/tree/v1.26.4/numpy/random](https://github.com/numpy/numpy/tree/v1.26.4/numpy/random)

### Mathematical principle

`rng_pcg64` follows the standard utility operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

C ABI entry points used by the language bindings:

```c
c4a_status_t c4a_rng_pcg64_advance(c4a_rng_pcg64_state_t* rng, uint64_t delta);
c4a_status_t c4a_rng_pcg64_create(uint64_t seed, c4a_rng_pcg64_state_t** out);
void c4a_rng_pcg64_destroy(c4a_rng_pcg64_state_t* rng);
c4a_status_t c4a_rng_pcg64_set_seed(c4a_rng_pcg64_state_t* rng, uint64_t seed);
c4a_status_t c4a_rng_pcg64_standard_normal_fill( c4a_rng_pcg64_state_t* rng, double* out, size_t n);
c4a_status_t c4a_rng_pcg64_uint64(c4a_rng_pcg64_state_t* rng, uint64_t* out);
c4a_status_t c4a_rng_pcg64_uint64_fill(c4a_rng_pcg64_state_t* rng, uint64_t* out, size_t n);
```

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | `c4a_rng_pcg64` | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
c4a_status_t c4a_rng_pcg64_advance(c4a_rng_pcg64_state_t* rng, uint64_t delta);
c4a_status_t c4a_rng_pcg64_create(uint64_t seed, c4a_rng_pcg64_state_t** out);
void c4a_rng_pcg64_destroy(c4a_rng_pcg64_state_t* rng);
c4a_status_t c4a_rng_pcg64_set_seed(c4a_rng_pcg64_state_t* rng, uint64_t seed);
c4a_status_t c4a_rng_pcg64_standard_normal_fill( c4a_rng_pcg64_state_t* rng, double* out, size_t n);
c4a_status_t c4a_rng_pcg64_uint64(c4a_rng_pcg64_state_t* rng, uint64_t* out);
c4a_status_t c4a_rng_pcg64_uint64_fill(c4a_rng_pcg64_state_t* rng, uint64_t* out, size_t n);
```

:::

::::


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
