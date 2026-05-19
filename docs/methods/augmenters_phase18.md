# `augmenters_phase18` — Phase 18 augmenters

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/augmenters_phase18.md`](../algorithms/augmenters_phase18.md)

## Description

Twelve data-augmentation operators that round out the unified `c4a_aug_*`
ABI category (Phase 15-18 contract).

All twelve operators share the universal ABI shape:

```c
typedef struct c4a_aug_<NAME>_handle_t c4a_aug_<NAME>_handle_t;

c4a_status_t c4a_aug_<NAME>_create(
    c4a_aug_<NAME>_handle_t** out,
    c4a_rng_pcg64_state_t* rng,   /* MUST be non-NULL */
    /* operator-specific parameters */);

c4a_status_t c4a_aug_<NAME>_apply(
    const c4a_aug_<NAME>_handle_t* handle,
    c4a_matrix_view_t X,
    [c4a_matrix_view_t wavelengths,]   /* only for wavelength-aware ops */
    c4a_matrix_view_t out);

void c4a_aug_<NAME>_destroy(c4a_aug_<NAME>_handle_t* handle);
```

The RNG handle is stored by reference; the augmenter does NOT own it. The
caller MUST keep the RNG alive for the augmenter's lifetime. With a fixed
RNG state at `_apply` time the output is bit-exact reproducible.

### Parameters

No public constructor parameters are required for the documented default call.

## Explanations

### Bibliographic source

- nirs4all 0.8.x —
  `nirs4all.operators.augmentation.edge_artifacts`,
  `nirs4all.operators.augmentation.splines`,
  `nirs4all.operators.augmentation.random`.
- `roadmap/phase-15-18-augmenters-abi-contract.md`
- `DEFERRALS.md`

### Mathematical principle

`augmenters_phase18` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

- All 10 fully-implemented operators are deterministic: re-seeding the
  RNG handle before two consecutive `_apply` calls yields bit-equal
  outputs. The smoke fixtures under `parity/fixtures/aug_*_v1.json`
  exercise the shape + finite-value path; bit-exact NumPy parity of
  `rng.uniform`, `rng.choice` is gated behind v2 of the augmenter ABI.
- The 2 v2-deferred operators (`Spline_X_Simplification`,
  `Spline_Curve_Simplification`) return `C4A_ERR_NOT_IMPLEMENTED` from
  `_apply` until v2; their `_create` / `_destroy` symbols are stable.

### Implementations

| Layer | Entry point | Language | Contract |
|-------|-------------|----------|----------|
| C ABI | — | C/C++ | Stable libc4a entry point family. |

### Usage

Every chemometrics4all binding dispatches into the same C kernel. The registry references are listed in the parity card below.

::::{tab-set}
:class: chemometrics4all-bindings


:::{tab-item} C ABI · libc4a
:sync: c
:class-label: lang-c

```c
/* C ABI prefix */
c4a_*
```

:::

::::


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
