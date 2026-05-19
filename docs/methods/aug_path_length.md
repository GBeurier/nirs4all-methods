# `aug_path_length` — PathLengthAugmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_path_length.md`](../algorithms/aug_path_length.md)

## Description

Applies a per-sample multiplicative path-length factor (with a lower clip to prevent sign inversion).

Algorithm:

1. Draw `n_samples` standard normals in one bulk call.
2. `L[i] = 1.0 + path_length_std * normal[i]`.
3. `L[i] = max(L[i], min_path_length)`.
4. `out[i, j] = L[i] * X[i, j]`.

Mirrors `nirs4all.operators.augmentation.synthesis.PathLengthAugmenter` with `variation_scope="sample"` (default).

### Parameters

- `path_length_std: double` (`>= 0.0`) — standard deviation of the per-sample factor (centered at 1.0).
- `min_path_length: double` — lower clip applied to `L` after the normal draw.

## Explanations

### Bibliographic source

- Frozen Python: `parity/python_generator/src/c4a_parity_augmenters_ref/path_length.py`.
- Parity tolerance: 1e-15 abs.

### Mathematical principle

`aug_path_length` follows the standard augmentation operator contract: an input matrix $\mathbf{X}\in\mathbb{R}^{n\times p}$ is transformed, scored, split, or filtered by the method-specific kernel while preserving the parity contract recorded by the benchmark snapshots. The precise numerical convention is the one exposed by the C ABI signatures and the registered external references below.

### Implementation

Implementation details are defined by the public C ABI and the generated binding wrappers.

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


**Registry parity references** 📐

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
