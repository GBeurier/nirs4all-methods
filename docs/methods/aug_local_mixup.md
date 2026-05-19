# `aug_local_mixup` — LocalMixupAugmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_local_mixup.md`](../algorithms/aug_local_mixup.md)

## Description

`aug_local_mixup` is a chemometrics4all augmentation method exposed through the C ABI and the generated language bindings.

### Parameters

| Parameter      | Constraint | Meaning |
|----------------|------------|---------|
| `alpha`        | $> 0$      | Shape of the symmetric Beta. |
| `k_neighbors`  | $\geq 1$ and $\leq n-1$ | Number of non-self neighbors to choose among. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.spectral.LocalMixupAugmenter`. The reference
implementation uses `sklearn.neighbors.NearestNeighbors` with default
parameters (brute force, Euclidean distance), then
`rng.choice(indices[i, 1:])` and `rng.beta(alpha, alpha)`.

### Mathematical principle

`c4a_aug_local_mixup` is a neighborhood-aware variant of mixup. For each
sample, it draws a random neighbor among the $k$ nearest non-self rows and
mixes the two with a $\mathrm{Beta}(\alpha, \alpha)$ weight.

For each $i \in \{0, \dots, n-1\}$:

1. Compute the squared-Euclidean distance from $X_{i,:}$ to every row of
   $X$.
2. Partial-sort to obtain the $k+1$ smallest indices (the first one is
   $i$ itself).
3. Draw `pick ∈ [0, k)` uniformly via PCG64 integers and select
   `neighbor = indices[1 + pick]`.
4. Draw $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$.
5. Write $X^{\mathrm{aug}}_{i,:} = \lambda \cdot X_{i,:} + (1-\lambda) \cdot X_{\mathrm{neighbor},:}$.

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


**Registry parity references** ◆

:::{card}
:class-card: external-refs

- ℹ No external parity reference row is registered for this public helper; the page is generated from the in-tree API and algorithm documentation.
:::

### Benchmarks

No cross-binding timing row is currently registered for this method. The implementation table above is still generated from the public API surface.

---

_See also_: [benchmark overview](../benchmarks/overview.md) · [methods index](index.md) · [interactive dashboard](../landing/dashboard.md)
