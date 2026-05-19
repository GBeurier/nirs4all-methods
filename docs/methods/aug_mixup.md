# `aug_mixup` — MixupAugmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_mixup.md`](../algorithms/aug_mixup.md)

## Description

`aug_mixup` is a chemometrics4all augmentation method exposed through the C ABI and the generated language bindings.

### Parameters

| Parameter | Constraint | Meaning |
|-----------|------------|---------|
| `alpha`   | $> 0$      | Shape of the symmetric Beta distribution used for $\lambda$. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.spectral.MixupAugmenter`. The reference
implementation uses `numpy.random.Generator.permutation` followed by
`Generator.beta(alpha, alpha, size=(n,1))`.

For a recent overview of mixup, see Zhang et al. 2018, "mixup: Beyond
Empirical Risk Minimization".

### Mathematical principle

`c4a_aug_mixup` augments a batch of spectra by replacing every row with a
convex combination of two original rows drawn from the same batch.

Given an input matrix $X \in \mathbb{R}^{n \times p}$, the algorithm:

1. Draws a Fisher-Yates permutation $\pi \in S_n$ of the sample indices.
2. Draws $\lambda_i \sim \mathrm{Beta}(\alpha, \alpha)$ for $i = 0, \dots, n-1$.
3. Writes
$$X^{\mathrm{aug}}_{i,:} = \lambda_i \cdot X_{i,:} + (1-\lambda_i) \cdot X_{\pi(i),:}$$

The output keeps the same shape $(n, p)$ as the input. **Each output row is
a mix of two rows from the SAME batch** — this is the canonical "within-batch
mixup" used in vision and audio augmentation pipelines, applied here to NIR
spectra.

### Implementation

A fixed PCG64 state at call time yields a bit-exact output. Seeding the
RNG with `c4a_rng_pcg64_set_seed(rng, S)` before each `_apply` produces a
reproducible stream regardless of prior consumption.

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
