# `aug_emsc_distort` — EMSCDistortionAugmenter

_Group_: **Augmentation** · _Registry tolerance_: `rtol=1e-5`, `atol=1e-8` · _Source_: [`docs/algorithms/aug_emsc_distort.md`](../algorithms/aug_emsc_distort.md)

## Description

`aug_emsc_distort` is a chemometrics4all augmentation method exposed through the C ABI and the generated language bindings.

### Parameters

| Parameter | Constraint | Meaning |
|-----------|------------|---------|
| `mult_low`, `mult_high` | `high >= low` | Range for multiplicative gain $b$. |
| `add_low`, `add_high`   | `high >= low` | Range for additive offset $a$. |
| `polynomial_order`      | $\geq 0$      | Polynomial order $d$. |
| `polynomial_strength`   |               | Base scale of $c_k$ draws. |
| `correlation`           | $[-1, 1]$     | $a$-$b$ correlation $\rho$. |

## Explanations

### Bibliographic source

`nirs4all.operators.augmentation.scattering.EMSCDistortionAugmenter`,
Martens et al. (2003).

### Mathematical principle

Simulates EMSC-style spectral distortion that EMSC is later designed to
correct:

$$X^{\mathrm{aug}}_{i,j} = a_i + b_i \cdot X_{i,j} + \sum_{k=1}^{d} c_{i,k} \cdot \tilde\lambda_j^k$$

where $\tilde\lambda_j = 2(\lambda_j - \lambda_{\min}) / (\lambda_{\max} - \lambda_{\min}) - 1$
is the normalised wavelength in $[-1, 1]$.

Per-sample parameter draws:

- $b_i \sim \mathrm{Normal}(\bar b, \sigma_b)$ where $\bar b = (b_{\mathrm{lo}} + b_{\mathrm{hi}})/2$ and $\sigma_b = (b_{\mathrm{hi}} - b_{\mathrm{lo}})/4$, then clipped to $[b_{\mathrm{lo}}, b_{\mathrm{hi}}]$.
- $\Delta_b = (b_i - 1) / \sigma_b$.
- $a_i \sim \mathrm{Normal}(\bar a - \rho \sigma_a \Delta_b, \sigma_a \sqrt{1 - \rho^2})$, then clipped to $[a_{\mathrm{lo}}, a_{\mathrm{hi}}]$.
- $c_{i,k} \sim \mathrm{Normal}(0, \mathrm{strength} / \sqrt{k})$ for $k = 1, \dots, d$.

The correlation parameter $\rho$ couples the additive and multiplicative
terms (positive $\rho$ → when $b$ is high, $a$ tends to be low).

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
