# EMSCDistortionAugmenter

## Algorithm

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

## Parameters

| Parameter | Constraint | Meaning |
|-----------|------------|---------|
| `mult_low`, `mult_high` | `high >= low` | Range for multiplicative gain $b$. |
| `add_low`, `add_high`   | `high >= low` | Range for additive offset $a$. |
| `polynomial_order`      | $\geq 0$      | Polynomial order $d$. |
| `polynomial_strength`   |               | Base scale of $c_k$ draws. |
| `correlation`           | $[-1, 1]$     | $a$-$b$ correlation $\rho$. |

## Reference

`nirs4all.operators.augmentation.scattering.EMSCDistortionAugmenter`,
Martens et al. (2003).
