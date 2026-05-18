# BatchEffectAugmenter

## Algorithm

Simulates batch/session measurement drift via a wavelength-normalised
additive offset, a slope, and a multiplicative gain.

Let $\tilde\lambda_j$ be the centred and range-normalised wavelength axis:
$$\tilde\lambda_j = (\lambda_j - \bar\lambda) / (\lambda_{\max} - \lambda_{\min} + 10^{-10}).$$
If `wavelengths == NULL`, the integer index $0, 1, \dots, p-1$ is used in
its place.

Two modes:

**Per-sample** (`variation_scope = 0`):
$$o_i \sim \mathrm{Normal}(0, \sigma_o),\ m_i \sim \mathrm{Normal}(0, \sigma_m),\ g_i \sim \mathrm{Normal}(1, \sigma_g)$$
$$X^{\mathrm{aug}}_{i,j} = X_{i,j} \cdot g_i + o_i + m_i \tilde\lambda_j$$

**Batch** (`variation_scope = 1`):
$$o \sim \mathrm{Normal}(0, \sigma_o),\ m \sim \mathrm{Normal}(0, \sigma_m),\ g \sim \mathrm{Normal}(1, \sigma_g)$$
$$X^{\mathrm{aug}}_{i,j} = X_{i,j} \cdot g + o + m \tilde\lambda_j$$

In per-sample mode the reference draws the parameters as three separate
length-$n$ normal vectors (offsets, then slopes, then gains). The C engine
mirrors this draw order so the PCG64 state evolution matches NumPy.

## Determinism

When $\sigma_o = \sigma_m = \sigma_g = 0$ the augmenter is the identity
transform (gain ~ N(1, 0) = 1, offset/slope ~ N(0, 0) = 0). The parity
fixture exercises this case (`batch_effect_zero_batch`).

## Reference

`nirs4all.operators.augmentation.synthesis.BatchEffectAugmenter`.
