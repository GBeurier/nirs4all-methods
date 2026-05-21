# ScatterSimulationMSC

## Algorithm

Simulates Multiplicative Scatter Correction (MSC) reversibility by drawing
a random additive offset and multiplicative gain per sample, then
applying

$$X^{\mathrm{aug}}_{i,:} = a_i + b_i \cdot X_{i,:}$$

with $a_i \sim \mathrm{Uniform}(a_{\mathrm{low}}, a_{\mathrm{high}})$ and
$b_i \sim \mathrm{Uniform}(b_{\mathrm{low}}, b_{\mathrm{high}})$.

The reference draws $n$ uniform `a` values followed by $n$ uniform `b`
values — the C engine mirrors this exact order so the PCG64 stream
consumption matches NumPy's vectorised `rng.uniform(low, high, size=n)`
calls.

## Parameters

| Parameter | Constraint | Meaning |
|-----------|------------|---------|
| `a_low`, `a_high` | `a_high >= a_low` | Additive-offset uniform range. |
| `b_low`, `b_high` | `b_high >= b_low` | Multiplicative-gain uniform range. |

## Determinism

When `a_low == a_high` and `b_low == b_high`, the augmenter is a
deterministic affine transform regardless of the RNG state. The parity
fixture exercises this case (`scatter_sim_constant` in
`aug_phase17_v1.json`).

## Reference

`nirs4all.operators.augmentation.spectral.ScatterSimulationMSC`.
