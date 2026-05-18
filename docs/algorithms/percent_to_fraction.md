# Percent To Fraction

## Algorithm

Element-wise division by 100:

$$
x'_{i,j} = \frac{x_{i,j}}{100}
$$

## Parameters

None — the transform is fully data-driven.

## Numerical contract

* Division is performed inline as `X[i] / 100.0` — not replaced with multiplication by `0.01`, because `0.01` has no exact binary representation. The two forms diverge in the last bit; the inline divide matches numpy's element-wise true-divide bit-for-bit.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

## Reference

Implementation matches `nirs4all.operators.transforms.signal_conversion.PercentToFraction` (0.8.11).
