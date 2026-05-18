# Fraction To Percent

## Algorithm

Element-wise multiplication by 100:

$$
x'_{i,j} = 100 \cdot x_{i,j}
$$

## Parameters

None — the transform is fully data-driven.

## Numerical contract

* The multiply uses `* 100.0`. `100.0` is exactly representable in IEEE 754 binary64, so the multiply is bit-exact — equivalent to a divide by `0.01` only in exact arithmetic; in floating point the two forms differ in the last bit. We use the literal multiply to match nirs4all's `X * 100.0`.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

## Reference

Implementation matches `nirs4all.operators.transforms.signal_conversion.FractionToPercent` (0.8.11).
