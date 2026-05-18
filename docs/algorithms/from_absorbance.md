# From Absorbance

## Algorithm

Element-wise inverse log-10: convert absorbance back to reflectance or transmittance.

$$
y_{i,j} = 10^{-x_{i,j}}, \qquad
\text{out}_{i,j} = \begin{cases} 100 \cdot y_{i,j} & \text{if `is\_percent`} \\ y_{i,j} & \text{otherwise} \end{cases}
$$

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `is_percent` | `false` | When set, multiply the post-power result by 100 (target signal type is `reflectance%` or `transmittance%`). |

## Numerical contract

* The power is computed as `pow(10.0, -x)` — element-wise `np.power(10.0, -X)` semantics. The alternative `exp(-x · ln10)` introduces one extra multiply rounding step that would diverge in the last bit.
* The percent multiply uses `* 100.0` as a separate inline operation (not folded into the exponent), mirroring `result = result * 100.0` in nirs4all.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

## Reference

Implementation matches `nirs4all.operators.transforms.signal_conversion.FromAbsorbance` (0.8.11).
