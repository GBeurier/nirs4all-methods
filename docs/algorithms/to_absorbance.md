# To Absorbance

## Algorithm

Element-wise log-10 conversion from reflectance (or transmittance) to absorbance:

$$
A_{i,j} = -\log_{10}\!\bigl(\max(\tilde{x}_{i,j},\, \varepsilon)\bigr), \qquad
\tilde{x}_{i,j} = \begin{cases} x_{i,j} / 100 & \text{if `is\_percent`} \\ x_{i,j} & \text{otherwise} \end{cases}
$$

The clamp `max(·, epsilon)` keeps the log domain strictly positive — values at or below `epsilon` are clipped before the log.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `is_percent` | `false` | When set, divide input by 100 before the log. |
| `epsilon` | `1e-10` | Positive clamp target. Must be `> 0`. |
| `clip_negative` | `true` | Preserved for API symmetry with `np.clip(X, eps, None)` / `np.maximum(X, eps)` — both forms produce identical output for a one-sided lower clamp. |

## Numerical contract

* `is_percent` divides by `100.0` element-wise (not multiplication by `0.01`, which has no exact binary representation).
* The clamp is one-sided: `if x < epsilon, x = epsilon`. The two nirs4all paths (`np.clip(X, epsilon, None)` and `np.maximum(X, epsilon)`) produce identical output, so the C engine takes a single branch regardless of `clip_negative`.
* `-log10(x)` is the literal IEEE 754 `-log10(·)` call to match numpy's vectorised `-np.log10(X)` rounding.
* Parity tolerance vs nirs4all reference: `1e-12 abs / 1e-13 rel`.

## Reference

Implementation matches `nirs4all.operators.transforms.signal_conversion.ToAbsorbance` (0.8.11).
