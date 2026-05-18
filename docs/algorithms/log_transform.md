# Log Transform

## Algorithm

Element-wise logarithm with optional fit-time safety offset:

1. Compute the **fitted offset**:

   $$
   \tilde{x}^{(\text{temp})}_{i,j} = x_{i,j} + \text{offset}, \qquad
   m^{(\text{temp})} = \min_{i, j} \tilde{x}^{(\text{temp})}_{i,j}
   $$

   $$
   \delta_{\text{auto}} =
   \begin{cases}
     \text{min\_value} - m^{(\text{temp})} & \text{if auto\_offset } \wedge \, m^{(\text{temp})} \le 0 \\
     0 & \text{otherwise}
   \end{cases}, \qquad
   \delta = \text{offset} + \delta_{\text{auto}}
   $$

2. Apply the fitted offset to the **raw** $x$ (not $\tilde{x}^{(\text{temp})}$):

   $$
   y_{i,j} = x_{i,j} + \delta, \qquad
   y_{i,j} \leftarrow \begin{cases}
     \text{min\_value} & \text{if } y_{i,j} \le 0 \\
     y_{i,j} & \text{otherwise}
   \end{cases}
   $$

3. Take the logarithm:

   $$
   x'_{i,j} =
   \begin{cases}
     \log y_{i,j} & \text{if natural log} \\
     \dfrac{\log y_{i,j}}{\log B} & \text{if base } B \neq e
   \end{cases}
   $$

Note: the fitted offset is computed against $x + \text{offset}$ but applied to the raw $x$. The two differ in floating-point rounding only when `offset != 0`, but the difference must be reproduced exactly to match nirs4all.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `base` | `0.0` | Logarithm base. `0.0` selects natural log; values $> 0, \neq 1$ select base-$B$. |
| `offset` | `0.0` | Manual additive offset. |
| `auto_offset` | `true` | Enable the fit-time safety offset that keeps the post-offset minimum at least `min_value`. |
| `min_value` | $10^{-8}$ | Minimum positive clamp target. |

## Numerical contract

* Natural-log fast path skips the `log(base)` division.
* For arbitrary bases, the change-of-base formula is `log(y) / log(base)` — a true element-wise division, matching `np.log(X) / np.log(base)`.
* `min(X + offset)` is computed by accumulating the per-element offset addition before the min, preserving nirs4all's rounding tree.

## Reference

Implementation matches `nirs4all.operators.transforms.nirs.LogTransform`.
