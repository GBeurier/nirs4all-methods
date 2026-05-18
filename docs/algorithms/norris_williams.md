# NorrisWilliams

## Algorithm

Two-pass derivative: a centred moving-average smoother followed by a gap
finite difference, applied `derivative_order` times.  Mirrors
`nirs4all.operators.transforms.norris_williams.norris_williams`.

Single pass on a row $x$:

$$
\tilde{x}_j = \frac{1}{\text{segment}} \sum_{k = -\lfloor \text{segment}/2 \rfloor}^{\lfloor \text{segment}/2 \rfloor} \overline{x}_{j+k},
\qquad
\hat{x}_j = \frac{\tilde{x}_{j + \text{gap}} - \tilde{x}_{j - \text{gap}}}{2 \cdot \text{gap} \cdot \delta}.
$$

where $\overline{x}$ extends $x$ to out-of-range indices by replicating
the boundary samples (edge padding).  When `derivative_order` is 2 the
(smooth, gap-diff) pair is applied a second time on the already-derived
row.

`segment = 1` disables the smoothing pass (the moving average over one
sample is the identity).  `gap` controls the spacing of the finite
difference and is independent of `segment`.

## Parameters

| Parameter | Constraint | Meaning |
|-----------|------------|---------|
| `segment` | odd, $\geq 1$ | Smoothing window. |
| `gap`     | $\geq 1$      | Finite-difference spacing. |
| `derivative_order` | $\in \{1, 2\}$ | Number of (smooth, diff) passes. |
| `delta`   | $\neq 0$      | Sample spacing for the derivative scaling. |

## Numerical contract

- `_create` returns `C4A_ERR_INVALID_ARGUMENT` for even or non-positive
  segments, $\text{gap} < 1$, $\text{derivative\_order} \notin \{1, 2\}$,
  or $\delta = 0$.
- Tolerance vs nirs4all (`np.pad` + arithmetic): $10^{-12}$ absolute /
  $10^{-13}$ relative.

## Reference

Norris, K. H. & Williams, P. C. (1984).  "Optimization of mathematical
treatments of raw near-infrared signal in the measurement of protein
in hard red spring wheat."  *Cereal Chemistry* 61, 158-165.
