# Wavelet

## Algorithm

Single-level discrete wavelet transform (DWT) applied row-by-row.  For
each row $x \in \mathbb{R}^n$ the operator computes:

$$
c_A[k] = \sum_{i=0}^{L-1} h_{\text{lo}}[L - 1 - i] \cdot \tilde{x}[2k + i - (L/2 - 1)],
\qquad
c_D[k] = \sum_{i=0}^{L-1} h_{\text{hi}}[L - 1 - i] \cdot \tilde{x}[2k + i - (L/2 - 1)],
$$

where $h_{\text{lo}}, h_{\text{hi}}$ are the analysis low-pass / high-pass
filters of length $L$ for the chosen wavelet family and $\tilde{x}$ is
the input signal under the chosen boundary extension.

The output is the row-major concatenation $[c_A \,\Vert\, c_D]$ of
length $2m$ where $m$ is the per-mode output length:

- **periodization**: $m = \lceil n / 2 \rceil$ (odd $n$ is internally
  padded with one copy of the last sample).
- **symmetric** / **zero**: $m = \lfloor (n + L - 1) / 2 \rfloor$.

Supported wavelet families: **haar** ($L=2$), **db4** ($L=8$), **sym4**
($L=8$), **coif1** ($L=6$). Filter banks are vendored from the frozen
PyWavelets 1.6.0 oracle; current PyWavelets package drift is checked
separately by the reference parity gate.

## Parameters

| Parameter | Type                       | Meaning                                  |
| --------- | -------------------------- | ---------------------------------------- |
| `family`  | enum `c4a_pp_wavelet_family_t`   | Wavelet family identifier          |
| `mode`    | enum `c4a_pp_wavelet_boundary_t` | Boundary extension mode            |

`_output_cols(input_cols)` returns the row width $2m$ used by the
caller to size the output matrix.

## Lifecycle

Stateless: `_create / _transform / _destroy` plus the
`_output_cols` helper.  No `_fit` step.

## Numerical contract

- `_transform` requires `out.cols == 2 * m` and `out.rows == X.rows`,
  otherwise returns `C4A_ERR_SHAPE_MISMATCH`.
- Parity tolerance against the internal parity fixture: `1e-10`
  absolute / `1e-11` relative.

## Reference

`nirs4all.operators.transforms.nirs.Wavelet` (single-level DWT;
chemometrics4all returns both `cA` and `cD` instead of resampling the
detail band). Filter banks: frozen PyWavelets 1.6.0 oracle, with current
upstream comparison handled by reference parity.
