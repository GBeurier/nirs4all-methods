# Haar

## Algorithm

Convenience wrapper around the `Wavelet` operator with the
`family = haar` and `mode = periodization` defaults.  Matches
`nirs4all.operators.transforms.nirs.Haar` (which itself is a
zero-argument subclass of `Wavelet`).

The Haar filter bank is the simplest orthogonal wavelet ($L = 2$):

$$
h_{\text{lo}} = (1/\sqrt 2,\; 1/\sqrt 2), \qquad
h_{\text{hi}} = (-1/\sqrt 2,\; 1/\sqrt 2).
$$

For an input row $x \in \mathbb{R}^n$ (with $n$ even, or
internally padded to even), the periodization Haar DWT reduces to:

$$
c_A[k] = \frac{x[2k] + x[2k+1]}{\sqrt 2}, \qquad
c_D[k] = \frac{x[2k+1] - x[2k]}{\sqrt 2}.
$$

Output is the concatenation $[c_A \,\Vert\, c_D]$ of length $2m$ with
$m = \lceil n / 2 \rceil$.

## Parameters

None.

## Lifecycle

Stateless: `_create / _transform / _destroy`.  `_output_cols(input_cols)`
returns the row width $2m$.

## Numerical contract

Same as Wavelet: parity tolerance `1e-10` absolute / `1e-11` relative
against the internal parity fixture.

## Reference

`nirs4all.operators.transforms.nirs.Haar`.
