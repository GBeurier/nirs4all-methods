Vendored FITPACK subset
=======================

This directory vendors the small FITPACK subset needed to reproduce SciPy's
`UnivariateSpline(x, y, s=1 / n_features)` contract for
`aug_spline_smooth`.

Source: SciPy 1.17.1, `scipy/interpolate/fitpack/`.
Original algorithm: P. Dierckx FITPACK curve fitting routines.

The files are kept as close as possible to the upstream SciPy copies. Local C
code calls `curfit` and `splev` with the same two-stage `nest` contract used by
SciPy's `fpcurf0`/`fpcurf1` wrappers.
