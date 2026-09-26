Vendored FITPACK subset (C translation)
=======================================

`fitpack.c` is a C translation of the FITPACK routines behind SciPy's
`UnivariateSpline(x, y, k=3, s=1 / n_features)` contract for
`aug_spline_smooth`: `curfit`, `fpcurf`, `fpback`, `fpbspl`, `fpdisc`,
`fpgivs`, `fpknot`, `fprati`, `fprota` and `splev`.

Source: SciPy 1.17.1, `scipy/interpolate/fitpack/` (Fortran).
Original algorithm: P. Dierckx FITPACK curve fitting routines.
License: BSD-3-Clause, see `LICENSE.scipy.txt`.

The translation keeps the Fortran arithmetic order and reproduces the Fortran
build (and SciPy) bit for bit. Only the paths used by the caller are
translated: `curfit` with `iopt` 0/1 and `s > 0`, and `splev` with `e = 0`.
`spline_smoothing.c` uses the same two-stage `nest` contract as SciPy's
`fpcurf0`/`fpcurf1` wrappers.
