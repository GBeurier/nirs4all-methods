# R Binding

The R package lives in `bindings/r/chemometrics4all` and vendors the `libc4a`
source for CRAN-style source builds.

It exposes two layers:

- Tier 1 `.Call` wrappers named after the C ABI symbols, for direct parity and
  benchmark work.
- Tier 2 idiomatic functions such as `snv()`, `wavelet()`, `epo()`,
  `flexible_pca()`, `resampler()`, `kbins_stratified()`, and
  `x_outlier_filter()`.

The dashboard timing matrix currently covers all 48 registered methods through
the installed public R binding. Inverse transforms are exposed where the C ABI
supports them, including `msc_inverse_transform()`,
`baseline_center_inverse_transform()`, and `epo_inverse_transform()`.

Local release smoke:

```bash
R CMD build bindings/r/chemometrics4all
R CMD check --no-manual --no-build-vignettes chemometrics4all_0.1.0.tar.gz
```
