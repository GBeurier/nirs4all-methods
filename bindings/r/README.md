# R binding (skeleton)

Phase 0 reserves this directory; the real binding lands in Phase 2.

The plan is to write a standard R package layout that calls into `libp4a`
via the native `.Call` interface (no Rcpp dependency, in line with the
project's zero-mandatory-deps policy). The skeleton mirrors what the
[`pls`](https://cran.r-project.org/package=pls) and
[`ropls`](https://www.bioconductor.org/packages/release/bioc/html/ropls.html)
packages expose, so users coming from those communities feel at home.

```
bindings/r/pls4all/
├── DESCRIPTION
├── NAMESPACE
├── R/
│   ├── pls4all-package.R
│   ├── version.R
│   ├── context.R
│   ├── config.R
│   └── model.R              # Phase 2
├── src/
│   ├── Makevars
│   ├── Makevars.win
│   ├── r_gateway.c          # .Call gateway
│   └── r_init.c             # R_registerRoutines
└── tests/testthat/
    └── test-version.R
```
