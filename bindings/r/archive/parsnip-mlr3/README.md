# Archived pls4all R parsnip/mlr3 adapters

These files (`parsnip.R`, `r6_learner.R`) are archived on purpose.
The first R binding target is the NIRS / chemometrics community, where
`pls`, `mdatools`, matrix workflows, and formula/S3 workflows are more
relevant than parsnip/mlr3.

They are kept here for later revival as real optional adapters. They are
not sourced, installed, exported, or benchmarked by the core `pls4all`
R package.

For local experimentation only:

```r
source("bindings/r/archive/parsnip-mlr3/parsnip.R")
source("bindings/r/archive/parsnip-mlr3/r6_learner.R")
```
