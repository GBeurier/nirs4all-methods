# pls4all R extensions (non-CRAN)

These files (`parsnip.R`, `r6_learner.R`) depend on R6, mlr3,
paradox, parsnip, rlang. They are not part of the CRAN-submitted
core package because pulling these heavy reverse-dependencies into
the core would block CRAN acceptance. They will ship as a separate
package `pls4all.extras` once the core is on CRAN.

To use them now, source the files directly:

```r
source("path/to/pls4all-extensions/parsnip.R")
source("path/to/pls4all-extensions/r6_learner.R")
```
