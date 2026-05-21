# n4m — full nirs4all-methods R binding (M11 scaffold)

**Status**: scaffold only. The vendored libn4m build (sources under
src/libn4m/), the roxygen docs for the ~200 exported functions, and the
parsnip integration land in the M11 focused session.

Generated from catalog/subsets/nirs4all_methods.yaml.

When activated:
```r
install.packages("n4m")
library(n4m)
n4m_preprocessing_snv_transform(X)
n4m_models_pls_simpls_fit(X, y, ncomp = 5)
# ... full catalog
```

CRAN-ready (per M11 gate: R CMD check --as-cran clean on R oldrel /
release / devel on Linux + Windows + macOS-ARM).
