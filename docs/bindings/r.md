# R binding

The current R package is
[`bindings/r/n4m`](https://github.com/GBeurier/nirs4all-methods/tree/main/bindings/r/n4m),
package name `n4m` (version 1.0.19 in this checkout). It builds and calls the
same ABI-2 `libn4m` engine as the other bindings. The older
`bindings/r/pls4all` tree is retained for migration history and is not the
package documented here.

Install from the package directory, then load the current namespace:

```r
install.packages("bindings/r/n4m", repos = NULL, type = "source")
library(n4m)
n4m_abi_version()
```

The package exports direct method-result fits such as `gpr_pls_fit`,
`sparse_simpls_fit`, `pls_diagnostics`, and `approximate_press`; selectors
such as `cars_select` and `bipls_select`; and formula-facing PLS helpers such
as `pls` and `pcr`. R source signatures are the authority for optional and
required arguments. Generated method pages show a current R symbol only when
it was found in this package's `NAMESPACE` and `R/` source.

```r
library(n4m)
X <- matrix(rnorm(200), nrow = 20)
split <- kennard_stone_split(X)
X_snv <- snv_transform(X)
```

Some catalog methods are exposed through R even when the full Python role
package has no wrapper. Conversely, a catalog entry without an R symbol has no
invented R call: use its linked C ABI surface or a binding that exports it.
