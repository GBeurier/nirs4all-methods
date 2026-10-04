# R binding

The `n4m` package is a thin `.Call` binding over the native engine. It exposes
PLS regression, preprocessing, estimator roles and pipelines, multimodal
regression/classification, AOM/POP and native optimization. The `pls4all`
package provides the slim PLS subset. Neither package implements numerical
kernels in R.

## Build / install

Source distributions vendor the native C/C++ sources and build a self-contained
R shared library. From a full source checkout, prepare this mode with:

```bash
N4M_R_VENDOR=1 R CMD INSTALL bindings/r/n4m
```

For a development installation using an external `libn4m`, build the C ABI
first:

```bash
cmake --build --preset dev-release --parallel
```

Then install the R package, pointing the include and lib paths at your
checkout:

```bash
cd bindings/r
R CMD INSTALL \
    --configure-vars="N4M_INCLUDE_DIR=$PWD/../../cpp/include \
                      N4M_LIB_DIR=$PWD/../../build/dev-release/cpp/src" \
    n4m
```

At load time R needs to find `libn4m`. Either install the shared library
on the system path or export `LD_LIBRARY_PATH` (Linux) /
`DYLD_LIBRARY_PATH` (macOS) / `PATH` (Windows).

The raw multimodal classifier requires ABI 2.17. Use
`n4m_multimodal_classifier(recipe, source_schemas)` with the native PLS-logistic
head, then `model <- n4m_fit(model, blocks, labels)`.
`predict(model, blocks, type = "prob")` returns columns in `n4m_classes(model)`
order. N4MC states contain the complete native encoder/head state and no
training rows; provide the original typed label table as `class_names` to
`n4m_multimodal_classifier_from_state` when restoring named classes.

## Smoke

```R
library(n4m)

n4m_version()
# "1.3.2+abi.2.17.0"

# Seeded native training-only augmentation of X (no paired Y output):
# n4m::n4m_augmentation_apply("gaussian_noise", train_X, 0.03, seed = 42)

n4m_abi_version()
# c(2, 17, 0)

set.seed(42)
X <- matrix(rnorm(2000), nrow = 200)
y <- X %*% rnorm(10) + 0.1 * rnorm(200)

model <- n4m_fit(X, y, algo = "pls_simpls", n_components = 5)
preds <- n4m_predict(model, X)
sqrt(mean((preds - y) ^ 2))
```

Preprocessing and splitters are exposed as thin wrappers over the same
libn4m C ABI:

```R
X_snv <- snv_transform(X)
X_sg <- savgol_transform(X, window_length = 11, polyorder = 3,
                         deriv = 0, mode = "interp")
X_local <- local_snv_transform(X, window = 11)
X_robust <- robust_snv_transform(X)
X_area <- area_normalization_transform(X, method = "trapz")
X_detrended <- detrend_transform(X, polyorder = 2)
msc_reference <- msc_fit(X)
X_msc <- msc_transform(X, msc_reference)
split <- kennard_stone_split(X, test_size = 0.3)
```

The four additional stateless operators plus MSC and EMSC use the same native
C ABI as Python `n4m` and have frozen cross-language matrix parity tests. MSC
and EMSC export their fitted reference vectors; only training spectra are used
to learn them. EMSC replay additionally requires the same polynomial degree.
Other stateful operators still need explicit state import/export surfaces.

## Available solvers

| `algo`                    | Algorithm/solver in libn4m                   |
|---------------------------|----------------------------------------------|
| `pls_nipals`              | `N4M_ALGO_PLS_REGRESSION + N4M_SOLVER_NIPALS`            |
| `pls_orthogonal_scores`   | `N4M_ALGO_PLS_REGRESSION + N4M_SOLVER_ORTHOGONAL_SCORES` |
| `pls_simpls`              | `N4M_ALGO_PLS_REGRESSION + N4M_SOLVER_SIMPLS`            |
| `pls_kernel_algorithm`    | `N4M_ALGO_PLS_REGRESSION + N4M_SOLVER_KERNEL_ALGORITHM`  |
| `pls_wide_kernel`         | `N4M_ALGO_PLS_REGRESSION + N4M_SOLVER_WIDE_KERNEL`       |
| `pls_svd`                 | `N4M_ALGO_PLS_REGRESSION + N4M_SOLVER_SVD`               |
| `pls_power`               | `N4M_ALGO_PLS_REGRESSION + N4M_SOLVER_POWER`             |
| `pls_randomized_svd`      | `N4M_ALGO_PLS_REGRESSION + N4M_SOLVER_RANDOMIZED_SVD`    |
| `pcr_svd`                 | `N4M_ALGO_PCR + N4M_SOLVER_SVD`                          |

## Scope

Fitted native handles are R external pointers with an explicit native lifetime;
serializing an external pointer with `saveRDS()` does not preserve its learned
state. Use `n4m_model_export()` / `n4m_model_import()` for portable N4MM model
bytes, or the corresponding estimator, role-pipeline and multimodal state APIs.
Import validates the native format and ABI before exposing the restored handle.
Cross-language parity fixtures and native lifecycle tests live under
`bindings/r/n4m/tests/`; numerical reference comparisons are profile-specific.
