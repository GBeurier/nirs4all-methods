# R binding

Phase 7c scaffolding. Minimal `.Call` gateway over `libn4m` with
fit/predict wrappers for every shipped PLS regression solver. Builds and
installs from `bindings/r/n4m/`.

## Build / install

The package needs a copy of `libn4m` already built. Build the C ABI
first:

```bash
cmake --build --preset dev-release --parallel
```

Then install the R package, pointing the include and lib paths at your
checkout:

```bash
cd bindings/r
R CMD INSTALL \
    --configure-vars="PLS4ALL_INCLUDE_DIR=$PWD/../../cpp/include \
                      PLS4ALL_LIB_DIR=$PWD/../../build/dev-release/cpp/src" \
    n4m
```

At load time R needs to find `libn4m`. Either install the shared library
on the system path or export `LD_LIBRARY_PATH` (Linux) /
`DYLD_LIBRARY_PATH` (macOS) / `PATH` (Windows).

## Smoke

```R
library(n4m)

n4m_version()
# "1.0.2+abi.2.0.0"

n4m_abi_version()
# c(2, 0, 0)

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
split <- kennard_stone_split(X, test_size = 0.3)
```

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

This phase ships the smallest viable surface to wire R into the
comprehensive benchmark matrix (see `benchmarks/`). AOM/POP wrappers,
sklearn-style classes, and parity tests against R `pls` / `ropls` /
`mixOmics` are deferred. See [`../../ROADMAP.md`](../../ROADMAP.md) for
the binding roadmap.
