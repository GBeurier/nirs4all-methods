# R binding

Phase 7c scaffolding. Minimal `.Call` gateway over `libp4a` with
fit/predict wrappers for every shipped PLS regression solver. Builds and
installs from `bindings/r/pls4all/`.

## Build / install

The package needs a copy of `libp4a` already built. Build the C ABI
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
    pls4all
```

At load time R needs to find `libp4a`. Either install the shared library
on the system path or export `LD_LIBRARY_PATH` (Linux) /
`DYLD_LIBRARY_PATH` (macOS) / `PATH` (Windows).

## Smoke

```R
library(pls4all)

pls4all_version()
# "0.91.0+abi.1.13.0"

pls4all_abi_version()
# c(1, 13, 0)

set.seed(42)
X <- matrix(rnorm(2000), nrow = 200)
y <- X %*% rnorm(10) + 0.1 * rnorm(200)

model <- pls4all_fit(X, y, algo = "pls_simpls", n_components = 5)
preds <- pls4all_predict(model, X)
sqrt(mean((preds - y) ^ 2))
```

## Available solvers

| `algo`                    | Algorithm/solver in libp4a                   |
|---------------------------|----------------------------------------------|
| `pls_nipals`              | `P4A_ALGO_PLS_REGRESSION + P4A_SOLVER_NIPALS`            |
| `pls_orthogonal_scores`   | `P4A_ALGO_PLS_REGRESSION + P4A_SOLVER_ORTHOGONAL_SCORES` |
| `pls_simpls`              | `P4A_ALGO_PLS_REGRESSION + P4A_SOLVER_SIMPLS`            |
| `pls_kernel_algorithm`    | `P4A_ALGO_PLS_REGRESSION + P4A_SOLVER_KERNEL_ALGORITHM`  |
| `pls_wide_kernel`         | `P4A_ALGO_PLS_REGRESSION + P4A_SOLVER_WIDE_KERNEL`       |
| `pls_svd`                 | `P4A_ALGO_PLS_REGRESSION + P4A_SOLVER_SVD`               |
| `pls_power`               | `P4A_ALGO_PLS_REGRESSION + P4A_SOLVER_POWER`             |
| `pls_randomized_svd`      | `P4A_ALGO_PLS_REGRESSION + P4A_SOLVER_RANDOMIZED_SVD`    |
| `pcr_svd`                 | `P4A_ALGO_PCR + P4A_SOLVER_SVD`                          |

## Scope

This phase ships the smallest viable surface to wire R into the
comprehensive benchmark matrix (see `benchmarks/`). AOM/POP wrappers,
sklearn-style classes, and parity tests against R `pls` / `ropls` /
`mixOmics` are deferred. See [`../../ROADMAP.md`](../../ROADMAP.md) for
the binding roadmap.
