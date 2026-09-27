# SPDX-License-Identifier: CECILL-2.1
library(n4m)
set.seed(27)
X <- matrix(rnorm(40 * 19), 40, 19)
V <- matrix(rnorm(7 * 19), 7, 19)
encoder <- lvse_fit(X, width = 8, rank = 2)
Z <- spectral_transform(encoder, V)
affine <- spectral_export_affine(encoder)
stopifnot(max(abs(Z - sweep(V %*% t(affine$operator), 2, affine$offset, "+"))) < 1e-10)
mu <- colMeans(X)
sd_pop <- sqrt(colMeans(sweep(X, 2, mu)^2))
reference <- NULL
for (start in seq(1, ncol(X), by = 8)) {
    cols <- start:min(start + 7, ncol(X))
    rank <- min(2, length(cols) - 1, nrow(X) - 1)
    if (rank < 1) next
    train <- sweep(sweep(X[, cols, drop = FALSE], 2, mu[cols]), 2, sd_pop[cols], "/")
    query <- sweep(sweep(V[, cols, drop = FALSE], 2, mu[cols]), 2, sd_pop[cols], "/")
    basis <- svd(train, nu = 0, nv = rank)$v
    reference <- cbind(reference, query %*% basis)
}
stopifnot(max(abs(tcrossprod(Z) - tcrossprod(reference))) < 1e-8)
gcu <- gcu_fit(X, rank = 3)
factors <- spectral_transform(gcu, V)
stopifnot(identical(dim(factors), c(7L, 3L)), all(is.finite(factors)), all(factors >= 0))
stopifnot(inherits(try(spectral_export_affine(gcu), silent = TRUE), "try-error"))
snv <- lvse_fit(X, width = 8, rank = 2, snv = TRUE)
stopifnot(inherits(try(spectral_export_affine(snv), silent = TRUE), "try-error"))
cat("LVSE/GCU R binding tests passed\n")
