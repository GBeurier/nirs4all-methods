# SPDX-License-Identifier: CECILL-2.1

skip_if_no_libp4a <- function() {
    so <- tryCatch(getLoadedDLLs()[["pls4all"]], error = function(e) NULL)
    if (is.null(so)) testthat::skip("pls4all DLL not loaded")
}

pls_compat_data <- function(n = 50, p = 8, seed = 10) {
    set.seed(seed)
    X <- matrix(stats::rnorm(n * p), n, p)
    y <- 1.5 * X[, 1] - 0.8 * X[, 3] + 0.05 * stats::rnorm(n)
    df <- data.frame(X, y = y)
    names(df)[seq_len(p)] <- paste0("x", seq_len(p))
    list(X = X, y = y, df = df)
}

test_that("plsr() exposes pls-like formula fit, predict and RMSEP", {
    skip_if_no_libp4a()
    xy <- pls_compat_data()
    fit <- plsr(y ~ ., data = xy$df, ncomp = 3L, method = "simpls")
    expect_s3_class(fit, "pls4all_mvr")
    preds <- predict(fit, newdata = xy$df, ncomp = 3L)
    expect_length(preds, nrow(xy$df))
    expect_lt(sqrt(mean((preds - xy$y) ^ 2)), 0.2)

    multi <- predict(fit, newdata = xy$df, ncomp = 1:3)
    expect_equal(dim(multi), c(nrow(xy$df), 1L, 3L))

    rmsep <- RMSEP(fit)
    expect_s3_class(rmsep, "pls4all_mvrVal")
    expect_equal(dim(rmsep$val), c(1L, 1L, 3L))
    expect_true(selectNcomp(fit) >= 1L)
})

test_that("pcr() runs through the PCR-compatible path", {
    skip_if_no_libp4a()
    xy <- pls_compat_data(seed = 11)
    fit <- pcr(y ~ ., data = xy$df, ncomp = 3L)
    preds <- predict(fit, newdata = xy$df)
    expect_length(preds, nrow(xy$df))
    expect_s3_class(summary(fit), "summary.pls4all_mvr")
})

test_that("pls-compatible prediction works when only final component is stored", {
    skip_if_no_libp4a()
    xy <- pls_compat_data(seed = 13)
    fit <- plsr(y ~ ., data = xy$df, ncomp = 2L, method = "simpls",
                fit_components = FALSE)
    preds <- predict(fit, newdata = xy$df, ncomp = 2L)
    expect_length(preds, nrow(xy$df))
    expect_error(predict(fit, newdata = xy$df, ncomp = 1L),
                 "requested ncomp was not fitted")
    rmsep <- RMSEP(fit)
    expect_equal(fit$ncomp, rmsep$comps)
})

test_that("plsr() can compute simple CV metrics", {
    skip_if_no_libp4a()
    xy <- pls_compat_data(n = 30, p = 6, seed = 12)
    fit <- plsr(y ~ ., data = xy$df, ncomp = 2L, method = "simpls",
                validation = "CV", segments = 3L)
    rmsep <- RMSEP(fit, estimate = "CV")
    expect_equal(dim(rmsep$val), c(1L, 1L, 2L))
})
