# SPDX-License-Identifier: CECILL-2.1

skip_if_no_libp4a <- function() {
    so <- tryCatch(getLoadedDLLs()[["n4m"]], error = function(e) NULL)
    if (is.null(so)) testthat::skip("n4m DLL not loaded")
}

mdatools_compat_data <- function(n = 45, p = 9, seed = 20) {
    set.seed(seed)
    X <- matrix(stats::rnorm(n * p), n, p)
    y <- 2 * X[, 2] - X[, 5] + 0.05 * stats::rnorm(n)
    list(X = X, y = y)
}

test_that("pls_mdatools() fits matrix-oriented NIRS workflow", {
    skip_if_no_libp4a()
    xy <- mdatools_compat_data()
    fit <- pls_mdatools(xy$X, xy$y, ncomp = 3L, cv = 3L)
    expect_s3_class(fit, "n4m_mdatools_pls")
    expect_named(fit$calres, c("y.pred", "residuals", "rmse"))
    expect_named(fit$cvres, c("y.pred", "rmse"))
    preds <- predict(fit, x = xy$X)
    expect_length(preds, nrow(xy$X))
    expect_lt(sqrt(mean((preds - xy$y) ^ 2)), 0.2)
})

test_that("pls() dispatches matrix calls to the mdatools-compatible path", {
    skip_if_no_libp4a()
    xy <- mdatools_compat_data(seed = 21)
    fit <- pls(xy$X, xy$y, ncomp = 2L, method = "simpls")
    expect_s3_class(fit, "n4m_mdatools_pls")
    preds <- predict(fit, x = xy$X)
    expect_length(preds, nrow(xy$X))
})

test_that("mdatools-compatible prediction works when only final component is stored", {
    skip_if_no_libp4a()
    xy <- mdatools_compat_data(seed = 23)
    fit <- pls_mdatools(xy$X, xy$y, ncomp = 2L, fit_components = FALSE)
    preds <- predict(fit, x = xy$X, ncomp = 2L)
    expect_length(preds, nrow(xy$X))
    expect_error(predict(fit, x = xy$X, ncomp = 1L),
                 "requested ncomp was not fitted")
})

test_that("pls_mdatools() supports exclusions and external test results", {
    skip_if_no_libp4a()
    xy <- mdatools_compat_data(seed = 22)
    fit <- pls_mdatools(xy$X, xy$y, ncomp = 2L,
                        exclcols = 1L,
                        x.test = xy$X[1:5, , drop = FALSE],
                        y.test = xy$y[1:5])
    expect_s3_class(fit, "n4m_mdatools_pls")
    expect_named(fit$testres, c("y.pred", "residuals", "rmse"))
})
