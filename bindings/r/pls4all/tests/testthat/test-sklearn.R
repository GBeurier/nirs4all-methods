context("tier-2 formula interface")

test_that("pls() formula fit + predict round-trip", {
    set.seed(0)
    n <- 80; p <- 10
    X <- matrix(rnorm(n * p), n, p)
    beta <- c(2, -1, 0.5, rep(0, p - 3))
    df <- as.data.frame(X)
    df$y <- as.vector(X %*% beta + 0.1 * rnorm(n))

    fit <- pls(y ~ ., data = df, ncomp = 3L)
    expect_s3_class(fit, "pls4all_fit")
    expect_equal(fit$n_features_in, p)
    expect_equal(fit$ncomp, 3L)
    expect_equal(fit$response_name, "y")

    preds <- predict(fit, newdata = df)
    expect_true(is.numeric(preds))
    expect_length(preds, n)
})

test_that("pls() vs pls4all_fit produce identical predictions", {
    set.seed(1)
    n <- 60; p <- 8
    X <- matrix(rnorm(n * p), n, p)
    y <- X[, 1] * 2 - X[, 3] + rnorm(n, sd = 0.1)
    df <- as.data.frame(cbind(X, y = y))
    colnames(df)[seq_len(p)] <- paste0("v", seq_len(p))

    # Tier-1 path
    X_t1 <- as.matrix(df[, paste0("v", seq_len(p))])
    fit_t1 <- pls4all_fit(X_t1, y, algo = "pls_nipals", n_components = 3L)
    preds_t1 <- as.vector(pls4all_predict(fit_t1, X_t1))

    # Tier-2 path
    fit_t2 <- pls(y ~ ., data = df, ncomp = 3L, algo = "pls_nipals")
    preds_t2 <- predict(fit_t2, newdata = df)

    # Bit-exact agreement — tier 2 is a pure reformatter.
    expect_equal(preds_t1, preds_t2)
})

test_that("predict() rejects newdata with wrong feature count", {
    set.seed(2)
    n <- 50; p <- 6
    X <- matrix(rnorm(n * p), n, p)
    df <- as.data.frame(X)
    df$y <- X[, 1] + rnorm(n, sd = 0.1)

    fit <- pls(y ~ ., data = df, ncomp = 2L)
    bad_df <- as.data.frame(matrix(rnorm(n * (p - 2)), n, p - 2))
    colnames(bad_df) <- paste0("V", seq_len(p - 2))
    expect_error(predict(fit, newdata = bad_df), regexp = "features")
})

test_that("summary() and print() produce structured output", {
    set.seed(3)
    df <- data.frame(
        x1 = rnorm(40), x2 = rnorm(40), x3 = rnorm(40), y = rnorm(40))
    fit <- pls(y ~ ., data = df, ncomp = 2L)
    out <- summary(fit)
    expect_s3_class(out, "summary.pls4all_fit")
    expect_equal(out$ncomp, 2L)
    expect_equal(out$n_features_in, 3L)
    expect_equal(out$response_name, "y")
})
