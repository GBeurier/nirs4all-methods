test_that("group sparse penalty changes the R native predictor", {
    X <- matrix(c(
        1, 0, 2, 3, 2, 1, 0, 2, 3, 2, 1, 0, 4, 1, 3, 1,
        5, 3, 2, 4, 6, 2, 4, 2, 7, 4, 1, 5, 8, 3, 5, 3
    ), ncol = 4L, byrow = TRUE)
    Y <- cbind(2 * X[, 1L] + 0.3 * X[, 2L] - X[, 3L] + 1,
               -X[, 1L] + 0.5 * X[, 3L] + 0.7 * X[, 4L] - 2)
    groups <- c(2L, 2L, 9L, 9L)
    raw <- group_sparse_pls_fit(X, Y, 2L, groups, group_lambda = 0)
    fit <- group_sparse_pls_fit(X, Y, 2L, groups, group_lambda = 0.2)
    expect_equal(fit$n_groups, 2)
    for (idx in list(1:2, 3:4)) {
        norm <- sqrt(sum(raw$coefficients[idx, , drop = FALSE]^2))
        factor <- if (norm > 0.2) 1 - 0.2 / norm else 0
        expect_equal(fit$coefficients[idx, , drop = FALSE],
                     raw$coefficients[idx, , drop = FALSE] * factor,
                     tolerance = 1e-10)
    }
    expected <- sweep(X, 2L, colMeans(X)) %*% fit$coefficients
    expected <- sweep(expected, 2L, colMeans(Y), `+`)
    expect_equal(fit$predictions, expected, tolerance = 1e-10)
    expect_equal(as.numeric(fit$predictions[1:2, ]),
                 c(2.211247731382924, 5.098353536109149,
                   -1.696732182318895, -2.609927236516501),
                 tolerance = 1e-10)
    expect_gt(max(abs(fit$predictions - raw$predictions)), 1e-6)
    zero <- group_sparse_pls_fit(X, Y, 2L, groups, group_lambda = 1e6)
    expect_equal(zero$coefficients, matrix(0, 4L, 2L))
    expect_equal(zero$predictions,
                 matrix(rep(colMeans(Y), each = nrow(X)), ncol = 2L),
                 tolerance = 1e-10)
    expect_error(group_sparse_pls_fit(X, Y, 2L, groups, group_lambda = -0.1))
})
