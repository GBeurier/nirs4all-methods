# SPDX-License-Identifier: CECILL-2.1

test_that("native row-mask roles retain held-out X/Y alignment", {
    X <- matrix(seq_len(30), nrow = 10L, ncol = 3L)
    Y <- matrix(c(1:9, 100), ncol = 1L)
    h <- n4m_sample_filter_create("y_outlier", 0, c(1.5, 5, 95))
    expect_error(n4m_sample_filter_apply(h, X, Y), "not fitted")
    expect_error(n4m_sample_filter_fit(h, X, cbind(Y, Y)), "shape mismatch")
    n4m_sample_filter_fit(h, X, Y)
    got <- n4m_sample_filter_apply(h, X[8:10, , drop = FALSE], Y[8:10, , drop = FALSE])
    expect_identical(got$mask, c(TRUE, TRUE, FALSE))
    expect_identical(got$stats[1:3], c(3, 2, 1))
    expect_error(n4m_sample_filter_apply(h, X[8:10, , drop = FALSE], Y), "shape mismatch")

    q <- n4m_sample_filter_create("spectral_quality", c(0, 0, 1),
                                   c(.1, .5, 1e-8, 0, 0))
    quality_X <- rbind(c(1, 2, 3), c(0, 0, 0), c(4, 5, 6))
    n4m_sample_filter_fit(q, quality_X)
    expect_identical(n4m_sample_filter_apply(q, quality_X)$mask,
                     c(TRUE, FALSE, TRUE))
    comp <- n4m_sample_filter_create("composite", 0, numeric())
    expect_error(n4m_sample_filter_fit(comp, quality_X), "invalid argument")
    n4m_sample_filter_add_child(comp, "spectral_quality", c(0, 0, 1),
                                c(.1, .5, 1e-8, 0, 0))
    n4m_sample_filter_fit(comp, quality_X)
    expect_identical(n4m_sample_filter_apply(comp, quality_X)$mask,
                     c(TRUE, FALSE, TRUE))

    train_X <- matrix(c(1,2,4, 2,5,3, 3,1,5, 4,7,2, 5,3,8,
                        6,8,6, 7,4,7, 8,9,10, 9,6,9, 10,10,11),
                      ncol = 3L, byrow = TRUE)
    x_filter <- n4m_sample_filter_create("x_outlier",
        c(0, 0, 0, 100, 256), c(0, .2), seed = 17)
    leverage <- n4m_sample_filter_create("high_leverage",
        c(0, 0, 0, 1), c(2, 0))
    n4m_sample_filter_fit(x_filter, train_X)
    n4m_sample_filter_fit(leverage, train_X)
    expect_length(n4m_sample_filter_apply(x_filter, train_X)$mask, 10L)
    expect_length(n4m_sample_filter_apply(leverage, train_X)$mask, 10L)
})

test_that("native column roles expose ordered selected indices", {
    X <- cbind(1:5, rep(4, 5), 2:6)
    Y <- matrix(1:5, ncol = 1L)
    held <- rbind(c(6, 9, 7), c(7, 9, 8))
    for (kind in c("variance", "correlation")) {
        h <- n4m_feature_filter_create(kind, threshold = .01)
        expect_error(n4m_feature_filter_indices(h), "not fitted")
        if (kind == "correlation")
            expect_error(n4m_feature_filter_fit(h, X, cbind(Y, Y)), "shape mismatch")
        n4m_feature_filter_fit(h, X, if (kind == "correlation") Y else NULL)
        expect_identical(n4m_feature_filter_indices(h), c(1L, 3L))
        expect_identical(unname(n4m_feature_filter_transform(h, held)),
                         unname(held[, c(1, 3), drop = FALSE]))
    }
})
