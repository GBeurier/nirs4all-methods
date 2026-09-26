# SPDX-License-Identifier: CECILL-2.1

test_that("one native contract exposes all nine ordered sample splitters", {
    samples <- 1:30
    X <- cbind(as.double(samples), as.double((samples * 7) %% 13))
    Y <- matrix(as.double(samples %% 11), ncol = 1L)
    groups <- rep(1:15, each = 2L)
    kinds <- c("kennard_stone", "spxy", "spxy_fold", "spxy_group_fold",
               "kmeans", "kbins_stratified", "binned_strat_group_fold",
               "systematic_circular", "data_twinning")
    for (kind in kinds) {
        code <- match(kind, kinds) - 1L
        uses_x <- code %in% c(0L, 1L, 2L, 3L, 4L, 8L)
        uses_y <- code %in% c(1L, 2L, 3L, 5L, 6L, 7L)
        grouped <- code %in% c(3L, 6L)
        args <- list(kind = kind, X = if (uses_x) X else NULL,
                     Y = if (uses_y) Y else NULL,
                     groups = if (grouped) groups else NULL,
                     n_splits = 3L, n_bins = 2L, seed = 42)
        first <- do.call(n4m_splitter_run, args)
        second <- do.call(n4m_splitter_run, args)
        expect_identical(first, second, info = kind)
        expect_equal(length(first$train) + length(first$test), 30L, info = kind)
        expect_setequal(c(first$train, first$test), 1:30)
        if (grouped)
            expect_length(intersect(groups[first$train], groups[first$test]),
                          0L)
        if (code %in% c(2L, 3L, 6L))
            expect_false(identical(first$test,
                do.call(n4m_splitter_run, c(args, list(fold = 2L)))$test),
                info = kind)
    }
    expect_identical(n4m_splitter_run("kennard_stone", X = X),
                     kennard_stone_split(X))
})

test_that("native splitter contract rejects missing targets and invalid groups", {
    X <- cbind(1:12, 12:1) * 1.0
    Y <- matrix(as.double(1:12), ncol = 1L)
    expect_error(n4m_splitter_run("spxy", X = X), "X/Y inputs")
    expect_error(n4m_splitter_run("spxy_group_fold", X = X, Y = Y,
                                  groups = 1:11), "group")
    expect_error(n4m_splitter_run("binned_strat_group_fold", Y = Y,
                                  groups = c(1:11, 2^53)), "exact integer")
    expect_error(n4m_splitter_run("spxy_fold", X = X, Y = Y,
                                  fold = 4L, n_splits = 3L), "parameters")
    expect_error(n4m_splitter_run("not_a_splitter", X = X), "unknown")
})
