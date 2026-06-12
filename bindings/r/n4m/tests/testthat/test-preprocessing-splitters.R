test_that("preprocessing wrappers keep matrix shape and finite values", {
    set.seed(42)
    X <- matrix(rnorm(8 * 15), 8, 15)

    snv <- n4m::snv_transform(X)
    expect_equal(dim(snv), dim(X))
    expect_true(all(is.finite(snv)))
    expect_equal(rowMeans(snv), rep(0, nrow(X)), tolerance = 1e-12)

    sg <- n4m::savgol_transform(X, window_length = 5L, polyorder = 2L,
                                deriv = 0L, mode = "interp")
    expect_equal(dim(sg), dim(X))
    expect_true(all(is.finite(sg)))
})

test_that("Kennard-Stone split returns idiomatic and parity indices", {
    set.seed(43)
    X <- matrix(rnorm(12 * 6), 12, 6)

    one_based <- n4m::kennard_stone_split(X, test_size = 0.25)
    zero_based <- n4m::kennard_stone_split(X, test_size = 0.25,
                                           zero_based = TRUE)

    expect_named(one_based, c("train", "test"))
    expect_named(zero_based, c("train", "test"))
    expect_equal(one_based$train, zero_based$train + 1L)
    expect_equal(one_based$test, zero_based$test + 1L)
    expect_equal(sort(c(one_based$train, one_based$test)),
                 seq_len(nrow(X)))
})
