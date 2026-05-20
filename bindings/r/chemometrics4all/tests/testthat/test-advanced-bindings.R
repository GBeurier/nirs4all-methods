# SPDX-License-Identifier: CECILL-2.1

test_that("advanced R bindings return expected shapes", {
  if (!exists("c4a_abi_version")) {
    library(chemometrics4all)
  }
  tryCatch(c4a_abi_version(), error = function(e) {
    testthat::skip("libc4a is not loaded")
  })
  set.seed(20260520)
  X <- matrix(runif(8 * 12, 0.2, 1.2), nrow = 8)
  Y <- X * 1.05 + 0.02
  y <- runif(8)
  weights <- runif(12)

  same_shape <- list(
    direct_standardization(X, Y),
    robust_direct_standardization(X, Y, max_iter = 2L),
    piecewise_direct_standardization(X, Y, window_size = 5L),
    score_augmented_projection_standardization(X, Y, n_components = 2L),
    local_centering(X, Y),
    weighted_snv(X, weights = weights),
    variable_sorting_normalization(X),
    piecewise_snv(X, window = 4L),
    piecewise_msc(X, window = 4L),
    localized_msc(X, window = 4L),
    cross_correlation_alignment(X, interval_size = 4L, max_shift = 2L),
    icoshift_alignment(X, interval_size = 4L, max_shift = 2L),
    dynamic_time_warping_alignment(X, interval_size = 4L, max_shift = 2L),
    correlation_optimized_warping(X, interval_size = 4L, max_shift = 2L)
  )
  for (out in same_shape) {
    expect_equal(dim(out), dim(X))
  }

  expect_length(slope_bias_correction(y, y * 2 + 1), length(y))
  expect_equal(dim(variance_filter(X, top_k = 3L)), c(nrow(X), 3L))
  expect_equal(dim(correlation_filter(X, y, top_k = 3L)), c(nrow(X), 3L))
  intervals <- interval_generator(X, interval_size = 4L, step = 3L)
  expect_type(intervals, "list")
  expect_equal(length(intervals), 4L)
  expect_equal(lapply(intervals, dim), list(c(8L, 4L), c(8L, 4L), c(8L, 4L), c(8L, 3L)))
})
