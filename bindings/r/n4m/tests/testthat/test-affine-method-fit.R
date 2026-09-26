test_that("marked group-sparse MethodResult predicts held-out via native model", {
  X <- outer(seq_len(21L), seq_len(12L), function(i, j)
    sin(i * j / 9) + cos(i + j / 7) + i * j / 100)
  colnames(X) <- paste0("wl", seq_len(ncol(X)))
  Y <- 1.3 + 0.7 * X[, 2L] - 0.4 * X[, 6L]
  held <- X[c(2L, 8L, 17L), , drop = FALSE] + 0.031
  groups <- stats::setNames(rep(0:2, each = 4L), colnames(X))
  expected <- c(1.2560303930849952, 1.8411427818857744,
                0.8454051444648407)
  expect_true("group_sparse_pls" %in% n4m_affine_supported_methods())
  fit <- n4m_affine_fit("group_sparse_pls", X, Y, 2L,
    list(group_assignment = groups, group_lambda = 0.05))
  expect_s3_class(fit, "n4m_affine_fit")
  expect_equal(predict(fit, held), expected, tolerance = 1e-10)
  expect_equal(as.numeric(n4m_predict(fit$native_model, held)), expected,
    tolerance = 1e-10)
  bytes <- n4m_affine_model_export(fit)
  expect_type(bytes, "raw")
  expect_equal(as.numeric(n4m_predict(n4m_model_import(bytes), held)),
    expected, tolerance = 1e-10)
  expect_error(predict(fit, held[, rev(seq_len(ncol(held))), drop = FALSE]),
    "feature names or order")
  expect_error(n4m_affine_fit("group_sparse_pls", X, Y, 2L,
    list(group_assignment = groups[-1L])), "one non-negative")
  expect_error(n4m_affine_fit("group_sparse_pls", X, Y, 2L,
    list(group_assignment = rev(groups))), "match X feature names")
  expect_error(n4m_affine_fit("group_sparse_pls", X, Y, 2L,
    list(group_assignment = groups, group_lambda = -1)), "non-negative")

  two_targets <- cbind(Y, 0.5 - 0.2 * X[, 4L] + 0.3 * X[, 10L])
  multi <- n4m_affine_fit("group_sparse_pls", X, two_targets, 2L,
    list(group_assignment = groups, group_lambda = 0.05))
  raw_multi <- n4m_method("group_sparse_pls", X, two_targets, 2L,
    params = list(group_assignment = unname(groups), group_lambda = 0.05))
  expected_multi <- sweep(held, 2L, as.numeric(raw_multi$x_mean), "-") %*%
    raw_multi$coefficients
  expected_multi <- sweep(expected_multi, 2L,
    as.numeric(raw_multi$y_mean), "+")
  expect_equal(predict(multi, held), expected_multi, tolerance = 1e-10)
  expect_equal(n4m_predict(n4m_model_import(n4m_affine_model_export(multi)),
    held), expected_multi, tolerance = 1e-10)
})

test_that("another marked affine method predicts held-out; unmarked methods refuse", {
  X <- outer(seq_len(21L), seq_len(12L), function(i, j)
    sin(i * j / 9) + cos(i + j / 7) + i * j / 100)
  Y <- 1.3 + 0.7 * X[, 2L] - 0.4 * X[, 6L]
  held <- X[c(2L, 8L, 17L), , drop = FALSE] + 0.031
  fitted <- n4m_affine_fit("fused_sparse_pls", X, Y, 2L)
  expected <- c(1.3373539467794611, 1.9377148952044152,
                0.7544322586932715)
  expect_equal(predict(fitted, held), expected, tolerance = 1e-10)
  expect_equal(as.numeric(n4m_predict(n4m_model_import(
    n4m_affine_model_export(fitted)), held)), expected, tolerance = 1e-10)
  expect_error(n4m_affine_fit("ridge", X, Y, 2L),
    "affine_predictor|affine")
  expect_error(n4m_affine_fit("kernel_pls", X, Y, 2L),
    "unsupported affine")
})
