test_that("N4MM round-trip preserves fitted predictions", {
  x <- matrix(seq(0.2, 3.1, length.out = 36), nrow = 12L, ncol = 3L)
  x[, 2L] <- x[, 2L]^2
  x[, 3L] <- sin(x[, 3L])
  y <- 0.7 + x[, 1L] - 0.4 * x[, 2L] + 0.2 * x[, 3L]
  model <- n4m_fit(x, y, algo = "pls_simpls", n_components = 2L)
  bytes <- n4m_model_export(model)
  expect_type(bytes, "raw")
  expect_gt(length(bytes), 32L)
  info <- n4m_model_inspect(bytes)
  expect_equal(info$format_version, 1)
  expect_equal(info$writer_abi[1L], n4m_abi_version()[1L])
  restored <- n4m_model_import(bytes)
  expect_equal(attr(restored, "n_features"), ncol(x))
  expect_equal(attr(restored, "n_targets"), 1L)
  expect_equal(n4m_predict(restored, x), n4m_predict(model, x), tolerance = 1e-12)
  expect_identical(n4m_model_export(restored), bytes)
})

test_that("N4MM import rejects malformed bytes", {
  expect_error(n4m_model_import(raw()), "non-empty raw")
  expect_error(n4m_model_import(as.raw(c(1, 2, 3, 4))))
  expect_error(n4m_model_import("not bytes"), "non-empty raw")
  expect_error(n4m_model_inspect(as.raw(c(1, 2, 3, 4))))
  expect_error(n4m_model_export(NULL), "external pointer")
})
