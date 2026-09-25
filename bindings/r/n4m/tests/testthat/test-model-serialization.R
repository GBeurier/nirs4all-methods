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

test_that("native affine import exports multi-target N4MM without transposing coefficients", {
  coefficients <- matrix(c(2, 0.5, -1, 3), nrow = 2L, byrow = TRUE)
  intercept <- c(1.5, -2)
  X <- matrix(c(1, 4, -2, 3, 0.5, -1), ncol = 2L, byrow = TRUE)
  expected <- sweep(X %*% coefficients, 2L, intercept, "+")
  model <- n4m_model_import_linear_predictor(coefficients, intercept,
                                             source_training_samples = 17L)
  expect_equal(n4m_predict(model, X), expected, tolerance = 1e-14)
  bytes <- n4m_model_export(model)
  descriptor <- n4m_model_descriptor(bytes)
  expect_identical(descriptor$format_version, 1L)
  expect_identical(descriptor$algorithm, 11L)
  expect_identical(descriptor$n_features, 2L)
  expect_identical(descriptor$n_targets, 2L)
  expect_identical(descriptor$n_components, 0L)
  expect_identical(descriptor$capabilities, 5)
  restored <- n4m_model_import(bytes)
  expect_equal(n4m_predict(restored, X), expected, tolerance = 1e-14)
  expect_identical(n4m_model_export(restored), bytes)
  unknown_count <- n4m_model_import_linear_predictor(coefficients, intercept)
  expect_equal(n4m_predict(unknown_count, X), expected, tolerance = 1e-14)
  expect_error(n4m_model_import_linear_predictor(coefficients, 1),
               "one finite numeric value")
  expect_error(n4m_model_import_linear_predictor(matrix(NaN, 1L), 0),
               "finite numeric matrix")
  expect_error(n4m_model_import_linear_predictor(coefficients, intercept,
                                                 source_training_samples = -1L),
               "non-negative integer")
})

test_that("embedded SNV-Savitzky-Golay state predicts raw spectra and round-trips", {
  x <- outer(seq_len(24L), seq_len(13L),
             function(i, j) sin(i * j / 11) + i * j / 170)
  y <- 1 + x[, 2L] - 0.3 * x[, 8L]
  embedded <- n4m_fit(x, y, "pls_simpls", 2L,
                      embedded_snv_savgol = c(5, 2))
  transformed <- savgol_transform(snv_transform(x), 5L, 2L, mode = "interp")
  plain <- n4m_fit(transformed, y, "pls_simpls", 2L)
  expect_equal(n4m_predict(embedded, x), n4m_predict(plain, transformed),
               tolerance = 1e-12)
  bytes <- n4m_model_export(embedded)
  info <- n4m_model_pipeline_info(bytes)
  descriptor <- n4m_model_descriptor(bytes)
  expect_true(info$present)
  expect_identical(info$semantic_profile, 1L)
  expect_identical(info$window_length, 5L)
  expect_identical(info$polyorder, 2L)
  expect_identical(info$raw_n_features, 13L)
  expect_identical(descriptor$format_version, 2L)
  expect_identical(descriptor$solver, 1L)
  expect_identical(descriptor$n_components, 2L)
  expect_equal(n4m_predict(n4m_model_import(bytes), x),
               n4m_predict(embedded, x), tolerance = 1e-12)
  expect_false(n4m_model_pipeline_info(n4m_model_export(plain))$present)
  expect_error(n4m_fit(x, y, "pls_simpls", 2L,
                       embedded_snv_savgol = c(4, 2)), "odd window")
  expect_error(n4m_model_pipeline_info(bytes[-length(bytes)]))
  expect_error(n4m_model_descriptor(bytes[-length(bytes)]))
})
