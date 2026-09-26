# SPDX-License-Identifier: CECILL-2.1

test_that("marked formula fits predict through one native N4MM model", {
  set.seed(11)
  X <- matrix(stats::rnorm(30L * 6L), 30L, 6L)
  y <- 0.3 + X[, 2L] - 0.2 * X[, 4L]
  data <- data.frame(X, y = y)
  heldout <- data.frame(X + 0.031, y = y)
  fits <- list(
    sparse = sparse_pls(y ~ ., data, ncomp = 2L),
    cppls = cppls(y ~ ., data, ncomp = 2L),
    mb = mb_pls(y ~ ., data, ncomp = 2L,
                block_sizes = c(3L, 3L)),
    mir = mir_pls(y ~ ., data, ncomp = 2L),
    di = di_pls(y ~ ., data, ncomp = 2L, X_target = X + 0.02)
  )
  legacy <- list(
    sparse_simpls_fit(X, y, 2L, 0.05),
    cppls_fit(X, y, 2L, 0.5),
    mb_pls_fit(X, y, 2L, c(3L, 3L)),
    mir_pls_fit(X, y, 2L),
    di_pls_fit(X, y, 2L, X + 0.02)
  )
  for (i in seq_along(fits)) {
    fit <- fits[[i]]
    expect_type(fit$native_model, "externalptr")
    expect_type(fit$native_model_bytes, "raw")
    expect_equal(fit$coefficients, legacy[[i]]$coefficients, tolerance = 1e-12)
    expect_equal(unname(predict(fit, data)), as.numeric(fit$predictions),
                 tolerance = 1e-10)
    imported <- n4m_model_import(n4m_model_export(fit$native_model))
    expect_equal(unname(predict(fit, heldout)),
                 as.numeric(n4m_predict(imported, as.matrix(heldout[, 1:6]))),
                 tolerance = 1e-12)
  }
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(list(fit = fits$sparse, heldout = heldout,
               expected = predict(fits$sparse, heldout)), path)
  restored <- readRDS(path)
  expect_equal(predict(restored$fit, restored$heldout), restored$expected,
               tolerance = 1e-12)
  code <- paste(
    "library(n4m); b <- readRDS(commandArgs(TRUE)[1]);",
    "stopifnot(isTRUE(all.equal(predict(b$fit, b$heldout),",
    "b$expected, tolerance = 1e-12))); cat('OK')")
  child <- suppressWarnings(system2(file.path(R.home("bin"), "Rscript"),
    c("--vanilla", "-e", shQuote(code), shQuote(path)),
    stdout = TRUE, stderr = TRUE))
  expect_identical(attr(child, "status"), NULL)
  expect_true("OK" %in% child)
})

test_that("unmarked formula fits retain the legacy prediction path", {
  set.seed(12)
  X <- matrix(stats::rnorm(30L * 6L), 30L, 6L)
  data <- data.frame(X, y = 0.3 + X[, 2L] - 0.2 * X[, 4L])
  weighted <- weighted_pls(y ~ ., data, ncomp = 2L,
                           weights = rep(1, nrow(data)))
  glm <- pls_glm(y ~ ., data, ncomp = 2L)
  for (fit in list(weighted, glm)) {
    expect_null(fit$native_model)
    expect_equal(unname(predict(fit, data)), as.numeric(fit$predictions),
                 tolerance = 1e-10)
  }
})

test_that("matrix-first affine fits rehydrate from N4MM after RDS", {
  X <- outer(seq_len(21L), seq_len(12L), function(i, j)
    sin(i * j / 9) + cos(i + j / 7) + i * j / 100)
  y <- 1.3 + 0.7 * X[, 2L] - 0.4 * X[, 6L]
  held <- X[c(2L, 8L, 17L), , drop = FALSE] + 0.031
  fit <- n4m_affine_fit("group_sparse_pls", X, y, 2L,
    list(group_assignment = rep(0:2, each = 4L)))
  restored <- unserialize(serialize(fit, NULL))
  expect_type(restored$native_model_bytes, "raw")
  expect_equal(predict(restored, held), predict(fit, held),
               tolerance = 1e-12)
  expect_identical(n4m_affine_model_export(restored),
                   n4m_affine_model_export(fit))
})
