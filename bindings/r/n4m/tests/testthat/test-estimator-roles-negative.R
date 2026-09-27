# Refusals of the generic estimator roles: the shared negative fixture (also
# replayed by Python, JS/WASM and Rust) and R's own conversions, which must
# never recycle a vector to the rows of X.
source(testthat::test_path("fixture-estimator-roles-negative.R"))

neg <- estimator_roles_negative

hex_raw <- function(hex) {
  as.raw(strtoi(substring(hex, seq(1, nchar(hex), 2), seq(2, nchar(hex), 2)), 16L))
}

build <- function(case) do.call(n4m_constructor(case$method_id), as.list(case$params))

target <- function(case) {
  if (!is.null(case$y_matrix)) return(case$y_matrix)
  if (!is.null(case$labels)) return(case$labels)
  case$y
}

for (case in neg$cases) {
  local({
    case <- case
    testthat::test_that(paste("shared negative fixture:", case$id), {
      if (case$step == "fit") {
        args <- list(build(case), neg$x_train, target(case))
        if (!is.null(case$sample_weight)) args$sample_weight <- case$sample_weight
        if (!is.null(case$fold_ids)) args$fold_ids <- case$fold_ids
        testthat::expect_error(do.call(n4m_estimator_fit, args), case$mentions)
      } else if (case$step == "import") {
        testthat::expect_error(n4m_estimator_import(hex_raw(case$n4me_hex)), case$mentions)
        control <- n4m_estimator_import(hex_raw(case$control_hex))
        testthat::expect_equal(predict(control, neg$x_test), case$control_predict,
                               tolerance = 1e-12)
      } else if (case$step == "output_view") {
        # R has no raw view API: the facade allocates the native widths.
        est <- n4m_estimator_import(hex_raw(case$n4me_hex))
        testthat::expect_identical(ncol(n4m_estimator_transform(est, neg$x_test)),
                                   as.integer(case$transform_cols))
        testthat::expect_length(predict(est, neg$x_test), nrow(neg$x_test))
      } else if (case$step == "export") {
        est <- n4m_estimator_fit(build(case), neg$x_train, target(case))
        testthat::expect_identical(n4m_contains_training_rows(est), case$contains_training_rows)
        testthat::expect_error(n4m_estimator_export(est), case$mentions)
        testthat::expect_length(n4m_estimator_export(est, allow_training_rows = TRUE),
                                case$export_size_with_opt_in)
      } else {
        testthat::expect_identical(case$step, "refit")
        est <- n4m_estimator_fit(build(case), neg$x_train, target(case))
        testthat::expect_error(est <- n4m_estimator_fit(est, neg$x_train, case$refit_labels),
                               case$mentions)
        testthat::expect_equal(n4m_classes(est), case$classes)
        testthat::expect_equal(predict(est, neg$x_test), case$predict_labels)
      }
    })
  })
}

testthat::test_that("targets and per-row inputs are never recycled", {
  X <- matrix(sin(seq_len(320)), 40, 8)
  y <- cos(seq_len(40))
  ridge <- n4m_ridge()
  # The audit reproduction: 7, 1:20 and 1:39 were recycled to 40 rows.
  for (bad in list(7, seq_len(20), seq_len(39), seq_len(80))) {
    testthat::expect_error(n4m_estimator_fit(ridge, X, bad), "y must have length 40")
  }
  testthat::expect_error(n4m_estimator_fit(ridge, X, matrix(y, 2, 20)), "y must have 40 rows")
  testthat::expect_error(n4m_estimator_fit(ridge, X, array(y, c(40, 1, 1))), "array")
  # One target: a vector or a one-column matrix; several: one column each.
  testthat::expect_equal(predict(n4m_estimator_fit(ridge, X, y), X),
                         predict(n4m_estimator_fit(ridge, X, matrix(y)), X)[, 1])
  two <- n4m_estimator_fit(n4m_pls_regression(), X, cbind(y, -y))
  testthat::expect_equal(dim(predict(two, X)), c(40L, 2L))
  testthat::expect_error(n4m_estimator_fit(n4m_weighted_pls(), X, y, sample_weight = rep(1, 20)),
                         "sample_weight must have length 40")
  testthat::expect_error(n4m_estimator_fit(n4m_weighted_pls(), X, y,
                                           sample_weight = matrix(1, 20, 2)),
                         "sample_weight must be a vector")
  lda <- n4m_pls_lda()
  labels <- ifelse(y > 0, "up", "down")
  testthat::expect_error(n4m_estimator_fit(lda, X, labels[1:20]), "class labels must have length 40")
  # Non-integer numbers are labels (a table), not truncated ids; missing,
  # non-finite and out-of-int64 labels are refused.
  testthat::expect_identical(n4m_classes(n4m_estimator_fit(lda, X, rep(c(0, 1.5), 20))), c(0, 1.5))
  testthat::expect_error(n4m_estimator_fit(lda, X, rep(c(0, NA), 20)), "must not be missing")
  testthat::expect_error(n4m_estimator_fit(lda, X, rep(c(0, Inf), 20)), "must be finite")
  testthat::expect_error(n4m_estimator_fit(lda, X, rep(c(0, 2^63), 20)), "must fit int64")
  testthat::expect_error(n4m_estimator_fit(n4m_stability(top_k = 3), X, y, fold_ids = rep(0:1, 10)),
                         "fold_ids must have length 40")
  filt <- n4m_estimator_fit(n4m_y_outlier(), X, y)
  testthat::expect_error(n4m_sample_mask(filt, X, y[1:20]), "y must have length 40")
  testthat::expect_error(n4m_split(n4m_kennard_stone(), X, y[1:20]), "y must have length 40")
  # Scalar parameters take one value, integers are whole numbers.
  testthat::expect_error(n4m_estimator_fit(n4m_pls_regression(n_components = c(2, 3)), X, y),
                         "n_components")
  testthat::expect_error(n4m_estimator_fit(n4m_pls_regression(n_components = 2.5), X, y),
                         "n_components")
})

testthat::test_that("a failed refit keeps the fitted object and its levels", {
  X <- matrix(sin(seq_len(320)), 40, 8)
  labels <- ifelse(X[, 1] > 0, "healthy", "diseased")
  est <- n4m_estimator_fit(n4m_pls_lda(), X, labels)
  before <- predict(est, X)
  testthat::expect_error(est <- n4m_estimator_fit(est, X, rep("other", 40)), "two classes")
  testthat::expect_identical(predict(est, X), before)
  testthat::expect_identical(n4m_classes(est), c("diseased", "healthy"))
})

testthat::test_that("seeds are optional and an unset seed runs as seed 0", {
  X <- matrix(sin(seq_len(320)), 40, 8)
  testthat::expect_null(formals(n4m_gaussian_noise)$seed)
  testthat::expect_identical(n4m_augment(n4m_gaussian_noise(), X),
                             n4m_augment(n4m_gaussian_noise(seed = 0), X))
  bag <- n4m_estimator_fit(n4m_bagging_pls(n_estimators = 3), X, X[, 1])
  testthat::expect_identical(n4m_estimator_import(n4m_estimator_export(bag))$params$seed, 0)
})

test_that("the legacy dispatcher refuses ids it would truncate", {
  set.seed(4)
  X <- matrix(stats::rnorm(120), 20, 6)
  y <- X[, 1] * 2 + X[, 2]
  expect_error(n4m_method("pls_logistic", X, y, 2,
                          params = list(y_labels = rep(c(0.2, 1.8), 10), n_classes = 2)),
               "y_labels must contain finite integers")
  expect_error(n4m_method("pls_lda", X, y, 2, params = list(y_labels = rep(c(0, NA), 10))),
               "y_labels must contain finite integers")
  expect_error(n4m_method("pls_lda", X, y, 2, params = list(y_labels = rep(c(0, 2^40), 10))),
               "y_labels must contain finite integers")
  expect_error(n4m_method("group_sparse_pls", X, y, 2,
                          params = list(group_assignment = c(0, 0.5, 1, 1, 2, 2))),
               "group_assignment must contain finite integers")
  expect_error(n4m_method("mb_pls", X, y, 2, params = list(block_sizes = c(2.5, 3.5))),
               "block_sizes must contain finite integers")
  expect_true(is.list(n4m_method("pls_logistic", X, y, 2,
                                 params = list(y_labels = rep(c(0, 1), 10), n_classes = 2))))
})
