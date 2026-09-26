# Generic estimator roles (ABI 2.13): R facade and cross-language N4ME states.
source(testthat::test_path("fixture-estimator-roles.R"))

fx <- estimator_roles_fixture
constructors <- lapply(n4m:::.n4m_method_constructors, get, envir = asNamespace("n4m"))

hex_to_raw <- function(hex) {
  as.raw(strtoi(substring(hex, seq(1, nchar(hex), 2), seq(2, nchar(hex), 2)), 16L))
}

fit_inputs <- function(names) {
  all <- list(feature_groups = fx$feature_groups, blocks = fx$blocks,
              X_target = fx$x_target)
  all[names]
}

testthat::test_that("every manifest estimator has a generated R constructor", {
  testthat::expect_setequal(names(constructors), names(n4m:::.n4m_method_roles))
  testthat::expect_setequal(names(constructors),
                            vapply(fx$cases, `[[`, "", "method_id"))
})

for (case in fx$cases) {
  local({
    case <- case
    testthat::test_that(paste("Python N4ME state predicts identically in R:", case$method_id), {
      bytes <- hex_to_raw(case$n4me)
      est <- n4m_estimator_import(bytes)
      testthat::expect_equal(inherits(est, "n4m_regressor"), !is.null(case$predict))
      if (!is.null(case$predict)) {
        testthat::expect_equal(predict(est, fx$x_test), case$predict, tolerance = 1e-12)
      }
      if (!is.null(case$selected_indices)) {
        testthat::expect_s3_class(est, "n4m_selector")
        testthat::expect_equal(n4m_selected_indices(est), case$selected_indices + 1)
      }
      if (!is.null(case$transform)) {
        testthat::expect_equal(n4m_estimator_transform(est, fx$x_test), case$transform,
                               tolerance = 1e-12)
      } else {
        testthat::expect_false(inherits(est, "n4m_transformer"))
      }
      testthat::expect_identical(n4m_estimator_export(est), bytes)
    })

    testthat::test_that(paste("R fit reproduces the Python fit:", case$method_id), {
      spec <- do.call(constructors[[case$method_id]], case$params)
      fitted <- do.call(n4m_estimator_fit, c(list(spec, fx$x_train, fx$y_train),
                                             fit_inputs(case$fit_inputs)))
      path <- tempfile(fileext = ".rds")
      saveRDS(fitted, path)
      restored <- readRDS(path)
      if (!is.null(case$predict)) {
        testthat::expect_equal(predict(fitted, fx$x_test), case$predict, tolerance = 1e-9)
        testthat::expect_identical(predict(restored, fx$x_test), predict(fitted, fx$x_test))
      }
      if (!is.null(case$selected_indices)) {
        testthat::expect_equal(n4m_selected_indices(fitted), case$selected_indices + 1)
        testthat::expect_identical(n4m_selected_indices(restored), n4m_selected_indices(fitted))
      }
    })
  })
}

testthat::test_that("roles and inputs are enforced natively", {
  est <- n4m_estimator_fit(n4m_cppls(), fx$x_train, fx$y_train)
  testthat::expect_error(n4m_estimator_transform(est, fx$x_test))
  testthat::expect_error(n4m_estimator_fit(n4m_group_sparse_pls(), fx$x_train, fx$y_train),
                         "feature_groups")
  testthat::expect_error(n4m_estimator_fit(n4m_cppls(), fx$x_train, fx$y_train,
                                           groups = rep(1, nrow(fx$x_train))),
                         "not used")
  testthat::expect_error(n4m_estimator_fit(n4m_pls_regression(solver = "bogus"),
                                           fx$x_train, fx$y_train), "solver")
  testthat::expect_error(n4m_estimator_fit(n4m_tensor_pls(), fx$x_train, fx$y_train),
                         "mode_j")
})
