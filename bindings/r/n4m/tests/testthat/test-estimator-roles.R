# Generic estimator roles (ABI 2.13): R facade and cross-language N4ME states.
source(testthat::test_path("fixture-estimator-roles.R"))

fx <- estimator_roles_fixture
constructors <- lapply(n4m:::.n4m_method_constructors, get, envir = asNamespace("n4m"))

hex_to_raw <- function(hex) {
  as.raw(strtoi(substring(hex, seq(1, nchar(hex), 2), seq(2, nchar(hex), 2)), 16L))
}

check_classifier <- function(est, case, tolerance) {
  testthat::expect_s3_class(est, "n4m_classifier")
  testthat::expect_equal(n4m_classes(est), case$classes)
  testthat::expect_equal(predict(est, fx$x_test), case$predict_labels)
  decision <- predict(est, fx$x_test, type = "decision")
  testthat::expect_equal(unname(decision), case$decision_function, tolerance = tolerance)
  testthat::expect_identical(colnames(decision), as.character(case$classes))
  if (is.null(case$predict_proba)) {
    testthat::expect_error(predict(est, fx$x_test, type = "prob"), "unsupported")
  } else {
    testthat::expect_equal(unname(predict(est, fx$x_test, type = "prob")), case$predict_proba,
                           tolerance = tolerance)
  }
}

fit_inputs <- function(names) {
  all <- list(feature_groups = fx$feature_groups, blocks = fx$blocks, axis = fx$axis,
              X_target = fx$x_target)
  all[names]
}

testthat::test_that("every manifest estimator has a generated R constructor", {
  testthat::expect_setequal(names(constructors), names(n4m:::.n4m_method_roles))
  testthat::expect_setequal(names(constructors),
                            c(vapply(fx$cases, `[[`, "", "method_id"),
                              vapply(fx$procedures, `[[`, "", "method_id")))
})

for (case in fx$cases) {
  local({
    case <- case
    testthat::test_that(paste("Python N4ME state predicts identically in R:", case$method_id), {
      testthat::skip_if(is.null(case$n4me), "train-only filter without a serializable state")
      bytes <- hex_to_raw(case$n4me)
      est <- n4m_estimator_import(bytes)
      testthat::expect_equal(inherits(est, "n4m_regressor"), !is.null(case[["predict"]]))
      if (!is.null(case[["predict"]])) {
        testthat::expect_equal(predict(est, fx$x_test), case[["predict"]], tolerance = 1e-12)
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
      if (!is.null(case$classes)) check_classifier(est, case, 1e-12)
      if (!is.null(case$mask)) {
        testthat::expect_identical(n4m_sample_mask(est, fx$x_test, fx$y_test), case$mask == 1)
      }
      testthat::expect_identical(n4m_estimator_export(est), bytes)
    })

    testthat::test_that(paste("R fit reproduces the Python fit:", case$method_id), {
      spec <- do.call(constructors[[case$method_id]], case$params)
      target <- if (is.null(case$classes)) fx$y_train else fx$labels_train
      fitted <- do.call(n4m_estimator_fit, c(list(spec, fx$x_train, target),
                                             fit_inputs(case$fit_inputs)))
      if (!is.null(case$mask)) {
        testthat::expect_identical(n4m_sample_mask(fitted, fx$x_test, fx$y_test), case$mask == 1)
      }
      if (is.null(case$n4me)) return(invisible())
      path <- tempfile(fileext = ".rds")
      saveRDS(fitted, path)
      restored <- readRDS(path)
      if (!is.null(case[["predict"]])) {
        testthat::expect_equal(predict(fitted, fx$x_test), case[["predict"]], tolerance = 1e-9)
        testthat::expect_identical(predict(restored, fx$x_test), predict(fitted, fx$x_test))
      }
      if (!is.null(case$selected_indices)) {
        testthat::expect_equal(n4m_selected_indices(fitted), case$selected_indices + 1)
        testthat::expect_identical(n4m_selected_indices(restored), n4m_selected_indices(fitted))
      }
      if (!is.null(case$transform)) {
        testthat::expect_equal(n4m_estimator_transform(fitted, fx$x_test), case$transform,
                               tolerance = 1e-9)
      }
      if (!is.null(case$classes)) {
        check_classifier(fitted, case, 1e-9)
        testthat::expect_identical(predict(restored, fx$x_test, type = "decision"),
                                   predict(fitted, fx$x_test, type = "decision"))
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

testthat::test_that("classifiers encode factor and character labels", {
  names <- c("high", "low", "mid")[fx$labels_train / 10]
  est <- n4m_estimator_fit(n4m_pls_qda(), fx$x_train, names)
  testthat::expect_identical(n4m_classes(est), c("high", "low", "mid"))
  pred <- predict(est, fx$x_test)
  testthat::expect_s3_class(pred, "factor")
  testthat::expect_identical(levels(pred), c("high", "low", "mid"))
  by_factor <- n4m_estimator_fit(n4m_pls_qda(), fx$x_train, factor(names))
  testthat::expect_identical(predict(by_factor, fx$x_test), pred)
  proba <- predict(est, fx$x_test, type = "prob")
  testthat::expect_identical(colnames(proba), c("high", "low", "mid"))
  testthat::expect_equal(unname(rowSums(proba)), rep(1, nrow(fx$x_test)), tolerance = 1e-12)
  testthat::expect_identical(as.character(pred), colnames(proba)[max.col(proba, "first")])
  testthat::expect_error(n4m_estimator_fit(n4m_pls_lda(), fx$x_train), "labels")
})

for (case in fx$procedures) {
  local({
    case <- case
    testthat::test_that(paste("R procedure reproduces the Python run:", case$method_id), {
      spec <- do.call(constructors[[case$method_id]], case$params)
      testthat::expect_false(inherits(spec, "n4m_estimator"))
      X <- fx[[case[["x"]]]]
      y <- if ("y" %in% case$inputs) fx$y_train
      groups <- if ("groups" %in% case$inputs) fx$groups
      axis <- if ("axis" %in% case$inputs) fx$axis
      if (!is.null(case$folds)) {
        folds <- n4m_split(spec, X, y, groups)
        testthat::expect_length(folds, length(case$folds))
        for (i in seq_along(folds)) {
          testthat::expect_equal(folds[[i]]$train, case$folds[[i]][[1L]] + 1)
          testthat::expect_equal(folds[[i]]$test, case$folds[[i]][[2L]] + 1)
        }
      }
      if (!is.null(case[["X"]])) {
        testthat::expect_equal(n4m_augment(spec, X, axis), case[["X"]], tolerance = 1e-12)
      }
      if (!is.null(case$outputs)) {
        args <- list(spec, X, y)
        if ("X_target" %in% case$inputs) args$X_target <- fx$x_target
        out <- do.call(n4m_run, args)
        testthat::expect_setequal(names(out), names(case$outputs))
        for (name in names(case$outputs)) {
          testthat::expect_equal(unname(out[[name]]), case$outputs[[name]], tolerance = 1e-9,
                                 label = name)
        }
      }
    })
  })
}

testthat::test_that("the native manifest and constructor lookup cover every method", {
  json <- n4m_manifest_json()
  testthat::expect_true(startsWith(json, "{\"abi\":\"2.13"))
  for (id in names(constructors)) {
    testthat::expect_true(grepl(paste0("\"method_id\":\"", id, "\""), json, fixed = TRUE))
    testthat::expect_identical(n4m_constructor(id), constructors[[id]])
  }
  testthat::expect_error(n4m_constructor("models.pls.missing"), "no n4m role class")
})
