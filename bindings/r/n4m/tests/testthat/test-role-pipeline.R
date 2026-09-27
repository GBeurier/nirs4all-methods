# Native role pipelines (ABI 2.14): R facade, strict inputs and the shared
# role-pipeline fixture written by the Python binding.
source(testthat::test_path("fixture-role-pipeline.R"))

rp <- role_pipeline_fixture
# Fixture outputs come from Linux x86-64; kernels drift by a few ulps on other
# platforms, so replays compare at 1e-9 (the exported bytes stay identical).
replay_tol <- 1e-9
refit_tol <- 1e-9

hex_to_raw <- function(hex) {
  as.raw(strtoi(substring(hex, seq(1, nchar(hex), 2), seq(2, nchar(hex), 2)), 16L))
}
states_of <- function(hexes) lapply(hexes, hex_to_raw)
named <- function(X, names = rp$feature_names) {
  colnames(X) <- names
  X
}

toy <- function(n = 40, p = 8) {
  set.seed(3)
  X <- matrix(2 + stats::rnorm(n * p), n, p)
  list(X = X, y = drop(X %*% stats::rnorm(p)))
}

testthat::test_that("the recipe is validated natively", {
  testthat::expect_error(n4m_role_pipeline(list()), "at least one step")
  testthat::expect_error(n4m_role_pipeline(list("preprocessing.scatter.snv")),
                         "ends with one regressor or classifier")
  testthat::expect_error(
    n4m_role_pipeline(list("models.regularized.ridge", "models.regularized.ridge")),
    "only the last step")
})

testthat::test_that("a fitted pipeline matches the chain run by hand", {
  d <- toy()
  steps <- list("preprocessing.scatter.snv",
                list(method_id = "models.pls.pls_regression", params = list(n_components = 3L)),
                list(class = "n4m:models.regularized.ridge"))
  fit <- n4m_estimator_fit(n4m_role_pipeline(steps), d$X, d$y)
  snv <- n4m_estimator_fit(n4m_snv(), d$X)
  pls <- n4m_estimator_fit(n4m_pls_regression(n_components = 3L),
                           n4m_estimator_transform(snv, d$X), d$y)
  scores <- n4m_estimator_transform(pls, n4m_estimator_transform(snv, d$X))
  ridge <- n4m_estimator_fit(n4m_ridge(), scores, d$y)
  testthat::expect_identical(predict(fit, d$X), predict(ridge, scores))
  testthat::expect_identical(n4m_estimator_transform(fit, d$X), scores)
  info <- n4m_role_pipeline_steps(fit)
  testthat::expect_identical(info$role, c("transformer", "transformer", "regressor"))
  testthat::expect_identical(info$state_index, 0:2)
})

testthat::test_that("targets are never recycled", {
  d <- toy()
  pipe <- n4m_role_pipeline(list("models.regularized.ridge"))
  testthat::expect_error(n4m_estimator_fit(pipe, d$X, d$y[1:20]), "20 values; X has 40 rows")
  testthat::expect_error(n4m_estimator_fit(pipe, d$X, 7), "1 values; X has 40 rows")
  testthat::expect_error(n4m_estimator_fit(pipe, d$X, matrix(d$y[1:20], 20)), "20 rows; X has 40")
  testthat::expect_error(n4m_estimator_fit(pipe, d$X, as.character(d$y)), "numeric vector or matrix")
  cls <- n4m_role_pipeline(list("models.classification.pls_lda"))
  testthat::expect_error(n4m_estimator_fit(cls, d$X, rep(c("a", "b"), 10)), "vector of 40 values")
  testthat::expect_error(n4m_estimator_fit(pipe, d$X, d$y, groups = seq_len(40)),
                         "not used by any step of the pipeline 'groups'")
})

testthat::test_that("column names are checked and training rows need an opt-in", {
  d <- toy()
  X <- named(d$X, paste0("w", 1:8))
  fit <- n4m_estimator_fit(n4m_role_pipeline(list("preprocessing.scatter.snv", "models.pls.kernel")),
                           X, d$y)
  testthat::expect_identical(predict(fit, X), predict(fit, d$X))  # positional arrays
  testthat::expect_error(predict(fit, X[, 8:1]), "the columns are reordered")
  testthat::expect_error(predict(fit, X[, -1]), "7 columns; the pipeline was fitted on 8")
  testthat::expect_identical(n4m_role_pipeline_steps(fit)$contains_training_rows, c(FALSE, TRUE))
  testthat::expect_error(n4m_role_pipeline_export(fit), "state retains training rows")
  states <- n4m_role_pipeline_export(fit, allow_training_rows = TRUE)
  back <- n4m_role_pipeline_import(fit$steps, states, feature_names = paste0("w", 1:8))
  testthat::expect_identical(predict(back, X), predict(fit, X))
  path <- tempfile(fileext = ".rds")
  saveRDS(fit, path)
  testthat::expect_identical(predict(readRDS(path), X), predict(fit, X))
})

testthat::test_that("the shared fixture's pipelines replay and refit identically", {
  reg <- rp$regression
  fit <- n4m_role_pipeline_import(reg$steps, states_of(reg$states), feature_names = rp$feature_names)
  testthat::expect_equal(predict(fit, named(rp$x_test)), reg$predict, tolerance = replay_tol)
  testthat::expect_equal(n4m_estimator_transform(fit, rp$x_test), reg$transform,
                         tolerance = replay_tol)
  exported <- n4m_role_pipeline_export(fit)
  testthat::expect_identical(vapply(exported, `[[`, "", "method_id"), reg$state_methods)
  testthat::expect_identical(lapply(exported, `[[`, "n4me"), states_of(reg$states))
  refit <- n4m_estimator_fit(n4m_role_pipeline(reg$steps), named(rp$x_train), rp$y_train)
  testthat::expect_equal(predict(refit, rp$x_test), reg$predict, tolerance = refit_tol)

  cls <- rp$classification
  fit <- n4m_role_pipeline_import(cls$steps, states_of(cls$states),
                                  feature_names = rp$feature_names, class_names = cls$class_names)
  testthat::expect_identical(as.character(predict(fit, rp$x_test)), cls$predict)
  testthat::expect_equal(unname(predict(fit, rp$x_test, type = "decision")), cls$decision_function,
                         tolerance = replay_tol)
  refit <- n4m_estimator_fit(n4m_role_pipeline(cls$steps), rp$x_train, rp$labels_train)
  testthat::expect_identical(as.character(predict(refit, rp$x_test)), cls$predict)
})

testthat::test_that("the shared fixture's negative cases are refused alike", {
  reg <- rp$regression
  fitted <- n4m_role_pipeline_import(reg$steps, states_of(reg$states),
                                     feature_names = rp$feature_names)
  for (case in rp$cases) {
    if (case$stage == "fit") {
      fit <- n4m_estimator_fit(n4m_role_pipeline(case$steps), rp$x_train, rp[[case$y]])
      testthat::expect_equal(predict(fit, rp$x_test), case$predict, tolerance = refit_tol,
                             label = case$name)
      next
    }
    run <- switch(case$stage,
      create = function() n4m_role_pipeline(case$steps),
      import = function() n4m_role_pipeline_import(case$steps, states_of(case$states)),
      predict = function() {
        names <- if (isTRUE(case$drop_last_column)) rp$feature_names[-12] else case$feature_names
        predict(fitted, named(rp$x_test[, seq_along(names)], names))
      },
      export = function() {
        n4m_role_pipeline_export(
          n4m_estimator_fit(n4m_role_pipeline(case$steps), rp$x_train, rp[[case$y]]))
      })
    testthat::expect_error(run(), case$message, fixed = TRUE, label = case$name)
  }
})

testthat::test_that("n4m_role_pipeline_info exposes width, column names and classes", {
  set.seed(1)
  X <- matrix(stats::rnorm(400), 80, 5, dimnames = list(NULL, paste0("w", 1:5)))
  y <- factor(sample(c("a", "b"), 80, TRUE))
  fit <- n4m_estimator_fit(n4m_role_pipeline(list("preprocessing.scatter.snv",
                                                  "models.classification.pls_qda")), X, y)
  info <- n4m_role_pipeline_info(fit)
  testthat::expect_identical(info$n_features, 5L)
  testthat::expect_identical(info$feature_names, paste0("w", 1:5))
  testthat::expect_identical(info$classes, c("a", "b"))
  testthat::expect_identical(n4m_classes(fit), c("a", "b"))
  testthat::expect_identical(info$label_names, c("a", "b"))
  testthat::expect_identical(nrow(info$steps), 2L)
  reg <- n4m_estimator_fit(n4m_role_pipeline(list("models.pls.cppls")), X, stats::rnorm(80))
  testthat::expect_null(n4m_role_pipeline_info(reg)$classes)
})

testthat::test_that("the shared fixture's label cases are refused alike", {
  # JSON has no NaN / infinity: `inject` puts one at position `at` (0-based).
  injected <- function(values, case) {
    if (!is.null(case$inject)) {
      values[[case$inject$at + 1L]] <- if (case$inject$value == "inf") Inf else NaN
    }
    values
  }
  for (case in rp$label_cases) {
    if (case$stage == "fit") {
      labels <- injected(case$labels, case)
      testthat::expect_error(
        n4m_estimator_fit(n4m_role_pipeline(rp$classification$steps), rp$x_train, labels),
        case$message, fixed = TRUE, label = case$name)
      # The estimator-level classifiers encode labels the same way.
      testthat::expect_error(n4m_estimator_fit(n4m_pls_lda(), rp$x_train, labels),
                             case$message, fixed = TRUE, label = case$name)
      next
    }
    source <- rp[[case$pipeline]]
    load <- function() {
      n4m_role_pipeline_import(source$steps, states_of(source$states),
                               class_names = injected(case$class_names, case))
    }
    if (isTRUE(case$accept)) {
      testthat::expect_identical(as.character(predict(load(), rp$x_test)), source$predict,
                                 label = case$name)
    } else {
      testthat::expect_error(load(), case$message, fixed = TRUE, label = case$name)
    }
  }
})

testthat::test_that("non-integer numeric labels become a label table", {
  cls <- rp$classification
  labels <- c(high = 2.5, low = 0.5, mid = 1.5)[rp$labels_train]
  fit <- n4m_estimator_fit(n4m_role_pipeline(cls$steps), rp$x_train, unname(labels))
  testthat::expect_identical(n4m_role_pipeline_info(fit)$label_names, c(0.5, 1.5, 2.5))
  back <- n4m_role_pipeline_import(cls$steps, n4m_role_pipeline_export(fit),
                                   class_names = n4m_role_pipeline_info(fit)$label_names)
  testthat::expect_identical(predict(back, rp$x_test), predict(fit, rp$x_test))
})

testthat::test_that("pipeline fit inputs are converted as strictly as the estimators'", {
  d <- toy()
  weighted <- list("models.regularized.weighted_pls")
  testthat::expect_error(n4m_estimator_fit(n4m_role_pipeline(weighted), d$X, d$y,
                                           sample_weight = matrix(1, 20, 2)),
                         "sample_weight must be a vector")
  testthat::expect_error(n4m_estimator_fit(n4m_role_pipeline(weighted), d$X, d$y,
                                           sample_weight = array(1, c(40, 1, 1))),
                         "sample_weight must be a vector")
  ridge <- n4m_role_pipeline(list("models.regularized.ridge"))
  testthat::expect_error(n4m_estimator_fit(ridge, d$X, d$y, groups = matrix(1, 20, 2)),
                         "groups must be a vector")
  testthat::expect_error(n4m_estimator_fit(ridge, d$X, d$y, fold_ids = rep(0.5, 40)),
                         "fold_ids must contain finite integers")
  testthat::expect_error(n4m_estimator_fit(ridge, d$X, d$y, axis = matrix(1, 4, 2)),
                         "axis must be a vector")
})

testthat::test_that("zero new rows give an empty output whatever the steps", {
  d <- toy()
  for (steps in list(list("models.regularized.ridge"),
                     list("preprocessing.scatter.snv", "models.regularized.ridge"))) {
    fit <- n4m_estimator_fit(n4m_role_pipeline(steps), d$X, d$y)
    testthat::expect_length(predict(fit, d$X[0, , drop = FALSE]), 0L)
  }
})
