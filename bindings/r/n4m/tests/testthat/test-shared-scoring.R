# SPDX-License-Identifier: CECILL-2.1
# Exercise the public R procedure against libn4m, with X=predicted and y=true.
testthat::test_that("regression metrics reject nonfinite values in either input", {
  refusals <- 0L
  for (field in c("y_true", "y_pred")) {
    for (bad in c(NaN, Inf, -Inf)) {
      truth <- c(0, 2, 4)
      predicted <- c(1, 2, 3)
      if (field == "y_true") truth[2] <- bad else predicted[2] <- bad
      testthat::expect_error(
        n4m::n4m_run(n4m::n4m_regression_metrics(), X = matrix(predicted, ncol = 1), y = truth),
        "invalid argument",
        info = paste(field, bad)
      )
      refusals <- refusals + 1L
    }
  }
  testthat::expect_equal(refusals, 6L)
})

fixture_path <- Sys.getenv("N4M_SHARED_SCORING_FIXTURE")
if (!nzchar(fixture_path)) {
  fixture_path <- system.file(
    "extdata", "shared_scoring_numerical.v1.json", package = "n4m"
  )
}
testthat::test_that("finite shared scalar cases preserve RMSE MAE and R squared", {
  testthat::expect_true(
    nzchar(fixture_path) && file.exists(fixture_path),
    info = "Required shared scoring fixture missing from n4m extdata or explicit override"
  )
  fixture <- jsonlite::fromJSON(fixture_path, simplifyVector = FALSE)
  cases <- Filter(function(case) {
    case$kind == "regression" && case$applicability$methods_scalar_abi == "supported"
  }, fixture$cases)
  testthat::expect_length(cases, 5L)
  for (case in cases) {
    actual <- n4m::n4m_run(
      n4m::n4m_regression_metrics(),
      X = matrix(unlist(case$y_pred), ncol = 1), y = unlist(case$y_true)
    )
    for (metric in c("rmse", "mae", "r2")) {
      expected <- case$expected[[metric]]
      testthat::expect_true(
        is.finite(actual[[metric]]) &&
          abs(actual[[metric]] - expected) <= fixture$tolerance$absolute +
            fixture$tolerance$relative * abs(expected),
        info = paste(case$name, metric, actual[[metric]], "expected", expected)
      )
    }
  }
})
