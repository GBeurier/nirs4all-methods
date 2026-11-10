# Generated constructor and shared native-state transport smoke.
testthat::test_that("RowL2Normalize uses the native transformer lifecycle", {
  X <- matrix(c(3, 4, 0, 0), nrow = 2L, byrow = TRUE)
  fit <- n4m_estimator_fit(n4m_row_l2_normalize(), X)
  actual <- n4m_estimator_transform(fit, X)
  testthat::expect_equal(unname(actual), matrix(c(0.6, 0.8, 0, 0), 2L, byrow = TRUE),
                         tolerance = 4e-15)
  state <- n4m_estimator_export(fit)
  replay <- n4m_estimator_import(state)
  testthat::expect_identical(n4m_estimator_transform(replay, X), actual)
})
