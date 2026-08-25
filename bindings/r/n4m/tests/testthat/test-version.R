testthat::test_that("version string is non-empty", {
  v <- n4m::n4m_version()
  testthat::expect_type(v, "character")
  testthat::expect_true(nchar(v) > 0)
})

testthat::test_that("abi version is a three-element integer vector", {
  abi <- n4m::n4m_abi_version()
  testthat::expect_type(abi, "integer")
  testthat::expect_length(abi, 3)
})

testthat::test_that("abi major/minor is the locked 2.3 ML namespace", {
  abi <- n4m::n4m_abi_version()
  testthat::expect_equal(abi[1:2], c(2L, 3L))
})

testthat::test_that("locked role-name aliases are exported", {
  exported <- getNamespaceExports("n4m")
  locked <- c(
    "n4m_transform_snv",
    "n4m_feature_select_cars",
    "n4m_model_selection_kennard_stone",
    "n4m_regression_ridge"
  )
  for (name in locked) {
    testthat::expect_true(name %in% exported, info = name)
    testthat::expect_true(is.function(get(name, envir = asNamespace("n4m"))), info = name)
  }
})
