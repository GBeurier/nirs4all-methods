testthat::test_that("version string is non-empty", {
  v <- pls4all::pls4all_version()
  testthat::expect_type(v, "character")
  testthat::expect_true(nchar(v) > 0)
})

testthat::test_that("abi version is a three-element integer vector", {
  abi <- pls4all::pls4all_abi_version()
  testthat::expect_type(abi, "integer")
  testthat::expect_length(abi, 3)
})
