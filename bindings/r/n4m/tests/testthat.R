# testthat entry point — R CMD check runs this file, which delegates to
# testthat::test_check() to discover and run everything under tests/testthat/.

library(testthat)
library(n4m)

test_check("n4m")
