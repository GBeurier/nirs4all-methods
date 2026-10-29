# SPDX-License-Identifier: CECILL-2.1
fixture_path <- Sys.getenv("N4M_INTRINSIC_FIXTURE")
if (!nzchar(fixture_path)) {
  fixture_path <- system.file("parity", "intrinsic_diagnostics.v1.json", package = "n4m")
}
fx <- jsonlite::fromJSON(fixture_path, simplifyVector = FALSE)
for (case in fx$cases) {
  testthat::test_that(paste("intrinsic native transport", case$method_id), {
    constructor <- get(n4m:::.n4m_method_constructors[[case$method_id]], asNamespace("n4m"))
    # Constructor names come from the single generated native manifest.
    object <- do.call(constructor, case$parameters)
    X <- do.call(rbind, lapply(case$X, unlist))
    args <- list(object = object, X = X)
    if (!is.null(case$X_target)) args$X_target <- do.call(rbind, lapply(case$X_target, unlist))
    actual <- do.call(n4m_run, args)
    testthat::expect_setequal(names(actual), names(case$expected))
    for (key in names(case$expected)) {
      expected <- case$expected[[key]]
      # JSON matrices are row-major lists; R's actual matrix is column-major.
      if (is.matrix(actual[[key]])) expected <- do.call(rbind, lapply(expected, unlist)) else expected <- unlist(expected)
      testthat::expect_equal(as.numeric(actual[[key]]), as.numeric(expected), tolerance = 1e-9, info = key)
      testthat::expect_true(all(is.finite(actual[[key]])))
    }
  })
}
