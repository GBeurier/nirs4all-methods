# Spectral conversions against their external references: the fixture written
# by parity/spectral_conversions/generate.py (photobiology, mdatools, astropy,
# scipy, specdal, spectrolab, prospectr, asdreader) replayed through the R roles.

fixture_path <- function() {
  root <- Sys.getenv("N4M_REPO_ROOT", unset = "")
  if (!nzchar(root)) root <- normalizePath(file.path(testthat::test_path(), "../../../../.."), mustWork = FALSE)
  path <- file.path(root, "parity", "fixtures", "spectral_conversions_external_v1.json")
  if (!file.exists(path)) testthat::skip("spectral conversion fixture not available off-tree")
  path
}

hex <- function(x) as.numeric(unlist(x))
hex_matrix <- function(rows) {
  if (is.list(rows[[1]])) return(do.call(rbind, lapply(rows, hex)))
  matrix(hex(rows), nrow = 1)
}

test_that("spectral conversions match their external references", {
  testthat::skip_if_not_installed("jsonlite")
  doc <- jsonlite::fromJSON(fixture_path(), simplifyVector = FALSE)
  constructors <- c(
    "preprocessing.signal_conversion.to_absorbance" = "n4m_to_absorbance",
    "preprocessing.signal_conversion.from_absorbance" = "n4m_from_absorbance",
    "preprocessing.signal_conversion.kubelka_munk" = "n4m_kubelka_munk",
    "preprocessing.signal_conversion.from_kubelka_munk" = "n4m_from_kubelka_munk",
    "preprocessing.signal_conversion.percent_to_fraction" = "n4m_percent_to_fraction",
    "preprocessing.signal_conversion.fraction_to_percent" = "n4m_fraction_to_percent",
    "preprocessing.signal_conversion.reflectance_from_references" = "n4m_reflectance_from_references",
    "preprocessing.signal_conversion.splice_correction" = "n4m_splice_correction",
    "preprocessing.resampling.resampler" = "n4m_resampler",
    "preprocessing.resampling.axis_unit_conversion" = "n4m_axis_unit_conversion"
  )
  for (case in doc$cases) {
    params <- lapply(case$params, function(v) if (is.list(v)) hex(v) else v)
    est <- do.call(get(constructors[[case$method_id]], envir = asNamespace("n4m")), params)
    X <- hex_matrix(case$X)
    est <- if (is.null(case$axis)) n4m_estimator_fit(est, X) else
      n4m_estimator_fit(est, X, axis = hex(case$axis))
    got <- n4m_estimator_transform(est, X)
    want <- hex_matrix(case$expected)
    expect_true(all(abs(got - want) <= case$atol + case$rtol * abs(want)), label = case$name)
    if (!is.null(case$expected_axis)) {
      axis <- as.vector(n4m_fit_result(est)$axis)
      expect_true(all(abs(axis - hex(case$expected_axis)) <= case$axis_rtol * hex(case$expected_axis)),
                  label = paste(case$name, "axis"))
    }
  }
})
