# SPDX-License-Identifier: CECILL-2.1
#
# R port of `parity/binding_parity.py` (Gate 1).
#
# Returns a list with the same fields as the Python dataclass:
#   ok              logical
#   max_abs_diff    numeric
#   max_rel_diff    numeric
#   shape_mismatch  logical
#   detail          character

binding_parity <- function(pred, ref, tolerance = 1e-6) {
  pred <- as.array(pred); ref <- as.array(ref)
  if (!identical(dim(pred), dim(ref))) {
    return(list(ok = FALSE, max_abs_diff = Inf, max_rel_diff = Inf,
                shape_mismatch = TRUE,
                detail = sprintf("shape mismatch: pred %s vs ref %s",
                                 paste(dim(pred), collapse = "x"),
                                 paste(dim(ref), collapse = "x"))))
  }
  if (anyNA(pred) || anyNA(ref) ||
      any(is.nan(pred)) || any(is.nan(ref))) {
    return(list(ok = FALSE, max_abs_diff = NaN, max_rel_diff = NaN,
                shape_mismatch = FALSE,
                detail = "NaN in pred or ref"))
  }
  if (length(pred) == 0L) {
    return(list(ok = TRUE, max_abs_diff = 0, max_rel_diff = 0,
                shape_mismatch = FALSE, detail = "empty"))
  }
  abs_d <- max(abs(pred - ref))
  rel_d <- abs_d / max(max(abs(ref)), 1e-300)
  list(ok = (abs_d <= tolerance),
       max_abs_diff = abs_d,
       max_rel_diff = rel_d,
       shape_mismatch = FALSE,
       detail = sprintf("max_abs=%.3e", abs_d))
}

#' testthat-friendly expectation wrapper.
expect_binding_parity <- function(pred, ref, tolerance = 1e-6,
                                  label = "binding parity") {
  res <- binding_parity(pred, ref, tolerance = tolerance)
  testthat::expect_true(
    res$ok,
    info = sprintf("%s: %s (max_abs=%.3e, max_rel=%.3e)",
                   label, res$detail, res$max_abs_diff, res$max_rel_diff))
  invisible(res)
}
