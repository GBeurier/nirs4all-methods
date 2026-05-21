# SPDX-License-Identifier: CECILL-2.1
#
# Fixture loader for the nirs4all-methods R binding parity tests.
#
# The committed JSON fixtures under `parity/fixtures/` encode matrices as
# arrays of 16-character hex strings, each representing one IEEE 754
# binary64 value in big-endian byte order (`ieee754_binary64_be_hex`).
# The helpers below decode these into numeric vectors / matrices and
# locate the fixture root via either the `N4M_FIXTURE_DIR`
# environment variable or by walking parents from this test file.

# Forward-compat shim: R 4.4 ships `%||%` in base; older releases don't.
if (!exists("%||%", envir = baseenv(), inherits = FALSE)) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

.n4m_locate_fixture_dir <- function() {
  env <- Sys.getenv("N4M_FIXTURE_DIR", unset = "")
  if (nzchar(env) && dir.exists(env)) return(env)

  local <- file.path(getwd(), "fixtures")
  if (dir.exists(local)) return(local)

  # When tests run inside `devtools::test()` or `R CMD check`, the working
  # directory is `tests/testthat/` (or its installed copy). Walk parents
  # looking first for packaged test fixtures, then for a repo-level
  # `parity/fixtures` sibling used by developer runs.
  search <- normalizePath(getwd(), mustWork = FALSE)
  for (i in seq_len(8L)) {
    candidate <- file.path(search, "tests", "testthat", "fixtures")
    if (dir.exists(candidate)) return(candidate)
    candidate <- file.path(search, "parity", "fixtures")
    if (dir.exists(candidate)) return(candidate)
    parent <- dirname(search)
    if (identical(parent, search)) break
    search <- parent
  }
  stop("nirs4all-methods tests: could not locate parity/fixtures/. ",
       "Set N4M_FIXTURE_DIR.",
       call. = FALSE)
}

#' Decode a vector of big-endian IEEE 754 binary64 hex strings.
#'
#' Each input element must be exactly 16 hex characters. Returns a
#' double-precision numeric vector of the same length.
hex_to_double <- function(hex) {
  hex <- as.character(hex)
  if (length(hex) == 0L) return(numeric(0))
  if (any(nchar(hex) != 16L)) {
    stop("hex_to_double: every element must be 16 characters", call. = FALSE)
  }
  raws <- lapply(hex, function(s) {
    as.raw(strtoi(substring(s, seq(1, 15, by = 2),
                                seq(2, 16, by = 2)), 16L))
  })
  flat <- unlist(raws, use.names = FALSE)
  if (.Platform$endian == "big") {
    readBin(flat, what = "double", n = length(hex), size = 8L,
            endian = "big")
  } else {
    # On little-endian, swap each 8-byte group.
    idx <- as.vector(matrix(seq_along(flat), nrow = 8)[8:1, , drop = FALSE])
    readBin(flat[idx], what = "double", n = length(hex), size = 8L,
            endian = "little")
  }
}

#' Decode a flat-row-major hex array to an R column-major matrix.
hex_to_matrix <- function(hex, rows, cols) {
  vals <- hex_to_double(hex)
  if (length(vals) != rows * cols) {
    stop(sprintf("hex_to_matrix: %d values, expected %d (= %dx%d)",
                 length(vals), rows * cols, rows, cols), call. = FALSE)
  }
  # Row-major: vals = X[0,0], X[0,1], ..., X[0,C-1], X[1,0], ...
  matrix(vals, nrow = rows, ncol = cols, byrow = TRUE)
}

#' Load a parity fixture JSON by stem name (without `_v1.json`).
load_fixture <- function(stem) {
  dir <- .n4m_locate_fixture_dir()
  path <- file.path(dir, paste0(stem, "_v1.json"))
  if (!file.exists(path)) {
    stop(sprintf("load_fixture: %s does not exist", path), call. = FALSE)
  }
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

#' Extract the primary input matrix from a fixture.
fixture_input_matrix <- function(fx) {
  hex_to_matrix(fx$input_hex, fx$rows, fx$cols)
}

#' Extract the optional fit input matrix (some fixtures ship a separate
#' training set under `fit_input_hex`).
fixture_fit_matrix <- function(fx) {
  if (is.null(fx$fit_input_hex)) return(NULL)
  hex_to_matrix(fx$fit_input_hex, fx$fit_rows, fx$fit_cols)
}

#' Extract the expected output matrix from a case dict, falling back to
#' the fixture-level rows / cols when the case omits them (e.g. SNV).
case_output_matrix <- function(case, fx) {
  rows <- case$output_rows %||% fx$rows
  cols <- case$output_cols %||% fx$cols
  hex_to_matrix(case$output_hex, rows, cols)
}
