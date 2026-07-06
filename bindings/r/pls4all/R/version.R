#' Runtime version string of the loaded libn4m.
#'
#' @return A character scalar like "1.0.3+abi.2.0.0".
#' @export
n4m_version <- function() {
  .Call("r_n4m_version", PACKAGE = "pls4all")
}


#' Loaded ABI version as an integer vector (major, minor, patch).
#'
#' @return An integer vector of length 3.
#' @export
n4m_abi_version <- function() {
  .Call("r_n4m_abi_version", PACKAGE = "pls4all")
}
