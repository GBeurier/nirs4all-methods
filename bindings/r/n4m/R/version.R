#' Runtime version string of the loaded libn4m.
#'
#' @return A character scalar like "0.67.0+abi.1.1.0".
#' @export
n4m_version <- function() {
  .Call("r_n4m_version", PACKAGE = "n4m")
}


#' Loaded ABI version as an integer vector (major, minor, patch).
#'
#' @return An integer vector of length 3.
#' @export
n4m_abi_version <- function() {
  .Call("r_n4m_abi_version", PACKAGE = "n4m")
}
