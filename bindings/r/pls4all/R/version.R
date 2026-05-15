#' Runtime version string of the loaded libp4a.
#'
#' @return A character scalar like "0.67.0+abi.1.1.0".
#' @export
pls4all_version <- function() {
  .Call("r_pls4all_version", PACKAGE = "pls4all")
}


#' Loaded ABI version as an integer vector (major, minor, patch).
#'
#' @return An integer vector of length 3.
#' @export
pls4all_abi_version <- function() {
  .Call("r_pls4all_abi_version", PACKAGE = "pls4all")
}
