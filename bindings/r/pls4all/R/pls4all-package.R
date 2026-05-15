#' pls4all: R binding for the libp4a C ABI
#'
#' Provides fit/predict wrappers for the PLS regression solvers shipped
#' by the upstream pls4all library. The shared library libp4a must be
#' findable at load time; set PLS4ALL_LIB_PATH to its absolute location
#' if the default search path does not work.
#'
#' @docType package
#' @name pls4all-package
#' @useDynLib pls4all, .registration = TRUE
NULL
