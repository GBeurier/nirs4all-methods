# SPDX-License-Identifier: CECILL-2.1
#
# Package load logic for chemometrics4all R binding (Phase 23).
#
# The CRAN package vendors the libc4a source and builds it into the native R
# extension. Developer builds may still load an external shared libc4a so tests
# can target an in-tree CMake build. Resolution paths, tried in order:
#
#   1. environment variable `CHEMOMETRICS4ALL_LIB_PATH` — developer/CI override.
#   2. shared object bundled under `inst/extdata/lib/`, if a binary package
#      ever ships one.
#   3. let the already-loaded package extension or the system loader resolve it.
#
# Any failure to load libc4a is fatal: the `.Call` entry points in
# `chemometrics4all.so` will fail with unresolved symbols on first call,
# so we surface the problem at package-load time with a clear message.

.c4a_env <- new.env(parent = emptyenv())

.c4a_resolve_lib_path <- function(pkgname) {
  env_path <- Sys.getenv("CHEMOMETRICS4ALL_LIB_PATH", unset = "")
  if (nzchar(env_path) && file.exists(env_path)) {
    return(env_path)
  }

  ext_dir <- system.file("extdata", "lib", package = pkgname)
  if (nzchar(ext_dir) && dir.exists(ext_dir)) {
    candidates <- list.files(ext_dir, pattern = "^libc4a\\.(so|dylib|dll)",
                             full.names = TRUE)
    if (length(candidates) > 0L) {
      return(candidates[[1L]])
    }
  }
  ""
}

.onLoad <- function(libname, pkgname) {
  lib_path <- .c4a_resolve_lib_path(pkgname)
  if (nzchar(lib_path)) {
    tryCatch(
      dyn.load(lib_path, local = FALSE, now = TRUE),
      error = function(e) {
        warning(sprintf(
          "chemometrics4all: failed to dyn.load(%s): %s. Falling back to the system loader.",
          lib_path, conditionMessage(e)),
          call. = FALSE)
        lib_path <<- ""
      })
  }
  assign("lib_path", lib_path, envir = .c4a_env)
}

.onUnload <- function(libpath) {
  lib_path <- get0("lib_path", envir = .c4a_env, ifnotfound = "")
  if (nzchar(lib_path) && file.exists(lib_path)) {
    tryCatch(dyn.unload(lib_path),
             error = function(e) invisible(NULL))
  }
}

#' Path to the dynamically-loaded libc4a.
#'
#' Returns the absolute path to the `libc4a` shared object that this
#' package loaded at startup, or an empty string when libc4a was resolved
#' by the system loader (i.e. via `LD_LIBRARY_PATH` or rpath).
#'
#' @return character(1)
#' @export
c4a_library_path <- function() {
  get0("lib_path", envir = .c4a_env, ifnotfound = "")
}

#' chemometrics4all ABI version (libc4a).
#'
#' @return Integer vector of length 3: c(major, minor, patch).
#' @export
c4a_abi_version <- function() {
  .Call(C_c4a_abi_version)
}

#' chemometrics4all version string (libc4a).
#'
#' @return character(1)
#' @export
c4a_version_string <- function() {
  .Call(C_c4a_version_string)
}
