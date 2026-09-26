# SPDX-License-Identifier: CECILL-2.1

.n4m_preprocess_kinds <- c(
    identity = 0L, center = 1L, autoscale = 2L, pareto_scale = 3L,
    snv = 4L, msc = 5L, emsc = 6L, detrend_poly = 7L,
    savgol_smooth = 8L, savgol_derivative = 9L, norris_williams = 10L,
    asls_baseline = 11L, osc = 12L, epo = 13L, wavelet_denoise = 14L,
    finite_difference = 15L, whittaker = 16L, fck = 17L, gaussian = 18L)

#' Native preprocessing pipeline capabilities
#'
#' The C ABI names 19 operator kinds. Fifteen currently support pipeline
#' fit/transform; the other four are reported as unavailable.
#' @return Data frame with `kind`, `code`, and `supported` columns.
#' @export
n4m_preprocess_capabilities <- function() {
    data.frame(kind = names(.n4m_preprocess_kinds),
               code = unname(.n4m_preprocess_kinds),
               supported = unname(.n4m_preprocess_kinds) <= 14L,
               row.names = NULL)
}

#' Define one native preprocessing pipeline operator
#'
#' Parameters follow the positional C ABI contract for the selected kind.
#' @param kind Name from [n4m_preprocess_capabilities()].
#' @param params Finite numeric C ABI parameter vector.
#' @return Operator specification for [n4m_preprocess_fit()].
#' @export
n4m_preprocess_step <- function(kind, params = numeric()) {
    if (!is.character(kind) || length(kind) != 1L || is.na(kind) ||
        !(kind %in% names(.n4m_preprocess_kinds)))
        stop("unknown native preprocessing kind", call. = FALSE)
    code <- unname(.n4m_preprocess_kinds[[kind]])
    if (code > 14L)
        stop(sprintf("native preprocessing kind '%s' is not implemented by pipeline fit/transform", kind),
             call. = FALSE)
    if (!is.numeric(params) || is.matrix(params) || anyNA(params) ||
        any(!is.finite(params)))
        stop("params must be a finite numeric vector", call. = FALSE)
    structure(list(kind = kind, code = code, params = as.numeric(params)),
              class = "n4m_preprocess_step")
}

#' Fit a native preprocessing pipeline
#'
#' The fitted state is an in-process native handle. It remains valid while this
#' object lives; it is not a portable serialized preprocessing artifact.
#' @param X Finite numeric training matrix.
#' @param steps Nonempty list of [n4m_preprocess_step()] specifications.
#' @param Y Optional finite numeric target matrix for supervised OSC/EPO.
#' @return Fitted native preprocessing state.
#' @export
n4m_preprocess_fit <- function(X, steps, Y = NULL) {
    if (!is.matrix(X) || !is.numeric(X) || nrow(X) < 1L || ncol(X) < 1L ||
        anyNA(X) || any(!is.finite(X)))
        stop("X must be a nonempty finite matrix", call. = FALSE)
    if (!is.list(steps) || length(steps) < 1L ||
        !all(vapply(steps, inherits, logical(1), "n4m_preprocess_step")))
        stop("steps must be a nonempty list of native preprocessing steps", call. = FALSE)
    if (!is.null(Y) && (!is.matrix(Y) || !is.numeric(Y) ||
                       nrow(Y) != nrow(X) || ncol(Y) < 1L ||
                       anyNA(Y) || any(!is.finite(Y))))
        stop("Y must be a finite matrix aligned with X", call. = FALSE)
    X <- matrix(as.double(X), nrow = nrow(X), dimnames = dimnames(X))
    if (!is.null(Y)) Y <- matrix(as.double(Y), nrow = nrow(Y))
    codes <- vapply(steps, `[[`, integer(1), "code")
    params <- lapply(steps, `[[`, "params")
    handle <- .Call("r_n4m_preprocess_fit", X, Y, codes, params, PACKAGE = "n4m")
    structure(list(handle = handle, n_features = ncol(X), steps = steps),
              class = "n4m_preprocess_fit")
}

#' Transform spectra with fitted native preprocessing state
#' @param object Result of [n4m_preprocess_fit()].
#' @param X Finite numeric matrix with the training feature width.
#' @return Transformed numeric matrix.
#' @export
n4m_preprocess_transform <- function(object, X) {
    if (!inherits(object, "n4m_preprocess_fit"))
        stop("object must be fitted native preprocessing state", call. = FALSE)
    if (!is.matrix(X) || !is.numeric(X) || nrow(X) < 1L ||
        ncol(X) != object$n_features || anyNA(X) || any(!is.finite(X)))
        stop("X must be finite and match the fitted feature width", call. = FALSE)
    X <- matrix(as.double(X), nrow = nrow(X), dimnames = dimnames(X))
    .Call("r_n4m_preprocess_transform", object$handle, X, PACKAGE = "n4m")
}
