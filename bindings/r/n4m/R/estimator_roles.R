# SPDX-License-Identifier: CECILL-2.1

# Generic native estimator roles (ABI 2.13). Each constructor generated in
# estimator_roles_generated.R returns an unfitted specification whose S3
# classes are its role interfaces (n4m_regressor, n4m_transformer, ...).
# Parameters, defaults, required inputs, fitting and the N4ME fitted state
# are native; this file only marshals R objects.

.n4m_role_classes <- c(regressor = "n4m_regressor", transformer = "n4m_transformer")

.n4m_estimator <- function(method_id, roles, params) {
  structure(
    list(method_id = method_id, params = params, state = NULL),
    class = c(unname(.n4m_role_classes[roles]), "n4m_estimator")
  )
}

.n4m_as_matrix <- function(X, name = "X") {
  X <- as.matrix(X)
  if (!is.numeric(X)) stop(name, " must be numeric", call. = FALSE)
  storage.mode(X) <- "double"
  X
}

# The external pointer does not survive saveRDS(); the N4ME bytes do.
.n4m_pointer <- function(object) {
  state <- object$state
  if (is.null(state)) stop("n4m estimator is not fitted", call. = FALSE)
  if (!isTRUE(.Call("r_n4m_estimator_alive", state$pointer, PACKAGE = "n4m"))) {
    state$pointer <- .Call("r_n4m_estimator_import", state$n4me, PACKAGE = "n4m")
  }
  state$pointer
}

#' Generic native estimator roles
#'
#' Constructors such as \code{n4m_pls_regression()} or \code{n4m_cppls()}
#' return an unfitted estimator whose classes are its role interfaces:
#' \code{n4m_regressor} (\code{predict}), \code{n4m_transformer}
#' (\code{n4m_estimator_transform}). Fit with \code{n4m_estimator_fit()}. The fitted state is kept
#' as portable N4ME bytes, so \code{saveRDS()}/\code{readRDS()} and the Python
#' and JS/WASM bindings reuse it without refitting.
#'
#' @param object An estimator from one of the generated constructors.
#' @param X Numeric matrix (rows are samples).
#' @param y Numeric response vector or matrix; required by regressors.
#' @param sample_weight,groups,feature_groups,blocks,axis,X_target,fold_ids
#'   Optional fit inputs; each method declares which ones it requires and the
#'   native core refuses the others.
#' @param seed Seed for stochastic methods.
#' @param newdata Numeric matrix of new samples.
#' @param bytes Raw vector produced by \code{n4m_estimator_export()}.
#' @param ... Unused.
#' @return \code{n4m_estimator_fit()} returns the fitted estimator; \code{predict()} and
#'   \code{n4m_estimator_transform()} return numeric results; \code{n4m_estimator_export()} a raw
#'   vector; \code{n4m_estimator_import()} a fitted estimator.
#' @name n4m_estimator_roles
NULL

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_fit <- function(object, X, y = NULL, ...) UseMethod("n4m_estimator_fit")

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_fit.n4m_estimator <- function(object, X, y = NULL, sample_weight = NULL, groups = NULL,
                                  feature_groups = NULL, blocks = NULL, axis = NULL,
                                  X_target = NULL, fold_ids = NULL, seed = 0, ...) {
  X <- .n4m_as_matrix(X)
  y_matrix <- if (is.null(y)) NULL else matrix(as.double(y), nrow = nrow(X))
  inputs <- list(sample_weight = sample_weight, groups = groups,
                 feature_groups = feature_groups, blocks = blocks, axis = axis,
                 X_target = if (is.null(X_target)) NULL else .n4m_as_matrix(X_target, "X_target"),
                 fold_ids = fold_ids)
  inputs <- inputs[!vapply(inputs, is.null, logical(1))]
  params <- object$params[!vapply(object$params, is.null, logical(1))]
  pointer <- .Call("r_n4m_estimator_fit", object$method_id, params, X, y_matrix,
                   inputs, as.double(seed), PACKAGE = "n4m")
  object$state <- list2env(list(
    pointer = pointer,
    n4me = .Call("r_n4m_estimator_export", pointer, PACKAGE = "n4m"),
    y_vector = !is.null(y) && is.null(dim(y))
  ))
  object
}

#' @rdname n4m_estimator_roles
#' @export
predict.n4m_regressor <- function(object, newdata, ...) {
  out <- .Call("r_n4m_estimator_predict", .n4m_pointer(object), .n4m_as_matrix(newdata),
               PACKAGE = "n4m")
  if (isTRUE(object$state$y_vector) && ncol(out) == 1L) out[, 1L] else out
}

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_transform <- function(object, X, ...) UseMethod("n4m_estimator_transform")

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_transform.n4m_transformer <- function(object, X, ...) {
  .Call("r_n4m_estimator_transform", .n4m_pointer(object), .n4m_as_matrix(X), PACKAGE = "n4m")
}

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_export <- function(object) {
  if (is.null(object$state)) stop("n4m estimator is not fitted", call. = FALSE)
  object$state$n4me
}

#' @rdname n4m_estimator_roles
#' @export
n4m_estimator_import <- function(bytes) {
  pointer <- .Call("r_n4m_estimator_import", bytes, PACKAGE = "n4m")
  info <- .Call("r_n4m_estimator_info", pointer, PACKAGE = "n4m")
  roles <- .n4m_method_roles[[info$method_id]]
  if (is.null(roles)) stop("no R role mapping for ", info$method_id, call. = FALSE)
  object <- .n4m_estimator(info$method_id, roles, info$params)
  object$state <- list2env(list(pointer = pointer, n4me = bytes,
                                y_vector = info$n_outputs == 1))
  object
}

#' @export
print.n4m_estimator <- function(x, ...) {
  roles <- sub("^n4m_", "", setdiff(class(x), "n4m_estimator"))
  cat("<n4m ", x$method_id, " (", paste(roles, collapse = ", "), ")",
      if (is.null(x$state)) ", unfitted" else ", fitted", ">\n", sep = "")
  invisible(x)
}
