# SPDX-License-Identifier: CECILL-2.1

# MethodResult regressors whose coefficients define a reusable affine model.
# The dispatcher remains the fitting source of truth; this S3 layer owns the
# native predictor handle used for held-out inference and N4MM export.
.n4m_affine_methods <- list(
  ridge = "ridge_lambda", ridge_pls = "ridge_lambda",
  robust_pls = c("huber_k", "max_irls_iter"), cppls = "gamma",
  sparse_simpls = "sparsity_lambda", ecr = "alpha",
  continuum_regression = "tau", mir_pls = character(),
  fused_sparse_pls = c("l1_lambda", "fusion_lambda"),
  bagging_pls = c("n_estimators", "seed"),
  boosting_pls = c("n_estimators", "learning_rate"),
  random_subspace_pls = c("n_estimators", "features_per_subspace", "seed"),
  n_pls = c("mode_j", "mode_k"), mb_pls = "block_sizes",
  di_pls = c("X_target", "di_lambda"),
  group_sparse_pls = c("group_assignment", "group_lambda"))

#' Native MethodResult methods currently marked for affine model conversion
#'
#' The native converter rejects unmarked results. This list lets product
#' controllers use the fitted-model path only where the producer has been
#' qualified for held-out prediction.
#' @return Character vector of marked method names.
#' @export
n4m_affine_supported_methods <- function() c(
  names(.n4m_affine_methods))

#' Fit a reusable native affine MethodResult regressor
#'
#' Fits a qualified native method and turns its coefficient result into a
#' libn4m model handle. `predict()` and [n4m_affine_model_export()] use that
#' handle; they do not repeat affine matrix arithmetic in R. The exported
#' N4MM bytes preserve prediction, not the original fitting algorithm. Only
#' producers listed by [n4m_affine_supported_methods()] currently carry the
#' native `affine_predictor` capability marker; others are refused.
#'
#' @param method Qualified n4m MethodResult regression method.
#' @param X Finite numeric samples-by-features matrix.
#' @param Y Finite numeric response vector or samples-by-targets matrix.
#' @param n_components Positive component count; ignored by ridge's solver.
#' @param params Named method-specific parameters.
#' @return An `n4m_affine_fit` object with a native prediction model.
#' @export
n4m_affine_fit <- function(method, X, Y, n_components = 2L, params = list()) {
  if (!is.character(method) || length(method) != 1L || is.na(method) ||
      !(method %in% names(.n4m_affine_methods)))
    stop("unsupported affine MethodResult regression method", call. = FALSE)
  if (!(method %in% n4m_affine_supported_methods()))
    stop("MethodResult is not marked affine_predictor", call. = FALSE)
  if (!is.matrix(X) || !is.numeric(X) || any(dim(X) < 1L) ||
      anyNA(X) || any(!is.finite(X)))
    stop("X must be a finite numeric matrix", call. = FALSE)
  if (!is.numeric(Y) || (!is.null(dim(Y)) && !is.matrix(Y)) ||
      length(Y) == 0L || anyNA(Y) || any(!is.finite(Y)))
    stop("Y must be a finite numeric vector or matrix", call. = FALSE)
  Y_matrix <- if (is.matrix(Y)) Y else matrix(Y, ncol = 1L)
  if (nrow(Y_matrix) != nrow(X) || ncol(Y_matrix) < 1L)
    stop("Y must have one row per X sample", call. = FALSE)
  if (!is.numeric(n_components) || length(n_components) != 1L ||
      !is.finite(n_components) || n_components < 1L ||
      n_components > .Machine$integer.max ||
      n_components != floor(n_components))
    stop("n_components must be a positive integer", call. = FALSE)
  if (!is.list(params) || (length(params) &&
      (is.null(names(params)) || anyNA(names(params)) ||
       any(!nzchar(names(params))) || anyDuplicated(names(params)) ||
       !all(names(params) %in% .n4m_affine_methods[[method]]))))
    stop("unsupported affine method parameters", call. = FALSE)
  if (identical(method, "group_sparse_pls")) {
    groups <- params$group_assignment
    if (is.null(groups) || !is.numeric(groups) ||
        length(groups) != ncol(X) || anyNA(groups) ||
        any(!is.finite(groups)) || any(groups < 0) ||
        any(groups != floor(groups)) || any(groups > .Machine$integer.max))
      stop("group_assignment must contain one non-negative int32 ID per X feature",
           call. = FALSE)
    if (!is.null(names(groups)) &&
        !identical(names(groups), colnames(X)))
      stop("group_assignment names must match X feature names in order",
           call. = FALSE)
    params$group_assignment <- as.integer(groups)
    if (!is.null(params$group_lambda) &&
        (!is.numeric(params$group_lambda) ||
         length(params$group_lambda) != 1L ||
         !is.finite(params$group_lambda) || params$group_lambda < 0))
      stop("group_lambda must be finite and non-negative", call. = FALSE)
  }
  if (identical(method, "n_pls") &&
      (is.null(params$mode_j) || is.null(params$mode_k) ||
       as.double(params$mode_j) * as.double(params$mode_k) != ncol(X)))
    stop("mode_j times mode_k must equal the X feature count", call. = FALSE)
  if (identical(method, "mb_pls") &&
      (is.null(params$block_sizes) || sum(params$block_sizes) != ncol(X)))
    stop("sum(block_sizes) must equal the X feature count", call. = FALSE)
  if (identical(method, "di_pls") &&
      (is.null(params$X_target) || !is.matrix(params$X_target) ||
       ncol(params$X_target) != ncol(X) ||
       (!is.null(colnames(params$X_target)) &&
        !identical(colnames(params$X_target), colnames(X)))))
    stop("X_target must match the X feature width and order", call. = FALSE)
  storage.mode(X) <- "double"
  storage.mode(Y_matrix) <- "double"
  result <- .Call("r_n4m_affine_dispatch_fit", method, X, Y_matrix,
    as.integer(n_components), params, 1L, 0L, 1L, 0L, PACKAGE = "n4m")
  coefficients <- as.matrix(result$coefficients)
  direct_intercept <- method %in% c("mb_pls", "ridge")
  x_mean <- if (direct_intercept) NULL else as.numeric(result$x_mean)
  y_mean <- if (direct_intercept) NULL else as.numeric(result$y_mean)
  if (!is.numeric(coefficients) ||
      !identical(dim(coefficients), c(ncol(X), ncol(Y_matrix))) ||
      any(!is.finite(coefficients)) ||
      (!direct_intercept &&
       (length(x_mean) != ncol(X) || length(y_mean) != ncol(Y_matrix) ||
        any(!is.finite(x_mean)) || any(!is.finite(y_mean)))) ||
      (direct_intercept &&
       (length(result$intercept) != ncol(Y_matrix) ||
        any(!is.finite(result$intercept)))))
    stop("n4m method returned an unsupported affine regression result",
         call. = FALSE)
  structure(list(method = method, n_components = as.integer(n_components),
    params = params, coefficients = coefficients, x_mean = x_mean,
    y_mean = y_mean, intercept = if (direct_intercept)
      as.numeric(result$intercept) else NULL,
    n_features_in = ncol(X), n_targets = ncol(Y_matrix),
    feature_names = colnames(X), source_training_samples = nrow(X),
    native_model = result$native_model), class = "n4m_affine_fit")
}

#' Predict from a fitted native affine MethodResult regressor
#' @param object An [n4m_affine_fit()] result.
#' @param newdata Numeric samples-by-features matrix.
#' @param ... Unused.
#' @return Numeric vector for one target, otherwise a matrix.
#' @export
predict.n4m_affine_fit <- function(object, newdata, ...) {
  if (!is.matrix(newdata) || !is.numeric(newdata) ||
      ncol(newdata) != object$n_features_in || anyNA(newdata) ||
      any(!is.finite(newdata)))
    stop("newdata must be a finite matrix with the fitted feature width",
         call. = FALSE)
  if (!is.null(object$feature_names) &&
      !identical(colnames(newdata), object$feature_names))
    stop("newdata feature names or order differ from the fitted X",
         call. = FALSE)
  predictions <- n4m_predict(object$native_model, newdata)
  if (object$n_targets == 1L) as.numeric(predictions) else predictions
}

#' Export a fitted affine MethodResult model as N4MM bytes
#' @param object An [n4m_affine_fit()] result.
#' @return Portable N4MM bytes for native prediction.
#' @export
n4m_affine_model_export <- function(object) {
  if (!inherits(object, "n4m_affine_fit"))
    stop("object must be an n4m_affine_fit", call. = FALSE)
  n4m_model_export(object$native_model)
}
