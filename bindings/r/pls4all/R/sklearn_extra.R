# SPDX-License-Identifier: CECILL-2.1
#
# Formula+S3 tier-2 wrappers for the remaining MethodResult-based
# algorithms. Each constructor returns a `n4m_method_fit` S3
# object that reuses the shared predict / coef / print / summary
# generics from sklearn_methods.R.
#
# The factory pattern: extract X / y via .n4m_method_design,
# call the corresponding tier-1 *_fit function, hand the result list
# to .n4m_method_fit_object.

#' Elastic Component Regression — formula entry point.
#'
#' @inheritParams pls
#' @return A `n4m_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param alpha Numeric in [0, 1]. Elastic-net / penalty mixing parameter.
#' @export
ecr <- function(formula, data, ncomp = 2L, alpha = 0.5,
                 na.action = stats::na.omit) {
    cl <- match.call()
    d <- .n4m_method_design(formula, data, na.action)
    res <- ecr_fit(d$X, d$y, ncomp, alpha)
    .n4m_method_fit_object(formula, cl, d, res, ncomp,
                                method = "ecr", extra = list(alpha = alpha))
}

#' Robust PLS — formula entry point.
#'
#' @inheritParams pls
#' @return A `n4m_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param huber_k Numeric >= 0. Huber loss tuning constant.
#' @param max_irls_iter Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
robust_pls <- function(formula, data, ncomp = 2L,
                        huber_k = 1.345, max_irls_iter = 20L,
                        na.action = stats::na.omit) {
    cl <- match.call()
    d <- .n4m_method_design(formula, data, na.action)
    res <- robust_pls_fit(d$X, d$y, ncomp, huber_k, max_irls_iter)
    .n4m_method_fit_object(formula, cl, d, res, ncomp,
                                method = "robust_pls",
                                extra = list(huber_k = huber_k,
                                              max_irls_iter = max_irls_iter))
}

#' Ridge PLS — formula entry point.
#'
#' @inheritParams pls
#' @return A `n4m_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param ridge_lambda Numeric >= 0. Ridge regularisation strength.
#' @export
ridge_pls <- function(formula, data, ncomp = 2L, ridge_lambda = 1.0,
                       na.action = stats::na.omit) {
    cl <- match.call()
    d <- .n4m_method_design(formula, data, na.action)
    res <- ridge_pls_fit(d$X, d$y, ncomp, ridge_lambda)
    .n4m_method_fit_object(formula, cl, d, res, ncomp,
                                method = "ridge_pls",
                                extra = list(ridge_lambda = ridge_lambda))
}

#' Continuum regression — formula entry point.
#'
#' @inheritParams pls
#' @return A `n4m_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param tau Numeric in [0, 1]. Continuum regression mixing parameter.
#' @export
continuum_regression <- function(formula, data, ncomp = 2L, tau = 0.5,
                                   na.action = stats::na.omit) {
    cl <- match.call()
    d <- .n4m_method_design(formula, data, na.action)
    res <- continuum_regression_fit(d$X, d$y, ncomp, tau)
    .n4m_method_fit_object(formula, cl, d, res, ncomp,
                                method = "continuum_regression",
                                extra = list(tau = tau))
}

#' Recursive PLS — formula entry point.
#'
#' Recursive PLS stores in-sample predictions rather than a reusable
#' coefficient model. `predict()` therefore accepts only the fitted
#' training design.
#'
#' @inheritParams pls
#' @param window_size Integer moving-window size.
#' @export
recursive_pls <- function(formula, data, ncomp = 2L, window_size = 50L,
                           na.action = stats::na.omit) {
    cl <- match.call()
    d <- .n4m_method_design(formula, data, na.action)
    res <- recursive_pls_fit(d$X, d$y, n_components = ncomp,
                              window_size = as.integer(window_size))
    out <- list(
        formula = formula,
        call = cl,
        terms = d$terms,
        xlevels = d$xlevels,
        response_name = d$response_name,
        n_features_in = ncol(d$X),
        ncomp = as.integer(ncomp),
        method = "recursive_pls",
        predictions = as.numeric(res$predictions),
        window_size = as.integer(window_size)
    )
    class(out) <- c("n4m_recursive_fit", "n4m_fit_base")
    out
}

#' @export
predict.n4m_recursive_fit <- function(object, newdata = NULL, ...) {
    if (is.null(newdata)) {
        stop("predict.recursive_pls requires the fitted training frame")
    }
    X <- .n4m_method_newdata(object, newdata)
    if (nrow(X) != length(object$predictions)) {
        stop("recursive_pls is in-sample only; newdata row count differs")
    }
    object$predictions
}

#' Bagging PLS — formula entry point.
#'
#' @inheritParams pls
#' @return A `n4m_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param n_estimators Integer >= 1. Number of bootstrap / boosting / random-subspace estimators.
#' @param seed Integer. Random seed for reproducibility.
#' @export
bagging_pls <- function(formula, data, ncomp = 2L,
                         n_estimators = 50L, seed = 0L,
                         na.action = stats::na.omit) {
    cl <- match.call()
    d <- .n4m_method_design(formula, data, na.action)
    res <- bagging_pls_fit(d$X, d$y, ncomp, n_estimators, seed)
    .n4m_method_fit_object(formula, cl, d, res, ncomp,
                                method = "bagging_pls",
                                extra = list(n_estimators = n_estimators))
}

#' Boosting PLS — formula entry point.
#'
#' @inheritParams pls
#' @return A `n4m_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param n_estimators Integer >= 1. Number of bootstrap / boosting / random-subspace estimators.
#' @param learning_rate Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
boosting_pls <- function(formula, data, ncomp = 2L,
                          n_estimators = 50L, learning_rate = 0.1,
                          na.action = stats::na.omit) {
    cl <- match.call()
    d <- .n4m_method_design(formula, data, na.action)
    res <- boosting_pls_fit(d$X, d$y, ncomp, n_estimators, learning_rate)
    .n4m_method_fit_object(formula, cl, d, res, ncomp,
                                method = "boosting_pls",
                                extra = list(n_estimators = n_estimators,
                                              learning_rate = learning_rate))
}

#' Random-subspace PLS — formula entry point.
#'
#' @inheritParams pls
#' @return A `n4m_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param n_estimators Integer >= 1. Number of bootstrap / boosting / random-subspace estimators.
#' @param features_per_subspace Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param seed Integer. Random seed for reproducibility.
#' @export
random_subspace_pls <- function(formula, data, ncomp = 2L,
                                  n_estimators = 50L,
                                  features_per_subspace = 10L,
                                  seed = 0L,
                                  na.action = stats::na.omit) {
    cl <- match.call()
    d <- .n4m_method_design(formula, data, na.action)
    res <- random_subspace_pls_fit(d$X, d$y, ncomp, n_estimators,
                                     features_per_subspace, seed)
    .n4m_method_fit_object(formula, cl, d, res, ncomp,
                                method = "random_subspace_pls",
                                extra = list(n_estimators = n_estimators,
                                              features_per_subspace =
                                                  features_per_subspace))
}

#' O2-PLS — formula entry point (uses n_predictive for component count).
#'
#' @inheritParams pls
#' @return A `n4m_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param n_predictive Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param n_x_orthogonal Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param n_y_orthogonal Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
o2pls <- function(formula, data, n_predictive = 2L,
                   n_x_orthogonal = 1L, n_y_orthogonal = 1L,
                   na.action = stats::na.omit) {
    cl <- match.call()
    d <- .n4m_method_design(formula, data, na.action)
    res <- o2pls_fit(d$X, d$y, n_predictive, n_x_orthogonal, n_y_orthogonal)
    .n4m_method_fit_object(formula, cl, d, res, n_predictive,
                                method = "o2pls",
                                extra = list(n_predictive = n_predictive,
                                              n_x_orthogonal = n_x_orthogonal,
                                              n_y_orthogonal = n_y_orthogonal))
}

#' Missing-aware NIPALS — formula entry point.
#'
#' @inheritParams pls
#' @return A `n4m_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @export
missing_aware_nipals <- function(formula, data, ncomp = 2L,
                                   na.action = stats::na.pass) {
    cl <- match.call()
    d <- .n4m_method_design(formula, data, na.action)
    res <- missing_aware_nipals_fit(d$X, d$y, ncomp)
    .n4m_method_fit_object(formula, cl, d, res, ncomp,
                                method = "missing_aware_nipals")
}
