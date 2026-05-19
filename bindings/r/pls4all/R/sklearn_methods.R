# SPDX-License-Identifier: CECILL-2.1
#
# Formula+S3 tier-2 wrappers for the MethodResult-based tier-1 fits in
# methods.R. Each algorithm gets its own constructor function returning
# an S3 object of class `pls4all_method_fit`. Predict is a single
# generic that applies the model's stored coefficients / centering.
#
# C ABI prediction convention (cpp/src/core/model.cpp::fill_prediction):
#   Y_pred = (X - x_mean) @ coef + y_mean
# i.e. coefficients are already in the ORIGINAL X/Y scale; x_scale was
# folded in at fit time and must NOT be re-applied. MB-PLS overrides
# this via a custom intercept stored alongside x_mean / x_scale.

# Internal: extract the (X, y) design from a formula + data and remember
# what we need for predict() on newdata.
.pls4all_method_design <- function(formula, data, na.action) {
    if (!inherits(formula, "formula"))
        stop("formula must be a two-sided formula (e.g. `y ~ .`)")
    mf <- stats::model.frame(formula, data = data, na.action = na.action)
    tt <- stats::terms(mf)
    if (attr(tt, "response") == 0L)
        stop("formula must have a response on the left-hand side")
    y <- stats::model.response(mf, type = "numeric")
    if (is.null(y))
        stop("response is missing or non-numeric; ",
             "regression only")
    X <- stats::model.matrix(tt, mf)
    icpt <- which(colnames(X) == "(Intercept)")
    if (length(icpt) > 0L) X <- X[, -icpt, drop = FALSE]
    if (ncol(X) < 1L)
        stop("design matrix has zero predictors after dropping intercept")
    list(X = X, y = y, terms = tt,
         xlevels = stats::.getXlevels(tt, mf),
         response_name = deparse(formula[[2L]]))
}

# Internal: build an S3 fit object from a MethodResult list.
# `method` describes the algorithm for print / summary; `extra` lets the
# caller store algorithm-specific state (e.g. block_sizes for MB-PLS).
.pls4all_method_fit_object <- function(formula, call, design, res, ncomp,
                                        method, extra = list(),
                                        use_intercept = FALSE) {
    out <- list(
        formula        = formula,
        call           = call,
        terms          = design$terms,
        xlevels        = design$xlevels,
        response_name  = design$response_name,
        n_features_in  = ncol(design$X),
        ncomp          = as.integer(ncomp),
        method         = method,
        coefficients   = res$coefficients,
        predictions    = res$predictions,
        x_mean         = res$x_mean,
        y_mean         = res$y_mean,
        x_scale        = res$x_scale,   # may be NULL
        intercept      = res$intercept,  # MB-PLS / GLM
        rmse           = res$rmse,
        block_weights  = res$block_weights,
        use_intercept  = isTRUE(use_intercept),
        extra          = extra
    )
    class(out) <- c("pls4all_method_fit", "pls4all_fit_base")
    out
}

# Internal: re-build the design for newdata, matching the formula.
.pls4all_method_newdata <- function(object, newdata) {
    tt <- stats::delete.response(object$terms)
    mf <- stats::model.frame(tt, newdata, xlev = object$xlevels,
                              na.action = stats::na.pass)
    X <- stats::model.matrix(tt, mf)
    icpt <- which(colnames(X) == "(Intercept)")
    if (length(icpt) > 0L) X <- X[, -icpt, drop = FALSE]
    if (ncol(X) != object$n_features_in)
        stop(sprintf("newdata has %d features, but model was fitted with %d",
                      ncol(X), object$n_features_in))
    X
}

#' Sparse SIMPLS — formula entry point.
#'
#' @inheritParams pls
#' @return A `pls4all_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param sparsity_lambda Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
sparse_pls <- function(formula, data, ncomp = 2L, sparsity_lambda = 0.05,
                        na.action = stats::na.omit) {
    cl <- match.call()
    d <- .pls4all_method_design(formula, data, na.action)
    res <- sparse_simpls_fit(d$X, d$y, n_components = ncomp,
                              sparsity_lambda = sparsity_lambda)
    .pls4all_method_fit_object(formula, cl, d, res, ncomp,
                                method = "sparse_simpls",
                                extra = list(sparsity_lambda = sparsity_lambda))
}

#' Canonical Powered PLS — formula entry point.
#'
#' @inheritParams pls
#' @return A `pls4all_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param gamma Numeric. CPPLS / kernel band parameter.
#' @export
cppls <- function(formula, data, ncomp = 2L, gamma = 0.5,
                   na.action = stats::na.omit) {
    cl <- match.call()
    d <- .pls4all_method_design(formula, data, na.action)
    res <- cppls_fit(d$X, d$y, n_components = ncomp, gamma = gamma)
    .pls4all_method_fit_object(formula, cl, d, res, ncomp,
                                method = "cppls",
                                extra = list(gamma = gamma))
}

#' Domain-invariant PLS -- formula entry point.
#' @param X_target Numeric matrix for the target domain.
#' @param di_lambda Numeric DI-PLS penalty.
#' @inheritParams pls
#' @export
di_pls <- function(formula, data, ncomp = 2L, X_target,
                   di_lambda = 1.0, na.action = stats::na.omit) {
    cl <- match.call()
    if (missing(X_target))
        stop("di_pls() requires an `X_target` matrix argument")
    d <- .pls4all_method_design(formula, data, na.action)
    res <- di_pls_fit(d$X, d$y, n_components = ncomp,
                       X_target = X_target, di_lambda = di_lambda)
    .pls4all_method_fit_object(formula, cl, d, res, ncomp,
                                method = "di_pls",
                                extra = list(di_lambda = di_lambda))
}

#' Sample-weighted PLS — formula entry point.
#' @param weights Numeric vector of length nrow(data) with sample weights.
#' @inheritParams pls
#' @export
weighted_pls <- function(formula, data, ncomp = 2L, weights,
                          na.action = stats::na.omit) {
    cl <- match.call()
    if (missing(weights))
        stop("weighted_pls() requires a `weights` numeric vector argument")
    d <- .pls4all_method_design(formula, data, na.action)
    if (length(weights) != nrow(d$X))
        stop(sprintf("length(weights) (%d) must equal nrow(data) after na.action (%d)",
                      length(weights), nrow(d$X)))
    res <- weighted_pls_fit(d$X, d$y, n_components = ncomp,
                             sample_weights = weights)
    .pls4all_method_fit_object(formula, cl, d, res, ncomp,
                                method = "weighted_pls",
                                extra = list(weights = weights))
}

#' Multi-block PLS — formula entry point.
#' @param block_sizes Integer vector summing to the number of predictors.
#' @inheritParams pls
#' @export
mb_pls <- function(formula, data, ncomp = 2L, block_sizes,
                    na.action = stats::na.omit) {
    cl <- match.call()
    if (missing(block_sizes))
        stop("mb_pls() requires a `block_sizes` integer vector argument")
    d <- .pls4all_method_design(formula, data, na.action)
    res <- mb_pls_fit(d$X, d$y, n_components = ncomp,
                       block_sizes = block_sizes)
    .pls4all_method_fit_object(formula, cl, d, res, ncomp,
                                method = "mb_pls",
                                extra = list(block_sizes = block_sizes),
                                use_intercept = TRUE)
}

#' PLS-GLM — formula entry point. Default is Gaussian; set
#' `family = "poisson"` for Poisson IRLS.
#'
#' @inheritParams pls
#' @return A `pls4all_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @param family Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
pls_glm <- function(formula, data, ncomp = 2L, family = "gaussian",
                     na.action = stats::na.omit) {
    cl <- match.call()
    family <- match.arg(family, c("gaussian", "poisson"))
    d <- .pls4all_method_design(formula, data, na.action)
    res <- pls_glm_fit(d$X, d$y, n_components = ncomp,
                        poisson = identical(family, "poisson"))
    .pls4all_method_fit_object(formula, cl, d, res, ncomp,
                                method = paste0("pls_glm_", family),
                                extra = list(family = family),
                                use_intercept = TRUE)
}

#' MIR-PLS — formula entry point.
#'
#' @inheritParams pls
#' @return A `pls4all_method_fit` object carrying the fitted model,
#'   in-sample predictions, training RMSE, and method-specific
#'   metadata. Use `predict()` for inference and `coef()` to
#'   extract regression coefficients.
#' @export
mir_pls <- function(formula, data, ncomp = 2L,
                     na.action = stats::na.omit) {
    cl <- match.call()
    d <- .pls4all_method_design(formula, data, na.action)
    res <- mir_pls_fit(d$X, d$y, n_components = ncomp)
    .pls4all_method_fit_object(formula, cl, d, res, ncomp,
                                method = "mir_pls")
}

# ---- S3 methods on the pls4all_method_fit class ----------------------

#' Predict from a MethodResult-based pls4all fit.
#' @param object Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param newdata Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param ... Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
predict.pls4all_method_fit <- function(object, newdata = NULL, ...) {
    if (is.null(newdata))
        stop("predict.pls4all_method_fit requires newdata.")
    X <- .pls4all_method_newdata(object, newdata)
    coef <- object$coefficients   # (p, q)
    if (object$use_intercept) {
        # MB-PLS / PLS-GLM: coefficients are in ORIGINAL X scale,
        # `intercept` already folds in y_mean - x_mean @ coef.
        intercept <- object$intercept
        if (length(intercept) == 1L) {
            preds <- X %*% coef + as.numeric(intercept)
        } else {
            preds <- sweep(X %*% coef, 2L, as.numeric(intercept), "+")
        }
    } else {
        # Standard convention: (X - x_mean) @ coef + y_mean.
        Xc <- sweep(X, 2L, as.numeric(object$x_mean), "-")
        ym <- as.numeric(object$y_mean)
        if (length(ym) == 1L) {
            preds <- Xc %*% coef + ym
        } else {
            preds <- sweep(Xc %*% coef, 2L, ym, "+")
        }
    }
    if (ncol(preds) == 1L) preds <- preds[, 1L]
    preds
}

#' @param object Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param ... Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
coef.pls4all_method_fit <- function(object, ...) object$coefficients

#' @param x Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param ... Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
print.pls4all_method_fit <- function(x, ...) {
    cat("pls4all method fit (", x$method, ", ncomp=", x$ncomp, ")\n",
        sep = "")
    cat("Call:\n  ")
    print(x$call)
    cat("Response:    ", x$response_name, "\n", sep = "")
    cat("Predictors:  ", x$n_features_in, "\n", sep = "")
    if (!is.null(x$rmse))
        cat("Train RMSE:  ", sprintf("%.4f", x$rmse), "\n", sep = "")
    invisible(x)
}

#' @param object Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param ... Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
summary.pls4all_method_fit <- function(object, ...) {
    out <- list(
        method        = object$method,
        ncomp         = object$ncomp,
        n_features_in = object$n_features_in,
        rmse          = object$rmse,
        call          = object$call,
        extra         = object$extra
    )
    class(out) <- "summary.pls4all_method_fit"
    out
}

#' @param x Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @param ... Method-specific parameter. See the underlying `*_fit()` function for the exact semantics.
#' @export
print.summary.pls4all_method_fit <- function(x, ...) {
    cat("pls4all method-fit summary\n")
    cat("--------------------------\n")
    cat("Method:      ", x$method, "\n", sep = "")
    cat("Components:  ", x$ncomp, "\n", sep = "")
    cat("Predictors:  ", x$n_features_in, "\n", sep = "")
    if (!is.null(x$rmse))
        cat("Train RMSE:  ", sprintf("%.4f", x$rmse), "\n", sep = "")
    if (length(x$extra) > 0L) {
        cat("Parameters:\n")
        for (nm in names(x$extra)) {
            cat("  ", nm, " = ", deparse(x$extra[[nm]]), "\n", sep = "")
        }
    }
    invisible(x)
}
