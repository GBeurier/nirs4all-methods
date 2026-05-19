# SPDX-License-Identifier: CECILL-2.1
#
# Compatibility facade for the common `mdatools::pls(x, y, ...)` idiom.
# This does not import mdatools; it mirrors the matrix-oriented signature
# used by NIRS/chemometrics workflows.

#' PLS regression with an `mdatools::pls`-like matrix signature.
#'
#' @param x Numeric predictor matrix.
#' @param y Numeric response vector or matrix.
#' @param ncomp Maximum number of components.
#' @param center Logical; center predictors and response.
#' @param scale Logical; standardize predictors and response.
#' @param cv `NULL`, an integer number of folds, or a list of assessment
#'   row indices.
#' @param exclcols Optional predictor columns to exclude.
#' @param exclrows Optional rows to exclude before fitting.
#' @param x.test Optional external test predictor matrix.
#' @param y.test Optional external test response vector or matrix.
#' @param method One of `"simpls"`, `"cppls"`, `"kernelpls"`,
#'   `"widekernelpls"`, `"oscorespls"`, `"nipals"` or `"pcr"`.
#' @param info Free-form model label.
#' @param ncomp.selcrit Stored for compatibility; currently not used.
#' @param lim.type Stored for compatibility; currently not used.
#' @param alpha Stored for compatibility with mdatools result objects.
#' @param gamma CPPLS gamma when `method = "cppls"`.
#' @param cv.scope Stored for compatibility; currently not used.
#' @param prep Optional preprocessing function applied to `x` and `x.test`.
#' @param fit_components Logical; fit component prefixes 1:ncomp.
#' @param ... Reserved for compatibility.
#' @return A `pls4all_mdatools_pls` object.
#' @export
pls_mdatools <- function(x, y,
                          ncomp = min(nrow(x) - 1L, ncol(x), 20L),
                          center = TRUE, scale = FALSE, cv = NULL,
                          exclcols = NULL, exclrows = NULL,
                          x.test = NULL, y.test = NULL,
                          method = c("simpls", "cppls", "kernelpls",
                                     "widekernelpls", "oscorespls",
                                     "nipals", "pcr"),
                          info = "", ncomp.selcrit = "min",
                          lim.type = "ddmoments", alpha = 0.05,
                          gamma = 0.5, cv.scope = "local",
                          prep = NULL, fit_components = TRUE, ...) {
    cl <- match.call()
    method <- match.arg(method)
    X <- as.matrix(x)
    Y <- .pls4all_as_response_matrix(y)
    storage.mode(X) <- "double"
    storage.mode(Y) <- "double"
    if (nrow(X) != nrow(Y)) {
        stop(sprintf("nrow(x) (%d) must equal nrow(y) (%d)",
                     nrow(X), nrow(Y)))
    }
    if (!is.null(exclrows)) {
        keep <- setdiff(seq_len(nrow(X)), as.integer(exclrows))
        X <- X[keep, , drop = FALSE]
        Y <- Y[keep, , drop = FALSE]
    }
    if (!is.null(exclcols)) {
        X <- X[, -as.integer(exclcols), drop = FALSE]
    }
    if (!is.null(prep)) {
        if (!is.function(prep)) {
            stop("prep must be a function when supplied")
        }
        X <- as.matrix(prep(X))
    }
    if (ncomp > min(nrow(X) - 1L, ncol(X))) {
        ncomp <- min(nrow(X) - 1L, ncol(X))
    }

    validation <- if (is.null(cv)) "none" else "CV"
    obj <- .pls4all_mvr_fit_object(
        X = X, Y = Y, formula = NULL, terms = NULL, xlevels = NULL,
        call = cl, ncomp = ncomp, method = method,
        model = if (identical(method, "pcr")) "pcr" else "plsr",
        center = center, scale = scale, validation = validation,
        segments = cv %||% 10L, fit_components = fit_components,
        gamma = gamma, sparsity_lambda = 0.0,
        response_names = colnames(Y) %||% "y",
        predictor_names = colnames(X) %||% paste0("x", seq_len(ncol(X))),
        class_prefix = "pls4all_mdatools_pls")

    obj$info <- info
    obj$ncomp.selcrit <- ncomp.selcrit
    obj$lim.type <- lim.type
    obj$alpha <- alpha
    obj$cv.scope <- cv.scope
    obj$exclcols <- exclcols
    obj$exclrows <- exclrows
    obj$calres <- list(
        y.pred = obj$fitted.values,
        residuals = obj$residuals,
        rmse = RMSEP(obj, estimate = "train")$val
    )
    if (!is.null(obj$cv_predictions)) {
        obj$cvres <- list(
            y.pred = .pls4all_drop_response_dim(
                obj$cv_predictions[, , obj$ncomp]),
            rmse = RMSEP(obj, estimate = "CV")$val
        )
    }
    if (!is.null(x.test)) {
        Xtest <- as.matrix(x.test)
        if (!is.null(exclcols)) {
            Xtest <- Xtest[, -as.integer(exclcols), drop = FALSE]
        }
        if (!is.null(prep)) {
            Xtest <- as.matrix(prep(Xtest))
        }
        ypred <- predict(obj, newdata = Xtest)
        obj$testres <- list(y.pred = ypred)
        if (!is.null(y.test)) {
            Ytest <- .pls4all_as_response_matrix(y.test)
            residuals <- Ytest - .pls4all_as_response_matrix(ypred)
            obj$testres$residuals <- .pls4all_drop_response_dim(residuals)
            obj$testres$rmse <- sqrt(colMeans(residuals ^ 2))
        }
    }
    obj
}

#' @export
predict.pls4all_mdatools_pls <- function(object, newdata = NULL,
                                          x = NULL, ncomp = object$ncomp,
                                          ...) {
    if (!is.null(x) && is.null(newdata)) {
        newdata <- x
    }
    ncomp <- as.integer(ncomp)
    if (any(ncomp < 1L | ncomp > object$ncomp)) {
        stop("ncomp must be between 1 and the fitted component count")
    }
    .pls4all_mvr_predict_matrix(object, newdata, ncomp)
}

#' @export
print.pls4all_mdatools_pls <- function(x, ...) {
    cat("pls4all mdatools-compatible PLS model\n")
    cat("Method:      ", x$method, "\n", sep = "")
    cat("Components:  ", x$ncomp, "\n", sep = "")
    cat("Predictors:  ", x$n_features_in, "\n", sep = "")
    if (nzchar(x$info)) cat("Info:        ", x$info, "\n", sep = "")
    invisible(x)
}
