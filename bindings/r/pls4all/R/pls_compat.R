# SPDX-License-Identifier: CECILL-2.1
#
# Compatibility facade for the R `pls` package idiom. This is not an
# import of `pls`: it provides familiar function signatures and S3
# methods backed by pls4all.

#' Partial least squares regression with a `pls::plsr`-like signature.
#'
#' @param formula A two-sided model formula.
#' @param ncomp Number of latent components.
#' @param data Data frame containing formula variables.
#' @param subset Optional row subset.
#' @param na.action Missing-value action.
#' @param method One of `"kernelpls"`, `"widekernelpls"`, `"simpls"`,
#'   `"oscorespls"`, `"cppls"` or `"nipals"`.
#' @param scale Logical; standardize predictors and response.
#' @param validation `"none"` or `"CV"`.
#' @param segments Number of contiguous CV segments, or a list of row
#'   indices used as assessment folds.
#' @param center Logical; center predictors and response.
#' @param fit_components Logical; fit component prefixes 1:ncomp so
#'   `predict(..., ncomp = k)` and RMSEP/R2 by component are available.
#' @param gamma CPPLS gamma, used when `method = "cppls"`.
#' @param sparsity_lambda Sparse-SIMPLS soft threshold used for
#'   `method = "simpls"`; default 0 gives a dense SIMPLS-compatible path.
#' @param ... Reserved for compatibility with `pls::plsr`.
#' @return A `pls4all_mvr` object.
#' @export
plsr <- function(formula, ncomp = 2L, data, subset,
                  na.action = stats::na.omit,
                  method = c("kernelpls", "widekernelpls", "simpls",
                             "oscorespls", "cppls", "nipals"),
                  scale = FALSE, validation = c("none", "CV"),
                  segments = 10L, center = TRUE,
                  fit_components = TRUE, gamma = 0.5,
                  sparsity_lambda = 0.0, ...) {
    method <- match.arg(method)
    validation <- match.arg(validation)
    cl <- match.call()
    subset_expr <- if (missing(subset)) NULL else substitute(subset)
    d <- .pls4all_formula_xy(formula, data, subset_expr, na.action)
    .pls4all_mvr_fit_object(
        X = d$X, Y = d$Y, formula = formula, terms = d$terms,
        xlevels = d$xlevels, call = cl, ncomp = ncomp,
        method = method, model = "plsr", center = center, scale = scale,
        validation = validation, segments = segments,
        fit_components = fit_components, gamma = gamma,
        sparsity_lambda = sparsity_lambda,
        response_names = d$response_names,
        predictor_names = d$predictor_names,
        class_prefix = "pls4all_plsr")
}

#' Principal component regression with a `pls::pcr`-like signature.
#'
#' @inheritParams plsr
#' @return A `pls4all_mvr` object.
#' @export
pcr <- function(formula, ncomp = 2L, data, subset,
                 na.action = stats::na.omit,
                 method = c("svdpc", "pcr"),
                 scale = FALSE, validation = c("none", "CV"),
                 segments = 10L, center = TRUE,
                 fit_components = TRUE, ...) {
    method <- match.arg(method)
    validation <- match.arg(validation)
    cl <- match.call()
    subset_expr <- if (missing(subset)) NULL else substitute(subset)
    d <- .pls4all_formula_xy(formula, data, subset_expr, na.action)
    .pls4all_mvr_fit_object(
        X = d$X, Y = d$Y, formula = formula, terms = d$terms,
        xlevels = d$xlevels, call = cl, ncomp = ncomp,
        method = "pcr", model = "pcr", center = center, scale = scale,
        validation = validation, segments = segments,
        fit_components = fit_components, gamma = 0.5,
        sparsity_lambda = 0.0,
        response_names = d$response_names,
        predictor_names = d$predictor_names,
        class_prefix = "pls4all_pcr")
}

#' Generic multivariate regression entry point.
#'
#' @inheritParams plsr
#' @param model `"plsr"` or `"pcr"`.
#' @return A `pls4all_mvr` object.
#' @export
mvr <- function(formula, ncomp = 2L, data, subset,
                 na.action = stats::na.omit,
                 method = c("kernelpls", "widekernelpls", "simpls",
                            "oscorespls", "cppls", "nipals", "pcr"),
                 scale = FALSE, validation = c("none", "CV"),
                 segments = 10L, center = TRUE,
                 model = c("plsr", "pcr"),
                 fit_components = TRUE, gamma = 0.5,
                 sparsity_lambda = 0.0, ...) {
    method <- match.arg(method)
    model <- match.arg(model)
    if (identical(method, "pcr")) model <- "pcr"
    validation <- match.arg(validation)
    cl <- match.call()
    subset_expr <- if (missing(subset)) NULL else substitute(subset)
    d <- .pls4all_formula_xy(formula, data, subset_expr, na.action)
    .pls4all_mvr_fit_object(
        X = d$X, Y = d$Y, formula = formula, terms = d$terms,
        xlevels = d$xlevels, call = cl, ncomp = ncomp,
        method = method, model = model, center = center, scale = scale,
        validation = validation, segments = segments,
        fit_components = fit_components, gamma = gamma,
        sparsity_lambda = sparsity_lambda,
        response_names = d$response_names,
        predictor_names = d$predictor_names,
        class_prefix = "pls4all_mvr")
}

.pls4all_mvr_fit_object <- function(X, Y, formula = NULL, terms = NULL,
                                     xlevels = NULL, call, ncomp,
                                     method, model, center, scale,
                                     validation, segments, fit_components,
                                     gamma, sparsity_lambda,
                                     response_names, predictor_names,
                                     class_prefix) {
    ncomp <- as.integer(ncomp)
    if (length(ncomp) != 1L || is.na(ncomp) || ncomp < 1L) {
        stop("ncomp must be a positive integer")
    }
    if (ncomp > min(nrow(X) - 1L, ncol(X))) {
        ncomp <- min(nrow(X) - 1L, ncol(X))
    }
    Y <- .pls4all_as_response_matrix(Y)
    main_fit <- .pls4all_fit_backend(X, Y, ncomp, method = method,
                                     center = center, scale = scale,
                                     gamma = gamma,
                                     sparsity_lambda = sparsity_lambda,
                                     model = model)
    component_fits <- if (isTRUE(fit_components)) {
        .pls4all_component_fits(X, Y, ncomp, method = method,
                                center = center, scale = scale,
                                gamma = gamma,
                                sparsity_lambda = sparsity_lambda,
                                model = model)
    } else {
        stats::setNames(list(main_fit), as.character(ncomp))
    }
    fitted_array <- .pls4all_predict_components(component_fits, X)
    cv_array <- NULL
    if (identical(validation, "CV")) {
        cv_array <- .pls4all_cv_predictions(X, Y, ncomp,
                                            method = method,
                                            center = center,
                                            scale = scale,
                                            gamma = gamma,
                                            sparsity_lambda = sparsity_lambda,
                                            model = model,
                                            segments = segments)
    }
    fitted_pos <- .pls4all_component_positions(fitted_array, ncomp)
    out <- list(
        call = call,
        formula = formula,
        terms = terms,
        xlevels = xlevels,
        ncomp = ncomp,
        method = method,
        model = model,
        center = isTRUE(center),
        scale = isTRUE(scale),
        validation = validation,
        segments = segments,
        fit = main_fit,
        component_fits = component_fits,
        fitted_array = fitted_array,
        fitted.values = .pls4all_drop_response_dim(fitted_array[, , fitted_pos]),
        residuals = .pls4all_drop_response_dim(Y - fitted_array[, , fitted_pos]),
        response = Y,
        response_names = response_names,
        predictor_names = predictor_names,
        n_features_in = ncol(X),
        cv_predictions = cv_array,
        coefficients = tryCatch(.pls4all_backend_coef(main_fit),
                                error = function(e) NULL)
    )
    class(out) <- unique(c(class_prefix, "pls4all_mvr", "mvr"))
    out
}

.pls4all_mvr_predict_matrix <- function(object, newdata, ncomp) {
    if (is.null(newdata)) {
        pos <- .pls4all_component_positions(object$fitted_array, ncomp)
        arr <- object$fitted_array[, , pos, drop = FALSE]
    } else {
        X <- if (!is.null(object$terms)) {
            .pls4all_newdata_from_terms(object, newdata)
        } else {
            X <- as.matrix(newdata)
            if (ncol(X) != object$n_features_in) {
                stop(sprintf("newdata has %d features, but model was fitted with %d",
                             ncol(X), object$n_features_in))
            }
            X
        }
        arr <- array(NA_real_,
                     dim = c(nrow(X), ncol(object$response), length(ncomp)))
        for (i in seq_along(ncomp)) {
            fit <- object$component_fits[[as.character(as.integer(ncomp[i]))]]
            if (is.null(fit)) {
                stop("requested ncomp was not fitted; refit with fit_components = TRUE")
            }
            arr[, , i] <- .pls4all_as_response_matrix(
                .pls4all_backend_predict(fit, X))
        }
    }
    if (length(ncomp) == 1L) {
        return(.pls4all_drop_response_dim(arr[, , 1L]))
    }
    arr
}

#' @export
predict.pls4all_mvr <- function(object, newdata = NULL, ncomp = object$ncomp,
                                  type = c("response"), ...) {
    type <- match.arg(type)
    ncomp <- as.integer(ncomp)
    if (any(ncomp < 1L | ncomp > object$ncomp)) {
        stop("ncomp must be between 1 and the fitted component count")
    }
    .pls4all_mvr_predict_matrix(object, newdata, ncomp)
}

#' @export
coef.pls4all_mvr <- function(object, ncomp = object$ncomp, ...) {
    ncomp <- as.integer(ncomp)
    fit <- object$component_fits[[as.character(ncomp)]]
    if (is.null(fit)) {
        stop("requested ncomp was not fitted; refit with fit_components = TRUE")
    }
    .pls4all_backend_coef(fit)
}

#' @export
print.pls4all_mvr <- function(x, ...) {
    cat("pls4all ", x$model, " fit (method=", x$method,
        ", ncomp=", x$ncomp, ")\n", sep = "")
    cat("Predictors:  ", x$n_features_in, "\n", sep = "")
    cat("Responses:   ", ncol(x$response), "\n", sep = "")
    cat("Validation:  ", x$validation, "\n", sep = "")
    invisible(x)
}

#' @export
summary.pls4all_mvr <- function(object, ...) {
    out <- list(
        method = object$method,
        model = object$model,
        ncomp = object$ncomp,
        n_features_in = object$n_features_in,
        responses = object$response_names,
        train_rmsep = .pls4all_metric_by_component(
            object$fitted_array, object$response, "rmsep")[,
                .pls4all_component_positions(object$fitted_array, object$ncomp)],
        validation = object$validation,
        call = object$call
    )
    class(out) <- "summary.pls4all_mvr"
    out
}

#' @export
print.summary.pls4all_mvr <- function(x, ...) {
    cat("pls4all PLS-compatible model summary\n")
    cat("------------------------------------\n")
    cat("Model:       ", x$model, "\n", sep = "")
    cat("Method:      ", x$method, "\n", sep = "")
    cat("Components:  ", x$ncomp, "\n", sep = "")
    cat("Predictors:  ", x$n_features_in, "\n", sep = "")
    cat("Validation:  ", x$validation, "\n", sep = "")
    cat("Train RMSEP: ", paste(sprintf("%.4f", x$train_rmsep),
                            collapse = ", "), "\n", sep = "")
    invisible(x)
}

#' Mean squared error of prediction.
#' @param object A `pls4all_mvr` object.
#' @param estimate `"train"` or `"CV"`.
#' @param ... Reserved for compatibility.
#' @export
MSEP <- function(object, ...) UseMethod("MSEP")

#' @export
MSEP.pls4all_mvr <- function(object, estimate = c("train", "CV"), ...) {
    estimate <- match.arg(estimate)
    arr <- if (identical(estimate, "CV") && !is.null(object$cv_predictions)) {
        object$cv_predictions
    } else {
        object$fitted_array
    }
    .pls4all_mvr_val(object, arr, estimate, "mse")
}

#' Root mean squared error of prediction.
#' @param object A `pls4all_mvr` object.
#' @param estimate `"train"` or `"CV"`.
#' @param ... Reserved for compatibility.
#' @export
RMSEP <- function(object, ...) UseMethod("RMSEP")

#' @export
RMSEP.pls4all_mvr <- function(object, estimate = c("train", "CV"), ...) {
    estimate <- match.arg(estimate)
    arr <- if (identical(estimate, "CV") && !is.null(object$cv_predictions)) {
        object$cv_predictions
    } else {
        object$fitted_array
    }
    .pls4all_mvr_val(object, arr, estimate, "rmsep")
}

#' Coefficient of determination by component.
#' @param object A `pls4all_mvr` object.
#' @param estimate `"train"` or `"CV"`.
#' @param ... Reserved for compatibility.
#' @export
R2 <- function(object, ...) UseMethod("R2")

#' @export
R2.pls4all_mvr <- function(object, estimate = c("train", "CV"), ...) {
    estimate <- match.arg(estimate)
    arr <- if (identical(estimate, "CV") && !is.null(object$cv_predictions)) {
        object$cv_predictions
    } else {
        object$fitted_array
    }
    .pls4all_mvr_val(object, arr, estimate, "r2")
}

.pls4all_mvr_val <- function(object, pred_array, estimate, metric) {
    vals <- .pls4all_metric_by_component(pred_array, object$response, metric)
    comps <- sub("^Comp ", "", colnames(vals))
    comps_int <- suppressWarnings(as.integer(comps))
    if (any(is.na(comps_int))) {
        comps_int <- seq_len(ncol(vals))
    }
    val <- array(vals, dim = c(1L, nrow(vals), ncol(vals)),
                 dimnames = list(estimate, rownames(vals), colnames(vals)))
    out <- list(val = val, comps = comps_int,
                type = metric, estimate = estimate, call = match.call())
    class(out) <- c("pls4all_mvrVal", "mvrVal")
    out
}

#' @export
print.pls4all_mvrVal <- function(x, ...) {
    metric <- toupper(x$type)
    cat(metric, " estimates\n", sep = "")
    print(x$val[1L, , , drop = FALSE])
    invisible(x)
}

#' Select the number of components.
#' @param object A `pls4all_mvr` object.
#' @param method Currently `"min"` selects the component with minimum RMSEP.
#' @param estimate `"CV"` if available, otherwise `"train"`.
#' @param ... Reserved for compatibility.
#' @export
selectNcomp <- function(object, method = c("min"), estimate = c("CV", "train"),
                         ...) {
    method <- match.arg(method)
    estimate <- match.arg(estimate)
    if (identical(estimate, "CV") && is.null(object$cv_predictions)) {
        estimate <- "train"
    }
    vals <- RMSEP(object, estimate = estimate)$val[1L, , , drop = FALSE]
    per_comp <- apply(vals, 3L, mean, na.rm = TRUE)
    as.integer(which.min(per_comp))
}
