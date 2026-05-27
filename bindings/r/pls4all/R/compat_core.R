# SPDX-License-Identifier: CECILL-2.1
#
# Shared helpers for R ecosystem compatibility facades. These helpers are
# intentionally package-local: they expose pls/plsr/mdatools-like entry
# points without depending on the upstream R packages.

`%||%` <- function(x, y) if (is.null(x)) y else x

.n4m_as_response_matrix <- function(y) {
    if (is.null(dim(y))) {
        y <- matrix(as.numeric(y), ncol = 1L)
    } else {
        y <- as.matrix(y)
        storage.mode(y) <- "double"
    }
    y
}

.n4m_formula_xy <- function(formula, data, subset_expr = NULL, na.action) {
    if (!inherits(formula, "formula")) {
        stop("formula must be a two-sided formula")
    }
    mf_call <- call("model.frame", formula = formula, data = data,
                    na.action = na.action)
    if (!is.null(subset_expr)) {
        mf_call$subset <- subset_expr
    }
    mf <- eval(mf_call, parent.frame())
    tt <- stats::terms(mf)
    if (attr(tt, "response") == 0L) {
        stop("formula must have a response on the left-hand side")
    }
    y <- .n4m_as_response_matrix(stats::model.response(mf, type = "numeric"))
    X <- stats::model.matrix(tt, mf)
    intercept_col <- which(colnames(X) == "(Intercept)")
    if (length(intercept_col) > 0L) {
        X <- X[, -intercept_col, drop = FALSE]
    }
    if (ncol(X) < 1L) {
        stop("design matrix has zero predictors after dropping intercept")
    }
    list(
        X = X,
        Y = y,
        terms = tt,
        xlevels = stats::.getXlevels(tt, mf),
        response_names = colnames(y) %||% deparse(formula[[2L]]),
        predictor_names = colnames(X)
    )
}

.n4m_newdata_from_terms <- function(object, newdata) {
    tt <- stats::delete.response(object$terms)
    mf <- stats::model.frame(tt, newdata,
                             xlev = object$xlevels,
                             na.action = stats::na.pass)
    X <- stats::model.matrix(tt, mf)
    intercept_col <- which(colnames(X) == "(Intercept)")
    if (length(intercept_col) > 0L) {
        X <- X[, -intercept_col, drop = FALSE]
    }
    if (ncol(X) != object$n_features_in) {
        stop(sprintf("newdata has %d features, but model was fitted with %d",
                     ncol(X), object$n_features_in))
    }
    X
}

.n4m_solver_from_pls_method <- function(method, model = "plsr") {
    method <- tolower(as.character(method %||% "kernelpls"))
    if (model == "pcr" || method %in% c("pcr", "svdpc")) {
        return(list(kind = "model", method = "pcr", algo = "pcr_svd"))
    }
    switch(method,
        "kernelpls" = list(kind = "model", method = "kernelpls",
                           algo = "pls_kernel_algorithm"),
        "widekernelpls" = list(kind = "model", method = "widekernelpls",
                               algo = "pls_wide_kernel"),
        "simpls" = list(kind = "method_result", method = "simpls",
                        algo = "sparse_simpls"),
        "oscorespls" = list(kind = "model", method = "oscorespls",
                            algo = "pls_orthogonal_scores"),
        "cppls" = list(kind = "method_result", method = "cppls",
                       algo = "cppls"),
        "nipals" = list(kind = "model", method = "nipals",
                        algo = "pls_nipals"),
        stop(sprintf("unsupported PLS method '%s'", method))
    )
}

.n4m_mdatools_method_from_algo <- function(algo) {
    algo <- as.character(algo %||% "pls_simpls")
    switch(algo,
        "pls_simpls" = "simpls",
        "pls_kernel_algorithm" = "kernelpls",
        "pls_wide_kernel" = "widekernelpls",
        "pls_orthogonal_scores" = "oscorespls",
        "pls_nipals" = "nipals",
        "pcr_svd" = "pcr",
        "simpls"
    )
}

.n4m_fit_backend <- function(X, Y, ncomp, method = "simpls",
                                  center = TRUE, scale = FALSE,
                                  gamma = 0.5, sparsity_lambda = 0.0,
                                  model = "plsr") {
    X <- as.matrix(X)
    Y <- .n4m_as_response_matrix(Y)
    storage.mode(X) <- "double"
    storage.mode(Y) <- "double"
    spec <- .n4m_solver_from_pls_method(method, model = model)
    if (spec$kind == "method_result") {
        params <- list()
        if (identical(spec$algo, "sparse_simpls")) {
            params$sparsity_lambda <- sparsity_lambda
        } else if (identical(spec$algo, "cppls")) {
            params$gamma <- gamma
        }
        res <- n4m_method(spec$algo, X, Y, as.integer(ncomp),
                              params = params,
                              center_x = center, scale_x = scale,
                              center_y = center, scale_y = scale)
        return(list(kind = "method_result", method = spec$method,
                    algo = spec$algo, result = res))
    }
    mdl <- n4m_fit(X, Y, algo = spec$algo,
                       n_components = as.integer(ncomp),
                       center_x = center, scale_x = scale,
                       center_y = center, scale_y = scale)
    list(kind = "model", method = spec$method, algo = spec$algo, model = mdl)
}

.n4m_backend_predict <- function(fit, X) {
    X <- as.matrix(X)
    storage.mode(X) <- "double"
    if (identical(fit$kind, "model")) {
        return(n4m_predict(fit$model, X))
    }
    res <- fit$result
    coef <- res$coefficients
    if (is.null(coef)) {
        stop("this backend did not expose regression coefficients")
    }
    if (!is.null(res$intercept)) {
        intercept <- as.numeric(res$intercept)
        out <- X %*% coef
        if (length(intercept) == 1L) {
            out <- out + intercept
        } else {
            out <- sweep(out, 2L, intercept, "+")
        }
        return(out)
    }
    x_mean <- as.numeric(res$x_mean %||% rep(0, ncol(X)))
    y_mean <- as.numeric(res$y_mean %||% 0)
    out <- sweep(X, 2L, x_mean, "-") %*% coef
    if (length(y_mean) == 1L) {
        out <- out + y_mean
    } else {
        out <- sweep(out, 2L, y_mean, "+")
    }
    out
}

.n4m_backend_coef <- function(fit) {
    if (!identical(fit$kind, "method_result")) {
        stop("coefficients are not exposed for this solver path yet")
    }
    fit$result$coefficients
}

.n4m_component_fits <- function(X, Y, ncomp, method, center, scale,
                                     gamma, sparsity_lambda, model) {
    out <- lapply(seq_len(as.integer(ncomp)), function(k) {
        .n4m_fit_backend(X, Y, k, method = method,
                             center = center, scale = scale,
                             gamma = gamma,
                             sparsity_lambda = sparsity_lambda,
                             model = model)
    })
    names(out) <- as.character(seq_len(as.integer(ncomp)))
    out
}

.n4m_predict_components <- function(fits, X) {
    preds <- lapply(fits, function(fit) .n4m_backend_predict(fit, X))
    q <- ncol(.n4m_as_response_matrix(preds[[1L]]))
    comp_names <- names(fits) %||% as.character(seq_along(preds))
    arr <- array(NA_real_, dim = c(nrow(X), q, length(preds)),
                 dimnames = list(NULL, NULL, comp_names))
    for (i in seq_along(preds)) {
        arr[, , i] <- .n4m_as_response_matrix(preds[[i]])
    }
    arr
}

.n4m_component_positions <- function(pred_array, ncomp) {
    ncomp <- as.integer(ncomp)
    comp_names <- dimnames(pred_array)[[3L]]
    if (!is.null(comp_names)) {
        pos <- match(as.character(ncomp), comp_names)
    } else {
        pos <- ncomp
    }
    if (any(is.na(pos)) || any(pos < 1L | pos > dim(pred_array)[3L])) {
        stop("requested ncomp was not fitted; refit with fit_components = TRUE")
    }
    pos
}

.n4m_make_segments <- function(n, segments) {
    if (is.list(segments)) {
        return(lapply(segments, as.integer))
    }
    k <- as.integer(segments %||% 10L)
    k <- max(2L, min(k, n))
    split(seq_len(n), cut(seq_len(n), breaks = k, labels = FALSE))
}

.n4m_cv_predictions <- function(X, Y, ncomp, method, center, scale,
                                     gamma, sparsity_lambda, model,
                                     segments) {
    Y <- .n4m_as_response_matrix(Y)
    folds <- .n4m_make_segments(nrow(X), segments)
    out <- array(NA_real_, dim = c(nrow(X), ncol(Y), as.integer(ncomp)))
    for (k in seq_len(as.integer(ncomp))) {
        for (idx in folds) {
            train <- setdiff(seq_len(nrow(X)), idx)
            fit <- .n4m_fit_backend(X[train, , drop = FALSE],
                                        Y[train, , drop = FALSE],
                                        k, method = method,
                                        center = center, scale = scale,
                                        gamma = gamma,
                                        sparsity_lambda = sparsity_lambda,
                                        model = model)
            out[idx, , k] <- .n4m_as_response_matrix(
                .n4m_backend_predict(fit, X[idx, , drop = FALSE]))
        }
    }
    out
}

.n4m_metric_by_component <- function(pred_array, Y, metric) {
    Y <- .n4m_as_response_matrix(Y)
    q <- ncol(Y)
    ncomp <- dim(pred_array)[3L]
    comp_names <- dimnames(pred_array)[[3L]] %||% as.character(seq_len(ncomp))
    out <- matrix(NA_real_, nrow = q, ncol = ncomp)
    for (j in seq_len(q)) {
        denom <- sum((Y[, j] - mean(Y[, j])) ^ 2)
        for (k in seq_len(ncomp)) {
            err <- pred_array[, j, k] - Y[, j]
            mse <- mean(err ^ 2)
            out[j, k] <- switch(metric,
                mse = mse,
                rmsep = sqrt(mse),
                r2 = if (denom > 0) 1 - sum(err ^ 2) / denom else NA_real_)
        }
    }
    rownames(out) <- colnames(Y) %||% paste0("Y", seq_len(q))
    colnames(out) <- paste0("Comp ", comp_names)
    out
}

.n4m_drop_response_dim <- function(pred) {
    pred <- .n4m_as_response_matrix(pred)
    if (ncol(pred) == 1L) pred[, 1L] else pred
}
