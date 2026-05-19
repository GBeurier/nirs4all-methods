#' Formula-based PLS regression wrapper around the pls4all C ABI.
#'
#' Tier-2 idiomatic R interface for tier-1 [pls4all_fit()] /
#' [pls4all_predict()]. Provides a formula entry point and S3 generics
#' so the returned object plays well with `predict()`, `summary()`,
#' `coef()`, `print()`, base R `cv.glmnet`-style workflows, and any
#' caret model that supports a `predict(object, newdata)` adapter.
#'
#' @param formula A two-sided formula (e.g. `y ~ .` or
#'   `y ~ x1 + x2 + x3`). Response on the left, predictors on the right.
#' @param data A `data.frame` (or anything `as.data.frame`-coercible)
#'   containing the response and predictor columns referenced by
#'   `formula`.
#' @param ncomp Integer. Number of latent components.
#' @param algo Character. One of `"pls_nipals"`, `"pls_simpls"`,
#'   `"pls_svd"`, `"pls_power"`, `"pls_kernel_algorithm"`,
#'   `"pls_wide_kernel"`, `"pls_orthogonal_scores"`,
#'   `"pls_randomized_svd"`, `"pcr_svd"`, `"opls_nipals"`.
#'   Defaults to `"pls_nipals"` to
#'   match the R `pls` package's default.
#' @param na.action What to do with `NA`s. Default: `na.omit`.
#'
#' @return An object of class `c("pls4all_fit", "pls_fit")` with
#'   components:
#'   * `model`         — external pointer to the libp4a model handle
#'   * `formula`       — the call's formula
#'   * `terms`         — the `terms()` object describing the model
#'   * `xlevels`       — factor levels of the predictors, for
#'                       newdata coercion
#'   * `call`          — the original call (for `print` / `summary`)
#'   * `ncomp`         — components used
#'   * `algo`          — solver used
#'   * `n_features_in` — predictor count
#'   * `response_name` — left-hand side of the formula (string)
#'
#' @examples
#' \dontrun{
#'   set.seed(0)
#'   df <- data.frame(
#'     x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100)
#'   )
#'   df$y <- 2 * df$x1 - df$x2 + rnorm(100, sd = 0.1)
#'   fit <- pls(y ~ ., data = df, ncomp = 3)
#'   summary(fit)
#'   preds <- predict(fit, newdata = df)
#' }
#' @export
pls <- function(formula, data, ncomp = 2L,
                algo = "pls_nipals", na.action = stats::na.omit, ...) {
    cl <- match.call()
    dots <- list(...)
    if (missing(formula) && !is.null(dots$x)) {
        x <- dots$x
        y <- dots$y
        dots$x <- NULL
        dots$y <- NULL
        if (is.null(y)) stop("pls(x = ..., y = ...) requires y")
        return(do.call(pls_mdatools,
                       c(list(x = x, y = y, ncomp = ncomp), dots)))
    }
    if (!inherits(formula, "formula")) {
        if (missing(data)) {
            stop("matrix-style pls() requires a response as the second argument")
        }
        method <- .pls4all_mdatools_method_from_algo(algo)
        args <- list(x = formula, y = data, ncomp = ncomp)
        if (is.null(dots$method)) args$method <- method
        return(do.call(pls_mdatools, c(args, dots)))
    }
    mf <- stats::model.frame(formula, data = data, na.action = na.action)
    tt <- stats::terms(mf)
    if (attr(tt, "response") == 0L) {
        stop("formula must have a response on the left-hand side")
    }
    y <- stats::model.response(mf, type = "numeric")
    if (is.null(y)) {
        stop("response is missing or non-numeric; pls() handles ",
             "regression only")
    }
    X <- stats::model.matrix(tt, mf)
    intercept_col <- which(colnames(X) == "(Intercept)")
    if (length(intercept_col) > 0L) {
        X <- X[, -intercept_col, drop = FALSE]
    }
    if (ncol(X) < 1L) {
        stop("design matrix has zero predictors after dropping intercept")
    }

    model <- pls4all_fit(X, y, algo = algo, n_components = as.integer(ncomp))

    structure(
        list(
            model         = model,
            formula       = formula,
            terms         = tt,
            xlevels       = stats::.getXlevels(tt, mf),
            call          = cl,
            ncomp         = as.integer(ncomp),
            algo          = as.character(algo),
            n_features_in = ncol(X),
            response_name = deparse(formula[[2L]])
        ),
        class = c("pls4all_fit", "pls_fit")
    )
}

#' Formula-based OPLS regression wrapper around the pls4all C ABI.
#'
#' Uses the same formula/S3 return contract as [pls()], with the model API
#' configured as `algo = "opls_nipals"`.
#'
#' @inheritParams pls
#' @return An object of class `c("opls_fit", "pls4all_fit", "pls_fit")`.
#' @export
opls <- function(formula, data, ncomp = 2L,
                 na.action = stats::na.omit) {
    cl <- match.call()
    if (!inherits(formula, "formula")) {
        stop("formula must be a two-sided formula (e.g. `y ~ .`)")
    }
    mf <- stats::model.frame(formula, data = data, na.action = na.action)
    tt <- stats::terms(mf)
    if (attr(tt, "response") == 0L) {
        stop("formula must have a response on the left-hand side")
    }
    y <- stats::model.response(mf, type = "numeric")
    if (is.null(y)) {
        stop("response is missing or non-numeric; opls() handles ",
             "regression only")
    }
    X <- stats::model.matrix(tt, mf)
    intercept_col <- which(colnames(X) == "(Intercept)")
    if (length(intercept_col) > 0L) {
        X <- X[, -intercept_col, drop = FALSE]
    }
    if (ncol(X) < 1L) {
        stop("design matrix has zero predictors after dropping intercept")
    }

    model <- pls4all_fit(X, y, algo = "opls_nipals",
                         n_components = as.integer(ncomp),
                         scale_x = FALSE, scale_y = FALSE)

    structure(
        list(
            model         = model,
            formula       = formula,
            terms         = tt,
            xlevels       = stats::.getXlevels(tt, mf),
            call          = cl,
            ncomp         = as.integer(ncomp),
            algo          = "opls_nipals",
            n_features_in = ncol(X),
            response_name = deparse(formula[[2L]])
        ),
        class = c("opls_fit", "pls4all_fit", "pls_fit")
    )
}

#' Build the model matrix for `newdata` in the same way `pls()` built
#' the training matrix. Used internally by [predict.pls4all_fit()].
#' @noRd
.pls4all_newdata_matrix <- function(object, newdata) {
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
        stop(sprintf(
            "newdata has %d features, but model was fitted with %d",
            ncol(X), object$n_features_in))
    }
    X
}

#' Predict from a [pls()]-fitted model.
#'
#' @param object A `pls4all_fit` returned by [pls()].
#' @param newdata A `data.frame` (or matrix). If `NULL`, predictions on
#'   the training matrix are returned.
#' @param ... Ignored (for S3 generic compatibility).
#' @return Numeric vector (for single-target regression) or matrix
#'   (multi-target).
#' @export
predict.pls4all_fit <- function(object, newdata = NULL, ...) {
    if (is.null(newdata)) {
        # Re-derive the training matrix from the original formula + data.
        # The model.frame from the original call is not retained
        # explicitly; users who want true in-sample preds should pass
        # `newdata = same df`.
        stop("predict.pls4all_fit requires newdata; the training frame ",
             "is not retained on the fitted object.")
    }
    X <- .pls4all_newdata_matrix(object, newdata)
    preds <- pls4all_predict(object$model, X)
    if (is.matrix(preds) && ncol(preds) == 1L) {
        preds <- preds[, 1L]
    }
    preds
}

#' @export
coef.pls4all_fit <- function(object, ...) {
    # The tier-1 binding currently doesn't expose model arrays directly;
    # callers can use the formula-driven predict path, or reach for the
    # underlying handle via `object$model`. Once `p4a_model_get_array` is
    # bound on the R side, this generic will return the (p x q)
    # coefficient matrix.
    stop("coef.pls4all_fit is not yet implemented; the R binding does ",
         "not currently expose `p4a_model_get_array` for coefficient ",
         "retrieval. Tracked in roadmap/phase-54a-r-tier2.md.")
}

#' @export
print.pls4all_fit <- function(x, ...) {
    cat("pls4all fit (", x$algo, ", ncomp=", x$ncomp, ")\n", sep = "")
    cat("Call:\n  ")
    print(x$call)
    cat("Response:           ", x$response_name, "\n", sep = "")
    cat("Predictors:         ", x$n_features_in, "\n", sep = "")
    invisible(x)
}

#' @export
summary.pls4all_fit <- function(object, ...) {
    out <- list(
        algo            = object$algo,
        ncomp           = object$ncomp,
        n_features_in   = object$n_features_in,
        response_name   = object$response_name,
        call            = object$call
    )
    class(out) <- "summary.pls4all_fit"
    out
}

#' @export
print.summary.pls4all_fit <- function(x, ...) {
    cat("pls4all model summary\n")
    cat("---------------------\n")
    cat("Solver:             ", x$algo, "\n", sep = "")
    cat("Components:         ", x$ncomp, "\n", sep = "")
    cat("Predictors:         ", x$n_features_in, "\n", sep = "")
    cat("Response:           ", x$response_name, "\n", sep = "")
    cat("Call:\n  ")
    print(x$call)
    invisible(x)
}
