#' Register pls4all as a tidymodels / parsnip engine.
#'
#' Parsnip exposes a single meta-model `pls_pls4all_reg()` that
#' dispatches to all 22 regression-mold pls4all algorithms via the
#' engine arg `algorithm`. This avoids declaring 22 separate parsnip
#' model specs (one per algorithm) while still working with workflows,
#' tune::tune_grid, and the rest of the tidymodels stack.
#'
#' Usage:
#'
#' ```r
#' library(tidymodels)
#' pls4all::register_parsnip()
#'
#' spec <- pls_pls4all_reg(num_comp = 5) %>%
#'   set_engine("pls4all", algorithm = "sparse_simpls",
#'              sparsity_lambda = 0.05)
#' wf <- workflow() %>%
#'   add_model(spec) %>%
#'   add_formula(y ~ .)
#' fit <- fit(wf, data = df)
#' preds <- predict(fit, new_data = test_df)
#' ```
#'
#' Supported `algorithm` values (regression):
#'   pls (default, equivalent to SIMPLS via the `pls()` formula entry),
#'   sparse_simpls, cppls, ecr, weighted_pls, robust_pls, ridge_pls,
#'   continuum_regression, mb_pls, pls_glm, mir_pls, missing_aware_nipals,
#'   bagging_pls, boosting_pls, random_subspace_pls, o2pls,
#'   n_pls, kernel_pls, di_pls, recursive_pls.
#'
#' Algorithm-specific tuning params (sparsity_lambda, gamma, tau, etc.)
#' are passed via `set_engine("pls4all", algorithm = "...", <param> = ...)`.

#' @export
register_parsnip <- function() {
    if (!requireNamespace("parsnip", quietly = TRUE)) {
        warning("parsnip is not installed; skipping registration. ",
                "Install it with install.packages('parsnip').")
        return(invisible(NULL))
    }
    if (isTRUE(.pls4all_env$.parsnip_registered)) return(invisible(NULL))

    parsnip::set_new_model("pls_pls4all_reg")
    parsnip::set_model_mode("pls_pls4all_reg", "regression")
    parsnip::set_model_engine("pls_pls4all_reg",
                               mode = "regression", eng = "pls4all")
    parsnip::set_dependency("pls_pls4all_reg", eng = "pls4all", pkg = "pls4all")

    parsnip::set_model_arg(
        model = "pls_pls4all_reg", eng = "pls4all",
        parsnip = "num_comp", original = "ncomp",
        func = list(pkg = "dials", fun = "num_comp"),
        has_submodel = FALSE
    )

    parsnip::set_fit(
        model = "pls_pls4all_reg", eng = "pls4all", mode = "regression",
        value = list(
            interface = "formula",
            protect = c("formula", "data"),
            func = c(pkg = "pls4all", fun = "pls4all_parsnip_fit"),
            defaults = list(algorithm = "pls")
        )
    )

    parsnip::set_encoding(
        model = "pls_pls4all_reg", eng = "pls4all", mode = "regression",
        options = list(
            predictor_indicators = "traditional",
            compute_intercept = FALSE,
            remove_intercept = FALSE,
            allow_sparse_x = FALSE
        )
    )

    parsnip::set_pred(
        model = "pls_pls4all_reg", eng = "pls4all", mode = "regression",
        type = "numeric",
        value = list(
            pre = NULL,
            post = function(results, object) results,
            func = c(fun = "predict"),
            args = list(
                object = quote(object$fit),
                newdata = quote(new_data)
            )
        )
    )

    .pls4all_env$.parsnip_registered <- TRUE
    invisible(NULL)
}

#' Parsnip-side fit dispatcher.
#'
#' Called by `parsnip::set_fit`. Routes to the right pls4all algorithm
#' based on `algorithm` engine arg and forwards algorithm-specific
#' params from the engine args.
#' @param formula A two-sided formula.
#' @param data A data.frame.
#' @param ncomp Number of latent components.
#' @param algorithm Algorithm name (see register_parsnip docs).
#' @param ... Algorithm-specific parameters forwarded to the underlying
#'   formula entry (e.g. `sparsity_lambda` for sparse_simpls).
#' @export
pls4all_parsnip_fit <- function(formula, data, ncomp = 2L,
                                  algorithm = "pls", ...) {
    args <- list(...)
    switch(algorithm,
        "pls"                  = pls(formula, data, ncomp = ncomp,
                                       algo = args$algo %||% "pls_simpls"),
        "sparse_simpls"        = sparse_pls(formula, data, ncomp,
                                              sparsity_lambda =
                                                  args$sparsity_lambda %||% 0.05),
        "cppls"                = cppls(formula, data, ncomp,
                                          gamma = args$gamma %||% 0.5),
        "ecr"                  = ecr(formula, data, ncomp,
                                       alpha = args$alpha %||% 0.5),
        "weighted_pls"         = weighted_pls(formula, data, ncomp,
                                                 weights = args$weights),
        "robust_pls"           = robust_pls(formula, data, ncomp,
                                              huber_k = args$huber_k %||% 1.345,
                                              max_irls_iter =
                                                  args$max_irls_iter %||% 20L),
        "ridge_pls"            = ridge_pls(formula, data, ncomp,
                                             ridge_lambda =
                                                 args$ridge_lambda %||% 1.0),
        "continuum_regression" = continuum_regression(formula, data, ncomp,
                                                        tau = args$tau %||% 0.5),
        "mb_pls"               = mb_pls(formula, data, ncomp,
                                          block_sizes = args$block_sizes),
        "pls_glm"              = pls_glm(formula, data, ncomp,
                                            family = args$family %||% "gaussian"),
        "mir_pls"              = mir_pls(formula, data, ncomp),
        "missing_aware_nipals" = missing_aware_nipals(formula, data, ncomp),
        "bagging_pls"          = bagging_pls(formula, data, ncomp,
                                                n_estimators =
                                                    args$n_estimators %||% 50L,
                                                seed = args$seed %||% 0L),
        "boosting_pls"         = boosting_pls(formula, data, ncomp,
                                                 n_estimators =
                                                     args$n_estimators %||% 50L,
                                                 learning_rate =
                                                     args$learning_rate %||% 0.1),
        "random_subspace_pls"  = random_subspace_pls(formula, data, ncomp,
                                                       n_estimators =
                                                           args$n_estimators %||% 50L,
                                                       features_per_subspace =
                                                           args$features_per_subspace %||% 10L,
                                                       seed = args$seed %||% 0L),
        "o2pls"                = o2pls(formula, data,
                                          n_predictive = ncomp,
                                          n_x_orthogonal =
                                              args$n_x_orthogonal %||% 1L,
                                          n_y_orthogonal =
                                              args$n_y_orthogonal %||% 1L),
        stop(sprintf("Unknown algorithm: %s", algorithm))
    )
}

`%||%` <- function(a, b) if (is.null(a)) b else a

#' @export
pls_pls4all_reg <- function(mode = "regression", num_comp = NULL) {
    if (!requireNamespace("parsnip", quietly = TRUE)) {
        stop("pls_pls4all_reg() requires parsnip; install it with ",
             "install.packages('parsnip').")
    }
    register_parsnip()
    args <- list(num_comp = rlang::enquo(num_comp))
    parsnip::new_model_spec(
        "pls_pls4all_reg",
        args = args, eng_args = NULL,
        mode = mode, user_specified_mode = TRUE,
        method = NULL, engine = NULL,
        user_specified_engine = TRUE
    )
}

# Internal package env to track registration state.
.pls4all_env <- new.env(parent = emptyenv())
