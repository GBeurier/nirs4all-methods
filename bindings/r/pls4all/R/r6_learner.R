#' R6 / mlr3 learner for pls4all PLS regression.
#'
#' Provides a unified R6 learner `regr.pls4all` that dispatches to all
#' 16 regression-mold pls4all algorithms via the `algorithm` parameter.
#' Same pattern as the parsnip side: one mlr3 learner, parameterized
#' by algorithm — keeps the mlr3 learner catalog small while still
#' exposing the full pls4all surface.
#'
#' Usage:
#'
#' ```r
#' library(mlr3)
#' pls4all::register_mlr3()
#'
#' task <- as_task_regr(df, target = "y")
#' learner <- lrn("regr.pls4all", ncomp = 5,
#'                algorithm = "sparse_simpls", sparsity_lambda = 0.05)
#' learner$train(task)
#' learner$predict(task)
#' ```
#'
#' Supported `algorithm` values:
#'   pls (default; SIMPLS via `pls()`), sparse_simpls, cppls, ecr,
#'   weighted_pls (requires `weights`), robust_pls, ridge_pls,
#'   continuum_regression, mb_pls (requires `block_sizes`),
#'   pls_glm, mir_pls, missing_aware_nipals,
#'   bagging_pls, boosting_pls, random_subspace_pls, o2pls.

#' @export
register_mlr3 <- function() {
    if (!requireNamespace("mlr3", quietly = TRUE) ||
        !requireNamespace("R6", quietly = TRUE) ||
        !requireNamespace("paradox", quietly = TRUE)) {
        warning("mlr3, R6, or paradox not installed; skipping registration. ",
                "Install with install.packages(c('mlr3', 'R6', 'paradox')).")
        return(invisible(NULL))
    }
    if (isTRUE(.pls4all_env$.mlr3_registered)) return(invisible(NULL))

    LearnerRegrPLS4all <- R6::R6Class(
        "LearnerRegrPLS4all",
        inherit = mlr3::LearnerRegr,
        public = list(
            initialize = function() {
                ps <- paradox::ps(
                    ncomp = paradox::p_int(
                        lower = 1L, upper = 200L,
                        default = 2L, tags = "train"),
                    algorithm = paradox::p_fct(
                        levels = c("pls", "sparse_simpls", "cppls", "ecr",
                                    "weighted_pls", "robust_pls", "ridge_pls",
                                    "continuum_regression", "mb_pls",
                                    "pls_glm", "mir_pls", "missing_aware_nipals",
                                    "bagging_pls", "boosting_pls",
                                    "random_subspace_pls", "o2pls"),
                        default = "pls", tags = "train"),
                    algo = paradox::p_fct(
                        levels = c("pls_nipals", "pls_orthogonal_scores",
                                    "pls_simpls", "pls_kernel_algorithm",
                                    "pls_wide_kernel", "pls_svd",
                                    "pls_power", "pls_randomized_svd",
                                    "pcr_svd"),
                        default = "pls_simpls", tags = "train"),
                    sparsity_lambda = paradox::p_dbl(0, default = 0.05, tags = "train"),
                    gamma           = paradox::p_dbl(0, 1, default = 0.5, tags = "train"),
                    alpha           = paradox::p_dbl(0, 1, default = 0.5, tags = "train"),
                    tau             = paradox::p_dbl(0, 1, default = 0.5, tags = "train"),
                    ridge_lambda    = paradox::p_dbl(0, default = 1.0, tags = "train"),
                    huber_k         = paradox::p_dbl(0, default = 1.345, tags = "train"),
                    max_irls_iter   = paradox::p_int(1L, default = 20L, tags = "train"),
                    n_estimators    = paradox::p_int(1L, default = 50L, tags = "train"),
                    learning_rate   = paradox::p_dbl(0, default = 0.1, tags = "train"),
                    features_per_subspace = paradox::p_int(1L, default = 10L, tags = "train"),
                    seed            = paradox::p_int(0L, default = 0L, tags = "train"),
                    n_x_orthogonal  = paradox::p_int(0L, default = 1L, tags = "train"),
                    n_y_orthogonal  = paradox::p_int(0L, default = 1L, tags = "train"),
                    family          = paradox::p_fct(
                        levels = c("gaussian", "poisson"),
                        default = "gaussian", tags = "train"),
                    block_sizes     = paradox::p_uty(default = NULL, tags = "train")
                )
                super$initialize(
                    id = "regr.pls4all",
                    param_set = ps,
                    feature_types = c("logical", "integer", "numeric"),
                    predict_types = "response",
                    properties = character(),
                    packages = "pls4all",
                    label = "pls4all PLS Regression (unified)"
                )
            }
        ),
        private = list(
            .train = function(task) {
                pars <- self$param_set$get_values(tags = "train")
                ncomp <- as.integer(pars$ncomp %||% 2L)
                algo  <- pars$algorithm %||% "pls"
                f <- task$formula()
                d <- task$data()
                switch(algo,
                    "pls"                  = pls4all::pls(f, d, ncomp,
                                                            algo = pars$algo %||% "pls_simpls"),
                    "sparse_simpls"        = pls4all::sparse_pls(f, d, ncomp,
                                                                   sparsity_lambda = pars$sparsity_lambda %||% 0.05),
                    "cppls"                = pls4all::cppls(f, d, ncomp,
                                                              gamma = pars$gamma %||% 0.5),
                    "ecr"                  = pls4all::ecr(f, d, ncomp,
                                                            alpha = pars$alpha %||% 0.5),
                    "weighted_pls"         = pls4all::weighted_pls(f, d, ncomp,
                                                                     weights = pars$weights),
                    "robust_pls"           = pls4all::robust_pls(f, d, ncomp,
                                                                   huber_k = pars$huber_k %||% 1.345,
                                                                   max_irls_iter = pars$max_irls_iter %||% 20L),
                    "ridge_pls"            = pls4all::ridge_pls(f, d, ncomp,
                                                                  ridge_lambda = pars$ridge_lambda %||% 1.0),
                    "continuum_regression" = pls4all::continuum_regression(f, d, ncomp,
                                                                             tau = pars$tau %||% 0.5),
                    "mb_pls"               = pls4all::mb_pls(f, d, ncomp,
                                                               block_sizes = pars$block_sizes),
                    "pls_glm"              = pls4all::pls_glm(f, d, ncomp,
                                                                family = pars$family %||% "gaussian"),
                    "mir_pls"              = pls4all::mir_pls(f, d, ncomp),
                    "missing_aware_nipals" = pls4all::missing_aware_nipals(f, d, ncomp),
                    "bagging_pls"          = pls4all::bagging_pls(f, d, ncomp,
                                                                    n_estimators = pars$n_estimators %||% 50L,
                                                                    seed = pars$seed %||% 0L),
                    "boosting_pls"         = pls4all::boosting_pls(f, d, ncomp,
                                                                     n_estimators = pars$n_estimators %||% 50L,
                                                                     learning_rate = pars$learning_rate %||% 0.1),
                    "random_subspace_pls"  = pls4all::random_subspace_pls(f, d, ncomp,
                                                                            n_estimators = pars$n_estimators %||% 50L,
                                                                            features_per_subspace = pars$features_per_subspace %||% 10L,
                                                                            seed = pars$seed %||% 0L),
                    "o2pls"                = pls4all::o2pls(f, d,
                                                              n_predictive = ncomp,
                                                              n_x_orthogonal = pars$n_x_orthogonal %||% 1L,
                                                              n_y_orthogonal = pars$n_y_orthogonal %||% 1L),
                    stop(sprintf("Unknown algorithm: %s", algo))
                )
            },
            .predict = function(task) {
                preds <- predict(self$model,
                                  newdata = task$data(
                                      cols = task$feature_names))
                list(response = as.numeric(preds))
            }
        )
    )

    mlr3::mlr_learners$add("regr.pls4all", LearnerRegrPLS4all)
    .pls4all_env$.mlr3_registered <- TRUE
    invisible(NULL)
}

`%||%` <- function(a, b) if (is.null(a)) b else a
