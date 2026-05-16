#' R6 / mlr3 learner for pls4all PLS regression.
#'
#' Provides an R6 class wrapping pls4all's `pls()` formula interface as
#' an mlr3-compatible learner. Lazy registration like the parsnip module:
#' `register_mlr3()` is the one-shot opt-in; it raises a clean warning
#' if mlr3 is not installed.
#'
#' Usage:
#'
#' ```r
#' library(mlr3)
#' pls4all::register_mlr3()
#'
#' task <- as_task_regr(df, target = "y")
#' learner <- lrn("regr.pls4all", ncomp = 5)
#' learner$train(task)
#' learner$predict(task)
#' ```

#' @export
register_mlr3 <- function() {
    if (!requireNamespace("mlr3", quietly = TRUE) ||
        !requireNamespace("R6", quietly = TRUE)) {
        warning("mlr3 and/or R6 not installed; skipping registration. ",
                "Install both with install.packages(c('mlr3', 'R6')).")
        return(invisible(NULL))
    }
    if (isTRUE(.pls4all_env$.mlr3_registered)) {
        return(invisible(NULL))
    }

    LearnerRegrPLS4all <- R6::R6Class(
        "LearnerRegrPLS4all",
        inherit = mlr3::LearnerRegr,
        public = list(
            initialize = function() {
                ps <- paradox::ps(
                    ncomp = paradox::p_int(
                        lower = 1L, upper = 200L,
                        default = 2L, tags = "train"),
                    algo = paradox::p_fct(
                        levels = c(
                            "pls_nipals", "pls_orthogonal_scores",
                            "pls_simpls", "pls_kernel_algorithm",
                            "pls_wide_kernel", "pls_svd",
                            "pls_power", "pls_randomized_svd",
                            "pcr_svd"),
                        default = "pls_nipals", tags = "train")
                )
                super$initialize(
                    id = "regr.pls4all",
                    param_set = ps,
                    feature_types = c("logical", "integer", "numeric"),
                    predict_types = "response",
                    properties = character(),
                    packages = "pls4all",
                    label = "pls4all PLS Regression"
                )
            }
        ),
        private = list(
            .train = function(task) {
                pars <- self$param_set$get_values(tags = "train")
                ncomp <- as.integer(pars$ncomp %||% 2L)
                algo <- pars$algo %||% "pls_nipals"
                f <- task$formula()
                pls4all::pls(
                    formula = f,
                    data = task$data(),
                    ncomp = ncomp,
                    algo = algo
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

# Tiny null-coalesce helper used inside .train (no rlang dep).
`%||%` <- function(a, b) if (is.null(a)) b else a
