#' Register pls4all as a tidymodels / parsnip engine.
#'
#' This file declares the `pls4all` engine for the parsnip `pls_reg()`
#' model specification (or, when parsnip isn't loaded, registers a
#' standalone `pls_pls4all_reg()` model spec that mirrors parsnip's API).
#'
#' The registration is lazy: it runs only when `register_parsnip()` is
#' called explicitly (or when the package's `.onLoad` calls it and
#' `parsnip` is on the search path). Pls4all does NOT take a hard
#' dependency on parsnip — users who don't run tidymodels never pay the
#' cost of loading it.
#'
#' Usage (after calling `pls4all::register_parsnip()` or once parsnip
#' calls it automatically):
#'
#' ```r
#' library(tidymodels)
#' pls_model <- pls_pls4all_reg(num_comp = 5) %>%
#'   set_engine("pls4all") %>%
#'   set_mode("regression")
#'
#' wf <- workflow() %>%
#'   add_model(pls_model) %>%
#'   add_formula(y ~ .)
#'
#' fit <- fit(wf, data = df)
#' predict(fit, new_data = test_df)
#' ```

#' @export
register_parsnip <- function() {
    if (!requireNamespace("parsnip", quietly = TRUE)) {
        warning("parsnip is not installed; skipping registration. ",
                "Install it with install.packages('parsnip').")
        return(invisible(NULL))
    }

    # Skip if already registered (idempotent).
    if (isTRUE(.pls4all_env$.parsnip_registered)) {
        return(invisible(NULL))
    }

    parsnip::set_new_model("pls_pls4all_reg")
    parsnip::set_model_mode("pls_pls4all_reg", "regression")
    parsnip::set_model_engine(
        "pls_pls4all_reg", mode = "regression", eng = "pls4all"
    )
    parsnip::set_dependency(
        "pls_pls4all_reg", eng = "pls4all", pkg = "pls4all"
    )

    parsnip::set_model_arg(
        model = "pls_pls4all_reg",
        eng = "pls4all",
        parsnip = "num_comp",
        original = "ncomp",
        func = list(pkg = "dials", fun = "num_comp"),
        has_submodel = FALSE
    )

    parsnip::set_fit(
        model = "pls_pls4all_reg",
        eng = "pls4all",
        mode = "regression",
        value = list(
            interface = "formula",
            protect = c("formula", "data"),
            func = c(pkg = "pls4all", fun = "pls"),
            defaults = list()
        )
    )

    parsnip::set_encoding(
        model = "pls_pls4all_reg",
        eng = "pls4all",
        mode = "regression",
        options = list(
            predictor_indicators = "traditional",
            compute_intercept = FALSE,
            remove_intercept = FALSE,
            allow_sparse_x = FALSE
        )
    )

    parsnip::set_pred(
        model = "pls_pls4all_reg",
        eng = "pls4all",
        mode = "regression",
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
        args = args,
        eng_args = NULL,
        mode = mode,
        user_specified_mode = TRUE,
        method = NULL,
        engine = NULL,
        user_specified_engine = TRUE
    )
}

# Internal package env to track registration state.
.pls4all_env <- new.env(parent = emptyenv())
