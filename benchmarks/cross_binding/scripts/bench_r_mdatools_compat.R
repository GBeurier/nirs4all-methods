# R mdatools-compatible benchmark column.
#
# For `pls`, `pcr`, and `cppls` this measures the actual
# `pls(x, y, ...)` / mdatools-style facade. For every other registry
# method, it enters through the same matrix-shaped X/Y contract and then
# uses the registry dispatcher so the dashboard has a complete R
# mdatools_compat column for the whole canonical method matrix.

suppressMessages(library(pls4all))
suppressMessages(library(jsonlite))

`%||%` <- function(x, y) if (is.null(x)) y else x

.script_dir <- function() {
    a <- commandArgs(trailingOnly = FALSE)
    f <- a[grep("--file=", a, fixed = TRUE)]
    if (length(f) > 0L) normalizePath(dirname(sub("^--file=", "", f[[1]]))) else "."
}
source(file.path(.script_dir(), "_npy.R"))

a <- pls4all_bench_parse_args()

params_env <- Sys.getenv("BENCH_R_PARAMS_JSON", unset = "")
if (nzchar(params_env)) {
    params <- fromJSON(params_env, simplifyVector = FALSE)
} else {
    params <- list()
}

.flatten_numeric <- function(value) {
    if (is.list(value) && length(value) > 0L &&
        all(vapply(value, is.numeric, logical(1L)))) {
        return(unlist(value))
    }
    value
}
.int_keys <- c("block_sizes", "n_components_per_block",
                "n_unique_per_block", "y_labels",
                "group_assignment", "n_blocks", "n_components",
                "max_iter", "max_irls_iter", "n_estimators",
                "n_neighbors", "n_iterations", "n_intervals",
                "window_size", "initial_intervals", "top_k",
                "min_selected", "seed", "n_classes",
                "n_predictive", "n_x_orthogonal", "n_y_orthogonal",
                "features_per_subspace", "n_steps", "n_folds",
                "noise_features", "noise_seed", "kernel_type",
                "degree", "n_targets", "mode_j", "mode_k",
                "window_half_width")
for (k in names(params)) {
    v <- .flatten_numeric(params[[k]])
    if (k %in% .int_keys) {
        v <- as.integer(v)
    } else if (is.numeric(v)) {
        v <- as.double(v)
    }
    params[[k]] <- v
}

nc <- if (!is.null(params$n_components)) as.integer(params$n_components) else a$nc

needs_labels       <- Sys.getenv("BENCH_R_NEEDS_LABELS", unset = "") == "1"
needs_sw           <- Sys.getenv("BENCH_R_NEEDS_SAMPLE_WEIGHTS", unset = "") == "1"
needs_groups       <- Sys.getenv("BENCH_R_NEEDS_GROUP_ASSIGNMENT", unset = "") == "1"
needs_x_target     <- Sys.getenv("BENCH_R_NEEDS_X_TARGET", unset = "") == "1"
x_target_dir       <- Sys.getenv("BENCH_R_X_TARGET_DIR", unset = "")
registry_pkey      <- Sys.getenv("BENCH_PREDICTION_KEY", unset = "predictions")

.formula_design <- function(xy) {
    df <- as.data.frame(xy$X)
    df$y <- xy$y
    mf <- stats::model.frame(y ~ ., data = df, na.action = stats::na.pass)
    tt <- stats::terms(mf)
    X <- stats::model.matrix(tt, mf)
    intercept_col <- which(colnames(X) == "(Intercept)")
    if (length(intercept_col) > 0L) {
        X <- X[, -intercept_col, drop = FALSE]
    }
    list(X = X, y = stats::model.response(mf, type = "numeric"))
}

.fold_rmse_matrix <- function(max_components, n_folds) {
    idx <- matrix(seq_len(max_components * n_folds) - 1L,
                  nrow = max_components, ncol = n_folds, byrow = TRUE)
    0.5 + 0.5 * ((idx * 37 + 11) %% 997) / 996
}

.materialize_result <- function(res, pkey, p, prefix) {
    if (pkey %in% c("selected_indices", "mask", "support")) {
        if (!is.null(res$mask)) return(matrix(as.numeric(res$mask), nrow = 1))
        if (!is.null(res$support)) return(matrix(as.numeric(res$support), nrow = 1))
        if (!is.null(res$selected_indices)) {
            sel <- as.integer(res$selected_indices)
            mask <- matrix(0.0, nrow = 1, ncol = p)
            idx <- sel[sel > 0L & sel <= p]
            if (length(idx) > 0L) mask[1, idx] <- 1.0
            return(mask)
        }
        return(matrix(0.0, nrow = 1, ncol = p))
    }
    if (pkey == "decision_scores" && !is.null(res$decision_scores)) {
        return(res$decision_scores)
    }
    if (!is.null(res[[pkey]])) return(res[[pkey]])
    if (!is.null(res$predictions)) return(res$predictions)
    if (!is.null(res$selected_indices)) {
        sel <- as.integer(res$selected_indices)
        mask <- matrix(0.0, nrow = 1, ncol = p)
        idx <- sel[sel > 0L & sel <= p]
        if (length(idx) > 0L) mask[1, idx] <- 1.0
        return(mask)
    }
    stop(sprintf("%s: returned no '%s' field", prefix, pkey))
}

.fit_simpls_model <- function(X, Y, n_components) {
    pls4all::pls4all_fit(X, Y,
                         algo = "pls_simpls",
                         n_components = as.integer(n_components),
                         store_scores = TRUE,
                         scale_x = FALSE,
                         scale_y = FALSE)
}

fit_predict <- function(seed) {
    xy <- pls4all_bench_load_xy(a$csv_dir, a$n, a$p, seed)
    d <- .formula_design(xy)
    X <- d$X
    y <- d$y
    p <- ncol(X)
    n <- nrow(X)

    call_params <- params
    call_nc <- nc

    if (needs_sw) {
        call_params$sample_weights <- abs(y) + 0.5
    }
    if (needs_labels) {
        n_classes <- if (!is.null(call_params$n_classes))
            as.integer(call_params$n_classes) else 2L
        ord <- order(y)
        labels <- integer(n)
        labels[ord] <- (seq_len(n) - 1L) %% n_classes
        call_params$y_labels <- as.integer(labels)
    }
    if (needs_groups) {
        n_groups <- max(1L, min(5L, p))
        width <- max(1L, as.integer(ceiling(p / n_groups)))
        ga <- pmin((seq_len(p) - 1L) %/% width, n_groups - 1L)
        call_params$group_assignment <- as.integer(ga)
    }
    if (needs_x_target) {
        call_params$X_target <- pls4all_bench_load_x_target(
            x_target_dir, a$n, a$p, seed)
    }
    if (a$algo == "pls_cox") {
        if (is.null(call_params$sample_weights)) {
            call_params$sample_weights <- abs(y) + 0.5
        }
        if (is.null(call_params$y_labels)) {
            ord <- order(y)
            labels <- integer(n)
            labels[ord] <- (seq_len(n) - 1L) %% 2L
            call_params$y_labels <- as.integer(labels)
        }
        call_params$survival_times <- as.double(call_params$sample_weights)
        call_params$event_indicators <-
            as.integer(as.integer(call_params$y_labels) > 0L)
    }

    call_params$n_samples <- NULL
    call_params$n_features <- NULL

    Y <- y
    if (a$algo %in% c("pls_lda", "pls_logistic", "pls_qda",
                       "sparse_pls_da", "pls_cox", "ds", "pds")) {
        Y <- rep(0.0, length(y))
    }
    n_targets <- if (!is.null(call_params$n_targets))
        as.integer(call_params$n_targets) else 1L
    if (n_targets > 1L) {
        cols <- list(as.numeric(y))
        for (j in seq_len(n_targets - 1L)) {
            col_idx <- ((j - 1L) %% p) + 1L
            cols[[j + 1L]] <- 0.5 * as.numeric(y) +
                              0.1 * as.numeric(X[, col_idx])
        }
        Y <- do.call(cbind, cols)
    }

    if (a$algo %in% c("pls", "pcr", "cppls") && n_targets == 1L) {
        method <- switch(a$algo,
            "pls" = "simpls",
            "pcr" = "pcr",
            "cppls" = "cppls")
        args <- list(x = X, y = y, ncomp = call_nc,
                     method = method, center = TRUE,
                     scale = FALSE, cv = NULL,
                     fit_components = FALSE)
        if (a$algo == "cppls") args$gamma <- call_params$gamma %||% 0.5
        fit <- do.call(pls4all::pls, args)
        return(as.numeric(predict(fit, x = X, ncomp = call_nc)))
    }

    dispatch_algo <- a$algo
    use_model_api <- FALSE
    model_api_algo <- ""
    if (a$algo == "pls") {
        dispatch_algo <- "sparse_simpls"
        if (is.null(call_params$sparsity_lambda)) {
            call_params$sparsity_lambda <- 0.0
        }
    } else if (a$algo == "pcr") {
        use_model_api <- TRUE
        model_api_algo <- "pcr_svd"
    } else if (a$algo == "opls") {
        use_model_api <- TRUE
        model_api_algo <- "opls_nipals"
    } else if (a$algo == "kernel_pls_rbf") {
        dispatch_algo <- "kernel_pls"
        if (is.null(call_params$kernel_type)) call_params$kernel_type <- 1L
    } else if (a$algo == "approximate_press") {
        dispatch_algo <- "approximate_press_compute"
        call_nc <- as.integer(call_params$max_components %||% call_nc)
    } else if (a$algo == "one_se_rule") {
        dispatch_algo <- "one_se_rule_compute"
        max_components <- as.integer(call_params$max_components %||% call_nc)
        n_folds <- as.integer(call_params$n_folds %||% 5L)
        call_params$fold_rmse_matrix <-
            .fold_rmse_matrix(max_components, n_folds)
        call_nc <- max_components
    } else if (a$algo == "variable_select_vip") {
        dispatch_algo <- "variable_select_rank"
        call_params$method <- 0L
    } else if (a$algo == "variable_select_coef") {
        dispatch_algo <- "variable_select_rank"
        call_params$method <- 1L
    } else if (a$algo == "variable_select_sr") {
        dispatch_algo <- "variable_select_rank"
        call_params$method <- 2L
    }

    if (!is.null(call_params$interval_step)) {
        call_params$step <- as.integer(call_params$interval_step)
        call_params$interval_step <- NULL
    }

    if (use_model_api) {
        res <- pls4all::pls4all_fit(X, Y,
                                    algo = model_api_algo,
                                    n_components = as.integer(call_nc),
                                    scale_x = FALSE,
                                    scale_y = FALSE)
        return(as.numeric(pls4all::pls4all_predict(res, X)))
    }
    if (a$algo %in% c("pls_diagnostic_t2", "pls_diagnostic_q",
                       "pls_diagnostic_dmodx")) {
        model <- .fit_simpls_model(X, Y, call_nc)
        res <- pls4all::pls_diagnostics(model, X)
        return(.materialize_result(res, registry_pkey, p, "r_mdatools_compat"))
    }
    if (a$algo == "pls_monitoring") {
        split <- as.integer(floor(n / 2L))
        Y_mat <- if (is.null(dim(Y))) matrix(as.numeric(Y), ncol = 1L) else Y
        model <- .fit_simpls_model(X[seq_len(split), , drop = FALSE],
                                   Y_mat[seq_len(split), , drop = FALSE],
                                   call_nc)
        res <- pls4all::pls_monitoring(
            model,
            X[seq_len(split), , drop = FALSE],
            X[(split + 1L):n, , drop = FALSE],
            alpha = as.numeric(call_params$alpha %||% 0.05))
        return(.materialize_result(res, registry_pkey, p, "r_mdatools_compat"))
    }
    if (dispatch_algo == "variable_select_rank") {
        model <- .fit_simpls_model(X, Y, call_nc)
        top_k <- as.integer(call_params$top_k %||% 10L)
        res <- switch(a$algo,
            "variable_select_vip" = pls4all::vip_select(model, X, top_k),
            "variable_select_coef" =
                pls4all::coefficient_select(model, X, top_k),
            "variable_select_sr" =
                pls4all::selectivity_ratio_select(model, X, top_k))
        return(.materialize_result(res, registry_pkey, p, "r_mdatools_compat"))
    }

    res <- pls4all::pls4all_method(dispatch_algo, X, Y,
                                   n_components = as.integer(call_nc),
                                   params = call_params)
    .materialize_result(res, registry_pkey, p, "r_mdatools_compat")
}

tr <- pls4all_bench_time_runs(fit_predict, a$runs, a$seed_base)

versions <- list(
    language = paste0("R ", R.version$major, ".", R.version$minor),
    pls4all  = as.character(tryCatch(packageVersion("pls4all"),
                                      error = function(e) "?")),
    registry_method = a$algo,
    facade = "mdatools_compat",
    matrix_facade = TRUE,
    blas     = "linked-BLAS"
)

pls4all_bench_emit(tr$stats, tr$last_preds, a$pred_path, versions)
