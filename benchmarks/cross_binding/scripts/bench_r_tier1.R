# R tier-1 — pls4all canonical R invocation via the registry-driven
# `pls4all::pls4all_method(algo, X, Y, n_components, params=...)` dispatcher.
#
# The orchestrator pushes a JSON-encoded params object via the
# `BENCH_R_PARAMS_JSON` env var (Python computes it via
# `bench_registry_common.adapted_params` so R and Python stay aligned).
# Methods that need data-shaped extras whose values can't be reproduced
# from y/p alone (X_target for di_pls/ds/pds) are read from a sidecar
# .npy file pointed to by `BENCH_R_X_TARGET_PATH`.

suppressMessages(library(pls4all))
suppressMessages(library(jsonlite))

`%||%` <- function(x, y) if (is.null(x)) y else x

.script_dir <- function() {
    a <- commandArgs(trailingOnly = FALSE)
    f <- a[grep("--file=", a, fixed = TRUE)]
    if (length(f) > 0L) {
        return(normalizePath(dirname(sub("^--file=", "", f[[1]]))))
    }
    "."
}
source(file.path(.script_dir(), "_npy.R"))

# Tiny .npy v1.0 reader for 2-D float64 (fortran_order False).
read_npy_f64 <- function(path) {
    con <- file(path, "rb")
    on.exit(close(con))
    magic <- readBin(con, "raw", 6L)
    if (!identical(magic, as.raw(c(0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59)))) {
        stop(sprintf("not a .npy file: %s", path))
    }
    version <- readBin(con, "raw", 2L)
    if (version[[1]] == as.raw(0x01)) {
        hl <- readBin(con, "integer", n = 1L, size = 2L,
                       endian = "little", signed = FALSE)
    } else {
        hl <- readBin(con, "integer", n = 1L, size = 4L,
                       endian = "little", signed = FALSE)
    }
    header <- rawToChar(readBin(con, "raw", hl))
    shape <- regmatches(header,
                        regexec("'shape':\\s*\\(([^)]*)\\)", header))[[1]][2]
    dims <- as.integer(trimws(strsplit(shape, ",")[[1]]))
    dims <- dims[!is.na(dims) & dims > 0L]
    total <- prod(dims)
    arr <- readBin(con, "double", n = total, size = 8L, endian = "little")
    if (length(dims) >= 2L) {
        matrix(arr, nrow = dims[1L], byrow = TRUE)
    } else {
        arr
    }
}

a <- pls4all_bench_parse_args()

# Pull per-algo params from the orchestrator (JSON env var). Empty
# defaults to an empty list so methods whose only param is n_components
# still work via the dispatcher.
params_env <- Sys.getenv("BENCH_R_PARAMS_JSON", unset = "")
if (nzchar(params_env)) {
    params <- fromJSON(params_env, simplifyVector = FALSE)
} else {
    params <- list()
}

# JSON list-of-numbers → integer / double vector for the C dispatcher.
# (`fromJSON(simplifyVector = FALSE)` produces list-of-scalars; the R
# `.Call` interface wants atomic vectors.)
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

# n_components — adapted_params on the Python side may have clamped
# this below the argv value (classifiers need n_components ≤
# n_classes - 1; n_pls is bounded by the (j, k) factor pair). Prefer
# the value from BENCH_R_PARAMS_JSON when present, fall back to argv.
nc <- if (!is.null(params$n_components)) as.integer(params$n_components) else a$nc

# Inline-deterministic extras: labels/sample_weights/group_assignment can
# be reproduced from y/p without depending on Python's PCG64. The
# orchestrator sets BENCH_R_NEEDS_* env vars when the active method needs
# them, so we don't speculatively attach extras the C dispatcher would
# reject.
needs_labels       <- Sys.getenv("BENCH_R_NEEDS_LABELS", unset = "") == "1"
needs_sw           <- Sys.getenv("BENCH_R_NEEDS_SAMPLE_WEIGHTS", unset = "") == "1"
needs_groups       <- Sys.getenv("BENCH_R_NEEDS_GROUP_ASSIGNMENT", unset = "") == "1"
needs_x_target     <- Sys.getenv("BENCH_R_NEEDS_X_TARGET", unset = "") == "1"
x_target_path     <- Sys.getenv("BENCH_R_X_TARGET_PATH", unset = "")
x_target_dir      <- Sys.getenv("BENCH_R_X_TARGET_DIR", unset = "")
registry_pkey      <- Sys.getenv("BENCH_PREDICTION_KEY", unset = "predictions")

.fold_rmse_matrix <- function(max_components, n_folds) {
    idx <- matrix(seq_len(max_components * n_folds) - 1L,
                  nrow = max_components, ncol = n_folds, byrow = TRUE)
    0.5 + 0.5 * ((idx * 37 + 11) %% 997) / 996
}

.materialize_result <- function(res, pkey, p, prefix) {
    if (pkey %in% c("selected_indices", "mask", "support")) {
        if (!is.null(res$mask)) {
            return(matrix(as.numeric(res$mask), nrow = 1))
        }
        if (!is.null(res$support)) {
            return(matrix(as.numeric(res$support), nrow = 1))
        }
        if (!is.null(res$selected_indices)) {
            sel <- as.integer(res$selected_indices)
            mask <- matrix(0.0, nrow = 1, ncol = p)
            idx <- sel[sel > 0L & sel <= p]
            if (length(idx) > 0L) mask[1, idx] <- 1.0
            return(mask)
        }
        # Empty selector outputs are valid all-zero masks.
        return(matrix(0.0, nrow = 1, ncol = p))
    }
    if (pkey == "decision_scores" && !is.null(res$decision_scores)) {
        return(res$decision_scores)
    }
    if (!is.null(res[[pkey]])) {
        return(res[[pkey]])
    }
    if (!is.null(res$predictions)) {
        return(res$predictions)
    }
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
    p <- a$p
    n <- a$n

    call_params <- params
    call_nc <- nc

    if (needs_sw) {
        call_params$sample_weights <- abs(xy$y) + 0.5
    }
    if (needs_labels) {
        n_classes <- if (!is.null(call_params$n_classes))
            as.integer(call_params$n_classes) else 2L
        ord <- order(xy$y)
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
    if (nzchar(x_target_path)) {
        if (!file.exists(x_target_path)) {
            stop(sprintf("X_target sidecar not found: %s", x_target_path))
        }
        call_params$X_target <- read_npy_f64(x_target_path)
    } else if (needs_x_target) {
        call_params$X_target <- pls4all_bench_load_x_target(
            x_target_dir, a$n, a$p, seed)
    }
    if (a$algo == "pls_cox") {
        if (is.null(call_params$sample_weights)) {
            call_params$sample_weights <- abs(xy$y) + 0.5
        }
        if (is.null(call_params$y_labels)) {
            ord <- order(xy$y)
            labels <- integer(n)
            labels[ord] <- (seq_len(n) - 1L) %% 2L
            call_params$y_labels <- as.integer(labels)
        }
        call_params$survival_times <- as.double(call_params$sample_weights)
        call_params$event_indicators <-
            as.integer(as.integer(call_params$y_labels) > 0L)
    }

    # Some methods (so_pls / on_pls) overload `n_components` semantics —
    # the JSON params already carry the canonical key, so any clash with
    # the positional `n_components` is resolved by stripping it here.
    call_params$n_samples <- NULL
    call_params$n_features <- NULL

    # Classifier algos use a labels y_labels and ignore Y; the dispatcher
    # in R wants a numeric Y placeholder.
    Y <- xy$y
    if (a$algo %in% c("pls_lda", "pls_logistic", "pls_qda",
                       "sparse_pls_da", "pls_cox", "ds", "pds")) {
        Y <- rep(0.0, length(xy$y))
    }
    # Multi-target methods (so_pls / rosa with n_targets > 1) need a
    # matrix Y matching the synthetic columns the Python registry
    # generates in benchmark_inputs: column 0 is y, column j (>0) is
    # 0.5*y + 0.1*X[:, (j-1) %% p]. Without this the C kernel runs on
    # 1-D y and diverges from the registry path.
    n_targets <- if (!is.null(call_params$n_targets))
        as.integer(call_params$n_targets) else 1L
    if (n_targets > 1L) {
        cols <- list(as.numeric(xy$y))
        for (j in seq_len(n_targets - 1L)) {
            col_idx <- ((j - 1L) %% a$p) + 1L
            cols[[j + 1L]] <- 0.5 * as.numeric(xy$y) +
                              0.1 * as.numeric(xy$X[, col_idx])
        }
        Y <- do.call(cbind, cols)
    }

    # Aliases for algos the R-side C dispatcher names differently than
    # the registry. "pls" runs through `sparse_simpls` with lambda=0
    # (numerically identical to Model.fit/SIMPLS used by the Python
    # registry function). "pcr" needs the Model API on the R side
    # because `pls4all_method`/`r_p4a_dispatch_fit` does not include
    # the PCR/SVD entry; we route to `pls4all_fit(...)` instead.
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
        if (is.null(call_params$kernel_type)) {
            call_params$kernel_type <- 1L  # RBF
        }
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

    # The C dispatcher uses `step`, not the registry's `interval_step`.
    if (!is.null(call_params$interval_step)) {
        call_params$step <- as.integer(call_params$interval_step)
        call_params$interval_step <- NULL
    }

    if (a$algo %in% c("pls_diagnostic_t2", "pls_diagnostic_q",
                       "pls_diagnostic_dmodx")) {
        model <- .fit_simpls_model(xy$X, Y, call_nc)
        res <- pls4all::pls_diagnostics(model, xy$X)
        return(.materialize_result(res, registry_pkey, a$p, "r_tier1"))
    }
    if (a$algo == "pls_monitoring") {
        split <- as.integer(floor(nrow(xy$X) / 2L))
        Y_mat <- if (is.null(dim(Y))) matrix(as.numeric(Y), ncol = 1L) else Y
        model <- .fit_simpls_model(xy$X[seq_len(split), , drop = FALSE],
                                   Y_mat[seq_len(split), , drop = FALSE],
                                   call_nc)
        res <- pls4all::pls_monitoring(
            model,
            xy$X[seq_len(split), , drop = FALSE],
            xy$X[(split + 1L):nrow(xy$X), , drop = FALSE],
            alpha = as.numeric(call_params$alpha %||% 0.05))
        return(.materialize_result(res, registry_pkey, a$p, "r_tier1"))
    }
    if (dispatch_algo == "variable_select_rank") {
        model <- .fit_simpls_model(xy$X, Y, call_nc)
        top_k <- as.integer(call_params$top_k %||% 10L)
        res <- switch(a$algo,
            "variable_select_vip" = pls4all::vip_select(model, xy$X, top_k),
            "variable_select_coef" =
                pls4all::coefficient_select(model, xy$X, top_k),
            "variable_select_sr" =
                pls4all::selectivity_ratio_select(model, xy$X, top_k))
        return(.materialize_result(res, registry_pkey, a$p, "r_tier1"))
    }

    if (use_model_api) {
        # Model API for PCR / OPLS via the high-level R interface.
        # The C dispatcher `pls4all_method` doesn't expose pcr_svd /
        # opls_nipals as named methods, so we go through the lower-level
        # `pls4all_fit` model wrapper instead.
        res <- pls4all::pls4all_fit(xy$X, Y,
                                      algo = model_api_algo,
                                      n_components = as.integer(call_nc),
                                      scale_x = FALSE,
                                      scale_y = FALSE)
        preds_vec <- as.numeric(pls4all::pls4all_predict(res, xy$X))
        return(preds_vec)
    }

    res <- pls4all::pls4all_method(dispatch_algo, xy$X, Y,
                                     n_components = as.integer(call_nc),
                                     params = call_params)
    .materialize_result(res, registry_pkey, a$p, "r_tier1")
}

tr <- pls4all_bench_time_runs(fit_predict, a$runs, a$seed_base)

versions <- list(
    language = paste0("R ", R.version$major, ".", R.version$minor),
    pls4all  = as.character(tryCatch(packageVersion("pls4all"),
                                      error = function(e) "?")),
    registry_method = a$algo,
    blas     = "linked-BLAS"
)

pls4all_bench_emit(tr$stats, tr$last_preds, a$pred_path, versions)
