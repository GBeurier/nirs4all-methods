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

# n_components handed in via argv (clamped by orchestrator).
nc <- a$nc

# Inline-deterministic extras: labels/sample_weights/group_assignment can
# be reproduced from y/p without depending on Python's PCG64. The
# orchestrator sets BENCH_R_NEEDS_* env vars when the active method needs
# them, so we don't speculatively attach extras the C dispatcher would
# reject.
needs_labels       <- Sys.getenv("BENCH_R_NEEDS_LABELS", unset = "") == "1"
needs_sw           <- Sys.getenv("BENCH_R_NEEDS_SAMPLE_WEIGHTS", unset = "") == "1"
needs_groups       <- Sys.getenv("BENCH_R_NEEDS_GROUP_ASSIGNMENT", unset = "") == "1"
x_target_path     <- Sys.getenv("BENCH_R_X_TARGET_PATH", unset = "")

fit_predict <- function(seed) {
    xy <- pls4all_bench_load_xy(a$csv_dir, a$n, a$p, seed)
    p <- a$p
    n <- a$n

    call_params <- params

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

    # Aliases for algos the R-side C dispatcher names differently than
    # the registry. "pls" runs through `sparse_simpls` with lambda=0
    # (numerically identical to Model.fit/SIMPLS used by the Python
    # registry function). "pcr" routes to the SVD-PCR path.
    dispatch_algo <- a$algo
    if (a$algo == "pls") {
        dispatch_algo <- "sparse_simpls"
        if (is.null(call_params$sparsity_lambda)) {
            call_params$sparsity_lambda <- 0.0
        }
    } else if (a$algo == "pcr") {
        dispatch_algo <- "pcr_svd"
    }

    res <- pls4all::pls4all_method(dispatch_algo, xy$X, Y,
                                     n_components = as.integer(nc),
                                     params = call_params)

    # Extract predictions: most methods expose `$predictions`; selectors
    # expose `$selected_indices` (1×p mask convention to match the
    # Python registry).
    if (!is.null(res$predictions)) {
        return(as.numeric(res$predictions))
    }
    if (!is.null(res$selected_indices)) {
        sel <- as.integer(res$selected_indices)
        mask <- double(a$p)
        idx <- sel[sel > 0L & sel <= a$p]
        if (length(idx) > 0L) mask[idx] <- 1.0
        return(mask)
    }
    stop(sprintf("r_tier1: %s returned neither predictions nor selected_indices",
                  a$algo))
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
