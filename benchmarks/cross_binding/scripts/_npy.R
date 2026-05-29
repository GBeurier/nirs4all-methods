# Tiny numpy .npy v1.0 writer for 1-D float64 arrays. Compatible with
# `numpy.load(path)`. Used by all bench_r_*.R scripts to persist the
# last-run predictions for post-hoc parity computation.

write_npy_f64 <- function(arr, path) {
    # R stores matrices column-major; numpy's default load convention is
    # C-order (row-major). If the input is a 2-D matrix, transpose so the
    # column-major-flatten produced by `as.double()` lands in the order a
    # Python `np.load(path).ravel()` would expect after a `.reshape(rows,
    # cols)`. We then declare the 2-D shape in the npy header so the
    # cross-binding parity comparison can reshape consistently.
    if (is.matrix(arr)) {
        rows <- nrow(arr)
        cols <- ncol(arr)
        flat <- as.double(t(arr))
        shape_str <- sprintf("(%d, %d)", rows, cols)
    } else {
        flat <- as.double(arr)
        shape_str <- sprintf("(%d,)", length(flat))
    }
    header_body <- sprintf(
        "{'descr': '<f8', 'fortran_order': False, 'shape': %s, }",
        shape_str)
    raw_magic <- as.raw(c(0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59))  # \x93NUMPY
    version <- as.raw(c(0x01, 0x00))
    # numpy spec: total header (magic + version + len(2) + header) is
    # multiple of 64. The header itself must end with '\n'.
    base_len <- length(raw_magic) + length(version) + 2L + nchar(header_body) + 1L
    pad <- (64L - (base_len %% 64L)) %% 64L
    header_full <- paste0(header_body, strrep(" ", pad), "\n")
    hl <- nchar(header_full)
    header_len_le <- as.raw(c(bitwAnd(hl, 0xff), bitwShiftR(hl, 8)))
    con <- file(path, "wb")
    on.exit(close(con))
    writeBin(raw_magic, con)
    writeBin(version, con)
    writeBin(header_len_le, con)
    writeBin(charToRaw(header_full), con)
    writeBin(flat, con, size = 8L, endian = "little")
}

# Make `pls4all_bench_emit(stats, last_preds, pred_path, versions)`
# available to bench scripts so we don't repeat the JSON+npy plumbing.
pls4all_bench_emit <- function(stats, last_preds, pred_path, versions) {
    suppressMessages(library(jsonlite))
    dir.create(dirname(pred_path), recursive = TRUE, showWarnings = FALSE)
    write_npy_f64(last_preds, pred_path)
    rec <- c(stats, list(predictions_path = pred_path, versions = versions))
    cat(toJSON(rec, auto_unbox = TRUE), "\n", sep = "")
}

# Load the orchestrator-cached CSV for (n, p, seed). All backends share
# the same CSV so cross-language parity makes sense.
pls4all_bench_load_xy <- function(csv_dir, n, p, seed) {
    # Force integer formatting (no scientific, no decimals) so the
    # filename matches the orchestrator's Python f-string.
    csv <- file.path(csv_dir, sprintf("data_%dx%d_seed%.0f.csv",
                                        n, p, seed))
    arr <- as.matrix(read.csv(csv, header = TRUE))
    list(X = arr[, -ncol(arr), drop = FALSE],
         y = arr[, ncol(arr)])
}

pls4all_bench_load_x_target <- function(x_target_dir, n, p, seed) {
    path <- file.path(x_target_dir, sprintf("xtarget_%dx%d_seed%.0f.csv",
                                             n, p, seed))
    if (!file.exists(path)) {
        stop(sprintf("X_target sidecar not found: %s", path))
    }
    as.matrix(read.csv(path, header = FALSE))
}

# Adaptive timed runs (same convention as Python _common).
pls4all_bench_now_ms <- function() {
    as.numeric(Sys.time()) * 1000.0
}

pls4all_bench_adaptive_target <- function(probe_ms, max_runs) {
    max_runs <- max(2L, as.integer(max_runs))
    if (probe_ms > 60000.0) {
        return(list(target = 2L, statistic = "single",
                    decision = "probe_gt_60s"))
    }
    if (probe_ms > 30000.0) {
        return(list(target = min(max_runs, 2L), statistic = "single",
                    decision = "probe_gt_30s"))
    }
    if (probe_ms > 5000.0) {
        return(list(target = min(max_runs, 3L), statistic = "mean",
                    decision = "probe_gt_5s"))
    }
    if (probe_ms > 1000.0) {
        return(list(target = min(max_runs, 10L), statistic = "median",
                    decision = "probe_gt_1s"))
    }
    if (probe_ms > 100.0) {
        return(list(target = min(max_runs, 20L), statistic = "median",
                    decision = "probe_gt_100ms"))
    }
    list(target = min(max_runs, 40L), statistic = "median",
         decision = "probe_le_100ms")
}

pls4all_bench_stats <- function(samples, statistic, warmup_ms, decision,
                                max_runs, total_runs,
                                warmup_included = FALSE,
                                prediction_seed = NA_real_) {
    if (length(samples) == 1L) {
        statistic <- "single"
    }
    reported <- if (identical(statistic, "mean")) mean(samples) else median(samples)
    list(
        ok = TRUE,
        n_runs = length(samples),
        # Always seed_base: the persisted parity prediction is the seed_base
        # run, comparable across backends regardless of adaptive run count.
        prediction_seed = prediction_seed,
        median_ms = unname(reported),
        reported_ms = unname(reported),
        sample_median_ms = unname(median(samples)),
        mean_ms = unname(mean(samples)),
        min_ms = unname(min(samples)),
        max_ms = unname(max(samples)),
        warmup_ms = unname(warmup_ms),
        warmup_included = warmup_included,
        timing_statistic = statistic,
        timing_decision = decision,
        max_runs = max(1L, as.integer(max_runs)),
        total_runs = max(1L, as.integer(total_runs))
    )
}

pls4all_bench_time_runs <- function(fit_predict_seeded, runs, seed_base) {
    max_runs <- max(1L, as.integer(runs))
    # Number of warm-up executions discarded before the first timed probe.
    # One warm-up is not enough to amortise R/.Call dispatch + BLAS spin-up
    # on the first few calls.
    warmup_runs <- suppressWarnings(
        as.integer(Sys.getenv("BENCH_WARMUP_RUNS", unset = "3")))
    if (is.na(warmup_runs) || warmup_runs < 1L) {
        warmup_runs <- 3L
    }

    # Discard `warmup_runs` executions on seed_base so the timed probe is warm.
    warmup_preds <- NULL
    warmup_ms <- 0.0
    for (w in seq_len(warmup_runs)) {
        t0 <- pls4all_bench_now_ms()
        warmup_preds <- fit_predict_seeded(seed_base)
        warmup_ms <- pls4all_bench_now_ms() - t0
    }
    if (warmup_ms > 300000.0) {
        return(list(
            stats = pls4all_bench_stats(
                c(warmup_ms), "single", warmup_ms, "warmup_gt_5min",
                max_runs, 1L, TRUE, prediction_seed = seed_base),
            last_preds = warmup_preds
        ))
    }
    if (max_runs < 2L) {
        return(list(
            stats = pls4all_bench_stats(
                c(warmup_ms), "single", warmup_ms,
                "max_runs_1_warmup_only", max_runs, 1L, TRUE,
                prediction_seed = seed_base),
            last_preds = warmup_preds
        ))
    }

    samples <- numeric(0)
    t0 <- pls4all_bench_now_ms()
    fit_predict_seeded(seed_base)
    samples <- c(samples, pls4all_bench_now_ms() - t0)

    target <- pls4all_bench_adaptive_target(samples[[1]], max_runs)
    target_samples <- max(1L, target$target - 1L)
    if (target_samples > 1L) {
        for (i in 2:target_samples) {
            t0 <- pls4all_bench_now_ms()
            fit_predict_seeded(seed_base + (i - 1L))
            samples <- c(samples, pls4all_bench_now_ms() - t0)
        }
    }
    # Parity prediction is always the seed_base run (warmup_preds), never the
    # last timed run, so it is comparable across backends.
    list(stats = pls4all_bench_stats(samples, target$statistic, warmup_ms,
                                     target$decision, max_runs,
                                     warmup_runs + length(samples),
                                     prediction_seed = seed_base),
         last_preds = warmup_preds)
}

# Common 8-arg parse — `runs` is the adaptive total-run cap.
pls4all_bench_parse_args <- function() {
    argv <- commandArgs(trailingOnly = TRUE)
    if (length(argv) != 8L) {
        stop("usage: SCRIPT algo csv_dir n p n_components max_total_runs seed_base predictions_path")
    }
    list(
        algo       = argv[[1]],
        csv_dir    = argv[[2]],
        n          = as.integer(argv[[3]]),
        p          = as.integer(argv[[4]]),
        nc         = as.integer(argv[[5]]),
        runs       = as.integer(argv[[6]]),
        seed_base  = as.numeric(argv[[7]]),
        pred_path  = argv[[8]]
    )
}

pls4all_bench_params <- function() {
    raw <- Sys.getenv("BENCH_R_PARAMS_JSON", unset = "")
    if (!nzchar(raw)) {
        return(list())
    }
    suppressMessages(library(jsonlite))
    fromJSON(raw, simplifyVector = FALSE)
}

pls4all_bench_param_value <- function(params, name, default) {
    value <- params[[name]]
    if (is.null(value)) {
        return(default)
    }
    unlist(value, use.names = FALSE)[[1]]
}

pls4all_bench_param_int <- function(params, name, default) {
    as.integer(pls4all_bench_param_value(params, name, default))
}

pls4all_bench_param_num <- function(params, name, default) {
    as.numeric(pls4all_bench_param_value(params, name, default))
}

pls4all_bench_labels <- function(y, n_classes) {
    y <- as.numeric(y)
    n <- length(y)
    n_classes <- as.integer(max(2L, n_classes))
    labels <- integer(n)
    order_idx <- order(y)
    labels[order_idx] <- (seq_len(n) - 1L) %% n_classes
    labels
}
