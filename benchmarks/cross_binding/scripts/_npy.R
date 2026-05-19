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

# Timed runs (small unmeasured warmup set, same convention as Python _common).
pls4all_bench_now_ms <- function() {
    as.numeric(Sys.time()) * 1000.0
}

pls4all_bench_warmup_count <- function(runs) {
    max(1L, min(3L, as.integer(runs)))
}

pls4all_bench_time_runs <- function(fit_predict_seeded, runs, seed_base) {
    if (runs < 1L) {
        stop("runs must be >= 1")
    }
    for (w in seq_len(pls4all_bench_warmup_count(runs))) {
        fit_predict_seeded(seed_base)
    }
    samples <- numeric(runs)
    last_preds <- NULL
    for (i in seq_len(runs)) {
        t0 <- pls4all_bench_now_ms()
        last_preds <- fit_predict_seeded(seed_base + (i - 1L))
        samples[i] <- pls4all_bench_now_ms() - t0
    }
    list(stats = list(
            ok = TRUE,
            n_runs = length(samples),
            median_ms = median(samples),
            min_ms = min(samples),
            max_ms = max(samples)
         ),
         last_preds = last_preds)
}

# Common 8-arg parse — returns a named list with algo / csv_dir / n / p /
# nc / runs / seed_base / pred_path.
pls4all_bench_parse_args <- function() {
    argv <- commandArgs(trailingOnly = TRUE)
    if (length(argv) != 8L) {
        stop("usage: SCRIPT algo csv_dir n p n_components n_runs seed_base predictions_path")
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
