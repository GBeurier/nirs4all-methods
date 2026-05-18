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

# Timed runs (warmup-aware median, same convention as Python _common).
pls4all_bench_time_runs <- function(fit_predict_seeded, runs, seed_base) {
    samples <- numeric(runs)
    last_preds <- NULL
    for (i in seq_len(runs)) {
        t0 <- proc.time()[["elapsed"]]
        last_preds <- fit_predict_seeded(seed_base + (i - 1L))
        samples[i] <- (proc.time()[["elapsed"]] - t0) * 1000
    }
    timed <- if (runs >= 3L) samples[-1L] else samples
    list(stats = list(
            ok = TRUE,
            n_runs = length(timed),
            median_ms = median(timed),
            min_ms = min(timed),
            max_ms = max(timed)
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
