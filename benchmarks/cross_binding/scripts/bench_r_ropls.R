# External R: ropls — covers OPLS regression.
suppressMessages(library(ropls))

.script_dir <- function() {
    a <- commandArgs(trailingOnly = FALSE)
    f <- a[grep("--file=", a, fixed = TRUE)]
    if (length(f) > 0L) normalizePath(dirname(sub("^--file=", "", f[[1]]))) else "."
}
source(file.path(.script_dir(), "_npy.R"))

a <- pls4all_bench_parse_args()

if (a$algo != "opls") {
    suppressMessages(library(jsonlite))
    cat(toJSON(list(ok = FALSE,
                     reason = sprintf("algo '%s' not implemented by ropls", a$algo),
                     skipped = TRUE,
                     versions = list(
                         language = paste0("R ", R.version$major, ".", R.version$minor),
                         ropls = as.character(packageVersion("ropls"))
                     )), auto_unbox = TRUE), "\n", sep = "")
    quit(save = "no")
}

fit_predict <- function(seed) {
    xy <- pls4all_bench_load_xy(a$csv_dir, a$n, a$p, seed)
    # ropls quietly prints progress; capture.output silences it.
    fit <- suppressMessages(suppressWarnings(
        invisible(capture.output(
            mod <- ropls::opls(x = xy$X, y = xy$y,
                                predI = 1L,
                                orthoI = max(1L, a$nc - 1L),
                                printL = FALSE, plotL = FALSE,
                                scaleC = "center")
        ))
    ))
    as.numeric(predict(mod, xy$X))
}

tr <- pls4all_bench_time_runs(fit_predict, a$runs, a$seed_base)
versions <- list(
    language = paste0("R ", R.version$major, ".", R.version$minor),
    ropls    = as.character(packageVersion("ropls"))
)
pls4all_bench_emit(tr$stats, tr$last_preds, a$pred_path, versions)
