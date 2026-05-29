# External R: `pls` package — SIMPLS, NIPALS via oscorespls, cppls, pcr.
suppressMessages(library(pls))

.script_dir <- function() {
    a <- commandArgs(trailingOnly = FALSE)
    f <- a[grep("--file=", a, fixed = TRUE)]
    if (length(f) > 0L) normalizePath(dirname(sub("^--file=", "", f[[1]]))) else "."
}
source(file.path(.script_dir(), "_npy.R"))

a <- pls4all_bench_parse_args()

supported <- c("pls", "cppls", "pcr")
if (!(a$algo %in% supported)) {
    suppressMessages(library(jsonlite))
    cat(toJSON(list(ok = FALSE,
                     reason = sprintf("algo '%s' not implemented by pls::", a$algo),
                     skipped = TRUE,
                     versions = list(
                         language = paste0("R ", R.version$major, ".", R.version$minor),
                         pls = as.character(packageVersion("pls"))
                     )), auto_unbox = TRUE), "\n", sep = "")
    quit(save = "no")
}

fit_predict <- function(seed) {
    xy <- pls4all_bench_load_xy(a$csv_dir, a$n, a$p, seed)
    df <- as.data.frame(xy$X)
    df$y <- xy$y
    fit <- switch(a$algo,
        "pls"   = pls::plsr(y ~ ., data = df, ncomp = a$nc,
                              method = "simpls",
                              validation = "none", scale = FALSE),
        "cppls" = pls::cppls(y ~ ., data = df, ncomp = a$nc,
                               validation = "none", scale = FALSE),
        "pcr"   = pls::pcr(y ~ ., data = df, ncomp = a$nc,
                             validation = "none", scale = FALSE)
    )
    as.numeric(predict(fit, newdata = df, ncomp = a$nc))
}

tr <- pls4all_bench_time_runs(fit_predict, a$runs, a$seed_base)
versions <- list(
    language = paste0("R ", R.version$major, ".", R.version$minor),
    pls      = as.character(packageVersion("pls"))
)
pls4all_bench_emit(tr$stats, tr$last_preds, a$pred_path, versions)
