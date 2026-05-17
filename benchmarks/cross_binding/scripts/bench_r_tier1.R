# R tier-1 dispatcher — uses _npy.R helpers for parsing / timing / npy
# write / JSON emit. Sources _npy.R from the same directory.
suppressMessages(library(pls4all))

# Robust path to _npy.R when called via Rscript:
.script_dir <- function() {
    a <- commandArgs(trailingOnly = FALSE)
    f <- a[grep("--file=", a, fixed = TRUE)]
    if (length(f) > 0L) {
        return(normalizePath(dirname(sub("^--file=", "", f[[1]]))))
    }
    return(".")
}
source(file.path(.script_dir(), "_npy.R"))

a <- pls4all_bench_parse_args()

fit_predict <- function(seed) {
    xy <- pls4all_bench_load_xy(a$csv_dir, a$n, a$p, seed)
    res <- switch(a$algo,
        "pls"                  = pls4all::sparse_simpls_fit(xy$X, xy$y, a$nc, 0.0),
        "sparse_simpls"        = pls4all::sparse_simpls_fit(xy$X, xy$y, a$nc, 0.05),
        "cppls"                = pls4all::cppls_fit(xy$X, xy$y, a$nc, 0.5),
        "ecr"                  = pls4all::ecr_fit(xy$X, xy$y, a$nc, 0.5),
        "mir_pls"              = pls4all::mir_pls_fit(xy$X, xy$y, a$nc),
        "ridge_pls"            = pls4all::ridge_pls_fit(xy$X, xy$y, a$nc, 1.0),
        "robust_pls"           = pls4all::robust_pls_fit(xy$X, xy$y, a$nc, 1.345, 20L),
        "missing_aware_nipals" = pls4all::missing_aware_nipals_fit(xy$X, xy$y, a$nc),
        # Generic fallback via the unified dispatcher.
        pls4all::pls4all_method(a$algo, xy$X, xy$y, a$nc, params = list())
    )
    as.numeric(res$predictions)
}

tr <- pls4all_bench_time_runs(fit_predict, a$runs, a$seed_base)

versions <- list(
    language = paste0("R ", R.version$major, ".", R.version$minor),
    pls4all  = as.character(tryCatch(packageVersion("pls4all"),
                                      error = function(e) "?")),
    blas     = "linked-BLAS (RhpcBLASctl not installed)"
)

pls4all_bench_emit(tr$stats, tr$last_preds, a$pred_path, versions)
