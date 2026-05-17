# R tier-2 dispatcher — formula+S3 wrappers (pls / sparse_pls / cppls / …).
suppressMessages(library(pls4all))

.script_dir <- function() {
    a <- commandArgs(trailingOnly = FALSE)
    f <- a[grep("--file=", a, fixed = TRUE)]
    if (length(f) > 0L) normalizePath(dirname(sub("^--file=", "", f[[1]]))) else "."
}
source(file.path(.script_dir(), "_npy.R"))

a <- pls4all_bench_parse_args()

fit_predict <- function(seed) {
    xy <- pls4all_bench_load_xy(a$csv_dir, a$n, a$p, seed)
    df <- as.data.frame(xy$X)
    df$y <- xy$y
    fit <- switch(a$algo,
        "pls"                  = pls(y ~ ., df, ncomp = a$nc, algo = "pls_simpls"),
        "sparse_simpls"        = sparse_pls(y ~ ., df, a$nc, 0.05),
        "cppls"                = cppls(y ~ ., df, a$nc, 0.5),
        "ecr"                  = ecr(y ~ ., df, a$nc, 0.5),
        "mir_pls"              = mir_pls(y ~ ., df, a$nc),
        "ridge_pls"            = ridge_pls(y ~ ., df, a$nc, 1.0),
        "robust_pls"           = robust_pls(y ~ ., df, a$nc, 1.345, 20L),
        "missing_aware_nipals" = missing_aware_nipals(y ~ ., df, a$nc),
        "continuum_regression" = continuum_regression(y ~ ., df, a$nc, 0.5),
        "bagging_pls"          = bagging_pls(y ~ ., df, a$nc, 50L, 0L),
        "boosting_pls"         = boosting_pls(y ~ ., df, a$nc, 50L, 0.1),
        "o2pls"                = o2pls(y ~ ., df, a$nc),
        stop(sprintf("r_tier2: unsupported algo '%s'", a$algo))
    )
    as.numeric(predict(fit, newdata = df))
}

tr <- pls4all_bench_time_runs(fit_predict, a$runs, a$seed_base)

versions <- list(
    language = paste0("R ", R.version$major, ".", R.version$minor),
    pls4all  = as.character(tryCatch(packageVersion("pls4all"),
                                      error = function(e) "?")),
    blas     = "linked-BLAS"
)

pls4all_bench_emit(tr$stats, tr$last_preds, a$pred_path, versions)
