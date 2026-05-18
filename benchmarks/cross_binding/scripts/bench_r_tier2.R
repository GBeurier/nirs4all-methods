# R tier-2 dispatcher — formula+S3 wrappers from `bindings/r/pls4all/R/sklearn*.R`.
#
# Limited by what the formula API exposes (≈15 methods). Pulls per-algo
# scalars from `BENCH_R_PARAMS_JSON` (same path as r_tier1) so the
# constants (lambda / gamma / n_estimators / seed / …) match the
# registry's `adapted_params`. Methods unwrapped on the formula side
# fail cleanly with "no formula wrapper" so the dashboard shows `—`.

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
    params <- fromJSON(params_env, simplifyVector = TRUE)
} else {
    params <- list()
}

nc <- if (!is.null(params$n_components)) as.integer(params$n_components) else a$nc

# Formula `y ~ .` collapses the response to a 1-D vector. Methods whose
# canonical cell uses multi-target Y (n_targets > 1) can't run
# faithfully through the formula API; fail with a clean reason so the
# dashboard shows `—` instead of a shape-mismatch crash downstream.
n_targets <- if (!is.null(params$n_targets)) as.integer(params$n_targets) else 1L
if (n_targets > 1L) {
    stop(sprintf("r_tier2: %s uses n_targets=%d; formula API takes 1-D y",
                  a$algo, n_targets))
}

fit_predict <- function(seed) {
    xy <- pls4all_bench_load_xy(a$csv_dir, a$n, a$p, seed)
    df <- as.data.frame(xy$X)
    df$y <- xy$y

    fit <- switch(a$algo,
        "pls" = pls(y ~ ., df, ncomp = nc,
                     algo = "pls_simpls"),
        "sparse_simpls" = sparse_pls(y ~ ., df, nc,
                                       params$sparsity_lambda %||% 0.05),
        "cppls" = cppls(y ~ ., df, nc, params$gamma %||% 0.5),
        "ecr"   = ecr(y ~ ., df, nc, params$alpha %||% 0.5),
        "mir_pls" = mir_pls(y ~ ., df, nc),
        "ridge_pls" = ridge_pls(y ~ ., df, nc,
                                  params$ridge_lambda %||% 1.0),
        "robust_pls" = robust_pls(y ~ ., df, nc,
                                    params$huber_k %||% 1.345,
                                    as.integer(params$max_irls_iter %||% 20L)),
        "missing_aware_nipals" = missing_aware_nipals(y ~ ., df, nc),
        "pcr" = pls(y ~ ., df, ncomp = nc, algo = "pcr_svd"),
        "continuum_regression" = continuum_regression(y ~ ., df, nc,
                                                       params$tau %||% 0.5),
        "bagging_pls" = bagging_pls(y ~ ., df, nc,
                                      as.integer(params$n_estimators %||% 50L),
                                      as.integer(params$seed %||% 0L)),
        "boosting_pls" = boosting_pls(y ~ ., df, nc,
                                        as.integer(params$n_estimators %||% 50L),
                                        params$learning_rate %||% 0.1),
        "random_subspace_pls" = random_subspace_pls(y ~ ., df, nc,
                                                     as.integer(params$n_estimators %||% 50L),
                                                     as.integer(params$features_per_subspace %||% 10L),
                                                     as.integer(params$seed %||% 0L)),
        "weighted_pls" = weighted_pls(y ~ ., df, ncomp = nc,
                                       weights = abs(df$y) + 0.5),
        "pls_glm" = pls_glm(y ~ ., df, ncomp = nc,
                              family = if (isTRUE(as.logical(params$poisson)))
                                  "poisson" else "gaussian"),
        "mb_pls" = mb_pls(y ~ ., df, ncomp = nc,
                            block_sizes = if (!is.null(params$block_sizes))
                                as.integer(params$block_sizes) else
                                as.integer(rep(ceiling(a$p / 3L), 3L))),
        "o2pls" = stop("r_tier2: o2pls needs multi-target Y (formula API limitation)"),
        stop(sprintf("r_tier2: no formula wrapper for '%s'", a$algo))
    )
    res <- as.numeric(predict(fit, newdata = df))
    res
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
