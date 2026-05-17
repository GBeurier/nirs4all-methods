# External R: mixOmics — sparse PLS regression / PLS-DA (where applicable).
suppressMessages(library(mixOmics))

.script_dir <- function() {
    a <- commandArgs(trailingOnly = FALSE)
    f <- a[grep("--file=", a, fixed = TRUE)]
    if (length(f) > 0L) normalizePath(dirname(sub("^--file=", "", f[[1]]))) else "."
}
source(file.path(.script_dir(), "_npy.R"))

a <- pls4all_bench_parse_args()

supported <- c("pls", "sparse_simpls", "sparse_pls_da", "pls_da")
if (!(a$algo %in% supported)) {
    suppressMessages(library(jsonlite))
    cat(toJSON(list(ok = FALSE,
                     reason = sprintf("algo '%s' not implemented by mixOmics", a$algo),
                     skipped = TRUE,
                     versions = list(
                         language = paste0("R ", R.version$major, ".", R.version$minor),
                         mixOmics = as.character(packageVersion("mixOmics"))
                     )), auto_unbox = TRUE), "\n", sep = "")
    quit(save = "no")
}

fit_predict <- function(seed) {
    xy <- pls4all_bench_load_xy(a$csv_dir, a$n, a$p, seed)
    if (a$algo == "pls") {
        fit <- mixOmics::pls(xy$X, xy$y, ncomp = a$nc, scale = FALSE,
                              mode = "regression")
        as.numeric(predict(fit, newdata = xy$X)$predict[, , a$nc])
    } else if (a$algo == "sparse_simpls") {
        keep <- rep(max(2L, floor(a$p / 4L)), a$nc)
        fit <- mixOmics::spls(xy$X, xy$y, ncomp = a$nc, keepX = keep,
                                scale = FALSE, mode = "regression")
        as.numeric(predict(fit, newdata = xy$X)$predict[, , a$nc])
    } else if (a$algo == "pls_da") {
        # Binary classification from y > 0.
        y_cls <- factor(ifelse(xy$y > 0, "pos", "neg"))
        fit <- mixOmics::plsda(xy$X, y_cls, ncomp = a$nc, scale = FALSE)
        as.numeric(as.integer(predict(fit, newdata = xy$X)$class$max.dist[, a$nc]))
    } else {  # sparse_pls_da
        y_cls <- factor(ifelse(xy$y > 0, "pos", "neg"))
        keep <- rep(max(2L, floor(a$p / 4L)), a$nc)
        fit <- mixOmics::splsda(xy$X, y_cls, ncomp = a$nc, keepX = keep,
                                  scale = FALSE)
        as.numeric(as.integer(predict(fit, newdata = xy$X)$class$max.dist[, a$nc]))
    }
}

tr <- pls4all_bench_time_runs(fit_predict, a$runs, a$seed_base)
versions <- list(
    language = paste0("R ", R.version$major, ".", R.version$minor),
    mixOmics = as.character(packageVersion("mixOmics"))
)
pls4all_bench_emit(tr$stats, tr$last_preds, a$pred_path, versions)
