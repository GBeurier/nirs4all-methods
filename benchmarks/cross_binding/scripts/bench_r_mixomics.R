# External R: mixOmics — sparse PLS regression / PLS-DA (where applicable).
suppressMessages(library(mixOmics))

.script_dir <- function() {
    a <- commandArgs(trailingOnly = FALSE)
    f <- a[grep("--file=", a, fixed = TRUE)]
    if (length(f) > 0L) normalizePath(dirname(sub("^--file=", "", f[[1]]))) else "."
}
source(file.path(.script_dir(), "_npy.R"))

a <- pls4all_bench_parse_args()
params <- pls4all_bench_params()
nc <- pls4all_bench_param_int(params, "n_components", a$nc)
n_classes <- pls4all_bench_param_int(params, "n_classes", 2L)
sparsity_lambda <- pls4all_bench_param_num(params, "sparsity_lambda", 0.05)

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

class_factor <- function(y) {
    labels <- pls4all_bench_labels(y, n_classes)
    factor(labels, levels = seq.int(0L, n_classes - 1L),
           labels = paste0("c", seq.int(0L, n_classes - 1L)))
}

one_hot <- function(pred, levels_ref) {
    idx <- match(as.character(pred), levels_ref)
    out <- matrix(0.0, nrow = length(idx), ncol = length(levels_ref))
    ok <- !is.na(idx)
    if (any(ok)) {
        out[cbind(which(ok), idx[ok])] <- 1.0
    }
    out
}

fit_predict <- function(seed) {
    xy <- pls4all_bench_load_xy(a$csv_dir, a$n, a$p, seed)
    if (a$algo == "pls") {
        fit <- mixOmics::pls(xy$X, xy$y, ncomp = nc, scale = FALSE,
                              mode = "regression")
        as.numeric(predict(fit, newdata = xy$X)$predict[, , nc])
    } else if (a$algo == "sparse_simpls") {
        keep_size <- max(2L, min(a$p, floor(a$p * max(0.05, 1.0 - sparsity_lambda))))
        keep <- rep(keep_size, nc)
        fit <- mixOmics::spls(xy$X, xy$y, ncomp = nc, keepX = keep,
                                scale = FALSE, mode = "regression")
        as.numeric(predict(fit, newdata = xy$X)$predict[, , nc])
    } else if (a$algo == "pls_da") {
        y_cls <- class_factor(xy$y)
        fit <- mixOmics::plsda(xy$X, y_cls, ncomp = nc, scale = FALSE)
        pred <- predict(fit, newdata = xy$X)$class$max.dist[, nc]
        one_hot(pred, levels(y_cls))
    } else {  # sparse_pls_da
        y_cls <- class_factor(xy$y)
        keep_size <- max(2L, min(a$p, floor(a$p * max(0.05, 1.0 - sparsity_lambda))))
        keep <- rep(keep_size, nc)
        fit <- mixOmics::splsda(xy$X, y_cls, ncomp = nc, keepX = keep,
                                  scale = FALSE)
        pred <- predict(fit, newdata = xy$X)$class$max.dist[, nc]
        one_hot(pred, levels(y_cls))
    }
}

tr <- pls4all_bench_time_runs(fit_predict, a$runs, a$seed_base)
versions <- list(
    language = paste0("R ", R.version$major, ".", R.version$minor),
    mixOmics = as.character(packageVersion("mixOmics"))
)
pls4all_bench_emit(tr$stats, tr$last_preds, a$pred_path, versions)
