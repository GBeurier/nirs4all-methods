# SPDX-License-Identifier: CECILL-2.1

.n4m_splitter_kinds <- c(
    kennard_stone = 0L, spxy = 1L, spxy_fold = 2L,
    spxy_group_fold = 3L, kmeans = 4L, kbins_stratified = 5L,
    binned_strat_group_fold = 6L, systematic_circular = 7L,
    data_twinning = 8L)

#' Split samples through a native n4m splitter
#'
#' The nine kinds use the same native result contract. `X` is required for
#' Kennard-Stone, SPXY, SPXY folds, KMeans and DataTwinning. `Y` is required
#' for SPXY, SPXY folds, KBins, binned group folds and SystematicCircular.
#' Group folds additionally require one exact integer group ID per row.
#' `fold` is one-based in R; results are one-based unless `zero_based=TRUE`.
#' No partitioning arithmetic is performed in R.
#'
#' @param kind One of the nine native splitter names.
#' @param X,Y Optional finite numeric matrices, as required by `kind`.
#' @param groups Optional exact signed integer group IDs, required by group folds.
#' @param fold One-based fold number for fold kinds; otherwise 1.
#' @param test_size Test fraction for non-fold kinds.
#' @param n_splits Number of folds for fold kinds.
#' @param y_metric 0=none, 1=Euclidean, 2=Hamming for SPXY folds.
#' @param aggregation 0=mean or 1=median for grouped SPXY folds.
#' @param n_bins,strategy Bin count and 0=uniform/1=quantile strategy.
#' @param shuffle Whether binned group folds shuffle with `seed`.
#' @param max_iter Maximum KMeans iterations.
#' @param seed Exact non-negative seed, at most `2^53-1` in R.
#' @param zero_based Return C-style zero-based indices.
#' @return A list of ordered native `train` and `test` integer indices.
#' @export
n4m_splitter_run <- function(kind, X = NULL, Y = NULL, groups = NULL,
                             fold = 1L, test_size = 0.25, n_splits = 3L,
                             y_metric = 1L, aggregation = 0L,
                             n_bins = 5L, strategy = 0L, shuffle = TRUE,
                             max_iter = 100L, seed = 0, zero_based = FALSE) {
    if (!is.character(kind) || length(kind) != 1L || is.na(kind) ||
        !(kind %in% names(.n4m_splitter_kinds)))
        stop("unknown native splitter kind", call. = FALSE)
    code <- unname(.n4m_splitter_kinds[[kind]])
    needs_x <- code %in% c(0L, 1L, 2L, 3L, 4L, 8L)
    needs_y <- code %in% c(1L, 2L, 3L, 5L, 6L, 7L)
    needs_groups <- code %in% c(3L, 6L)
    fold_kind <- code %in% c(2L, 3L, 6L)
    valid_matrix <- function(value) is.matrix(value) && is.numeric(value) &&
        all(dim(value) > 0L) && !anyNA(value) && all(is.finite(value))
    if (needs_x != !is.null(X) || needs_y != !is.null(Y) ||
        (needs_x && !valid_matrix(X)) || (needs_y && !valid_matrix(Y)))
        stop("X/Y inputs must match the native splitter kind and be finite matrices",
             call. = FALSE)
    if (needs_x && needs_y && nrow(X) != nrow(Y))
        stop("X and Y must have matching rows", call. = FALSE)
    if (needs_groups != !is.null(groups))
        stop("groups must be supplied only for group fold kinds", call. = FALSE)
    if (needs_groups && (!is.numeric(groups) || anyNA(groups) ||
        length(groups) != (if (needs_x) nrow(X) else nrow(Y)) ||
        any(!is.finite(groups)) || any(groups != floor(groups)) ||
        any(abs(groups) > 2^53 - 1)))
        stop("groups must contain one exact integer ID per row", call. = FALSE)
    scalars <- c(n_splits, y_metric, aggregation, n_bins, strategy,
                 as.integer(shuffle), max_iter, test_size, seed, fold)
    if (anyNA(scalars) || any(!is.finite(scalars)) ||
        any(scalars[c(1:7, 10)] != floor(scalars[c(1:7, 10)])) ||
        any(abs(scalars[1:7]) > .Machine$integer.max) ||
        !fold_kind && fold != 1L || fold_kind && (fold < 1L || fold > n_splits) ||
        seed < 0 || seed > 2^53 - 1 || seed != floor(seed))
        stop("invalid native splitter parameters", call. = FALSE)
    if (!is.null(X)) storage.mode(X) <- "double"
    if (!is.null(Y)) storage.mode(Y) <- "double"
    .Call("r_n4m_splitter_run", as.integer(code), X, Y,
          if (is.null(groups)) NULL else as.double(groups),
          as.double(scalars[1:9]), as.integer(fold - 1L),
          as.logical(zero_based), PACKAGE = "n4m")
}
