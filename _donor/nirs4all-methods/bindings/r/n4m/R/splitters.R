# SPDX-License-Identifier: CECILL-2.1

#' Kennard-Stone train/test split.
#'
#' Deterministic split of `X` into train and test subsets that maximises
#' the spread of the training set (Kennard & Stone 1969).
#'
#' @param X numeric matrix.
#' @param test_size numeric(1). Fraction of samples assigned to the test set.
#' @return list with two integer vectors of 1-based indices: `train_idx` and `test_idx`.
#' @export
kennard_stone <- function(X, test_size = 0.25) {
  n4m_split_kennard_stone_split(X, test_size = test_size)
}

#' SPXY (Sample set Partitioning based on X and Y) split.
#'
#' Variant of Kennard-Stone that also incorporates the response `Y`.
#'
#' @param X numeric matrix.
#' @param Y numeric matrix or vector.
#' @param test_size numeric(1).
#' @return list with `train_idx` and `test_idx` integer vectors.
#' @export
spxy <- function(X, Y, test_size = 0.25) {
  n4m_split_spxy_split(X, Y, test_size = test_size)
}

#' K-bins stratified train/test split.
#'
#' @param Y numeric response vector or matrix.
#' @param test_size numeric(1).
#' @param seed numeric(1).
#' @param n_bins integer(1).
#' @param strategy character(1) or integer(1). One of "uniform", "quantile".
#' @return list with `train_idx` and `test_idx` integer vectors.
#' @export
kbins_stratified <- function(Y, test_size = 0.25, seed = 0,
                             n_bins = 5L, strategy = "uniform") {
  strategy_int <- if (is.character(strategy)) {
    switch(strategy,
           "uniform"  = 0L,
           "quantile" = 1L,
           stop(sprintf("kbins_stratified: unknown strategy '%s'", strategy),
                call. = FALSE))
  } else {
    as.integer(strategy)[1L]
  }
  n4m_split_kbins_stratified_split(Y, test_size = test_size, seed = seed,
                                   n_bins = n_bins, strategy = strategy_int)
}
