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
  c4a_split_kennard_stone_split(X, test_size = test_size)
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
  c4a_split_spxy_split(X, Y, test_size = test_size)
}
