\name{n4m_splitter_run}
\alias{n4m_splitter_run}
\title{Run one of nine native sample splitters}
\usage{
n4m_splitter_run(kind, X = NULL, Y = NULL, groups = NULL,
  fold = 1L, test_size = 0.25, n_splits = 3L, y_metric = 1L,
  aggregation = 0L, n_bins = 5L, strategy = 0L, shuffle = TRUE,
  max_iter = 100L, seed = 0, zero_based = FALSE)
}
\arguments{
\item{kind}{One of \code{kennard_stone}, \code{spxy}, \code{spxy_fold},
\code{spxy_group_fold}, \code{kmeans}, \code{kbins_stratified},
\code{binned_strat_group_fold}, \code{systematic_circular}, or
\code{data_twinning}.}
\item{X, Y}{Finite numeric matrices required by the selected kind.}
\item{groups}{Exact integer group ID per sample for group-fold kinds.}
\item{fold}{One-based fold number for fold kinds.}
\item{test_size}{Fraction assigned to the test set for non-fold kinds.}
\item{n_splits}{Number of folds for fold kinds.}
\item{y_metric}{SPXY-fold Y metric: 0 none, 1 Euclidean, 2 Hamming.}
\item{aggregation}{Grouped SPXY aggregation: 0 mean, 1 median.}
\item{n_bins}{Number of target bins for binned splitters.}
\item{strategy}{Bin strategy: 0 uniform, 1 quantile.}
\item{shuffle}{Whether binned group folds shuffle.}
\item{max_iter}{Maximum KMeans iterations.}
\item{seed}{Exact nonnegative random seed.}
\item{zero_based}{Return zero-based indices when \code{TRUE}.}
}
\value{A list with native-ordered integer vectors \code{train} and \code{test}.}
\description{Runs the selected existing libn4m splitter via one C ABI dispatch.
R only marshals data and indices; partitioning remains native.}
