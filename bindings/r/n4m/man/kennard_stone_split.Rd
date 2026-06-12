\name{kennard_stone_split}
\alias{kennard_stone_split}
\title{Kennard-Stone train/test split}
\usage{
kennard_stone_split(X, test_size = 0.25, zero_based = FALSE)
}
\arguments{
\item{X}{Numeric matrix.}
\item{test_size}{Fraction of samples assigned to the test set.}
\item{zero_based}{Logical; return 0-based indices when \code{TRUE}. The
default \code{FALSE} returns idiomatic 1-based R indices.}
}
\value{
Named list with integer vectors \code{train} and \code{test}.
}
\description{
Thin R wrapper over libn4m's Kennard-Stone splitter C ABI. The R layer performs
matrix layout conversion only; all split logic is delegated to libn4m.
}
