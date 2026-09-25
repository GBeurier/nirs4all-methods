\name{area_normalization_transform}
\alias{area_normalization_transform}
\title{Area normalization transform}
\description{Normalize spectra using libn4m's row area operator.}
\usage{area_normalization_transform(X, method = "sum")}
\arguments{
\item{X}{Numeric matrix.}
\item{method}{One of \code{"sum"}, \code{"abs_sum"}, or \code{"trapz"}.}
}
\value{Numeric matrix with the same shape as \code{X}.}
