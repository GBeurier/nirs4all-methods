\name{detrend_transform}
\alias{detrend_transform}
\title{Polynomial detrend transform}
\description{Remove a per-spectrum polynomial baseline with libn4m.}
\usage{detrend_transform(X, polyorder = 1L)}
\arguments{
\item{X}{Numeric matrix.}
\item{polyorder}{Non-negative baseline polynomial order.}
}
\value{Numeric matrix with the same shape as \code{X}.}
