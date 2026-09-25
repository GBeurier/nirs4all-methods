\name{emsc_transform}
\alias{emsc_transform}
\title{Apply fitted Extended Multiplicative Scatter Correction}
\usage{emsc_transform(X, reference, degree = 2L)}
\arguments{
\item{X}{Finite numeric matrix in training feature order.}
\item{reference}{Reference returned by \code{emsc_fit()}.}
\item{degree}{Same polynomial degree used at fit time.}
}
\value{Numeric matrix with the same dimensions as \code{X}.}
\description{Applies native n4m EMSC using a saved training reference.}
