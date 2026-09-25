\name{emsc_fit}
\alias{emsc_fit}
\title{Fit Extended Multiplicative Scatter Correction}
\usage{emsc_fit(X, degree = 2L)}
\arguments{
\item{X}{Finite numeric training matrix.}
\item{degree}{Positive polynomial degree; at least \code{degree + 2} features are required.}
}
\value{Numeric fitted reference spectrum of length \code{ncol(X)}.}
\description{Fits the native n4m EMSC kernel on training spectra. Save the
reference and pass it with the same degree to \code{emsc_transform()} for
validation or future samples.}
