\name{msc_fit}
\alias{msc_fit}
\title{Fit native Multiplicative Scatter Correction}
\description{Fit libn4m MSC on training spectra and return only the learned reference spectrum. Do not refit on validation or prediction samples.}
\usage{msc_fit(X)}
\arguments{\item{X}{Finite numeric training matrix with at least two features.}}
\value{Numeric fitted reference spectrum, length \code{ncol(X)}.}
