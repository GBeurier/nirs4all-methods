\name{msc_transform}
\alias{msc_transform}
\title{Apply a fitted MSC reference to spectra}
\description{Restore a libn4m MSC handle from the supplied training reference and transform new spectra without learning from them.}
\usage{msc_transform(X, reference)}
\arguments{
\item{X}{Finite numeric matrix with the same feature order as training.}
\item{reference}{Numeric reference from \code{msc_fit()}.}
}
\value{Numeric matrix with the same shape as \code{X}.}
