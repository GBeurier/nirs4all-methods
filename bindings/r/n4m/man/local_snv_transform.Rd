\name{local_snv_transform}
\alias{local_snv_transform}
\title{Local Standard Normal Variate transform}
\description{Apply libn4m's sliding-window SNV to each spectrum.}
\usage{local_snv_transform(X, window = 11L, pad_mode = "reflect", constant_value = 0)}
\arguments{
\item{X}{Numeric matrix.}
\item{window}{Odd sliding window length.}
\item{pad_mode}{One of \code{"reflect"}, \code{"edge"}, or \code{"constant"}.}
\item{constant_value}{Padding value in constant mode.}
}
\value{Numeric matrix with the same shape as \code{X}.}
