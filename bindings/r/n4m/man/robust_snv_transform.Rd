\name{robust_snv_transform}
\alias{robust_snv_transform}
\title{Robust Standard Normal Variate transform}
\description{Apply libn4m's median and robust-dispersion row normalization.}
\usage{robust_snv_transform(X, with_center = TRUE, with_scale = TRUE, k = 1.4826)}
\arguments{
\item{X}{Numeric matrix.}
\item{with_center}{Center each row by its median.}
\item{with_scale}{Scale each row by robust dispersion.}
\item{k}{Robust scale factor.}
}
\value{Numeric matrix with the same shape as \code{X}.}
