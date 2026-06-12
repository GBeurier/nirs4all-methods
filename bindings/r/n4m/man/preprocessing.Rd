\name{snv_transform}
\alias{snv_transform}
\alias{savgol_transform}
\title{Portable preprocessing transforms}
\usage{
snv_transform(X, with_mean = TRUE, with_std = TRUE, ddof = 0L)

savgol_transform(
  X,
  window_length,
  polyorder = 2L,
  deriv = 0L,
  delta = 1,
  mode = "mirror",
  cval = 0
)
}
\arguments{
\item{X}{Numeric matrix.}
\item{with_mean}{Logical; center each row before scaling.}
\item{with_std}{Logical; scale each row by its standard deviation.}
\item{ddof}{Integer degrees-of-freedom correction.}
\item{window_length}{Odd integer Savitzky-Golay window length.}
\item{polyorder}{Savitzky-Golay polynomial order.}
\item{deriv}{Savitzky-Golay derivative order.}
\item{delta}{Sample spacing.}
\item{mode}{Boundary mode: \code{"mirror"}, \code{"constant"},
\code{"nearest"}, \code{"wrap"}, or \code{"interp"}.}
\item{cval}{Constant fill value used when \code{mode = "constant"}.}
}
\value{
Numeric matrix with the same dimensions as \code{X}.
}
\description{
Thin R wrappers over libn4m's preprocessing C ABI. The R layer performs
matrix layout conversion only; all numerical work is delegated to libn4m.
}
