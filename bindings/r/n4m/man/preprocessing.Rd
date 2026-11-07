\name{snv_transform}
\alias{snv_transform}
\alias{savgol_transform}
\alias{n4m_transform_snv}
\alias{n4m_transform_savgol}
\title{Portable preprocessing transforms}
\usage{
snv_transform(X, with_mean = TRUE, with_std = TRUE, ddof = 0L)

savgol_transform(
  X,
  window_length,
  polyorder = 3L,
  deriv = 0L,
  delta = 1,
  mode = "mirror",
  cval = 0
)
}
\arguments{
\item{X}{Numeric matrix.}
\item{with_mean}{Unclassed logical or exact numeric 0/1 scalar; center each row.}
\item{with_std}{Unclassed logical or exact numeric 0/1 scalar; scale by row standard deviation.}
\item{ddof}{Unclassed exact numeric/integer scalar in [0, INT32_MAX], not logical or missing.
With scaling, it must be smaller than the feature count even for constant or nonfinite rows.
Centering-only does not use variance and accepts the full nonnegative range.}
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
