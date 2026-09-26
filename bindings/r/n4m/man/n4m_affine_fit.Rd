\name{n4m_affine_fit}
\alias{n4m_affine_fit}
\alias{predict.n4m_affine_fit}
\title{Fit and predict a native affine MethodResult regressor}
\usage{
n4m_affine_fit(method, X, Y, n_components = 2L, params = list())
\method{predict}{n4m_affine_fit}(object, newdata, ...)
}
\arguments{
\item{method}{A qualified n4m MethodResult regression method.}
\item{X}{A finite numeric samples-by-features matrix.}
\item{Y}{A finite numeric response vector or samples-by-targets matrix.}
\item{n_components}{A positive component count; ridge ignores it.}
\item{params}{A named list of method-specific parameters.}
\item{object}{An \code{n4m_affine_fit} object.}
\item{newdata}{A finite numeric matrix in the fitted feature order.}
\item{...}{Unused.}
}
\value{
\code{n4m_affine_fit()} returns an S3 object with a native prediction
model. \code{predict()} returns a numeric vector for one target or a matrix
for multiple targets.
}
\description{
Fits a MethodResult through libn4m and converts a natively marked affine
result to a native prediction model. Held-out predictions use libn4m.
}
\details{
Only methods listed by \code{n4m_affine_supported_methods()} are currently
qualified for this conversion. The native converter refuses unmarked results.
}
