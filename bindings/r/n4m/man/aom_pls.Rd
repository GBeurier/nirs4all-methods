\name{aom_pls}
\alias{aom_pls}
\alias{aompls}
\alias{pop_pls}
\alias{poppls}
\title{AOM-PLS and POP-PLS adaptive operator selection}
\usage{
aom_pls(X, Y, max_components = 3L, n_operators = 9L, cv = 3L)

aompls(X, Y, max_components = 3L, n_operators = 9L, cv = 3L)

pop_pls(X, Y, max_components = 3L, n_operators = 9L, cv = 3L)

poppls(X, Y, max_components = 3L, n_operators = 9L, cv = 3L)
}
\arguments{
\item{X}{Numeric matrix of spectra, with rows as samples and columns as wavelengths.}
\item{Y}{Numeric vector or matrix of responses, with one row per sample.}
\item{max_components}{Maximum number of latent PLS components.}
\item{n_operators}{Number of compact AOM bank operators to expose.}
\item{cv}{Number of contiguous cross-validation folds.}
}
\value{
A named list. Both functions return \code{predictions}, \code{operator_kinds},
component-selection diagnostics, cross-validation scores, and selected
component metadata.
}
\description{
\code{aom_pls()} (alias \code{aompls()}) runs global Adaptive Operator
Mixture PLS selection:
one preprocessing operator is selected for the whole PLS model.
\code{pop_pls()} (alias \code{poppls()}) runs per-component operator
selection, where each retained PLS component may use a different
preprocessing operator.
}
\details{
The operator bank mirrors the compact public \code{nirs4all} AOM bank:
identity, Savitzky-Golay smoothers, Savitzky-Golay derivatives, polynomial
detrending, and finite difference. The implementation is provided by the
native n4m AOM selector ABI.
}
\examples{
set.seed(1)
X <- matrix(rnorm(40 * 20), nrow = 40)
Y <- as.numeric(X[, 1] + rnorm(40, sd = 0.1))
fit <- aom_pls(X, Y, max_components = 2L)
dim(fit$predictions)
fit2 <- pop_pls(X, Y, max_components = 2L)
dim(fit2$predictions)
}
\keyword{models}
