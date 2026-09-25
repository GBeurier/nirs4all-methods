\name{n4m_model_import_linear_predictor}
\alias{n4m_model_import_linear_predictor}
\title{Import an affine predictor as a portable N4MM model}
\usage{
n4m_model_import_linear_predictor(coefficients, intercept,
                                  source_training_samples = 0L)
}
\arguments{
\item{coefficients}{Finite numeric features-by-targets matrix.}
\item{intercept}{Finite numeric vector, one value per target.}
\item{source_training_samples}{Non-negative training row count, or zero when
the source format does not attest it.}
}
\value{A native model pointer accepted by \code{n4m_predict()} and
\code{n4m_model_export()}.}
\description{Calls the n4m C ABI affine importer without fitting coefficients.
Predictions are \code{X \%*\% coefficients + intercept}. The resulting model
is exportable to portable N4MM but has no PLS latent decomposition.}
