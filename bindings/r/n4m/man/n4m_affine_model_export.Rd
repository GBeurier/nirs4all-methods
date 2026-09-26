\name{n4m_affine_model_export}
\alias{n4m_affine_model_export}
\title{Export a fitted affine MethodResult predictor}
\usage{n4m_affine_model_export(object)}
\arguments{\item{object}{An \code{n4m_affine_fit} object.}}
\value{Portable N4MM raw bytes for native prediction.}
\description{
Exports the fitted native affine model. The bytes preserve prediction, not
the original fitting algorithm or its parameters.
}
