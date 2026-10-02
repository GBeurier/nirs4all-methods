\name{n4m_multimodal_pipeline}
\alias{n4m_multimodal_pipeline}
\alias{n4m_multimodal_pipeline_from_state}
\alias{n4m_export_state}
\alias{n4m_export_state.n4m_multimodal_pipeline}
\alias{n4m_close}
\alias{n4m_close.n4m_multimodal_pipeline}
\alias{n4m_estimator_fit.n4m_multimodal_pipeline}
\alias{n4m_estimator_transform.n4m_multimodal_pipeline}
\alias{predict.n4m_multimodal_pipeline}
\title{Complete Native Raw Multimodal Pipeline}
\usage{
n4m_multimodal_pipeline(recipe, source_schemas)
n4m_multimodal_pipeline_from_state(state, recipe, source_schemas)
n4m_export_state(object)
n4m_close(object)
\method{n4m_estimator_fit}{n4m_multimodal_pipeline}(object, X, y = NULL, ...)
\method{predict}{n4m_multimodal_pipeline}(object, newdata,
  source_schemas = object$source_schemas, ...)
\method{n4m_estimator_transform}{n4m_multimodal_pipeline}(object, X,
  source_schemas = object$source_schemas, ...)
}
\arguments{
\item{recipe}{Closed version-one early-fusion recipe: ordered nir, image,
series and metadata encoders, source weights and one centered, unscaled-X Ridge.}
\item{source_schemas}{Named raw source descriptors containing representation_id,
input_shape, dtype and the unchanged canonical IO identity string.}
\item{state}{Raw vector containing a bounded native N4MF fitted state.}
\item{object}{A native multimodal pipeline.}
\item{X, newdata}{Named list of IO-aligned sample-first raw numeric arrays and
an n-by-two character matrix (or declared numeric/string data frame) for metadata.}
\item{y}{One numeric target per sample.}
\item{...}{Unsupported additional options are refused.}
}
\description{
Native population scaling, learned PCA and training-only UTF-8 one-hot encoding
precede weighted early fusion and one Ridge predictor. Unknown categories encode
as zero. No host encoders or training rows are stored in the portable state.
Import validates the expected recipe and full source identity without fitting.
}
\value{
Constructors and fit return a pipeline; predict returns a numeric vector,
transform returns the actual weighted encoded matrix, export returns raw N4MF
bytes, and close releases native resources invisibly and idempotently.
}
\details{
Requires ABI 2.16. Numeric arrays use R's double storage. Raw descriptors and
array layouts are translated only; all learned arithmetic belongs to libn4m.
The existing \code{n4m_fit(object, blocks, y)} entry point delegates this class
to \code{n4m_estimator_fit}. Calls with extra constructor controls are refused.
}
