\name{n4m_preprocess_fit}
\alias{n4m_preprocess_capabilities}
\alias{n4m_preprocess_step}
\alias{n4m_preprocess_fit}
\alias{n4m_preprocess_transform}
\alias{n4m_preprocess_export}
\alias{n4m_preprocess_import}
\alias{n4m_preprocess_plan}
\title{Generic native preprocessing pipeline}
\usage{
n4m_preprocess_capabilities()

n4m_preprocess_step(kind, params = numeric())

n4m_preprocess_fit(X, steps, Y = NULL)

n4m_preprocess_transform(object, X)

n4m_preprocess_export(object)

n4m_preprocess_import(bytes)

n4m_preprocess_plan(object)
}
\arguments{
\item{kind}{Operator name from \code{n4m_preprocess_capabilities()}.}
\item{params}{Finite numeric vector of positional C ABI parameters for the operator.}
\item{X}{Finite numeric matrix. At transform time, its feature width must match training.}
\item{steps}{Nonempty list of \code{n4m_preprocess_step()} specifications.}
\item{Y}{Optional finite numeric response matrix, required for supervised OSC and EPO.}
\item{object}{Fitted native pipeline returned by \code{n4m_preprocess_fit()}.}
\item{bytes}{A raw vector containing a fitted native N4MP pipeline.}
}
\value{
\code{n4m_preprocess_capabilities()} returns a data frame of names, numeric
codes, and support flags. \code{n4m_preprocess_step()} returns an operator
specification. \code{n4m_preprocess_fit()} returns fitted native state.
\code{n4m_preprocess_transform()} returns a numeric matrix.
\code{n4m_preprocess_export()} returns portable raw bytes;
\code{n4m_preprocess_import()} restores a fitted native pipeline.
\code{n4m_preprocess_plan()} returns its ordered original operator plan.
}
\description{
A thin R interface to the libn4m C ABI pipeline. The C ABI names 19 operator
kinds; 15 currently implement generic pipeline fit and transform. The other
four are reported as unsupported by \code{n4m_preprocess_capabilities()} and
rejected by \code{n4m_preprocess_step()}.
}
\details{
All numerical fitting and transformation is performed by libn4m. The fitted
state is an in-process native pointer; save the exported N4MP raw bytes,
not this pointer, for use in another R session or supported language binding.

Parameter vectors are positional; use \code{numeric()} for the native default.
The contract is defined by \code{n4m_operator_kind_t} and the native pipeline
parameter parsers in \code{cpp/src/core/pipeline.cpp}:
\describe{
\item{No parameters}{identity, center, autoscale, pareto_scale, snv, msc.}
\item{Optional one integer}{emsc polynomial degree (default 2, range 0--5);
detrend_poly degree (default 1, range 0--5).}
\item{Savitzky--Golay}{savgol_smooth takes window, polynomial degree (defaults
11, 3); savgol_derivative takes window, polynomial degree, derivative order,
and optional delta (defaults 11, 3, 1, 1). Window must be odd; delta is the
positive feature spacing.}
\item{Other unsupervised}{norris_williams takes odd segment, odd gap, derivative
order (defaults 5, 3, 1); asls_baseline takes lambda, asymmetry, iterations
(defaults 100000, 0.001, 10); wavelet_denoise takes levels, nonnegative
threshold (defaults 1, 0).}
\item{Supervised}{osc and epo take max_iter, positive tolerance (defaults 100,
1e-10); each requires \code{Y} at fit time.}
}
The native fit rejects invalid arity or parameter ranges; R does not duplicate
the numerical contract.
}
