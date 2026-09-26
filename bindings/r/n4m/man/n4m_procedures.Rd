\name{n4m_procedures}
\alias{n4m_procedures}
\alias{n4m_split}
\alias{n4m_split.n4m_splitter}
\alias{n4m_augment}
\alias{n4m_augment.n4m_augmenter}
\alias{n4m_run}
\alias{n4m_run.n4m_procedure}
\title{Generic native procedures}
\usage{
n4m_split(object, X, y = NULL, groups = NULL)

\method{n4m_split}{n4m_splitter}(object, X, y = NULL, groups = NULL)

n4m_augment(object, X, axis = NULL)

\method{n4m_augment}{n4m_augmenter}(object, X, axis = NULL)

n4m_run(object, X, y = NULL, ...)

\method{n4m_run}{n4m_procedure}(object, X, y = NULL, ...)
}
\arguments{
\item{object}{A procedure from one of the generated constructors
(\code{\link{n4m_estimator_role_constructors}}).}
\item{X}{Numeric matrix (rows are samples).}
\item{y}{Optional target vector or matrix, when the method uses it.}
\item{groups}{Optional sample groups for group-aware splitters.}
\item{axis}{Optional spectral axis for axis-dependent augmenters.}
\item{...}{Further named inputs (\code{X_target}, \code{fold_ids}, ...).}
}
\value{
\code{n4m_split()} a list of folds, each \code{list(train, test)} of 1-based
row indices; \code{n4m_augment()} the augmented matrix; \code{n4m_run()} a
named list of the native outputs.
}
\description{
Constructors such as \code{n4m_kennard_stone()}, \code{n4m_gaussian_noise()}
or \code{n4m_regression_metrics()} return a procedure: a catalog method run
once, without fitted state. Its class is its role: \code{n4m_splitter}
(\code{n4m_split}), \code{n4m_augmenter} (\code{n4m_augment}, train-only) or
\code{n4m_procedure} (\code{n4m_run}). Parameters, seeds included, are native
and shared with the Python and JS/WASM bindings.
}
