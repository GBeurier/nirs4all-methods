\name{n4m_estimator_roles}
\alias{n4m_estimator_roles}
\alias{n4m_estimator_fit}
\alias{n4m_estimator_fit.n4m_estimator}
\alias{predict.n4m_regressor}
\alias{n4m_estimator_transform}
\alias{n4m_estimator_transform.n4m_transformer}
\alias{n4m_estimator_transform.n4m_selector}
\alias{n4m_selected_indices}
\alias{n4m_selected_indices.n4m_selector}
\alias{n4m_estimator_export}
\alias{n4m_estimator_import}
\alias{print.n4m_estimator}
\title{Generic native estimator roles}
\usage{
n4m_estimator_fit(object, X, y = NULL, ...)

\method{n4m_estimator_fit}{n4m_estimator}(object, X, y = NULL, sample_weight = NULL,
  groups = NULL, feature_groups = NULL, blocks = NULL, axis = NULL,
  X_target = NULL, fold_ids = NULL, ...)

\method{predict}{n4m_regressor}(object, newdata, ...)

n4m_estimator_transform(object, X, ...)

\method{n4m_estimator_transform}{n4m_transformer}(object, X, ...)

\method{n4m_estimator_transform}{n4m_selector}(object, X, ...)

n4m_selected_indices(object)

\method{n4m_selected_indices}{n4m_selector}(object)

n4m_estimator_export(object)

n4m_estimator_import(bytes)

\method{print}{n4m_estimator}(x, ...)
}
\arguments{
\item{object}{An estimator from one of the generated constructors
(\code{\link{n4m_estimator_role_constructors}}).}
\item{X}{Numeric matrix (rows are samples).}
\item{y}{Numeric response vector or matrix; required by regressors.}
\item{sample_weight, groups, feature_groups, blocks, axis, X_target, fold_ids}{
Optional fit inputs; each method declares which ones it requires and the
native core refuses the others.}
\item{newdata}{Numeric matrix of new samples.}
\item{bytes}{Raw vector produced by \code{n4m_estimator_export()}.}
\item{x}{An estimator.}
\item{...}{Unused.}
}
\value{
\code{n4m_estimator_fit()} returns the fitted estimator; \code{predict()} and
\code{n4m_estimator_transform()} return numeric results;
\code{n4m_estimator_export()} a raw vector; \code{n4m_estimator_import()} a
fitted estimator.
}
\description{
Constructors such as \code{n4m_pls_regression()} or \code{n4m_cppls()} return
an unfitted estimator whose classes are its role interfaces:
\code{n4m_regressor} (\code{predict}), \code{n4m_transformer}
(\code{n4m_estimator_transform}) and \code{n4m_selector}
(\code{n4m_estimator_transform}, which returns the selected columns in input
order, and \code{n4m_selected_indices}, 1-based in native selection order). The fitted state is kept as portable N4ME
bytes, so \code{saveRDS()}/\code{readRDS()} and the Python and JS/WASM
bindings reuse it without refitting.
}
