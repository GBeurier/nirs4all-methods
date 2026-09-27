\name{n4m_estimator_roles}
\alias{n4m_estimator_roles}
\alias{n4m_estimator_fit}
\alias{n4m_estimator_fit.n4m_estimator}
\alias{predict.n4m_regressor}
\alias{predict.n4m_classifier}
\alias{n4m_classes}
\alias{n4m_classes.n4m_classifier}
\alias{n4m_sample_mask}
\alias{n4m_sample_mask.n4m_sample_filter}
\alias{n4m_estimator_transform}
\alias{n4m_estimator_transform.n4m_transformer}
\alias{n4m_estimator_transform.n4m_selector}
\alias{n4m_selected_indices}
\alias{n4m_selected_indices.n4m_selector}
\alias{n4m_estimator_export}
\alias{n4m_contains_training_rows}
\alias{n4m_estimator_import}
\alias{print.n4m_method}
\title{Generic native estimator roles}
\usage{
n4m_estimator_fit(object, X, y = NULL, ...)

\method{n4m_estimator_fit}{n4m_estimator}(object, X, y = NULL, sample_weight = NULL,
  groups = NULL, feature_groups = NULL, blocks = NULL, axis = NULL,
  X_target = NULL, fold_ids = NULL, ...)

\method{predict}{n4m_regressor}(object, newdata, ...)

\method{predict}{n4m_classifier}(object, newdata, type = c("class", "prob", "decision"), ...)

n4m_classes(object)

\method{n4m_classes}{n4m_classifier}(object)

n4m_sample_mask(object, X, y = NULL)

\method{n4m_sample_mask}{n4m_sample_filter}(object, X, y = NULL)

n4m_estimator_transform(object, X, ...)

\method{n4m_estimator_transform}{n4m_transformer}(object, X, ...)

\method{n4m_estimator_transform}{n4m_selector}(object, X, ...)

n4m_selected_indices(object)

\method{n4m_selected_indices}{n4m_selector}(object)

n4m_estimator_export(object, allow_training_rows = FALSE)

n4m_contains_training_rows(object)

n4m_estimator_import(bytes)

\method{print}{n4m_method}(x, ...)
}
\arguments{
\item{object}{An estimator from one of the generated constructors
(\code{\link{n4m_estimator_role_constructors}}).}
\item{X}{Numeric matrix (rows are samples).}
\item{y}{Numeric response vector (one value per row of \code{X}) or matrix
(one row per row of \code{X}) for regressors; class labels (factor, character
or numeric vector, one per row) for classifiers: integer labels are the class
ids (within int64), strings and other finite numbers become a label table;
missing (NA) and non-finite labels are refused. Nothing is recycled: another
length is an error.}
\item{sample_weight, groups, feature_groups, blocks, axis, X_target, fold_ids}{
Optional fit inputs; each method declares which ones it requires and the
native core refuses the others. Per-row inputs have one entry per row of
\code{X}, \code{feature_groups} and \code{axis} one per column.}
\item{newdata}{Numeric matrix of new samples.}
\item{type}{For classifiers: \code{"class"} (labels, a factor when the
estimator was fitted on a factor or character vector), \code{"prob"} (class
probabilities, for methods that define them) or \code{"decision"}
(method-defined class scores). Matrix columns are named by class.}
\item{allow_training_rows}{\code{n4m_estimator_export()} refuses a state that
embeds training rows (\code{n4m_contains_training_rows()}, e.g. kernel PLS,
GPR-PLS, LW-PLS) unless this is \code{TRUE}: sharing the export shares those
rows. \code{saveRDS()} keeps them, as a checkpoint of the in-process object.}
\item{bytes}{Raw vector produced by \code{n4m_estimator_export()}.}
\item{x}{An estimator.}
\item{...}{Unused.}
}
\value{
\code{n4m_estimator_fit()} returns the fitted estimator (a failed fit leaves
the object passed in unchanged); \code{predict()} and
\code{n4m_estimator_transform()} return numeric results (class labels for
\code{predict(type = "class")}); \code{n4m_classes()} the fitted classes; \code{n4m_sample_mask()} a logical
vector (\code{TRUE} keeps the row);
\code{n4m_estimator_export()} a raw vector; \code{n4m_contains_training_rows()}
a logical; \code{n4m_estimator_import()} a fitted estimator.
}
\description{
Constructors such as \code{n4m_pls_regression()} or \code{n4m_cppls()} return
an unfitted estimator whose classes are its role interfaces:
\code{n4m_regressor} (\code{predict}), \code{n4m_classifier}
(\code{predict}, \code{n4m_classes}), \code{n4m_transformer}
(\code{n4m_estimator_transform}) and \code{n4m_selector}
(\code{n4m_estimator_transform}, which returns the selected columns in input
order, and \code{n4m_selected_indices}, 1-based in native selection order) and
\code{n4m_sample_filter} (\code{n4m_sample_mask}, a logical keep mask; filters
are train-only, filters on the target read \code{y}). The fitted state is kept as portable N4ME
bytes, so \code{saveRDS()}/\code{readRDS()} and the Python and JS/WASM
bindings reuse it without refitting.
}
