\name{n4m_role_pipeline}
\alias{n4m_role_pipeline}
\alias{n4m_estimator_fit.n4m_role_pipeline}
\alias{predict.n4m_role_pipeline}
\alias{n4m_estimator_transform.n4m_role_pipeline}
\alias{n4m_role_pipeline_export}
\alias{n4m_role_pipeline_import}
\alias{n4m_role_pipeline_steps}
\alias{print.n4m_role_pipeline}
\title{Native role pipelines}
\usage{
n4m_role_pipeline(steps)

\method{n4m_estimator_fit}{n4m_role_pipeline}(object, X, y = NULL, sample_weight = NULL,
  groups = NULL, feature_groups = NULL, blocks = NULL, axis = NULL,
  X_target = NULL, fold_ids = NULL, ...)

\method{predict}{n4m_role_pipeline}(object, newdata, type = c("class", "prob", "decision"), ...)

\method{n4m_estimator_transform}{n4m_role_pipeline}(object, X, ...)

n4m_role_pipeline_export(object, allow_training_rows = FALSE)

n4m_role_pipeline_import(steps, states, feature_names = NULL, class_names = NULL)

n4m_role_pipeline_steps(object)

\method{print}{n4m_role_pipeline}(x, ...)
}
\arguments{
\item{steps}{A list of steps: method ids (\code{"models.pls.cppls"} or
\code{"n4m:models.pls.cppls"}), \code{list(method_id =, params = list())}
or recipe tokens \code{list(class = "n4m:<id>", params = list())}.}

\item{object}{A role pipeline.}

\item{X, newdata}{Numeric matrix or data frame (rows are samples).}

\item{y}{Responses (regression) or class labels (classification).}

\item{sample_weight, groups, feature_groups, blocks, axis, X_target, fold_ids}{Optional
fit inputs, routed natively to the steps that declare them.}

\item{type}{For a final classifier: \code{"class"}, \code{"prob"} or
\code{"decision"}.}

\item{allow_training_rows}{Export states that embed training rows.}

\item{states}{List of N4ME raw vectors, one per stateful step (or the list
returned by \code{n4m_role_pipeline_export()}).}

\item{feature_names}{Optional input column names of the imported pipeline.}

\item{class_names}{Optional label names of an imported classifier.}

\item{x}{A role pipeline to print.}

\item{...}{Unused.}
}
\value{
\code{n4m_role_pipeline()} an unfitted pipeline;
\code{n4m_estimator_fit()} and \code{n4m_role_pipeline_import()} a fitted
one; \code{predict()} and \code{n4m_estimator_transform()} numeric results
(class labels for \code{type = "class"}); \code{n4m_role_pipeline_export()}
a list of \code{list(method_id, n4me, contains_training_rows)};
\code{n4m_role_pipeline_steps()} a data frame describing each step.
}
\description{
\code{n4m_role_pipeline()} builds a trained linear recipe of catalog
estimators: sample filters (training rows only), transformers and selectors,
then one regressor or classifier. The recipe is validated natively (role
order, final step, parameters). \code{n4m_estimator_fit()} fits every step
natively: \code{y} holds responses for a final regressor (a vector or a
matrix, one row per sample) or class labels for a final classifier; steps
that require \code{y} receive every response column (or the class ids),
sample filters drop training rows of every row-aligned input, and inputs
no step uses are refused. Column names of \code{X} are stored and checked
at every later call (a reordered, renamed or missing column is refused);
matrices without column names are positional. The fitted state is one N4ME
payload per stateful step: \code{n4m_role_pipeline_export()} refuses states
that embed training rows unless \code{allow_training_rows = TRUE}, and
\code{n4m_role_pipeline_import()} refuses states that contradict the recipe
(method, parameters, widths). The same states are read by the Python,
JS/WASM and Rust bindings.
}
