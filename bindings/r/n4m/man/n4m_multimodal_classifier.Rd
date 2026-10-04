\name{n4m_multimodal_classifier}
\alias{n4m_multimodal_classifier}
\alias{n4m_multimodal_classifier_from_state}
\title{Native Raw Multimodal PLS-logistic Classifier}
\usage{
n4m_multimodal_classifier(recipe, source_schemas)
n4m_multimodal_classifier_from_state(state, recipe, source_schemas,
                                   class_names = NULL)
}
\arguments{
\item{recipe}{Closed version-one early-fusion recipe using the existing native
PLS-logistic head, with positive integer n_components and max_iter.}
\item{source_schemas}{Named raw source declarations in recipe source order.}
\item{state}{Bounded raw N4MC bytes containing encoders and classifier state.}
\item{class_names}{Optional unique string or exact integer original labels in
native class-column order. Without this table, native integer IDs are exposed.}
}
\description{The portable C ABI learns raw StandardScaler, TensorPCA and mixed
metadata encoders and a genuine PLS-logistic classifier. Bindings marshal raw
arrays and typed label tables; they contain no learned numerical logic.}
\details{Use \code{object <- n4m_fit(object, X, y)}. Failed fitting leaves the
supplied object usable. \code{predict(object, X, type = "prob")} returns columns
in \code{n4m_classes(object)} order; \code{type = "class"} decodes original labels.
N4MC bytes contain no training rows. Carry class_names beside the portable state
to restore original labels. Recipe, schema and native class consistency are
validated before hydration. ABI 2.17 is required.}
\value{An n4m_multimodal_classifier object with a native owner.}
