\name{n4m_sample_filter_create}
\alias{n4m_sample_filter_create}
\alias{n4m_sample_filter_add_child}
\alias{n4m_sample_filter_fit}
\alias{n4m_sample_filter_apply}
\alias{n4m_feature_filter_create}
\alias{n4m_feature_filter_fit}
\alias{n4m_feature_filter_indices}
\alias{n4m_feature_filter_transform}
\title{Opaque native row and column filter roles}
\usage{
n4m_sample_filter_create(kind, integers, doubles, seed = 0)
n4m_sample_filter_add_child(filter, kind, integers, doubles, seed = 0)
n4m_sample_filter_fit(filter, X, Y = NULL)
n4m_sample_filter_apply(filter, X, Y = NULL)
n4m_feature_filter_create(kind, threshold = 0, top_k = -1L)
n4m_feature_filter_fit(filter, X, Y = NULL)
n4m_feature_filter_indices(filter)
n4m_feature_filter_transform(filter, X)
}
\arguments{
\item{kind}{A native filter name: row kinds are \code{y_outlier},
\code{x_outlier}, \code{high_leverage}, \code{spectral_quality}, or
\code{composite}; column kinds are \code{variance} or \code{correlation}.}
\item{integers}{Exact positional integer parameters specified in the
\code{n4m/filter_role.h} constructor schema.}
\item{doubles}{Finite positional double parameters in the same schema.}
\item{seed}{Exact nonnegative integer seed up to \code{2^53-1}.}
\item{filter}{An opaque native filter handle returned by a create function.}
\item{X}{Nonempty numeric matrix of spectra.}
\item{Y}{Optional one-column numeric matrix aligned with \code{X}; required for
\code{y_outlier} and \code{correlation}.}
\item{threshold}{Finite variance or absolute-correlation threshold.}
\item{top_k}{Number of columns to retain, or \code{-1} for threshold mode.}
}
\value{Create, add-child and fit functions return the opaque filter handle.
Row apply returns a list with a logical \code{mask} and numeric \code{stats}.
Column indices returns one-based selected input columns in native transform
order. Column transform returns a numeric matrix.}
\description{Two closed fit/apply roles reuse seven existing libn4m filter
kernels. Only layout conversion and one-based index presentation occur in R;
all selection and masking numerics remain native. A composite handle owns its
leverage/quality children. Neither role exports fitted state.}
