\name{n4m_augmentation_apply}
\alias{n4m_augmentation_apply}
\title{Apply a seeded native X-only training augmentation}
\usage{
n4m_augmentation_apply(kind, X, params = numeric(), seed = 0)
}
\arguments{
\item{kind}{One of the 22 supported native augmentation names.}
\item{X}{Finite numeric matrix of training spectra.}
\item{params}{Finite positional parameter vector in the order of the corresponding
libn4m augmenter create function.}
\item{seed}{Exact nonnegative integer seed up to \code{2^53-1}.}
}
\value{A new numeric matrix with the same shape as \code{X}.}
\description{One-shot X-only augmentation by an existing seeded libn4m kernel.
There is no fitted state, held-out transform, or paired Y output. Mixup and
LocalMixup are excluded because their C kernels do not expose pairing or
weights needed to keep labels aligned. Use only on training spectra.}
