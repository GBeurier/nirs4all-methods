\name{spectral_encoding}
\alias{lvse_fit}
\alias{gcu_fit}
\alias{spectral_transform}
\alias{spectral_export_affine}
\title{Native fitted spectral representations}
\description{Fit LVSE local subspaces or GCU global nonnegative factors using only training spectra. All numerical operations use the native C ABI.}
\usage{
lvse_fit(X, width = 64L, rank = 4L, overlap = 0,
         standardize = TRUE, snv = FALSE)
gcu_fit(X, rank = 16L, max_iter = 60L, tol = 0.001)
spectral_transform(encoder, X)
spectral_export_affine(encoder)
}
\arguments{
\item{X}{Numeric matrix, samples by channels.}
\item{width}{Local window width.}
\item{rank}{Maximum local LVSE rank or global GCU rank.}
\item{overlap}{Window overlap fraction, in [0,1).}
\item{standardize}{Use training channel standard deviations.}
\item{snv}{Apply per-spectrum SNV before LVSE.}
\item{max_iter}{Coordinate descent iteration limit.}
\item{tol}{Relative projected-gradient tolerance.}
\item{encoder}{Fitted native spectral encoder.}
}
\value{Fit functions return an external pointer. Transform returns a matrix. Affine export returns a list with operator (outputs by channels) and offset (one row).}
\details{LVSE caps each block rank at min(training rows, block width)-1 and skips singleton blocks. GCU uses training minima, a global standard deviation, exact NNDSVDa initialization and cyclic coordinate descent. All supplied rows are used. Fit inside each training fold to prevent validation leakage. Affine export is supported only for LVSE without SNV. External pointers must be refitted after restarting R.}
