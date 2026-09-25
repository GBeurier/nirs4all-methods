\name{n4m_model_import}
\alias{n4m_model_import}
\title{Import a fitted model from portable N4MM bytes}
\usage{n4m_model_import(bytes)}
\arguments{\item{bytes}{Non-empty raw N4MM vector.}}
\value{An external pointer with the same lifecycle as \code{n4m_fit()}.}
\description{Imports a fitted model through the native libn4m parser. Treat bytes from external sources as untrusted input.}
\seealso{\code{\link{n4m_model_export}}}
