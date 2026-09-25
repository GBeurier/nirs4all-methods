\name{n4m_model_inspect}
\alias{n4m_model_inspect}
\title{Inspect portable N4MM model metadata}
\usage{n4m_model_inspect(bytes)}
\arguments{\item{bytes}{Non-empty raw N4MM vector.}}
\value{A list with \code{format_version} and \code{writer_abi}.}
\description{Checks the N4MM header through the native libn4m parser before full import.}
\seealso{\code{\link{n4m_model_import}}}
