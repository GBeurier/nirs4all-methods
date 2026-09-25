\name{n4m_model_export}
\alias{n4m_model_export}
\title{Export a fitted n4m model as portable N4MM bytes}
\usage{n4m_model_export(model)}
\arguments{\item{model}{External pointer returned by \code{n4m_fit()} or \code{n4m_model_import()}.}}
\value{A raw vector containing an N4MM model.}
\description{Exports the native fitted model to the portable libn4m format. Unlike an R external pointer, the bytes can be imported by another libn4m binding.}
\seealso{\code{\link{n4m_model_import}}}
