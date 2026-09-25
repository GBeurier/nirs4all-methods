\name{n4m_model_pipeline_info}
\alias{n4m_model_pipeline_info}
\title{Inspect the native preprocessing state embedded in N4MM bytes}
\usage{n4m_model_pipeline_info(bytes)}
\arguments{\item{bytes}{Non-empty raw N4MM vector.}}
\value{A list with \code{present}, \code{semantic_profile}, Savitzky-Golay
parameters, raw/model feature widths, and \code{fingerprint}.}
\description{Uses the authoritative libn4m decoder and rejects malformed
payloads. Format-1 models return \code{present = FALSE}. For the supported
format-2 profile, the fingerprint is a hexadecimal FNV-1a-64 identity.}
