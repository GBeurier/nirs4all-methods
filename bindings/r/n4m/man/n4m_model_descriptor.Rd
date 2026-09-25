\name{n4m_model_descriptor}
\alias{n4m_model_descriptor}
\title{Inspect a fully validated N4MM model descriptor}
\usage{n4m_model_descriptor(bytes)}
\arguments{\item{bytes}{Non-empty raw N4MM vector.}}
\value{A list of wire version, algorithm, solver, deflation,
feature/target dimensions, component count, and capability mask.}
\description{Validates all N4MM sections before reporting the numerical
recipe and capabilities. Enum values are the versioned C ABI values from
\code{n4m.h}.}
