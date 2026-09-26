\name{n4m_manifest}
\alias{n4m_manifest}
\alias{n4m_manifest_json}
\alias{n4m_constructor}
\title{Native manifest and method lookup}
\usage{
n4m_manifest_json()

n4m_constructor(method_id)
}
\arguments{
\item{method_id}{Catalog method id, for example \code{"models.pls.cppls"}.}
}
\value{
A JSON string, or a constructor function.
}
\description{
\code{n4m_manifest_json()} returns the native manifest (every catalog method's
roles, DAG-ML node kinds, fit inputs and typed parameters with defaults) as a
JSON string, the same document every n4m binding reads.
\code{n4m_constructor()} returns the generated constructor of a catalog method
id, so pipelines can be built from method ids.
}
