# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#
# External R references of the spectral conversion parity fixture.
# Reads the job list written by generate.py and writes one output per job.
#   photobiology::T2A / A2T  absorbance <-> transmittance (-log10 T, 10^-A)
#   mdatools::prep.ref2km    Kubelka-Munk function (1 - R)^2 / (2 R)
#   stats::uniroot over mdatools::prep.ref2km   inverse Kubelka-Munk
#   prospectr::readASD, asdreader::get_spectra  ASD reflectance from the file's
#                                               target and white reference
#   spectrolab::match_sensors                   multiplicative tapered splice correction
suppressPackageStartupMessages({
  library(jsonlite)
  library(photobiology)
  library(mdatools)
  library(prospectr)
  library(asdreader)
  library(spectrolab)
})

args <- commandArgs(trailingOnly = TRUE)
jobs <- fromJSON(args[[1]], simplifyVector = FALSE)
as_matrix <- function(rows) do.call(rbind, lapply(rows, unlist))

inverse_km <- function(f) {
  # Bracketing root of prep.ref2km(r) = f on (0, 1]; prep.ref2km decreases.
  km <- function(r) as.numeric(prep.ref2km(matrix(r, 1, 1)))
  if (f == 0) return(1)
  lower <- 1e-300
  uniroot(function(r) km(r) - f, c(lower, 1), tol = 1e-300, maxiter = 10000)$root
}

run <- function(job) {
  if (job$kind == "T2A") {
    x <- as_matrix(job$X)
    return(matrix(T2A(as.vector(x)), nrow(x)))
  }
  if (job$kind == "A2T") {
    x <- as_matrix(job$X)
    return(matrix(A2T(as.vector(x)), nrow(x)))
  }
  if (job$kind == "ref2km") return(unclass(prep.ref2km(as_matrix(job$X))))
  if (job$kind == "inverse_km") {
    x <- as_matrix(job$X)
    return(matrix(vapply(as.vector(x), inverse_km, 0.0), nrow(x)))
  }
  if (job$kind == "prospectr_asd") {
    s <- readASD(job$path, in_format = "binary", out_format = "list")[[1]]
    return(list(radiance = s$radiance, reference = s$reference,
                reflectance = s$reflectance, wavelength = s$wavelength))
  }
  if (job$kind == "asdreader_asd") {
    return(list(radiance = as.vector(get_spectra(job$path, type = "radiance")),
                reference = as.vector(get_spectra(job$path, type = "white_reference")),
                reflectance = as.vector(get_spectra(job$path, type = "reflectance"))))
  }
  if (job$kind == "match_sensors") {
    x <- as_matrix(job$X)
    s <- spectra(value = x, bands = unlist(job$axis), names = paste0("s", seq_len(nrow(x))))
    m <- match_sensors(s, splice_at = unlist(job$splices), fixed_sensor = 2,
                       interpolate_wvl = job$window)
    return(unname(value(m)))
  }
  stop(paste("unknown job", job$kind))
}

versions <- list(
  R = paste(R.version$major, R.version$minor, sep = "."),
  photobiology = as.character(packageVersion("photobiology")),
  mdatools = as.character(packageVersion("mdatools")),
  prospectr = as.character(packageVersion("prospectr")),
  asdreader = as.character(packageVersion("asdreader")),
  spectrolab = as.character(packageVersion("spectrolab"))
)
# Doubles travel as C99 hexadecimal floats ("%a"), exactly.
hex <- function(x) {
  if (is.list(x)) return(lapply(x, hex))
  if (is.matrix(x)) return(lapply(seq_len(nrow(x)), function(i) sprintf("%a", x[i, ])))
  sprintf("%a", x)
}
out <- list(versions = versions, results = lapply(lapply(jobs, run), hex))
writeLines(toJSON(out, auto_unbox = FALSE), args[[2]])
