# SPDX-License-Identifier: CECILL-2.1
##
## Nim binding around libp4a's `p4a_pls_fit_simple` C ABI helper.
## Uses Nim's built-in `{.importc, dynlib.}` pragmas — no nimble
## packages required. Parity-gated against the shared cross-binding
## fixture `bindings/js/test/parity_fixture.json` at machine epsilon.

import std/strformat

# libp4a shared-object name. Override at compile time with
# `nim c -d:libp4aName=/abs/path/libp4a.so.0.93.0 ...` for an
# explicit path; the runtime loader handles standard-name fallback
# via LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / PATH.
const libp4aName {.strdefine: "libp4aName".} = "libp4a.so"

{.push importc, dynlib: libp4aName.}

proc p4a_get_version_string(): cstring
proc p4a_get_abi_version_major(): cuint
proc p4a_get_abi_version_minor(): cuint
proc p4a_get_abi_version_patch(): cuint

proc p4a_pls_fit_simple(
  x: ptr cdouble; y: ptr cdouble;
  n, p, q, nComponents: cint;
  coefficientsOut: ptr cdouble;
  xMeanOut: ptr cdouble;
  yMeanOut: ptr cdouble;
  predictionsOut: ptr cdouble): cint

{.pop.}

type FitResult* = object
  n*, p*, q*, nComponents*: int
  coefficients*: seq[float64]   ## length p*q (row-major)
  xMean*: seq[float64]          ## length p
  yMean*: seq[float64]          ## length q
  predictions*: seq[float64]    ## length n*q (row-major)

proc version*(): string =
  ## Returns the libp4a runtime version, e.g. "0.93.0+abi.1.13.0".
  let raw = p4a_get_version_string()
  if raw.isNil: result = ""
  else: result = $raw

proc abiVersion*(): tuple[major, minor, patch: int] =
  ## Returns the libp4a ABI (major, minor, patch).
  result = (
    int(p4a_get_abi_version_major()),
    int(p4a_get_abi_version_minor()),
    int(p4a_get_abi_version_patch()),
  )

proc plsFit*(
    x, y: openArray[float64];
    n, p, q, nComponents: int): FitResult =
  ## Fit a SIMPLS PLS regression on row-major matrices `x` and `y`.
  ##
  ## `x` must have length n*p; `y` must have length n*q.
  if x.len != n * p:
    raise newException(ValueError,
      &"x length {x.len} != n*p ({n * p})")
  if y.len != n * q:
    raise newException(ValueError,
      &"y length {y.len} != n*q ({n * q})")
  if nComponents < 1:
    raise newException(ValueError, "nComponents must be >= 1")

  var
    coefs = newSeq[float64](p * q)
    xMean = newSeq[float64](p)
    yMean = newSeq[float64](q)
    preds = newSeq[float64](n * q)

  # `openArray[float64]` is contiguous, so the address of element 0
  # is the row-major buffer head. Same convention as Rust / Go.
  let status = p4a_pls_fit_simple(
    cast[ptr cdouble](unsafeAddr x[0]),
    cast[ptr cdouble](unsafeAddr y[0]),
    cint(n), cint(p), cint(q), cint(nComponents),
    cast[ptr cdouble](addr coefs[0]),
    cast[ptr cdouble](addr xMean[0]),
    cast[ptr cdouble](addr yMean[0]),
    cast[ptr cdouble](addr preds[0]),
  )
  if status != 0:
    raise newException(IOError,
      &"p4a_pls_fit_simple failed with status {status}")

  result = FitResult(
    n: n, p: p, q: q, nComponents: nComponents,
    coefficients: coefs,
    xMean: xMean,
    yMean: yMean,
    predictions: preds,
  )
