# SPDX-License-Identifier: CeCILL-2.1
##
## Cross-binding parity gate for the Nim binding. Builds the same
## deterministic (X, Y) the Python generator writes, fits the model
## through {.importc.}, and compares against
## bindings/js/test/parity_fixture.json. Exits non-zero on failure.

import std/[math, json, os, strformat]

import pls4all

const Tolerance = 1e-12

proc rmsRel(a, b: openArray[float64]): float64 =
  doAssert a.len == b.len, &"length mismatch {a.len} vs {b.len}"
  var diff, refSq: float64
  for i in 0 ..< a.len:
    let d = a[i] - b[i]
    diff += d * d
    refSq += b[i] * b[i]
  let denom = max(sqrt(refSq), 2.225e-308)
  result = sqrt(diff) / denom

proc resolveRepoRoot(): string =
  let env = getEnv("REPO_ROOT")
  if env.len > 0: return env
  var cur = getCurrentDir()
  while cur.len > 0 and cur != "/":
    if fileExists(cur / "bindings/js/test/parity_fixture.json"):
      return cur
    let parent = parentDir(cur)
    if parent == cur: break
    cur = parent
  raise newException(IOError,
    "could not locate repo root containing the shared fixture")

let
  repoRoot = resolveRepoRoot()
  fixturePath = repoRoot / "bindings/js/test/parity_fixture.json"
  fixture = parseFile(fixturePath)

let
  n = fixture["n"].getInt
  p = fixture["p"].getInt
  q = fixture["q"].getInt
  nComponents = fixture["n_components"].getInt
  fixtureVersion = fixture["pls4all_version"].getStr

proc parseArray(node: JsonNode): seq[float64] =
  result = newSeqOfCap[float64](node.len)
  for v in node:
    result.add(v.getFloat)

let
  fxCoefs = parseArray(fixture["coefficients"])
  fxXMean = parseArray(fixture["x_mean"])
  fxYMean = parseArray(fixture["y_mean"])
  fxPreds = parseArray(fixture["predictions"])

var
  x = newSeq[float64](n * p)
  y = newSeq[float64](n * q)
for i in 0 ..< n:
  for j in 0 ..< p:
    x[i * p + j] = sin(float((i + 1) * (j + 1)) * 0.3)
  y[i * q + 0] = x[i * p + 0] + 0.5 * x[i * p + 1] - 0.3 * x[i * p + 2]

let fit = plsFit(x, y, n, p, q, nComponents)

let
  errC = rmsRel(fit.coefficients, fxCoefs)
  errX = rmsRel(fit.xMean, fxXMean)
  errY = rmsRel(fit.yMean, fxYMean)
  errP = rmsRel(fit.predictions, fxPreds)

let abi = abiVersion()
echo "pls4all.version           = ", version()
echo "pls4all.abiVersion        = ", abi.major, ".", abi.minor, ".", abi.patch
echo "fixture pls4all_version   = ", fixtureVersion
echo &"rmse_rel coefficients    = {errC:.3e}"
echo &"rmse_rel x_mean          = {errX:.3e}"
echo &"rmse_rel y_mean          = {errY:.3e}"
echo &"rmse_rel predictions     = {errP:.3e}"

if errC > Tolerance or errX > Tolerance or errY > Tolerance or errP > Tolerance:
  stderr.write &"PARITY GATE FAIL (tol = {Tolerance:.1e})\n"
  quit(1)
echo "PARITY GATE PASS"
