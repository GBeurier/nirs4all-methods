# SPDX-License-Identifier: CECILL-2.1
##
## Idiomatic Nim tier-2 wrapper around the pls4all module's
## `pls_fit_simple` proc. Provides a `PLSRegression` ref object with
## `fit`, `predict`, `score` methods, suitable for arraymancer Tensor
## integration when the caller wraps the in/out in `toRawSeq` /
## `fromRawSeq`.
##
## Example:
##
## ```nim
## import pls4all_sklearn
## var m = newPLSRegression(nComponents = 5)
## m.fit(x, y, n = 100, p = 30)
## let preds = m.predict(xNew, nNew = 20, p = 30)
## let r2 = m.score(xTest, yTest, nTest = 20, p = 30)
## ```

import pls4all  # tier-1 wrapper from the same package

type
    PLSRegression* = ref object
        nComponents*: int
        coefficients*: seq[float64]
        xMean*: seq[float64]
        yMean*: seq[float64]
        nFeaturesIn*: int
        nTargets*: int

proc newPLSRegression*(nComponents: int = 2): PLSRegression =
    PLSRegression(nComponents: nComponents,
                   coefficients: @[], xMean: @[], yMean: @[],
                   nFeaturesIn: 0, nTargets: 0)

proc fit*(m: PLSRegression, x, y: openArray[float64],
           n, p: int): PLSRegression {.discardable.} =
    if n <= 0 or p <= 0:
        raise newException(ValueError, "n and p must be positive")
    if x.len != n * p:
        raise newException(ValueError,
            "x.len (" & $x.len & ") must equal n*p (" & $(n * p) & ")")
    if y.len == 0 or y.len mod n != 0:
        raise newException(ValueError,
            "y.len must be a positive multiple of n (" & $n & ")")
    let q = y.len div n
    let res = pls4all.plsFit(x, y, n, p, q, m.nComponents)
    m.coefficients = res.coefficients
    m.xMean = res.xMean
    m.yMean = res.yMean
    m.nFeaturesIn = p
    m.nTargets = q
    return m

proc predict*(m: PLSRegression, x: openArray[float64],
               nPred, p: int): seq[float64] =
    if m.coefficients.len == 0:
        raise newException(ValueError, "Model is not fitted")
    if p != m.nFeaturesIn:
        raise newException(ValueError,
            "X has " & $p & " features but model was fitted with " &
            $m.nFeaturesIn)
    let q = m.nTargets
    result = newSeq[float64](nPred * q)
    for i in 0 ..< nPred:
        for t in 0 ..< q:
            var s = m.yMean[t]
            for j in 0 ..< p:
                let centered = x[i * p + j] - m.xMean[j]
                s += centered * m.coefficients[j * q + t]
            result[i * q + t] = s

proc score*(m: PLSRegression, x, y: openArray[float64],
             n, p: int): float64 =
    let preds = m.predict(x, n, p)
    let q = m.nTargets
    var r2Sum = 0.0
    for t in 0 ..< q:
        var mean = 0.0
        for i in 0 ..< n: mean += y[i * q + t]
        mean /= float64(n)
        var ssRes, ssTot = 0.0
        for i in 0 ..< n:
            let dr = preds[i * q + t] - y[i * q + t]
            let dt = y[i * q + t] - mean
            ssRes += dr * dr
            ssTot += dt * dt
        if ssTot != 0.0:
            r2Sum += 1.0 - ssRes / ssTot
    return r2Sum / float64(q)
