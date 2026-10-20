// SPDX-License-Identifier: CECILL-2.1
// Verify the public WASM diagnostic wrappers against hand-computed results.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
    loadModule, abiVersion, preflightPca, preflightRepeatability,
    preflightFactorAssociation, preflightHolmAdjust, statsCorrelationTest, statsBhAdjust,
    statsFactorAssociationExact, statsCorrelationExact,
    preflightNoiseProxy, preflightPairsWithinTolerance,
    preflightSummary, preflightNearestNeighbors, preflightLocalContrast,
    clusterHdbscan, preflightAdjustedRandIndex, statsPermanova, statsPermdisp, statsPermanovaFreedmanLane, statsGroupMeans, statsRegressionScores, Status, N4mError,
    statsEnergyDistance, statsPcaMonitor,
    statsVarianceComponentsReml, statsVarianceComponentsIntervals,
    ppCreate, ppTransform, ppDestroy, preflightSnv,
    clusterJaccardMatrix, linearSumAssignment,
    randomMultivariateNormal, randomPermutations, getModule, makeMatrixView, Context, HotellingT2, QResiduals,
} from "../dist/index.js";

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9,
    `${actual} != ${expected}`);
const relativeNear = (actual, expected) => assert.ok(
    Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)),
    `${actual} != ${expected}`);
const relativeNearTol = (actual, expected) => assert.ok(
    Math.abs(actual - expected) <= 1e-12 * Math.abs(expected), `${actual} != ${expected}`);
await loadModule();
assert.deepEqual(abiVersion(), [2, 20, 0]);
const pca = preflightPca({
    data: new Float64Array([2, 1, 2, -1, -2, 1, -2, -1]), rows: 4, cols: 2,
}, 1);
near(pca.eigenvalues[0], 16 / 3);
near(pca.fullMoments[0], 20 / 3);
near(pca.fullMoments[1], 272 / 9);
// +-1 spectra, row 2 = -row 0 (rank-deficient): used to fail to converge.
// Eigenvalues are NumPy 2.2.6 eigvalsh goldens (same case as test_preflight.cpp).
const pm1 = preflightPca({
    data: new Float64Array([1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1,
                            -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1]),
    rows: 4, cols: 6,
}, 3);
near(pm1.eigenvalues[0], 4.451454311908496);
near(pm1.eigenvalues[1], 2.943891106099754);
near(pm1.eigenvalues[2], 0.2713212486584137);
near(pm1.fullMoments[0], 23 / 3);
// PCA numerical domain D (same cases as test_preflight.cpp): nonzero |x| in
// [2^-128, 2^128], max/min nonzero |x| <= 2^32. Inside: exact power-of-two
// equivariance and analytic values; outside: exactly N4M_ERR_UNSUPPORTED.
const UNSUPPORTED = 10;
const pcaBase = [3, -1, 4, 1, -5, 9, 2, -6, 5, 3, -5, 8, 9, -7, 9, 3, 2, -3, 8, 4,
                 -6, 2, 6, 4, -3, 3, 3, -8, 3, 2];
const pcaOf = (values, rows = 6, cols = 5, k = 3) =>
    preflightPca({data: new Float64Array(values), rows, cols}, k);
const statusOf = (values, rows = 6, cols = 5, k = 3) => {
    try { pcaOf(values, rows, cols, k); return 0; } catch (e) { return e.status; }
};
const pow2 = (s) => 2 ** (s / 2 | 0) * 2 ** (s - (s / 2 | 0));
const base = pcaOf(pcaBase);
for (const s of [-128, -100, 0, 100, 124]) {
    const f = pow2(s);
    const r = pcaOf(pcaBase.map((v) => v * f));
    for (let a = 0; a < 3; ++a) assert.equal(r.eigenvalues[a], base.eigenvalues[a] * f * f);
    assert.equal(r.fullMoments[0], base.fullMoments[0] * f * f);
    assert.equal(r.fullMoments[1], base.fullMoments[1] * f * f * f * f);
    r.scores.data.forEach((v, i) => assert.equal(v, base.scores.data[i] * f));
    assert.deepEqual(Array.from(r.loadings.data), Array.from(base.loadings.data));
}
for (const s of [-1014, -1022, -500, -129, 125, 500, 1000]) {
    assert.equal(statusOf(pcaBase.map((v) => v * pow2(s))), UNSUPPORTED, `scale 2^${s}`);
}
const hiD = [2 ** 128, 2 ** 96, -(2 ** 128), 2 ** 97, 2 ** 127, -(2 ** 96),
             0, 3 * 2 ** 96, 2 ** 126, 2 ** 98, -(2 ** 127), 0];
const loD = hiD.map((v) => v * 2 ** -112 * 2 ** -112);
assert.equal(statusOf(hiD, 6, 2, 2), 0);
assert.equal(statusOf(loD, 6, 2, 2), 0);
assert.equal(statusOf([2 ** 128 * (1 + 2 ** -52), ...hiD.slice(1)], 6, 2, 2), UNSUPPORTED);
assert.equal(statusOf(loD.map((v) => v / 2), 6, 2, 2), UNSUPPORTED);
assert.equal(statusOf([hiD[0], 2 ** 95, ...hiD.slice(2)], 6, 2, 2), UNSUPPORTED);
const pcaShifted = pcaOf(pcaBase.map((v) => 16 * v + 1e16));
for (let a = 0; a < 3; ++a) relativeNearTol(pcaShifted.eigenvalues[a], 256 * base.eigenvalues[a]);
assert.equal(statusOf(pcaBase.map((v) => v * pow2(950) + 1e300)), UNSUPPORTED);
const withConstants = (c0, c1) => {
    const out = [];
    for (let i = 0; i < 6; ++i) out.push(c0, ...pcaBase.slice(5 * i, 5 * i + 5), c1, 0.5, 0);
    return out;
};
const c = pcaOf(withConstants(2 ** 30, -3 * 2 ** 28), 6, 9);
for (let a = 0; a < 3; ++a) relativeNearTol(c.eigenvalues[a], base.eigenvalues[a]);
relativeNearTol(c.fullMoments[1], base.fullMoments[1]);
for (let a = 0; a < 3; ++a) {
    for (const j of [0, 6, 7, 8]) assert.ok(c.loadings.data[9 * a + j] === 0);
}
assert.equal(statusOf(withConstants(1e300, 0.5), 6, 9), UNSUPPORTED);
assert.equal(statusOf(withConstants(2 ** 34, 0.5), 6, 9), UNSUPPORTED);
// Review cases: [A; -A], A = [[-1,1,-1],[-1,-1,-1],[1,-1,1],[-t,t,t]].
const s17 = Math.sqrt(17);
for (const t of [2 ** -30, 2 ** -40, 2 ** -60, 1e-40]) {
    const a4 = [-1, 1, -1, -1, -1, -1, 1, -1, 1, -t, t, t];
    const x = [...a4, ...a4.map((v) => -v)];
    // Spreads 2^40, 2^60 and 1e40 exceed 2^32 (rule 15).
    if (t !== 2 ** -30) { assert.equal(statusOf(x, 8, 3, 3), UNSUPPORTED); continue; }
    const r = pcaOf(x, 8, 3, 3);
    relativeNearTol(r.eigenvalues[0], 2 * (s17 + 1) ** 2 / 28);
    relativeNearTol(r.eigenvalues[1], 2 * (s17 - 1) ** 2 / 28);
    // 4t^2/7 ~ 2^-61 lambda_1: below the resolution, flagged unresolved.
    assert.equal(r.eigenvalues[2], 0);
    assert.deepEqual(Array.from(r.unresolved), [0, 0, 1]);
}
assert.equal(statusOf([2 ** 200, 0, -(2 ** 200), 0, 0, 2 ** -450, 0, -(2 ** -450)], 4, 2, 2),
             UNSUPPORTED);
// Rule 14 through the 2.20 entry point, against the mpmath variances of
// cpp/tests/svd_degenerate_goldens.inc (generated by
// parity/scripts/svd_degenerate_reference.py; mpmath 1.3.0, 600 digits).
// Families inside D are computed; the review families with spread > 2^32
// (rule 15) are refused.
const bidiag = (n, t) => Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j) => (i === j ? t : j === i + 1 ? 1 : 0)));
const stacked = (rows, s) => [...rows.map((r) => r.map((v) => s * v)),
                              ...rows.map((r) => r.map((v) => -s * v))];
const kronB6 = (t) => {
    const b = bidiag(6, t), kk = [[1, 0.75], [0.5, 1]];
    const a = Array.from({length: 12}, (_, i) =>
        Array.from({length: 12}, (_, j) => b[i >> 1][j >> 1] * kk[i & 1][j & 1]));
    return stacked(a, 1).map((r) => [...r].reverse());
};
const weakRow = (t) => stacked([[-1, 1, -1], [-1, -1, -1], [1, -1, 1], [-t, t, t]], 1);
for (const rows of [stacked(bidiag(8, 2 ** -63), 2 ** 63), stacked(bidiag(8, 2 ** -60), 2 ** -67),
                    kronB6(2 ** -50)]) {
    assert.equal(statusOf(rows.flat(), rows.length, rows[0].length, rows[0].length), UNSUPPORTED);
}
const mpCases = [
    [stacked(bidiag(8, 2 ** -30), 1), [0.1333333335627813, 0.1333333335089452, 0.13333333342837378, 0.13333333333333333, 0.13333333323829288, 0.13333333315772147, 0.13333333310388537, 4.271110603057282e-146], 1],
    [stacked(bidiag(8, 2 ** -30), 2 ** -97), [5.3103063795798e-60, 5.310306377435653e-60, 5.310306374226712e-60, 5.310306370441508e-60, 5.310306366656303e-60, 5.310306363447362e-60, 5.310306361303215e-60, 1.7010679383206515e-204], 1],
    [kronB6(2 ** -30), [0.23182415456279407, 0.23182415440474205, 0.23182415418883898, 0.23182415397293593, 0.23182415381488392, 0.012741063223017941, 0.012741063214331399, 0.012741063202465357, 0.012741063190599318, 0.012741063181912773, 9.870979384799337e-110, 5.425093543510313e-111], 2],
    [weakRow(2 ** -20), [1.8747293750882725, 0.6966991963405587, 5.197112581557818e-13], 0],
];
for (const [rows, ref, nUnresolved] of mpCases) {
    const n = rows.length, p = rows[0].length;
    for (const perm of [0, 1, 2, 3]) {
        const x = new Float64Array(n * p);
        rows.forEach((r, i) => r.forEach((v, j) => {
            x[((perm & 1) ? n - 1 - i : i) * p + ((perm & 2) ? p - 1 - j : j)] = v;
        }));
        const r = preflightPca({data: x, rows: n, cols: p}, p);
        assert.equal(r.resolution, 16 * Math.max(n, p) * Number.EPSILON * r.eigenvalues[0]);
        let unresolved = 0;
        for (let a = 0; a < p; ++a) {
            if (ref[a] > 2 * r.resolution) assert.equal(r.unresolved[a], 0);
            if (ref[a] < 0.5 * r.resolution) assert.equal(r.unresolved[a], 1);
            if (r.unresolved[a] === 0) {
                assert.ok(Math.abs(r.eigenvalues[a] - ref[a]) <= r.resolution, `component ${a}`);
                assert.ok(r.eigenvalues[a] >= 2.2250738585072014e-308);
            } else {
                ++unresolved;
                assert.equal(r.eigenvalues[a], 0);
                for (let i = 0; i < n; ++i) assert.equal(r.scores.data[i * p + a], 0);
            }
        }
        assert.equal(unresolved, nUnresolved);
    }
}
// Weak decoupled variance: 1.689e-19 (NumPy 2.2.6) = 2.6e-18 lambda_1 is
// below the resolution (rule 14): flagged unresolved and published as 0.
const weak = new Float64Array(96);
[1, 0, 0, 0, 1e-9, 0, 0, 1e-9, 1e-9].forEach((v, i) => { weak[i] = v; weak[9 + i] = -v; });
const wk = preflightPca({data: weak, rows: 32, cols: 3}, 2);
relativeNearTol(wk.eigenvalues[0], 0.06451612903225808);
assert.equal(wk.eigenvalues[1], 0);
assert.deepEqual(Array.from(wk.unresolved), [0, 1]);
assert.equal(pca.scores.rows, 4);
assert.equal(pca.loadings.cols, 2);
const repeat = preflightRepeatability({
    data: new Float64Array([0, 2, 4, 6]), rows: 4, cols: 1,
}, [10, 10, 20, 20]);
near(repeat.pooledSd[0], Math.sqrt(2));
near(repeat.icc[0], 7 / 9);
assert.equal(repeat.bandStatus[0], 0);
const unbalanced = preflightRepeatability({
    data: new Float64Array([0, 2, 4, 6]), rows: 4, cols: 1,
}, [10, 10, 20, 20], new Uint8Array([1, 1, 1, 0]));
assert.equal(unbalanced.bandStatus[0], 2);
assert.ok(Number.isNaN(unbalanced.icc[0]));
const summary = preflightSummary(new Float64Array([0, 2, 4, 6]));
near(summary[5], 1.5);
near(summary[6], 3);
near(summary[7], 4.5);
// Domain edge 2^256 (rule 13): [-a, 0, a] has SD a and quartiles -a/2, a/2.
const summaryEdge = 2 ** 256;
const extremeSummary = preflightSummary(new Float64Array([-summaryEdge, 0, summaryEdge]));
assert.ok(extremeSummary[6] === 0 && extremeSummary[4] === summaryEdge);
assert.ok(extremeSummary[5] === -summaryEdge / 2 && extremeSummary[7] === summaryEdge / 2);
assert.throws(() => preflightSummary(new Float64Array([Number.MAX_VALUE, -Number.MAX_VALUE])),
    (error) => error.status === 10);
const extremeRepeat = preflightRepeatability({
    data: new Float64Array([1e308, -1e308, 1e308, -1e308]), rows: 4, cols: 1,
}, [10, 10, 20, 20]);
relativeNear(extremeRepeat.pooledSd[0], Math.SQRT2 * 1e308);
near(extremeRepeat.icc[0], -1);
const neighbors = preflightNearestNeighbors(
    {data: new Float64Array([0, 0]), rows: 1, cols: 2},
    {data: new Float64Array([0, 0, 1, 0]), rows: 2, cols: 2},
    2, [9], [9, 10],
);
assert.deepEqual(neighbors.indices, [1, -1]);
near(neighbors.distances[0], 1);
assert.ok(Number.isNaN(neighbors.distances[1]));
const extremeNeighbors = preflightNearestNeighbors(
    {data: new Float64Array([summaryEdge, summaryEdge]), rows: 1, cols: 2},
    {data: new Float64Array([0, 0]), rows: 1, cols: 2}, 1);
assert.ok(Math.abs(extremeNeighbors.distances[0] - Math.SQRT2 * summaryEdge) <= 4e-16 * Math.SQRT2 * summaryEdge);
assert.throws(() => preflightNearestNeighbors(
    {data: new Float64Array([Number.MAX_VALUE]), rows: 1, cols: 1},
    {data: new Float64Array([-Number.MAX_VALUE]), rows: 1, cols: 1}, 1),
    (error) => error.status === 10);
const continuous = preflightLocalContrast(new Float64Array([0, 1, 2]),
    [1, 2, 1], 1, false);
near(continuous.contrast, 0.5);
near(continuous.fractionRowsWithNeighbors, 1);
assert.ok(Number.isNaN(continuous.observedSameClassFraction));
const categorical = preflightLocalContrast(new Float64Array([0, 0, 1, 1]),
    [2, 2, 3, 2], 1, true, [1, 1, 2, 3]);
// Same hand values and NumPy/SciPy permutation goldens as test_preflight.cpp.
const assoc = preflightFactorAssociation([4, 4, 4, -9, -9, -9],
    new Float64Array([1, 2, 3, 7, 8, 9]), false, 999, 17);
near(assoc.effectSize, 27 / 29);
near(assoc.statistic, 54);
near(assoc.log10DistinctPermutations, Math.log10(20));
assert.equal(assoc.nAtLeastObserved, 93);
assert.equal(assoc.nPermutationsExecuted, 999);
near(assoc.pValue, 94 / 1000);
assert.equal(preflightFactorAssociation([4, 4, 4, -9, -9, -9],
    new Float64Array([1, 2, 3, 7, 8, 9]), false, 999, 18446744073709551557n).nAtLeastObserved, 88);
const classAssoc = preflightFactorAssociation([0, 0, 0, 1, 1, 1, 1],
    new Float64Array([0, 0, 1, 1, 1, 1, 0]), true, 999, 17);
near(classAssoc.effectSize, 5 / 12);
near(classAssoc.statistic, 175 / 144);
assert.equal(classAssoc.nAtLeastObserved, 477);
const holm = preflightHolmAdjust(new Float64Array([0.01, 0.04, 0.03, 0.005]));
[0.03, 0.06, 0.06, 0.02].forEach((value, i) => near(holm[i], value));
near(categorical.contrast, 0.4);
near(categorical.observedSameClassFraction, 0.5);
near(categorical.expectedSameClassFraction, 1 / 6);
// Window 5 / quadratic SG weights (-3,12,17,12,-3)/35; gain^2 = 18/35.
// Row 0 quadratic, row 1 linear plus impulse 35 at band 5.
const noiseRows = [];
for (let j = 0; j < 11; j++) noiseRows.push(j * j);
for (let j = 0; j < 11; j++) noiseRows.push(1 + 2 * j + (j === 5 ? 35 : 0));
const noise = preflightNoiseProxy({data: new Float64Array(noiseRows), rows: 2, cols: 11}, 5, 2);
near(noise.gain, Math.sqrt(18 / 35));
near(noise.bandNoise[5], Math.sqrt(315));
near(noise.bandNoise[3], Math.sqrt(8.75));
near(noise.pooled, Math.sqrt(87.5));
assert.ok(Number.isNaN(noise.bandNoise[0]) && noise.bandCounts[0] === 0 && noise.bandCounts[5] === 2);
// Review b1 #5: arguments are range-checked before Emscripten's int32
// conversion (4294967301 would wrap to a window of 5) and against the SG domain.
const noiseX = {data: new Float64Array(noiseRows), rows: 2, cols: 11};
for (const [w, o] of [[4294967301, 2], [2 ** 31 + 5, 2], [-(2 ** 32) + 5, 2], [13, 2], [4, 2],
    [11, 10], [5, 4294967298], [5, -1], [5.5, 2]]) {
    assert.throws(() => preflightNoiseProxy(noiseX, w, o), RangeError, `${w}/${o}`);
}
near(preflightNoiseProxy(noiseX, 11, 9).bandCounts[5], 2);
const ones41 = {data: new Float64Array(41).fill(1), rows: 1, cols: 41};
assert.throws(() => preflightNoiseProxy(ones41, 41, 38), RangeError);
assert.ok(Math.abs(preflightNoiseProxy(ones41, 41, 10).pooled) < 1e-12);
assert.throws(() => preflightNearestNeighbors({data: new Float64Array([0, 1]), rows: 2, cols: 1},
    {data: new Float64Array([0, 1]), rows: 2, cols: 1}, 2 ** 32 + 1), RangeError);
assert.throws(() => preflightPca({data: new Float64Array([1, 2, 3, 4]), rows: 2, cols: 2},
    2 ** 32 + 1), RangeError);
// Relative error against a nonzero expectation: 0 and sign flips fail.
const rel = (actual, expected, r, what = "") => assert.ok(
    Number.isFinite(actual) && Math.abs(actual - expected) <= r * Math.abs(expected),
    `${what} ${actual} != ${expected}`);
const row = (values) => ({data: new Float64Array(values), rows: 1, cols: values.length});
const pair = (q, r, tol) => preflightPairsWithinTolerance(row(q), tol, 1, row(r));
// Exact status codes (types.ts Status): 10 = N4M_ERR_UNSUPPORTED (outside the
// supported numerical domain), never a trap or a silently wrong value.
const unsupported = (fn) => assert.throws(fn, (e) => e.status === 10 &&
    /outside the supported numerical domain/.test(e.message));
// Supported domain D (rules 13/15): nonzero |x| in [2^-256, 2^256], per-row spread <= 2^32.
const A = 2 ** 200, a = 2 ** -200, lo = 2 ** -256, hi = 2 ** 256;
const huge = pair([A, 0], [-A, 0], 1.5 * A);
assert.equal(huge.totalPairs, 1);
rel(huge.distances[0], Math.SQRT2 * A, 4e-16);
assert.equal(pair([A, 0], [-A, 0], 1.414 * A).totalPairs, 0);
// Tolerance-invariant distance a/sqrt(2), both sides of the threshold.
for (const tol of [0.70711 * a, 1e-30, 1, A, hi]) {
    const tiny = pair([a, 0], [0, 0], tol);
    assert.equal(tiny.totalPairs, 1);
    rel(tiny.distances[0], a / Math.SQRT2, 4e-16, `tol ${tol}`);
}
assert.equal(pair([a, 0], [0, 0], 0.70710 * a).totalPairs, 0);
assert.equal(pair([2 * lo, 0], [0, 0], lo).totalPairs, 0);
// Domain boundaries on both sides.
assert.equal(pair([lo, 0], [0, 0], 1).totalPairs, 1);
assert.equal(pair([hi, 0], [0, 0], 1).totalPairs, 0);
unsupported(() => pair([lo / 2, 0], [0, 0], 1));
unsupported(() => pair([hi * 2, 0], [0, 0], 1));
assert.equal(pair([1, 2 ** 32], [0, 0], 1).totalPairs, 0);
unsupported(() => pair([1, 2 ** 32 + 2 ** -20], [0, 0], 1));
unsupported(() => pair([Number.MAX_VALUE, 0], [-Number.MAX_VALUE, 0], 1));
// Magnitude sweep inside D: q - r = [3, 4, 0, 12] has RMS 6.5 at every scale.
for (const k of [-256, -200, -100, 0, 100, 200, 248]) {
    const s = 2 ** k, q = [5, 4, 9, 12].map((v) => v * s), r = [2, 0, 9, 0].map((v) => v * s);
    const result = pair(q, r, 6.5 * s);
    assert.equal(result.totalPairs, 1, `k ${k}`);
    assert.equal(result.distances[0], 6.5 * s);
    assert.equal(pair(q, r, 6 * s).totalPairs, 0, `k ${k}`);
}
// Test-only goldens inside the supported domain: exact rational decisions
// (Python 3.10 fractions) and 400-digit decimal square roots, rounded to binary64.
const pairGoldens = [
    [[1.0, 1.0], [-5.551115123125783e-17, -5.551115123125783e-17], 0.9999999999999999, false, 1.0],
    [[1.0, 1.0], [-5.551115123125783e-17, -5.551115123125783e-17], 1.0, false, 1.0],
    [[1.0, 1.0], [-5.551115123125783e-17, -5.551115123125783e-17], 1.0000000000000002, true, 1.0],
    [[1.0, 1.0, 1.4901161193847656e-08], [0.0, 0.0, 0.0], 0.8164965809277259, false, 0.816496580927726],
    [[1.0, 1.0, 1.4901161193847656e-08], [0.0, 0.0, 0.0], 0.816496580927726, false, 0.816496580927726],
    [[1.0, 1.0, 1.4901161193847656e-08], [0.0, 0.0, 0.0], 0.8164965809277261, true, 0.816496580927726],
    [[1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0], 0.9999999999999999, false, 1.0],
    [[1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0], 1.0, true, 1.0],
    [[1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0], 1.0000000000000002, true, 1.0],
    [[1073741824.0, 0.5, -0.75], [1073741824.0, 0.0, 0.25], 0.6454972243679027, false, 0.6454972243679028],
    [[1073741824.0, 0.5, -0.75], [1073741824.0, 0.0, 0.25], 0.6454972243679028, false, 0.6454972243679028],
    [[1073741824.0, 0.5, -0.75], [1073741824.0, 0.0, 0.25], 0.6454972243679029, true, 0.6454972243679028],
    [[1.8092513943330656e+75, 1.725436586697641e+69, 2.5881548800464614e+69], [-1.8092513943330656e+75, 1.725436586697641e+69, 0.0], 2.0891435591003365e+75, false, 2.089143559100337e+75],
    [[1.8092513943330656e+75, 1.725436586697641e+69, 2.5881548800464614e+69], [-1.8092513943330656e+75, 1.725436586697641e+69, 0.0], 2.089143559100337e+75, true, 2.089143559100337e+75],
    [[1.8092513943330656e+75, 1.725436586697641e+69, 2.5881548800464614e+69], [-1.8092513943330656e+75, 1.725436586697641e+69, 0.0], 2.0891435591003373e+75, true, 2.089143559100337e+75],
    [[3.0, -4.0, 1.862645149230957e-09], [-3.0, 4.0, 0.0], 5.773502691896256, false, 5.773502691896257],
    [[3.0, -4.0, 1.862645149230957e-09], [-3.0, 4.0, 0.0], 5.773502691896257, false, 5.773502691896257],
    [[3.0, -4.0, 1.862645149230957e-09], [-3.0, 4.0, 0.0], 5.773502691896258, true, 5.773502691896257],
    [[6.223015277861142e-61, 5.795634610449096e-70], [0.0, -5.795634610449096e-70], 4.4003363024031e-61, false, 4.400336302403101e-61],
    [[6.223015277861142e-61, 5.795634610449096e-70], [0.0, -5.795634610449096e-70], 4.400336302403101e-61, true, 4.400336302403101e-61],
    [[6.223015277861142e-61, 5.795634610449096e-70], [0.0, -5.795634610449096e-70], 4.4003363024031016e-61, true, 4.400336302403101e-61],
    [[1e+16, 1.0000000000000004e+16], [9999999999999998.0, 1e+16], 3.162277660168379, false, 3.1622776601683795],
    [[1e+16, 1.0000000000000004e+16], [9999999999999998.0, 1e+16], 3.1622776601683795, true, 3.1622776601683795],
    [[1e+16, 1.0000000000000004e+16], [9999999999999998.0, 1e+16], 3.16227766016838, true, 3.1622776601683795],
    [[1.157920892373162e+77, 2.695994666715064e+67], [-1.157920892373162e+77, 2.695994666715064e+67], 1.6375474301492824e+77, false, 1.6375474301492827e+77],
    [[1.157920892373162e+77, 2.695994666715064e+67], [-1.157920892373162e+77, 2.695994666715064e+67], 1.6375474301492827e+77, true, 1.6375474301492827e+77],
    [[1.157920892373162e+77, 2.695994666715064e+67], [-1.157920892373162e+77, 2.695994666715064e+67], 1.637547430149283e+77, true, 1.6375474301492827e+77],
    [[8.636168555094445e-78, 3.7092061506874214e-68], [0.0, -3.7092061506874214e-68], 5.245609643939853e-68, false, 5.245609643939854e-68],
    [[8.636168555094445e-78, 3.7092061506874214e-68], [0.0, -3.7092061506874214e-68], 5.245609643939854e-68, true, 5.245609643939854e-68],
    [[8.636168555094445e-78, 3.7092061506874214e-68], [0.0, -3.7092061506874214e-68], 5.2456096439398546e-68, true, 5.245609643939854e-68],
];
const snvGoldens = [
    [[1073741824.0, 0.5, -0.5, 0.0], [1.7320508075688772, -0.5773502681142271, -0.5773502702650244, -0.5773502691896257]],
    [[1.157920892373162e+77, 1.7668470647783843e+72, 2.695994666715064e+67, 0.0], [1.732050807389645, -0.577326776747033, -0.577362015052455, -0.5773620155901571]],
    [[8.636168555094445e-78, 5.795634610449096e-70, 2.5908505665283334e-77, 0.0], [-0.5773502806605448, 1.732050807568876, -0.5773502118350281, -0.5773503150733033]],
    [[1e+16, 1.0000000000000002e+16, 9999999999999998.0, 1.0000000000000004e+16], [-0.4472135954999579, 0.4472135954999579, -1.3416407864998738, 1.3416407864998738]],
    [[2147483648.0, -2147483648.0, 0.5, 0.0], [1.414213562290777, -1.414213562455413, 2.469541904935197e-10, -8.231806349783991e-11]],
    [[1.0, 1.0000000000000002, 1.0, 1.0], [-0.5773502691896257, 1.7320508075688772, -0.5773502691896257, -0.5773502691896257]],
    [[-3.7092061506874214e-68, 8.636168555094445e-78, 1.727233711018889e-77, 0.0], [-1.7320508075688772, 0.5773502691896257, 0.5773502697273251, 0.5773502686519264]],
];
// Exact threshold decisions on mixed-exponent rows inside D.
for (const [q, r, tol, within, rms] of pairGoldens) {
    const got = pair(q, r, tol);
    assert.equal(got.totalPairs === 1, within, `${q} ${r} ${tol}`);
    if (!within) continue;
    assert.ok(got.distances[0] > 0 && got.distances[0] <= tol);
    assert.ok(Math.abs(got.distances[0] - rms) <= 2 ** -50 * rms);
}
// SNV of mixed-exponent rows within the reported bound of 400-digit references.
for (const [x, z] of snvGoldens) {
    const out = preflightSnv(row(x));
    assert.ok(out.errorBounds[0] < 1e-14);
    z.forEach((v, j) => assert.ok(Math.abs(out.matrix.data[j] - v) <= out.errorBounds[0] + 2 ** -53 * Math.abs(v), `${x} ${j}`));
}
unsupported(() => preflightSnv(row([1, 0, 0, Number.MAX_VALUE])));
unsupported(() => preflightSnv(row([1, 0, 0, 2 ** -1074])));
unsupported(() => preflightSnv(row([1, 0, 0, 2 ** 32 + 2 ** -20])));
assert.ok(preflightSnv(row([1, 0, 0, 2 ** 32])).errorBounds[0] < 1e-14);
// SNV at the domain edges, bounds and aggregation of very unbalanced repeats
// (a [1,2,4], a = 1, 2, 4, 8; 1, 10000, 1, 1 copies).
const snvRows = preflightSnv({data: new Float64Array([lo, 0, -lo, 0.1, 0.1, 0.1]), rows: 2, cols: 3});
rel(snvRows.matrix.data[0], Math.sqrt(1.5), 1e-15);
rel(snvRows.matrix.data[2], -Math.sqrt(1.5), 1e-15);
assert.ok(snvRows.matrix.data.slice(3).every((v) => v === 0) && snvRows.errorBounds[1] === 0);
assert.ok(snvRows.errorBounds[0] > 0 && snvRows.errorBounds[0] < 1e-14);
{
    const reps = [1, 10000, 1, 1], values = [], gid = [];
    reps.forEach((n, g) => { for (let i = 0; i < n; i++) { values.push(2 ** g, 2 ** (g + 1), 2 ** (g + 2)); gid.push(g); } });
    const agg = preflightSnv({data: new Float64Array(values), rows: gid.length, cols: 3}, gid, 4);
    const mu = 7 / 3, sd = Math.sqrt(((1 - mu) ** 2 + (2 - mu) ** 2 + (4 - mu) ** 2) / 3);
    for (let g = 0; g < 4; g++) {
        assert.ok(agg.errorBounds[g] > 0 && agg.errorBounds[g] < 1e-14);
        [1, 2, 4].forEach((v, j) => assert.ok(
            Math.abs(agg.matrix.data[g * 3 + j] - (v - mu) / sd) <= agg.errorBounds[g] + 4e-16 * Math.abs((v - mu) / sd)));
    }
    assert.throws(() => preflightSnv(row([1, 2]), [0], 2), RangeError);
}
// Noise proxy inside D: power-of-two equivariance, exact-offset invariance,
// window-local residuals, and refusal just outside D.
{
    const base = [];
    for (let j = 0; j < 11; j++) base.push(j * j);
    for (let j = 0; j < 11; j++) base.push(1 + 2 * j + (j === 5 ? 35 : 0));
    const ref = preflightNoiseProxy({data: new Float64Array(base), rows: 2, cols: 11}, 5, 2);
    for (const k of [-256, -200, -100, 0, 100, 200, 248]) {
        const s = 2 ** k;
        const out = preflightNoiseProxy({data: new Float64Array(base.map((v) => v * s)), rows: 2, cols: 11}, 5, 2);
        assert.equal(out.pooled, ref.pooled * s, `k ${k}`);
    }
    unsupported(() => preflightNoiseProxy({data: new Float64Array(base.map((v) => v * 2 ** 250)), rows: 2, cols: 11}, 5, 2));
    const shifted = preflightNoiseProxy({data: new Float64Array(base.map((v) => v * 2 ** 148 + 2 ** 200)), rows: 2, cols: 11}, 5, 2);
    rel(shifted.pooled, ref.pooled * 2 ** 148, 1e-15);
    // [H,0,0,0,0,1,0,...] with index 1 masked: H is outside every evaluable window.
    const mask = new Uint8Array(11).fill(1);
    mask[1] = 0;
    for (const big of [0, 1, 1e9, 2 ** 32, -(2 ** 31)]) {
        const out = preflightNoiseProxy(row([big, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 3, 1, mask);
        rel(out.bandNoise[5], Math.sqrt(2 / 3), 1e-15, `H ${big}`);
        rel(out.pooled, 1 / Math.sqrt(7), 1e-15, `H ${big}`);
    }
    unsupported(() => preflightNoiseProxy(row([2 ** 33, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 3, 1, mask));
    // Spread 2^32 inside one window: [-2^32, 2^32, 1, 0, 0], window 5 / order 1 -> sqrt(4/5).
    const spread = preflightNoiseProxy(row([-(2 ** 32), 2 ** 32, 1, 0, 0]), 5, 1);
    unsupported(() => preflightNoiseProxy(row([-(2 ** 33), 2 ** 33, 1, 0, 0]), 5, 1));
    rel(spread.bandNoise[2], Math.sqrt(0.8), 1e-14);
    // Outside D (review b14fix3 #3/#4): explicit refusal.
    unsupported(() => preflightNoiseProxy(row([-(2 ** 100), 2 ** 100, 1, 0, 0]), 5, 1));
    unsupported(() => preflightNoiseProxy(row([-Number.MAX_VALUE, 2 ** -1074, Number.MAX_VALUE]), 3, 1));
}
// Test-only goldens inside the supported domain (Python 3.10 fractions,
// 400-digit decimal square roots, rounded to binary64).
const summaryGoldens = [
    [[1e+16, 1.0000000000000002e+16, 1.0000000000000004e+16, 1.0000000000000008e+16, 9999999999999998.0], [1.0000000000000002e+16, 3.847076812334269, 1e+16, 1.0000000000000002e+16, 1.0000000000000004e+16]],
    [[1.152921504606847e+18, 0.125, -0.0625, 0.0, 1.0], [2.305843009213694e+17, 5.1560217140444934e+17, 0.0, 0.125, 1.0]],
    [[0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6], [0.35000000000000003, 0.18708286933869706, 0.22500000000000003, 0.35000000000000003, 0.475]],
    [[8.636168555094445e-78, 2.5908505665283334e-77, 6.223015277861142e-61, 0.0], [1.5557538194652854e-61, 3.111507638930571e-61, 6.477126416320833e-78, 1.727233711018889e-77, 1.5557538194652858e-61]],
    [[-1.6069380442589903e+60, 1.6069380442589903e+60, 1.42724769270596e+45, -1.393796574908164e+42, 1.0727468786061222e+55], [2.1454937574974152e+54, 1.1362767880523081e+60, -1.393796574908164e+42, 1.42724769270596e+45, 1.0727468786061222e+55]],
    [[1.0000000000000002, 1.0, 0.9999999999999999, 1.0000000000000004], [1.0000000000000002, 2.461759443637518e-16, 1.0, 1.0, 1.0000000000000002]],
];
const knnGoldens = [
    [[1.0000000000000024, 1.0000000000000089, 1.0000000000000029], [1.5000000000000062, 1.5000000000000135, 0.9999999999999969, 0.499999999999996, 0.5000000000000095, 1.500000000000002, 0.49999999999999756, 1.0000000000000004, 1.5000000000000095, 0.4999999999999971, 1.0000000000000153, 1.5000000000000095, 0.49999999999999556, 0.5000000000000073, 1.0000000000000049], [4, 0, 2, 3, 1], [0.7071067811865535, 0.8660254037844415, 0.7071067811865557, 0.707106781186556, 0.7071067811865535]],
    [[1.000000000000007, 1.0000000000000042, 1.000000000000007], [1.0000000000000064, 0.49999999999999667, 1.4999999999999984, 1.0000000000000158, 0.5000000000000075, 1.5000000000000093, 1.5000000000000098, 1.0000000000000095, 1.000000000000002, 1.5000000000000013, 1.5000000000000062, 1.50000000000001, 0.5000000000000098, 1.0000000000000022, 1.5], [2, 4, 1, 0, 3], [0.7071067811865467, 0.7071067811865467, 0.5000000000000027, 0.8660254037844382, 0.7071067811865406]],
    [[1.000000000000008, 1.000000000000008, 1.0000000000000042], [1.000000000000004, 0.5000000000000091, 0.9999999999999998, 1.0000000000000027, 0.5000000000000084, 1.5000000000000118, 1.0000000000000129, 1.5000000000000098, 0.500000000000004, 0.5000000000000111, 0.5000000000000093, 1.500000000000002, 0.5000000000000064, 1.0000000000000118, 0.5000000000000038], [0, 4, 2, 1, 3], [0.4999999999999989, 0.7071067811865526, 0.7071067811865489, 0.8660254037844348, 0.7071067811865489]],
    [[1.0000000000000113, 1.0000000000000069, 1.0000000000000022], [0.5000000000000107, 0.5000000000000004, 0.5000000000000044, 1.5000000000000135, 0.9999999999999976, 1.5000000000000095, 1.5000000000000193, 1.000000000000002, 0.9999999999999971, 1.0000000000000173, 1.4999999999999982, 0.9999999999999991, 1.500000000000016, 0.500000000000002, 1.0000000000000089], [3, 2, 4, 1, 0], [0.8660254037844415, 0.7071067811865542, 0.500000000000008, 0.49999999999999134, 0.7071067811865542]],
    [[1.0000000000000016, 1.000000000000005, 1.0000000000000033], [0.4999999999999971, 1.4999999999999976, 1.5000000000000075, 1.5000000000000018, 1.5000000000000007, 1.5000000000000087, 1.0000000000000093, 1.500000000000012, 1.000000000000006, 1.5000000000000029, 1.0000000000000007, 1.0000000000000127, 0.5000000000000093, 0.9999999999999969, 1.4999999999999964], [3, 2, 4, 1, 0], [0.8660254037844393, 0.8660254037844393, 0.5000000000000069, 0.5000000000000013, 0.7071067811865371]],
    [[1.0000000000000107, 1.000000000000009, 1.0000000000000056], [1.0000000000000024, 0.5000000000000049, 1.0000000000000133, 1.0000000000000044, 1.5000000000000133, 1.0000000000000082, 0.5000000000000107, 1.0000000000000155, 0.5000000000000051, 0.5000000000000109, 1.0000000000000115, 1.500000000000011, 1.5000000000000142, 0.5000000000000031, 1.5000000000000064], [1, 0, 2, 3, 4], [0.5000000000000042, 0.5000000000000042, 0.7071067811865478, 0.7071067811865513, 0.8660254037844447]],
    [[0.1, 0.7], [0.4, 1.1, -0.30000000000000004, 1.0, 0.1, 0.19999999999999996], [2, 1, 0], [0.5000000000000001, 0.5000000000000001, 0.5]],
];
const contrastGoldens = [
    [[1e+16, 1.0000000000000002e+16, 1.0000000000000004e+16, 1.0000000000000006e+16, 1.0000000000000008e+16], [1, 0, 3, 2, 3], 0.8],
    [[0.1, 0.2, 0.30000000000000004, 0.7, 0.9], [1, 2, 1, 4, 3], 0.9067796610169491],
    [[1073741824.0, 0.5, -0.25, 1.0], [2, 3, 1, 0], -7.761021435854282e-11],
];
// Summary, nearest neighbours and local contrast (ABI 2.18) inside D against
// exact rational / 400-digit references, and refusal just outside D.
const ulpClose = (x, y, ulps) => Number.isFinite(x) && Math.abs(x - y) <= ulps * 2 ** -52 * Math.max(Math.abs(y), 2 ** -1022);
for (const [x, [mean, sd, q1, med, q3]] of summaryGoldens) {
    const out = preflightSummary(new Float64Array(x));
    assert.ok(ulpClose(out[3], mean, 1), `mean ${x}`);
    rel(out[4], sd, 1e-15, `sd ${x}`);
    [q1, med, q3].forEach((q, c) => assert.ok(ulpClose(out[5 + c], q, 2), `q ${x}`));
}
// The summary has no spread bound: 1e-20 next to 1 against the rational oracle.
{
    const out = preflightSummary(new Float64Array([1e-20, 1, 0.5, 1e-20, 2]));
    assert.ok(ulpClose(out[3], 0.7, 1) && ulpClose(out[5], 1e-20, 2));
    rel(out[4], 0.8366600265340756, 1e-15);
    assert.equal(preflightSummary(new Float64Array([1, 0, 2 ** 64 + 2 ** 12]))[0], 3);
}
unsupported(() => preflightSummary(new Float64Array([2 ** -257, 0])));
unsupported(() => preflightSummary(new Float64Array([1e-200, 1])));
unsupported(() => preflightSummary(new Float64Array([1e200, 1])));
for (const [q, refs, order, dist] of knnGoldens) {
    const p = q.length, m = refs.length / p;
    const out = preflightNearestNeighbors(row(q), {data: new Float64Array(refs), rows: m, cols: p}, m);
    assert.deepEqual(out.indices, order);
    out.indices.forEach((idx, a) => rel(out.distances[a], dist[idx], 1e-15));
}
unsupported(() => preflightNearestNeighbors(row([2 ** -257, 0]), row([0, 0]), 1));
unsupported(() => preflightNearestNeighbors(row([1, 0]), row([2 ** 257, 0]), 1));
// Neighbours have no spread bound: rounding-level components next to large ones.
rel(preflightNearestNeighbors(row([10.9, 1.75, 1.4e-16, 1.6e-17]), row([8.9, 0.06, 2.2e-16, 7e-17]), 1).distances[0], Math.hypot(10.9 - 8.9, 1.75 - 0.06), 1e-15);
for (const [y, nb, contrast] of contrastGoldens) {
    const out = preflightLocalContrast(new Float64Array(y), nb, 1, false);
    assert.ok(Math.abs(out.contrast - contrast) <= 1e-15 * Math.max(1, Math.abs(contrast)), `${y}`);
}
unsupported(() => preflightLocalContrast(new Float64Array([1, 0, 2 ** -257, 1, 2]), [1, 0, 3, 2, 2], 1, false));
assert.ok(Number.isFinite(preflightLocalContrast(new Float64Array([1, 0, 2 ** 32, 1, 2]), [1, 0, 3, 2, 2], 1, false).contrast));
unsupported(() => preflightLocalContrast(new Float64Array([1, 0, 2 ** 32 + 2 ** -20, 1, 2]), [1, 0, 3, 2, 2], 1, false));
const noiseGoldens = [
    [[0.0, 1.0, 0.0], 3, 1, 0.816496580927726, 2.538181250286301e-15],
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 31, 4, 0.0, 1.6667789926024464e-15],
    [[0.0, 1.0, 4.0, 2.0, 2.0, 4.0, 6.0, 0.0, 1.0, 4.0, 2.0, 2.0, 4.0], 13, 10, 1.766567619513078, 3.265413853461328e-13],
    [[-4.0, -3.0, 4.0, 1.0, 5.0, 0.0, 3.0, -2.0, 2.0, -1.0, 6.0, -4.0, -3.0, 4.0, 1.0], 13, 6, 2.724890161911477, 1.3503489675058813e-13],
];
// Noise proxy against exact rational SG residuals within the honest bound
// 2 W eps sum|c_k| max|w_k| / gain (+ a few eps relative).
for (const [x, w, o, pooled, bound] of noiseGoldens) {
    const out = preflightNoiseProxy(row(x), w, o);
    assert.ok(Math.abs(out.pooled - pooled) <= bound, `${w}/${o}: ${out.pooled} vs ${pooled}`);
}
// Review b1 #1: the registry SNV keeps a constant row flat and resists overflow.
const snvOp = ppCreate("StandardNormalVariate");
try {
    assert.ok(ppTransform(snvOp, new Float64Array(11).fill(0.1), 1, 11).every((v) => v === 0));
    const extreme = ppTransform(snvOp, new Float64Array([1e200, 0, -1e200]), 1, 3);
    relativeNear(extreme[0], Math.sqrt(1.5));
    relativeNear(extreme[2], -Math.sqrt(1.5));
    const subnormal = ppTransform(snvOp, new Float64Array([2 ** -1074, 0, -(2 ** -1074)]), 1, 3);
    relativeNear(subnormal[0], Math.sqrt(1.5));
} finally {
    ppDestroy(snvOp);
}
const pairs = preflightPairsWithinTolerance({
    data: new Float64Array([0, 0, 0.1, 0, 0, 0.3, 5, 5, 0, 0]), rows: 5, cols: 2,
}, 0.1, 1, undefined, [1, 2, 3, 4, 1]);
assert.equal(pairs.totalPairs, 2);
assert.equal(pairs.candidatePairs, 9);
assert.deepEqual(pairs.pairs, [0, 1]);
near(pairs.distances[0], Math.sqrt(0.005));
assert.deepEqual(Array.from(pairs.queryFlags), [1, 1, 0, 0, 1]);
const cross = preflightPairsWithinTolerance({data: new Float64Array([0, 0]), rows: 1, cols: 2},
    0.1, 0, {data: new Float64Array([0, 0.1, 1, 1]), rows: 2, cols: 2});
assert.equal(cross.totalPairs, 1);
assert.deepEqual(Array.from(cross.referenceFlags), [1, 0]);
const blobs = clusterHdbscan({data: new Float64Array([0, 0, 1, 1, 5, 5, 6, 6]), rows: 4, cols: 2},
    {minClusterSize: 2, minSamples: 2});
assert.equal(blobs.nClusters, 2);
assert.equal(blobs.labels[0], blobs.labels[1]);
assert.equal(blobs.labels[2], blobs.labels[3]);
assert.notEqual(blobs.labels[0], blobs.labels[2]);
assert.equal(blobs.tree[0].parent, -1);
assert.equal(blobs.tree[0].size, 4);
assert.equal(blobs.persistence.length, 2);
assert.equal(blobs.membership.length, 4);
assert.throws(() => clusterHdbscan({data: new Float64Array([0, 1]), rows: 2, cols: 1},
    {minClusterSize: 1, minSamples: 1}), RangeError);
near(preflightAdjustedRandIndex([0, 0, 1, 2], [0, 0, 1, 1]), 0.5714285714285714);
near(preflightAdjustedRandIndex([0, 0, 1, 1], [0, 1, 0, 1]), -0.5);
assert.ok(Number.isNaN(preflightAdjustedRandIndex([3, 3, 3], [1, 1, 1])));
// Same frozen HDBSCAN reference gate as the native n4m_hdbscan_tests.
const hex = (h) => Buffer.from(h, "hex").readDoubleBE(0);
const same = (a, b) => Number.isNaN(b) ? Number.isNaN(a) : a === b ||
    Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b));
const fixture = JSON.parse(readFileSync(
    new URL("../../../parity/fixtures/hdbscan_v1.json", import.meta.url), "utf8"));
for (const c of fixture.cases) {
    const got = clusterHdbscan({data: new Float64Array(c.input_hex.map(hex)), rows: c.rows, cols: c.cols}, {
        minClusterSize: c.params.min_cluster_size, minSamples: c.params.min_samples,
        selection: c.params.selection_method === 1 ? "leaf" : "eom",
        allowSingleCluster: c.params.allow_single_cluster === 1,
    });
    const map = new Map();
    got.labels.forEach((label, i) => {
        assert.equal(label < 0, c.labels[i] < 0, c.name);
        if (label >= 0) assert.equal(map.get(label) ?? c.labels[i], c.labels[i], c.name);
        if (label >= 0) map.set(label, c.labels[i]);
    });
    assert.equal(new Set(map.values()).size, map.size, c.name);
    assert.equal(got.nClusters, c.n_clusters, c.name);
    c.membership_hex.map(hex).forEach((v, i) => assert.ok(same(got.membership[i], v), c.name));
    c.outlier_scores_hex.map(hex).forEach((v, i) => assert.ok(same(got.outlierScores[i], v), c.name));
    assert.deepEqual(got.tree.map((node) => node.parent), c.tree_parent, c.name);
}
// Same hand values and SciPy/NumPy/statsmodels goldens as test_preflight.cpp.
const pearson = statsCorrelationTest(new Float64Array([1, 2, 3, 4, 5]),
    new Float64Array([2, 4, 5, 4, 5]), "pearson", 999, 17);
near(pearson.coefficient, 6 / Math.sqrt(60));
near(pearson.log10DistinctPermutations, Math.log10(30));
assert.equal(pearson.nAtLeastObserved, 133);
assert.equal(pearson.nPermutationsExecuted, 999);
near(pearson.pValue, 134 / 1000);
const spearman = statsCorrelationTest(new Float64Array([1, 2, 3, 4, 5]),
    new Float64Array([2, 4, 5, 4, 5]), "spearman", 999, 18446744073709551557n);
near(spearman.coefficient, 7 / Math.sqrt(90));
assert.equal(spearman.nAtLeastObserved, 190);
const tiedRho = statsCorrelationTest(new Float64Array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
    new Float64Array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3]), "spearman", 999, 17);
near(tiedRho.coefficient, 0.4099457958749614);
assert.equal(tiedRho.nAtLeastObserved, 241);
assert.ok(Number.isNaN(statsCorrelationTest(new Float64Array([1, 1, 1]),
    new Float64Array([1, 2, 3]), "pearson", 99, 1).coefficient));
const bh = statsBhAdjust(new Float64Array([0.001, 0.2, 0.5, 0.03, 0.04, 0.049]));
[0.006, 0.24, 0.5, 0.0735, 0.0735, 0.0735].forEach((value, i) => near(bh[i], value));
// Translation invariance and exact enumeration (brute-force SciPy goldens).
const same16 = preflightFactorAssociation([0, 0, 0, 1, 1, 1],
    new Float64Array([1e16, 1e16 + 2, 1e16 + 4, 1e16, 1e16 + 2, 1e16 + 4]), false, 0, 1);
assert.ok(Math.abs(same16.effectSize) < 1e-15 && Math.abs(same16.statistic) < 1e-15);
const perfect = statsFactorAssociationExact([0, 0, 0, 0, 1, 1, 1, 1],
    new Float64Array([0, 0, 0, 0, 1, 1, 1, 1]), true, 1000);
assert.equal(perfect.nArrangements, 70);
near(perfect.minimumPValue, 1 / 35);
const exactRho = statsCorrelationExact(new Float64Array([1, 2, 3, 4, 5]),
    new Float64Array([2, 4, 5, 4, 5]), "spearman", 1000);
assert.equal(exactRho.nArrangements, 30);
near(exactRho.pValue, 0.2);
near(exactRho.minimumPValue, 1 / 15);
// Oversized categorical space (review P0): 65536 levels x 65536 classes
// overflow 32-bit sizes; both APIs refuse before allocating.
const bigCodes = Array.from({length: 65536}, (_, i) => i);
const bigClasses = Float64Array.from(bigCodes);
// Only the explicit status is accepted: a WASM trap (RuntimeError) fails.
const status = (code) => (error) => error?.name === "N4mError" && error.status === code;
assert.throws(() => statsFactorAssociationExact(bigCodes, bigClasses, true, 100000), status(10));
assert.throws(() => preflightFactorAssociation(bigCodes, bigClasses, true, 0, 1), status(10));
assert.ok(statsFactorAssociationExact([0, 0, 1, 1], new Float64Array([0, 0, 1, 1]), true, 10)
    .nArrangements === 6);
// Supported numerical domain D (rule 13; same cases as test_preflight.cpp):
// nonzero |x| in [2^-256, 2^256], max/min nonzero |x| <= 2^32 per array (rule 15).
// Inside D results match exact references; outside D only N4M_ERR_UNSUPPORTED
// (status 10) is accepted, never a trap or a silent value.
const relNear = (a, b) => assert.ok(Math.abs(a - b) <= 1e-12 * Math.abs(b), `${a} vs ${b}`);
const sp = 2 ** 32;  // lo = 2^-256 and hi = 2^256 are declared above
const xz4 = new Float64Array([1, 2, 3, 4]);
const insideD = [[lo, 2 * lo, 3 * lo, 0], [hi / 4, hi / 2, hi, 0], [1, 2, sp, 0], [-lo, 0, lo * sp, -lo]];
const outsideD = [[lo * (1 - 2 ** -53), 2 * lo, 3 * lo, 0], [hi / 4, hi / 2, hi * (1 + 2 ** -52), 0],
    [1, 2, sp * (1 + 2 ** -52), 0], [-lo, 0, 2 * lo * sp, -lo], [1, 0, 2 ** 33, 0]];
for (const v of insideD) {
    const y = new Float64Array(v);
    preflightFactorAssociation([0, 0, 1, 1], y, false, 0, 1);
    statsFactorAssociationExact([0, 0, 1, 1], y, false, 100);
    statsCorrelationTest(y, xz4, "pearson", 0, 1);
    statsCorrelationExact(xz4, y, "spearman", 100);
}
for (const v of outsideD) {
    const y = new Float64Array(v);
    assert.throws(() => preflightFactorAssociation([0, 0, 1, 1], y, false, 0, 1), status(10));
    assert.throws(() => statsFactorAssociationExact([0, 0, 1, 1], y, false, 100), status(10));
    assert.throws(() => statsCorrelationTest(y, xz4, "pearson", 0, 1), status(10));
    assert.throws(() => statsCorrelationExact(xz4, y, "spearman", 100), status(10));
}
// Holm/BH: no spread limit (p = 1 next to 1e-25 or 2^-1022); subnormal p refused.
assert.deepEqual(Array.from(preflightHolmAdjust(new Float64Array([1e-25, 1]))), [2e-25, 1]);
assert.deepEqual(Array.from(statsBhAdjust(new Float64Array([1e-25, 1]))), [2e-25, 1]);
assert.equal(preflightHolmAdjust(new Float64Array([2 ** -1022, 1]))[0], 2 ** -1021);
assert.throws(() => preflightHolmAdjust(new Float64Array([2 ** -1022 * (1 - 2 ** -52), 1])), status(10));
assert.throws(() => statsBhAdjust(new Float64Array([2 ** -1074, 1])), status(10));
for (const k of [-240, -100, 0, 100, 240]) {
    const ys = Float64Array.from([1, 2, 3, 7, 8, 9], (v) => v * 2 ** k);
    const sweep = preflightFactorAssociation([4, 4, 4, -9, -9, -9], ys, false, 999, 17);
    relNear(sweep.effectSize, 27 / 29);
    relNear(sweep.statistic, 54);
    assert.equal(sweep.nAtLeastObserved, 93);
    const rho = statsCorrelationTest(Float64Array.from([1, 2, 3, 4, 5], (v) => v * 2 ** k),
        new Float64Array([2, 4, 5, 4, 5]), "spearman", 999, 17);
    relNear(rho.coefficient, 7 / Math.sqrt(90));
    assert.equal(rho.nAtLeastObserved, 201);
}
for (const k of [-1060, -300, 300, 1000]) {
    const ys = Float64Array.from([1, 2, 3, 7, 8, 9], (v) => v * 2 ** k);
    assert.throws(() => preflightFactorAssociation([4, 4, 4, -9, -9, -9], ys, false, 999, 17), status(10));
}
const shifted = preflightFactorAssociation([4, 4, 4, -9, -9, -9],
    Float64Array.from([1, 2, 3, 7, 8, 9], (v) => 2 ** 200 + v * 2 ** 150), false, 999, 17);
relNear(shifted.effectSize, 27 / 29);
assert.equal(shifted.nAtLeastObserved, 93);
// Mixed exponents inside D (exact references) and the review's extreme
// mixtures, now outside D.
relNear(preflightFactorAssociation([0, 0, 1], new Float64Array([0, 2 ** -30, 1]), false, 0, 1)
    .statistic, 1.5372286713774735e18);
relNear(preflightFactorAssociation([0, 0, 1], new Float64Array([-(2 ** -15), 2 ** -15, 2 ** 15]),
    false, 0, 1).statistic, (2 / 3) * 2 ** 59);
for (const y of [[0, 1e-100, 1e300], [0, 2 ** -1074, 1], [0, 2 ** -520, 1], [0, 2 ** -60, 1]]) {
    assert.throws(() => preflightFactorAssociation([0, 0, 1], new Float64Array(y), false, 0, 1), status(10));
}
const tinyHolm = preflightHolmAdjust(Float64Array.from([0.01, 0.04, 0.03, 0.005], (p) => p * 2 ** -200));
[0.03, 0.06, 0.06, 0.02].forEach((v, i) => relNear(tinyHolm[i] * 2 ** 200, v));
const d20 = 2 ** -20;
const nearZero = preflightFactorAssociation([0, 0, 1, 1], new Float64Array([0, 2, d20, 2 + d20]), false, 0, 1);
relNear(nearZero.effectSize, d20 * d20 / (4 + d20 * d20));
relNear(nearZero.statistic, d20 * d20 / 2);
const independent = preflightFactorAssociation(Array.from({length: 18}, (_, i) => Math.floor(i / 9)),
    Float64Array.from({length: 18}, (_, i) => i % 9), true, 0, 1);
assert.equal(independent.effectSize, 0);
assert.equal(independent.statistic, 0);
const nearIndependent = [];
[[0, 0, 50], [0, 1, 51], [1, 0, 51], [1, 1, 50]].forEach(([r, c, k]) => {
    for (let j = 0; j < k; ++j) nearIndependent.push([r, c]);
});
relNear(preflightFactorAssociation(nearIndependent.map(([r]) => r),
    Float64Array.from(nearIndependent, ([, c]) => c), true, 0, 1).effectSize, 1 / 101);
relNear(statsCorrelationTest(new Float64Array([1, 2, 2 ** 20]), new Float64Array([1, 2, 3]),
    "pearson", 0, 1).coefficient, 0.8660258167378265);
assert.equal(statsCorrelationTest(new Float64Array([1, 2, 3, 4]), new Float64Array([1, -1, -1, 1]),
    "pearson", 0, 1).coefficient, 0);
// PERMANOVA / PERMDISP: hand values of test_permanova.cpp, then the
// permanova_v1 reference gate (NumPy lstsq, SciPy, scikit-bio, NumPy draws).
const oneWay = statsPermanova({data: new Float64Array([1, 2, 3, 7, 8, 9]), rows: 6, cols: 1},
    [4, 4, 4, -9, -9, -9], undefined, 999, 17);
near(oneWay.ssTerm, 54);
near(oneWay.pseudoF, 54);
assert.equal(oneWay.nAtLeastObserved, 93);
const crossed = statsPermanova({data: new Float64Array([1, 3, 5, 7, 2, 4, 6, 8]), rows: 8, cols: 1},
    [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], 0, 1);
near(crossed.pseudoF, 20);
near(crossed.ssStrata, 2);
const aliased = statsPermanova({data: new Float64Array([1, 3, 5, 7, 2, 4, 6, 8]), rows: 8, cols: 1},
    [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], 999, 1);
assert.equal(aliased.dfTerm, 0);
assert.ok(Number.isNaN(aliased.pValue));
const disp = statsPermdisp({data: new Float64Array([0, 2, 10, 14, 18]), rows: 5, cols: 1},
    [7, 7, 3, 3, 3], 0, 1);
near(disp.f, 15 / 16);
[8 / 3, 1].forEach((v, i) => near(disp.groupMeanDistances[i], v));
// Robustness rules 13 and 15 (same cases as test_permanova.cpp): inside the
// supported domain D (nonzero |x| in [2^-256, 2^256], max/min nonzero
// |x| <= 2^32) results are exact up to the published exponent; outside D
// every kernel refuses with status 10 and the domain message.
const outsideDomain = (error) => error?.name === "N4mError" && error.status === 10 &&
    error.message.includes("input outside the supported numerical domain");
const kTerm = [0, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 2];
const kStrata = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
const kBase = [3, 1, 5, 2, 9, 4, 11, 7, 13, 12, 15, 9, 4, 6, 10, 3, 12, 8, 16, 14, 17, 10, 21, 11];
const contract = (values, e) => values.every((v) => v >= 0 && Number.isFinite(v) &&
    !(e === 0 && v > 0 && v < 2.2250738585072014e-308));
const kernels = {
    permanova: (x, cols) => { const r = statsPermanova({data: Float64Array.from(x), rows: 12, cols}, kTerm, kStrata, 199, 7); return {ss: [r.ssTotal, r.ssTerm, r.ssResidual], e: r.ssExponent, ratio: [r.pseudoF, r.partialR2, Number(r.fNumericallyUnresolved)], b: r.nAtLeastObserved}; },
    freedmanLane: (x, cols) => { const r = statsPermanovaFreedmanLane({data: Float64Array.from(x), rows: 12, cols}, kTerm, [kStrata], 199, 7); return {ss: [r.ssTotal, r.ssTerm, r.ssResidual], e: r.ssExponent, ratio: [r.pseudoF, r.partialR2, Number(r.fNumericallyUnresolved)], b: r.nAtLeastObserved}; },
    permdisp: (x, cols) => { const r = statsPermdisp({data: Float64Array.from(x), rows: 12, cols}, kTerm, 199, 7); return {ss: Array.from(r.distances), e: r.distanceExponent, ratio: [r.f, Number(r.fNumericallyUnresolved)], b: r.nAtLeastObserved}; },
};
const ldexp = (v, k) => { let out = v; for (let i = 0; i < Math.abs(k); i++) out = k > 0 ? out * 2 : out / 2; return out; };
const sameScaled = (got, base, shift, tag) => {
    assert.ok(contract(got.ss, got.e), tag);
    // Tolerance relative to each value and to the largest one (a zero
    // distance may become a tiny positive one next to a 2^-56 column).
    const want = base.ss.map((v) => ldexp(ldexp(v, base.e), shift));
    const largest = Math.max(...want);
    got.ss.forEach((v, i) => {
        const have = ldexp(v, got.e);
        assert.ok(Math.abs(have - want[i]) <= 1e-9 * Math.max(want[i], 1e-6 * largest), tag);
        assert.ok(!(want[i] > 0) || have > 0, tag);
    });
    got.ratio.forEach((v, i) => assert.ok(Math.abs(v - base.ratio[i]) <= 1e-9 * Math.max(1, Math.abs(base.ratio[i])), tag));
    assert.equal(got.b, base.b, tag);
};
for (const [name, run] of Object.entries(kernels)) {
    const power = name === "permdisp" ? 1 : 2;
    const base = run(kBase, 2);
    // Magnitude sweep inside D (data 1..21: k in [-256, 251]) and outside.
    for (const k of [-256, -200, -100, 0, 100, 200, 251]) {
        sameScaled(run(kBase.map((v) => ldexp(v, k)), 2), base, power * k, `${name} k=${k}`);
    }
    for (const k of [-1069, -257, 252, 1000]) {
        assert.throws(() => run(kBase.map((v) => ldexp(v, k)), 2), outsideDomain, `${name} k=${k}`);
    }
    // Translation: offsets 1e16 (even data) and 2^200 (data * 2^148) inside
    // D; 2^300 and 1e300 outside.
    for (const [offset, k] of [[1e16, 1], [2 ** 200, 148]]) {
        sameScaled(run(kBase.map((v) => offset + ldexp(v, k)), 2), base, power * k, `${name} offset ${offset}`);
    }
    for (const [offset, k] of [[2 ** 300, 248], [1e300, 944]]) {
        assert.throws(() => run(kBase.map((v) => offset + ldexp(v, k)), 2), outsideDomain, `${name} offset ${offset}`);
    }
    // Constant columns inside D (spread 2^31) change nothing; 2^30 (spread
    // 2^33) and 1e308 constant columns are refused.
    for (const huge of [2 ** 28, 2 ** 30, 1e308]) {
        const wide = [];
        for (let i = 0; i < 12; i++) wide.push(huge, kBase[2 * i], -(2 ** 27), kBase[2 * i + 1], 0.125);
        if (huge < 2 ** 29) sameScaled(run(wide, 5), base, 0, `${name} constant columns`);
        else assert.throws(() => run(wide, 5), outsideDomain, `${name} constant ${huge}`);
    }
    // Mixed signed exponents: a 2^13 column next to a -2^-13 (term + 1)
    // column (constant within term levels); 2^28 / -2^-28 is refused.
    const one = run(kBase.filter((_, i) => i % 2 === 0), 1);
    const mixed = kBase.map((v, i) => i % 2 === 0 ? ldexp(v, 13) : -ldexp(kTerm[(i - 1) / 2] + 1, -13));
    sameScaled(run(mixed, 2), one, power * 13, `${name} mixed`);
    assert.throws(() => run(kBase.map((v, i) => i % 2 === 0 ? ldexp(v, 28) : -ldexp(v, -28)), 2),
        outsideDomain, `${name} wide mixed`);
}
// Value-range boundaries: 2^+-256 inside D (SS_total = 6 (a/2)^2), 2^+-257 refused.
for (const e of [256, -256, 257, -257]) {
    const a = ldexp(1, e);
    const call = () => statsPermanova({data: new Float64Array([0, 0, 0, a, a, a]), rows: 6, cols: 1},
        [4, 4, 4, -9, -9, -9], undefined, 0, 1);
    if (Math.abs(e) === 256) {
        const r = call();
        assert.equal(r.ssExponent, 0);
        assert.equal(r.ssTotal, 6 * ldexp(1, 2 * (e - 1)));
        assert.equal(r.r2, 1);
    } else {
        assert.throws(call, outsideDomain, `value range ${e}`);
    }
}
{
    const dev = [-3e-8, -1e-8, 1e-8, 3e-8];
    const a = [], b = [], y = [];
    for (let i = 0; i < 16; i++) { a.push(Math.floor(i / 4) % 2); b.push(Math.floor(i / 8)); y.push(10 * b[i] + 3 * a[i] + dev[i % 4]); }
    const fl = statsPermanovaFreedmanLane({data: Float64Array.from(y), rows: 16, cols: 1}, a, [b], 99, 3);
    assert.ok(fl.ssExponent === 0 && fl.ssResidual > 7e-15 && fl.ssResidual < 9e-15);
    assert.ok(Number.isFinite(fl.pseudoF) && Math.abs(fl.pseudoF / 5.85e16 - 1) < 1e-2);
    assert.equal(fl.fNumericallyUnresolved, false);
}
{
    // Dyadic fixture (same as test_permanova.cpp) at the spread boundary
    // (p = 16, q = -17, max/min nonzero |x| = 2^16 / 2^-16 = 2^32), 4, 6 and
    // 8 rows per cell: SS_A = SS_res = n 2^-34, F = n - 3, partial R^2 1/2;
    // PERMDISP by B gives distances 2^-17 |A + r|. (17, -17), (33, -32) and
    // (600, -500) are outside D.
    const fixture = (p, q, perCell) => {
        const x = [], a = [], b = [];
        for (let i = 0; i < 4 * perCell; i++) {
            const cell = Math.floor(i / perCell);
            a.push(cell < 2 ? -1 : 1); b.push(cell % 2 === 0 ? -1 : 1);
            x.push(ldexp(b[i], p), ldexp(a[i] + (i % 2 === 0 ? 1 : -1), q));
        }
        return {X: {data: Float64Array.from(x), rows: 4 * perCell, cols: 2}, a, b};
    };
    for (const perCell of [4, 6, 8]) {
        const {X, a, b} = fixture(16, -17, perCell), n = 4 * perCell;
        for (const r of [statsPermanovaFreedmanLane(X, a, [b], 99, 5), statsPermanova(X, a, b, 99, 5)]) {
            assert.equal(r.ssExponent, 0);
            relNear(r.ssTerm, n * 2 ** -34);
            relNear(r.ssResidual, n * 2 ** -34);
            relNear(r.pseudoF, n - 3);
            relNear(r.partialR2, 0.5);
            assert.equal(r.fNumericallyUnresolved, false);
        }
        const d = statsPermdisp(X, b, 0, 1);
        assert.equal(d.distanceExponent, 0);
        d.distances.forEach((v, i) => assert.equal(v, Math.abs(a[i] + (i % 2 === 0 ? 1 : -1)) * 2 ** -17));
        assert.equal(d.f, 0);
    }
    for (const [p, q] of [[17, -17], [33, -32], [600, -500]]) {
        const out = fixture(p, q, 4);
        assert.throws(() => statsPermanovaFreedmanLane(out.X, out.a, [out.b], 0, 1), outsideDomain);
        assert.throws(() => statsPermanova(out.X, out.a, out.b, 0, 1), outsideDomain);
        assert.throws(() => statsPermdisp(out.X, out.b, 0, 1), outsideDomain);
    }
}
{
    // Offset stratum next to a varying one (rational oracle): SS_A = 2,
    // SS_res = 3, F = 10/3, partial R^2 0.4, PERMDISP distances by stratum
    // [0 0 0 0 1.5 0.5 0.5 1.5] with c = 2^31; c = 2^60 is refused.
    const term = [0, 0, 1, 1, 0, 0, 1, 1], strata = [0, 0, 0, 0, 1, 1, 1, 1];
    const X = (c) => ({data: new Float64Array([c, c, c, c, 1, 2, 3, 4]), rows: 8, cols: 1});
    for (const r of [statsPermanova(X(2 ** 31), term, strata, 0, 1),
        statsPermanovaFreedmanLane(X(2 ** 31), term, [strata], 0, 1)]) {
        relNear(r.ssTerm, 2);
        relNear(r.ssResidual, 3);
        relNear(r.pseudoF, 10 / 3);
        relNear(r.partialR2, 0.4);
    }
    assert.deepEqual(Array.from(statsPermdisp(X(2 ** 31), strata, 0, 1).distances),
        [0, 0, 0, 0, 1.5, 0.5, 0.5, 1.5]);
    assert.throws(() => statsPermanova(X(2 ** 60), term, strata, 0, 1), outsideDomain);
    assert.throws(() => statsPermdisp(X(2 ** 60), strata, 0, 1), outsideDomain);
}
{
    // F is published whenever the residual SS is positive: resolved for
    // A next to 2^-31 r (F = 14 2^62), flagged for one-ulp residuals, NaN
    // only for an exactly zero residual SS.
    const term = Array.from({length: 16}, (_, i) => i < 8 ? -1 : 1);
    const x = [];
    term.forEach((t, i) => x.push(t, (i % 2 === 0 ? 1 : -1) * 2 ** -31));
    const fine = statsPermanova({data: Float64Array.from(x), rows: 16, cols: 2}, term, undefined, 0, 1);
    relNear(fine.pseudoF, 14 * 2 ** 62);
    assert.ok(fine.residualResolution < 1e-3 && !fine.fNumericallyUnresolved);
    const ulp = statsPermanova({data: new Float64Array([1, 1 + 2 ** -52, 1, 1, 3, 3, 3, 3]), rows: 8, cols: 1},
        [0, 0, 0, 0, 1, 1, 1, 1], undefined, 99, 1);
    assert.ok(ulp.ssResidual > 0 && Number.isFinite(ulp.pseudoF) && ulp.fNumericallyUnresolved);
    assert.ok(ulp.residualResolution >= 1 && ulp.nPermutationsExecuted === 99);
    const exact = statsPermanova({data: new Float64Array([1, 1, 3, 3]), rows: 4, cols: 1}, [0, 0, 1, 1],
        undefined, 99, 1);
    assert.ok(Number.isNaN(exact.pseudoF) && exact.residualResolution === Infinity && exact.fNumericallyUnresolved);
}
const permanovaFixture = JSON.parse(readFileSync(
    new URL("../../../parity/fixtures/permanova_v1.json", import.meta.url), "utf8"));
for (const c of permanovaFixture.cases) {
    const X = {data: new Float64Array(c.X.map(hex)), rows: c.rows, cols: c.cols};
    const got = statsPermanova(X, c.term, c.strata ?? undefined, c.n_permutations, c.seed);
    const want = c.permanova, total = hex(want.ss_total);
    const close = (a, b, scale = 1) => Number.isNaN(b) ? Number.isNaN(a)
        : Math.abs(a - b) <= 1e-10 * Math.max(scale, Math.abs(b));
    [["ssTotal", "ss_total"], ["ssStrata", "ss_strata"], ["ssTerm", "ss_term"],
        ["ssResidual", "ss_residual"]].forEach(([k, w]) => assert.ok(close(got[k], hex(want[w]), total), c.name));
    [["r2", "r2"], ["partialR2", "partial_r2"], ["pseudoF", "pseudo_f"]]
        .forEach(([k, w]) => assert.ok(close(got[k], hex(want[w])), c.name));
    assert.equal(got.dfTerm, want.df_term, c.name);
    assert.equal(got.dfResidual, want.df_residual, c.name);
    assert.equal(got.nAtLeastObserved, want.b, c.name);
    assert.equal(got.nPermutationsExecuted, want.executed, c.name);
    if (c.freedman_lane) {
        const fl = c.freedman_lane;
        const r = statsPermanovaFreedmanLane(X, c.term, fl.conditions, c.n_permutations, fl.seed);
        assert.ok(close(r.ssTerm, hex(fl.ss_term), hex(fl.ss_total)), c.name);
        assert.ok(close(r.pseudoF, hex(fl.pseudo_f)), c.name);
        assert.equal(r.dfTerm, fl.df_term, c.name);
        assert.equal(r.rankReduced, fl.rank_reduced, c.name);
        assert.equal(r.nAtLeastObserved, fl.b, c.name);
        assert.equal(r.nPermutationsExecuted, fl.executed, c.name);
    }
    if (c.permdisp) {
        const d = statsPermdisp(X, c.term, c.n_permutations, c.permdisp.seed);
        assert.ok(close(d.f, hex(c.permdisp.f)), c.name);
        c.permdisp.distances.map(hex).forEach((v, i) => assert.ok(close(d.distances[i], v), c.name));
        assert.equal(d.nAtLeastObserved, c.permdisp.b, c.name);
    }
}
// Energy distance and PCA monitoring gate shared with n4m_domain_shift_tests.
{
    const hexd = (h) => Buffer.from(h, "hex").readDoubleBE(0);
    const EPS = 2 ** -52;
    const closeTo = (a, b, floor = 1) => Number.isNaN(b) ? Number.isNaN(a) :
        Math.abs(a - b) <= 1e-12 * Math.max(floor, Math.abs(b));
    const isStatus = (code) => (e) => e instanceof N4mError && e.status === code;
    const shift = JSON.parse(readFileSync(
        new URL("../../../parity/fixtures/domain_shift_v1.json", import.meta.url), "utf8"));
    assert.equal(shift.cases.length, 20);
    for (const c of shift.cases) {
        const X = {data: new Float64Array(c.input_hex.map(hexd)), rows: c.rows, cols: c.cols};
        if (c.kind === "energy") {
            const got = statsEnergyDistance(X, c.codes, c.n_groups, c.n_permutations, BigInt(c.seed));
            const g = c.n_groups, rational = c.oracle === "rational";
            c.energy_hex.map(hexd).forEach((v, i) => {
                if (rational) assert.ok(Math.abs(got.energy[i] - v) <= got.energyBounds[i] + EPS / 2 * Math.abs(v), c.name);
                else assert.ok(closeTo(got.energy[i], v), c.name);
                assert.ok(got.energy[i] >= 0 && got.energyBounds[i] >= 0, c.name);
                assert.equal(got.energy[i], got.energy[(i % g) * g + Math.floor(i / g)], `${c.name} symmetry`);
            });
            c.mean_distances_hex.map(hexd).forEach((v, i) => assert.ok(closeTo(got.meanDistances[i], v), c.name));
            assert.ok(closeTo(got.pValue, hexd(c.p_value_hex[0])), c.name);
            assert.equal(got.nAtLeastObserved, c.n_at_least_observed, c.name);
            assert.equal(got.nPermutationsExecuted, c.n_permutations, c.name);
            if (c.n_permutations) assert.ok(got.pValueConservative >= got.pValue && got.nUndecidable >= 0);
        } else {
            const M = {data: new Float64Array(c.monitored_hex.map(hexd)), rows: c.monitored_rows,
                cols: c.cols};
            const got = statsPcaMonitor(X, M, c.k, c.n_folds, hexd(c.alpha_hex[0]));
            const tau = got.tolerance, rel = Math.max(tau, 1e-12), qs = Math.max(got.qToleranceScale, 1);
            assert.ok(Math.abs(tau - hexd(c.tau_hex[0])) <= (1e-6 + tau) * tau, c.name);
            assert.equal(got.effectiveComponents, c.effective_k, c.name);
            assert.deepEqual([got.nBeyondT2, got.nBeyondQ, got.nBeyondEither, got.nNearT2,
                got.nNearQ, got.nNearEither], c.counts, c.name);
            c.monitored_q_hex.map(hexd).forEach((v, i) => {
                assert.ok(Math.abs(got.monitoredQ[i] - v) <= rel * qs, c.name);
                assert.ok(got.monitoredQ[i] >= 0, c.name);
            });
            c.monitored_t2_hex.map(hexd).forEach((v, i) => assert.ok(Number.isNaN(v) ?
                Number.isNaN(got.monitoredT2[i]) : Math.abs(got.monitoredT2[i] - v) <= rel * Math.max(Math.abs(v), 1), c.name));
            const out = c.out_hex.map(hexd);
            assert.ok(closeTo(got.referenceTotalVariance, out[2]) && closeTo(got.monitoredTotalVariance, out[3]), c.name);
        }
    }
    // Analytic cases: local differences, exact cancellation, symmetry.
    const col = (...v) => ({data: new Float64Array(v), rows: v.length, cols: 1});
    let e = statsEnergyDistance(col(2 ** 31, 0, 1), [0, 1, 2], 3, 0, 0);
    assert.equal(e.energy[1 * 3 + 2], 2);
    e = statsEnergyDistance(col(0, -4, -7, 2, 2, 2 ** 31, 1, -4, -7, 2, 2, 2 ** 31),
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 2, 0, 0);
    assert.ok(Math.abs(e.energy[1] - 1 / 18) <= e.energyBounds[1] && e.energy[1] === e.energy[2]);
    // Declared tie rule (rational oracle 476 / 999).
    e = statsEnergyDistance(col(0, 1, 2, 2 ** 31, 100, 101, 102, 2 ** 31), [0, 0, 0, 0, 1, 1, 1, 1], 2, 999, 42n);
    assert.equal(e.nAtLeastObserved, 476);
    assert.ok(e.pValueConservative >= e.pValue);
    const line = [];
    for (let i = -5; i <= 5; i++) line.push(i, 0);
    let mon = statsPcaMonitor({data: new Float64Array(line), rows: 11, cols: 2},
        {data: new Float64Array([0, 100, 1, 2 ** -18, 2, 0]), rows: 3, cols: 2}, 1, 5, 0.05);
    assert.equal(mon.monitoredQ[1], 2 ** -36);
    assert.ok(mon.nBeyondQ === 2 && mon.nNearQ === 1);
    mon = statsPcaMonitor(col(...[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].map((i) => 2 ** 31 + 256 * i)),
        col(2 ** 31 - 2 ** 20, 2 ** 31 - 2 ** 20 + 1), 1, 5, 0.05);
    assert.equal(mon.monitoredTotalVariance, 0.5);
    // Domain boundaries: inside accepted, outside refused with status 10.
    const pair = (a, b) => statsEnergyDistance(col(a, b, a, b), [0, 1, 1, 0], 2, 0, 0);
    const after = 1 + EPS;
    pair(2 ** -256, 2 ** -250);
    assert.throws(() => pair(2 ** -256 / after, 2 ** -250), isStatus(10));
    pair(2 ** 256, 2 ** 250);
    assert.throws(() => pair(2 ** 256 * after, 2 ** 250), isStatus(10));
    pair(1, 2 ** 32);
    assert.throws(() => pair(1, 2 ** 32 * after), isStatus(10));
    assert.throws(() => pair(1, 2 ** 60), isStatus(10));
    pair(0, 2 ** -256);
    assert.throws(() => pair(0, 2 ** -1074), isStatus(10));
    assert.throws(() => pair(-1e308, 1e308), isStatus(10));
    // Budgets refuse before allocation (status 10), never a trap.
    assert.throws(() => statsEnergyDistance({data: new Float64Array(100000), rows: 100000, cols: 1},
        Array.from({length: 100000}, (_, i) => i), 100000, 0, 0), isStatus(10));
    assert.throws(() => statsEnergyDistance(col(0, 1), [0, 2], 2, 0, 0), RangeError);
    // Magnitude sweep on integer-grid data: exact power-of-two images inside
    // the domain, explicit refusal (status 10) outside.
    const n = 12, p = 3, base = new Float64Array(n * p);
    for (let i = 0; i < n; i++) for (let j = 0; j < p; j++)
        base[i * p + j] = ((i * 37 + j * 11) % 23) - 11 + (i >= 8 ? 9 * (j + 1) : 0);
    const codes = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
    const runBoth = (data, cols) => ({
        e: statsEnergyDistance({data, rows: n, cols}, codes, 2, 99, 7n),
        m: statsPcaMonitor({data: data.slice(0, 8 * cols), rows: 8, cols},
            {data: data.slice(8 * cols), rows: 4, cols}, 2, 4, 0.1),
    });
    const unit = runBoth(base, p);
    for (const s of [-1062, -300, -230, -100, 0, 100, 200, 300, 1000]) {
        const x = base.map((v) => v * 2 ** s);
        if (s < -256 || s + 6 > 256) {
            assert.throws(() => runBoth(x, p), isStatus(10), `shift ${s}`);
            continue;
        }
        const r = runBoth(x, p);
        assert.equal(r.e.energy[1], unit.e.energy[1] * 2 ** s, `energy ${s}`);
        assert.equal(r.e.nAtLeastObserved, unit.e.nAtLeastObserved);
        assert.equal(r.m.effectiveComponents, unit.m.effectiveComponents);
        assert.deepEqual([r.m.nBeyondT2, r.m.nBeyondQ, r.m.nNearT2, r.m.nNearQ],
            [unit.m.nBeyondT2, unit.m.nBeyondQ, unit.m.nNearT2, unit.m.nNearQ]);
    }
    for (const level of [2 ** 31, 2 ** 33]) {
        const wide = new Float64Array(n * (p + 1));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < p; j++) wide[i * (p + 1) + j] = base[i * p + j];
            wide[i * (p + 1) + p] = level;
        }
        if (level > 2 ** 32) {
            assert.throws(() => runBoth(wide, p + 1), isStatus(10));
            continue;
        }
        const w = runBoth(wide, p + 1);
        assert.deepEqual([...w.e.energy], [...unit.e.energy]);
        assert.deepEqual([w.m.nBeyondQ, w.m.nBeyondT2], [unit.m.nBeyondQ, unit.m.nBeyondT2]);
    }
}
// Overflowing squared distances once produced 7 tree nodes for 4 rows; such
// inputs are now outside the rule-13 domain and refused with status 10.
const unit = clusterHdbscan({data: new Float64Array([0, 1, 2, 3]), rows: 4, cols: 1},
    {minClusterSize: 2, minSamples: 2});
const edge = clusterHdbscan({data: new Float64Array([0, 2 ** 254, 2 ** 255, 3 * 2 ** 254]), rows: 4, cols: 1},
    {minClusterSize: 2, minSamples: 2});
assert.deepEqual(edge.labels, unit.labels);
for (const data of [[0, 1e155, 2e155, 3e155], [-1e308, 1e308]]) {
    assert.throws(() => clusterHdbscan({data: new Float64Array(data), rows: data.length, cols: 1},
        {minClusterSize: 2, minSamples: 2}), (e) => e.status === 10);
}
// Review 2026-10-13 P0: a 65536 x 65536 overlapping covariance view with n = 0
// made d*d wrap to 0 in WASM32 and the Cholesky factor be written into an empty
// vector. The limits are now checked first and the call is refused.
{
    const m = getModule(), ctx = Context.create();
    const backing = makeMatrixView(new Float64Array(131071).fill(1), 131071, 1);
    const mean = m._malloc(65536 * 8);
    try {
        m.HEAPF64.fill(0, mean / 8, mean / 8 + 65536);
        assert.equal(m.ccall("n4m_matrix_view_init_strided", "number",
            ["number", "number", "i64", "i64", "i64", "i64", "number"],
            [backing.viewPtr, backing.dataPtr, 65536n, 65536n, 1n, 1n, 1]), 0);
        const before = m.HEAPF64[0];
        assert.equal(m.ccall("n4m_random_multivariate_normal", "number",
            ["number", "i64", "number", "number", "i64", "number"],
            [ctx.handle, 1n, mean, backing.viewPtr, 0n, 0]), 10 /* N4M_ERR_UNSUPPORTED */);
        assert.equal(m.HEAPF64[0], before);
    } finally {
        m._free(mean); backing.free(); ctx.destroy();
    }
}
// Rules 13/15 domain (same rules as n4m_cluster_robustness_tests): finite,
// nonzero |x| in [2^-256, 2^256], max/min nonzero |x| <= 2^32. Inside it the
// results are exact power-of-two rescalings or exact optima; outside it every
// kernel refuses with N4M_ERR_UNSUPPORTED (10) and no trap.
{
    const status = (fn) => { try { fn(); return 0; } catch (e) { return e.status ?? -1; } };
    const line = [0, 1, 3, 10, 11, 14];
    const unitLine = clusterHdbscan({data: new Float64Array(line), rows: 6, cols: 1}, {minClusterSize: 2, minSamples: 2});
    for (const k of [-256, -100, 100, 252]) {
        const got = clusterHdbscan({data: new Float64Array(line.map((v) => v * 2 ** k)), rows: 6, cols: 1},
            {minClusterSize: 2, minSamples: 2});
        assert.deepEqual(got.labels, unitLine.labels);
        assert.deepEqual(Array.from(got.persistence), Array.from(unitLine.persistence));
        got.tree.forEach((node, i) => assert.equal(node.stability, unitLine.tree[i].stability * 2 ** -k));
    }
    // Outside: below 2^-256, above 2^256, and the review's 1e-308 / 1e-310.
    for (const factor of [2 ** -257, 2 ** 253, 1e-308, 1e-310]) {
        const code = status(() => clusterHdbscan({data: new Float64Array(line.map((v) => v * factor)), rows: 6, cols: 1},
            {minClusterSize: 2, minSamples: 2}));
        assert.equal(code, 10, `hdbscan x ${factor}`);
    }
    // Spread edge: max/min nonzero |x| = 2^32 accepted, one ulp more refused.
    const hdbscanStatus = (values) => status(() => clusterHdbscan({data: new Float64Array(values), rows: values.length, cols: 1},
        {minClusterSize: 2, minSamples: 2}));
    assert.equal(hdbscanStatus([1, 2, 2 ** 32]), 0);
    assert.equal(hdbscanStatus([1, 2, 2 ** 32 * (1 + 2 ** -52)]), 10);
    const base = Array.from({length: 35}, (_, i) => 1 + ((i * 7.31) % 9));
    const want = linearSumAssignment({data: new Float64Array(base), rows: 5, cols: 7});
    for (const k of [-256, -100, 100, 252]) {
        assert.deepEqual(linearSumAssignment({data: new Float64Array(base.map((v) => v * 2 ** k)), rows: 5, cols: 7}).colInd,
            want.colInd);
    }
    for (const k of [-257, 253]) {
        assert.equal(status(() => linearSumAssignment({data: new Float64Array(base.map((v) => v * 2 ** k)), rows: 5, cols: 7})), 10);
    }
    assert.equal(status(() => linearSumAssignment({data: new Float64Array([1, 2 ** 32, 2, 3]), rows: 2, cols: 2})), 0);
    assert.equal(status(() => linearSumAssignment({data: new Float64Array([1, 2 ** 32 * (1 + 2 ** -52), 2, 3]), rows: 2, cols: 2})), 10);
    // Exact oracle: a double is an integer multiple of 2^-1074 (BigInt sums).
    const exactBig = (x) => {
        if (x === 0) return 0n;
        const dv = new DataView(new ArrayBuffer(8));
        dv.setFloat64(0, x);
        const bits = dv.getBigUint64(0);
        const exponent = (bits >> 52n) & 0x7ffn;
        let mantissa = bits & ((1n << 52n) - 1n);
        if (exponent !== 0n) mantissa = (mantissa | (1n << 52n)) << (exponent - 1n);
        return bits >> 63n ? -mantissa : mantissa;
    };
    const optimal = (cost, rows, cols, maximize = false) => {
        const got = linearSumAssignment({data: new Float64Array(cost), rows, cols}, maximize).colInd;
        const sign = maximize ? -1n : 1n;
        const total = (columns) => columns.reduce((acc, j, i) => acc + sign * exactBig(cost[i * cols + j]), 0n);
        let best = null;
        const visit = (prefix) => {
            if (prefix.length === rows) {
                const t = total(prefix);
                if (best === null || t < best) best = t;
                return;
            }
            for (let j = 0; j < cols; ++j) if (!prefix.includes(j)) visit([...prefix, j]);
        };
        visit([]);
        assert.equal(total(got), best);
        return got;
    };
    // Review 2026-10-13 P0: global centring returned [0, 1, 2] for the 1e20
    // matrix and declared the +-1e308 matrix infeasible. Both are outside the
    // domain now; their analogues at the domain edges keep the exact optimum.
    assert.equal(status(() => linearSumAssignment({data: new Float64Array([-1e20, 1e20, 1e20, 1e20, 1, 0, 1e20, 0, 1]), rows: 3, cols: 3})), 10);
    assert.equal(status(() => linearSumAssignment({data: new Float64Array([-1e308, 1e308, -1e308, 1e308]), rows: 2, cols: 2})), 10);
    assert.deepEqual(optimal([-(2 ** 32), 2 ** 32, 2 ** 32, 2 ** 32, 1, 0, 2 ** 32, 0, 1], 3, 3), [0, 2, 1]);
    optimal([-(2 ** 256), 2 ** 256, -(2 ** 256), 2 ** 256], 2, 2);
    let state = 12345;
    const random = () => { state = (state * 1103515245 + 12345) % 2147483648; return state / 2147483648; };
    for (let trial = 0; trial < 150; ++trial) {
        const rows = 2 + Math.floor(random() * 3), cols = rows + Math.floor(random() * 3);
        // Full 52-bit mantissas, exponents in [low, low + 31]: spread below 2^32.
        const low = Math.floor(random() * 481) - 256;
        const cost = Array.from({length: rows * cols}, () => {
            if (random() < 0.1) return 0;
            const exponent = low + Math.floor(random() * 32);
            const fraction = (Math.floor(random() * 2 ** 26) * 2 ** 26 + Math.floor(random() * 2 ** 26)) / 2 ** 52;
            return (random() < 0.5 ? -1 : 1) * (1 + fraction) * 2 ** exponent;
        });
        optimal(cost, rows, cols, trial % 2 === 1);
    }
    // Review 2026-10-13 P0 (memory): a 1 x 32,000,000 zero matrix needed more
    // than 2 GiB in WASM32; it is refused before any allocation.
    {
        const m = getModule(), ctx = Context.create();
        const backing = m._malloc(32000000 * 8);
        try {
            assert.notEqual(backing, 0);
            m.HEAPF64.fill(0, backing / 8, backing / 8 + 32000000);
            const view = m._malloc(64), rows = m._malloc(8), cols = m._malloc(8), count = m._malloc(8);
            try {
                assert.equal(m.ccall("n4m_matrix_view_init_rowmajor", "number",
                    ["number", "number", "i64", "i64", "number"], [view, backing, 1n, 32000000n, 1]), 0);
                assert.equal(m.ccall("n4m_linear_sum_assignment", "number",
                    ["number", "number", "number", "number", "number", "number"],
                    [ctx.handle, view, 0, rows, cols, count]), 10);
            } finally {
                for (const p of [view, rows, cols, count]) m._free(p);
            }
        } finally {
            m._free(backing);
            ctx.destroy();
        }
    }
    const cov = [2.0, 0.3, 0.1, 0.3, 1.0, -0.2, 0.1, -0.2, 0.5];
    const unitDraw = randomMultivariateNormal(new Float64Array(3), {data: new Float64Array(cov), rows: 3, cols: 3}, 6, 42n);
    for (const k of [-126, 127]) {
        const got = randomMultivariateNormal(new Float64Array(3),
            {data: new Float64Array(cov.map((v) => v * 4 ** k)), rows: 3, cols: 3}, 6, 42n);
        got.data.forEach((v, i) => assert.equal(v, unitDraw.data[i] * 2 ** k));
    }
    for (const k of [-127, 128]) {
        assert.equal(status(() => randomMultivariateNormal(new Float64Array(3),
            {data: new Float64Array(cov.map((v) => v * 4 ** k)), rows: 3, cols: 3}, 6, 42n)), 10);
    }
    const identity = {data: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), rows: 3, cols: 3};
    assert.equal(status(() => randomMultivariateNormal(new Float64Array([1, 2 ** 32, 0]), identity, 3, 1n)), 0);
    assert.equal(status(() => randomMultivariateNormal(new Float64Array([1, 2 ** 32 * (1 + 2 ** -52), 0]), identity, 3, 1n)), 10);
}
// Same assignment, Jaccard, normal-draw and Gaussian HDBSCAN gate as the
// native n4m_cluster_matching_null_tests (NumPy 2.4.6, SciPy 1.18.1).
const samePartition = (got, want, name) => {
    const forward = new Map(), backward = new Map();
    got.forEach((label, i) => {
        assert.equal(label < 0, want[i] < 0, name);
        if (label < 0) return;
        assert.equal(forward.get(label) ?? want[i], want[i], name);
        assert.equal(backward.get(want[i]) ?? label, label, name);
        forward.set(label, want[i]); backward.set(want[i], label);
    });
};
const nullFixture = JSON.parse(readFileSync(
    new URL("../../../parity/fixtures/cluster_matching_null_v1.json", import.meta.url), "utf8"));
for (const c of nullFixture.cases) {
    if (c.kind === "assignment") {
        const cost = {data: new Float64Array(c.cost_hex.map(hex)), rows: c.rows, cols: c.cols};
        if (c.error) {
            assert.throws(() => linearSumAssignment(cost, c.maximize === 1), c.name);
            continue;
        }
        const got = linearSumAssignment(cost, c.maximize === 1);
        assert.deepEqual(got.rowInd, c.row_ind, c.name);
        assert.deepEqual(got.colInd, c.col_ind, c.name);
    } else if (c.kind === "jaccard") {
        const got = clusterJaccardMatrix(c.labels_a, c.labels_b, c.n_clusters_a, c.n_clusters_b);
        assert.deepEqual(got.intersections, c.intersections, c.name);
        assert.deepEqual(got.sizesA, c.sizes_a, c.name);
        assert.deepEqual(got.sizesB, c.sizes_b, c.name);
        c.jaccard_hex.map(hex).forEach((v, i) =>
            assert.ok(Number.isNaN(v) ? Number.isNaN(got.jaccard.data[i]) : got.jaccard.data[i] === v, c.name));
    } else if (c.kind === "normal") {
        const got = randomMultivariateNormal(new Float64Array(c.mean_hex.map(hex)),
            {data: new Float64Array(c.cov_hex.map(hex)), rows: c.d, cols: c.d}, c.n, BigInt(c.seed));
        c.draws_hex.map(hex).forEach((v, i) => relativeNear(got.data[i], v));
    } else if (c.kind === "multiscale_hdbscan") {
        const got = clusterHdbscan({data: new Float64Array(c.input_hex.map(hex)), rows: c.rows, cols: c.cols},
            {minClusterSize: c.min_cluster_size, minSamples: c.min_samples});
        samePartition(got.labels, c.labels, c.name);
        c.membership_hex.map(hex).forEach((v, i) => relativeNear(got.membership[i], v));
    } else if (c.kind === "ari") {
        const a = Array.from({length: c.n}, (_, i) => i % c.ka);
        const b = Array.from({length: c.n}, (_, i) => (Math.floor(i / c.ka) + c.shift * (i % 3)) % c.kb);
        const want = hex(c.ari_hex[0]);
        assert.ok(Math.abs(preflightAdjustedRandIndex(a, b) - want) <= 1e-14 * Math.abs(want), c.name);
    } else if (c.kind === "permutations") {
        assert.deepEqual(randomPermutations(BigInt(c.seed), c.n, c.n_permutations), c.draws, c.name);
    } else {
        const x = randomMultivariateNormal(new Float64Array(c.d),
            {data: new Float64Array(c.cov_hex.map(hex)), rows: c.d, cols: c.d}, c.n, BigInt(c.seed));
        const got = clusterHdbscan(x, {minClusterSize: c.min_cluster_size, minSamples: c.min_samples});
        assert.equal(got.nClusters, c.n_clusters, c.name);
        samePartition(got.labels, c.labels, c.name);
    }
}

// One-way REML and intervals: same independent references as
// cpp/tests/test_variance_components.cpp (dense REML score equations with
// NumPy 2.5.4 / SciPy 1.18.1; SciPy chi2/F quantiles; NumPy-drawn bootstrap).
const vcNear = (actual, expected, tol) => assert.ok(
    Math.abs(actual - expected) <= tol * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
const ga = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4];
const ya = { data: new Float64Array([1.0, 1.4, 2.2, 2.0, 2.9, 0.3, 0.9, 0.4, 0.8, 1.7, 3.1, 2.5]), rows: 12, cols: 1 };
const vc = statsVarianceComponentsReml(ya, ga);
assert.equal(vc.status[0], 0);
vcNear(vc.sigma2Between[0], 0.7446105326226952, 1e-10);
vcNear(vc.sigma2Within[0], 0.13630617544724544, 1e-10);
vcNear(vc.icc[0], 0.8452678054592837, 1e-10);
vcNear(vc.logLikelihood[0], -10.963218720046495, 1e-12);
assert.deepEqual([vc.nGroups[0], vc.nObservations[0]], [5, 12]);
const vci = statsVarianceComponentsIntervals(ya, ga, {
    confidence: 0.95, nBootstrap: 40, seed: 12345n,
    clusterIds: [7, 7, 7, 7, 7, 3, 3, 3, 3, 9, 5, 5],
});
vcNear(vci.sdChiSquare[0], 0.24570003389152809, 1e-10);
vcNear(vci.sdChiSquare[1], 0.7563308737449301, 1e-10);
assert.ok(Number.isNaN(vci.iccExactF[0]));
assert.equal(vci.nValidReplicates[0], 40);
vcNear(vci.sdBootstrap[0], 0.3094525017261184, 1e-10);
vcNear(vci.sdBootstrap[1], 0.42426406871192857, 1e-10);
assert.equal(vci.iccBootstrap[0], 0);
vcNear(vci.iccBootstrap[1], 0.9233717840878614, 1e-8);
const balanced = statsVarianceComponentsIntervals({
    data: new Float64Array([1.0, 1.2, 0.9, 2.0, 2.3, 2.1, 0.2, 0.1, 0.5, 1.4, 1.6, 1.3]), rows: 12, cols: 1,
}, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], { confidence: 0.95, nBootstrap: 0, seed: 0 });
vcNear(balanced.iccExactF[0], 0.7838941809128407, 1e-10);
vcNear(balanced.iccExactF[1], 0.9968006295172774, 1e-10);
assert.equal(balanced.nValidReplicates[0], 0);
// Searle bounds with 3 groups use F(2, nu) quantiles: closed form
// (nu/2) expm1(-2 log(p)/nu) for the upper tail and reciprocity for
// F(nu, 2). Data 10 i +- 1: MSB = 100 r, MSW = r/(r-1), F = 100 (r-1).
for (const r of [2, 10, 1000]) {
    const data = [], groups = [];
    for (let i = 0; i < 3; i++) for (let j = 0; j < r; j++) { groups.push(i); data.push(10 * i + (j % 2 === 0 ? 1 : -1)); }
    const nu = 3 * (r - 1), F = 100 * (r - 1), p = 0.025;
    const fl = F / (0.5 * nu * Math.expm1(-2 * Math.log(p) / nu));
    const fu = F / (0.5 * nu * Math.expm1(-2 * Math.log1p(-p) / nu));
    const ci = statsVarianceComponentsIntervals({ data: new Float64Array(data), rows: data.length, cols: 1 }, groups, { confidence: 0.95, nBootstrap: 0, seed: 0 });
    vcNear(ci.iccExactF[0], (fl - 1) / (fl + r - 1), 1e-10);
    vcNear(ci.iccExactF[1], (fu - 1) / (fu + r - 1), 1e-10);
}
// Rules 13/15 domain: nonzero |x| in [2^-256, 2^256] and spread <= 2^32 (raw
// values and the differences the statistic uses). Inside: exact 2^k
// identities; outside: N4mError with status 10 (UNSUPPORTED), never a value.
const twoGroups = [0, 0, 1, 1];
const off = 2 ** 255, u = 2 ** 210;
const top = statsVarianceComponentsReml({ data: new Float64Array([off - 3 * u, off - u, off + u, off + 3 * u]), rows: 4, cols: 1 }, twoGroups);
assert.equal(top.status[0], 0);
vcNear(top.sigma2Between[0], 7 * 2 ** 420, 1e-12);
vcNear(top.sigma2Within[0], 2 * 2 ** 420, 1e-12);
vcNear(top.icc[0], 7 / 9, 1e-12);
const refused = (data, groups) => {
    for (const call of [
        () => statsVarianceComponentsReml({ data: new Float64Array(data), rows: data.length, cols: 1 }, groups),
        () => statsVarianceComponentsIntervals({ data: new Float64Array(data), rows: data.length, cols: 1 }, groups, { confidence: 0.95, nBootstrap: 0, seed: 0 }),
    ]) {
        assert.throws(call, e => e.status === 10 && /outside the supported numerical domain/.test(e.message));
    }
};
const accepted = (data, groups) => {
    assert.equal(statsVarianceComponentsReml({ data: new Float64Array(data), rows: data.length, cols: 1 }, groups).status.length, 1);
};
const g3 = [0, 1, 1], g001 = [0, 0, 1];
accepted([2 ** 256, 2 ** 255, 2 ** 254], g3);
refused([2 ** 256 * (1 + 2 ** -52), 2 ** 255, 2 ** 254], g3);
accepted([2 ** -256, 2 ** -255, 2 ** -254], g3);
refused([2 ** -257, 2 ** -255, 2 ** -254], g3);
accepted([1, 2 ** 32, 2], g3);
refused([1, 2 ** 33, 2], g3);
accepted([1, 1 + 2 ** -20, 2 ** 10], g001);
refused([1, 1 + 2 ** -22, 2 ** 12], g001);
refused([-1e300, -9e299, 9e299, 1e300], twoGroups);
// Group means: exact dyadic values (same case as test_group_means.cpp).
const gm = statsGroupMeans({data: new Float64Array([1, 2, 3, 6, 10, -4, 0, 0, 1, 1, 2, 8]),
    rows: 6, cols: 2}, [1, 1, 0, 2, 2, 2], 3);
assert.deepEqual(Array.from(gm.means.data), [10, -4, 2, 4, 1, 3]);
assert.deepEqual(gm.counts, [1, 2, 3]);
// 2^52 offset: mean(2^52 + {1, 2, 4}) = 2^52 + 7/3 rounds to 2^52 + 2.
const offset = statsGroupMeans({data: new Float64Array([2 ** 52 + 1, 2 ** 52 + 2, 2 ** 52 + 4]),
    rows: 3, cols: 1}, [0, 0, 0], 1);
assert.equal(offset.means.data[0], 2 ** 52 + 2);
// Declared domain: inside at its boundary, refused outside with ERR_UNSUPPORTED.
assert.equal(statsGroupMeans({data: new Float64Array([1, 2 ** 100]), rows: 2, cols: 1}, [0, 0], 1).counts[0], 2);
const domainError = (fn) => assert.throws(fn, (e) => e instanceof N4mError && e.status === Status.ERR_UNSUPPORTED);
domainError(() => statsGroupMeans({data: new Float64Array([2 ** 300]), rows: 1, cols: 1}, [0], 1));
domainError(() => statsGroupMeans({data: new Float64Array([Number.MAX_VALUE]), rows: 1, cols: 1}, [0], 1));
assert.throws(() => statsGroupMeans({data: new Float64Array([1, 2]), rows: 2, cols: 1}, [0, 2], 3));
assert.throws(() => statsGroupMeans({data: new Float64Array([1, 2]), rows: 2, cols: 1}, [0, 0], 2), N4mError);
// Regression scores: y = (1, 2, 3, 4), prediction (1, 3, 3, 2): SSE = 5, SAE = 3, SST = 5.
for (const k of [-200, 0, 200]) {
    const s = 2 ** k;
    const rs = statsRegressionScores(new Float64Array([1, 2, 3, 4].map((v) => v * s)),
        new Float64Array([1, 3, 3, 2].map((v) => v * s)));
    assert.equal(rs.rmse, Math.sqrt(5 / 4) * s);
    assert.equal(rs.mae, 0.75 * s);
    assert.equal(rs.r2, 0);
}
const shiftedScores = statsRegressionScores(new Float64Array([1, 2, 3, 4].map((v) => v + 2 ** 52)),
    new Float64Array([1, 3, 3, 2].map((v) => v + 2 ** 52)));
assert.equal(shiftedScores.r2, 0);
assert.ok(Number.isNaN(statsRegressionScores(new Float64Array([2, 2]), new Float64Array([1, 3])).r2));
domainError(() => statsRegressionScores(new Float64Array([1, 2 ** 300]), new Float64Array([1, 1])));
// Spread limit 2^32 (rule 15), both sides.
assert.equal(statsRegressionScores(new Float64Array([1, 2 ** 32]), new Float64Array([1, 1])).mae, (2 ** 32 - 1) / 2);
domainError(() => statsRegressionScores(new Float64Array([1, 2 ** 33]), new Float64Array([1, 1])));
// Every former automatic solver class of the PCA on analytic spectra:
// X = [D 0; -D 0] (2q x p) has centred variances 2 d_i^2 / (2q - 1) and
// zeros (mirrors pca_solver_class_tests in cpp/tests/test_preflight.cpp).
// q = 128, d_i = 1 + i/128, k = 10 (review b8fix3) was 7 % low through the
// randomized sketch.
for (const [q, p, k, zeroTail] of [[128, 128, 10, 0], [32, 32, 10, 0], [100, 300, 10, 0],
                                   [150, 400, 10, 0], [24, 24, 24, 4], [10, 40, 20, 0]]) {
    const n = 2 * q;
    const d = Array.from({length: q}, (_, i) => (i < q - zeroTail ? 1 + i / q : 0));
    const x = new Float64Array(n * p);
    d.forEach((v, i) => { x[i * p + i] = v; x[(q + i) * p + i] = -v; });
    const ref = new Array(Math.min(n, p)).fill(0);
    d.forEach((v, i) => { ref[i] = 2 * v * v / (n - 1); });
    ref.sort((a, b) => b - a);
    const r = preflightPca({data: x, rows: n, cols: p}, k);
    assert.equal(r.resolution, 16 * Math.max(n, p) * Number.EPSILON * r.eigenvalues[0]);
    for (let a = 0; a < k; ++a) {
        if (ref[a] > 0) {
            assert.equal(r.unresolved[a], 0, `${n}x${p} component ${a}`);
            assert.ok(Math.abs(r.eigenvalues[a] - ref[a]) <= r.resolution, `${n}x${p} component ${a}`);
        } else {
            assert.equal(r.unresolved[a], 1);
            assert.equal(r.eigenvalues[a], 0);
            for (let i = 0; i < n; ++i) assert.equal(r.scores.data[i * k + a], 0);
        }
    }
    for (let a = 0; a < k; ++a) for (let b = 0; b <= a; ++b) {
        let dot = 0;
        for (let j = 0; j < p; ++j) dot += r.loadings.data[a * p + j] * r.loadings.data[b * p + j];
        assert.ok(Math.abs(dot - (a === b ? 1 : 0)) <= 1e-12);
    }
}
// k is validated before any allocation (k = INT_MAX used to abort).
assert.throws(() => preflightPca({data: new Float64Array([1, 2]), rows: 2, cols: 1}, 2147483647));

// Hotelling T² keeps only resolved directions (review b8fix3): column 3 =
// column 1 + column 2, rank 2; exact rational references, UCL with r = 2.
{
    const out = new HotellingT2({n_components: 3, alpha: 0.05})
        .run({data: new Float64Array([1, 2, 3, 4, 5, 9, 6, 3, 9, 2, 8, 10]), rows: 4, cols: 3});
    assert.equal(out.n_components_used, 2);
    [5439, 417, 4113, 4599].forEach((num, i) =>
        assert.ok(Math.abs(out.t2.data[i] - num / 2428) <= 1e-12));
    assert.ok(Math.abs(out.ucl - 71.25) <= 1e-9);
}

// Review b8fix4 P0-1 / P1 (mirrors pca_completion_tests in
// cpp/tests/test_preflight.cpp): the unresolved loading rows of a wide input
// are a Householder completion, unit and orthogonal to the resolved rows.
// X = I_8 (k = 8) used to publish a last loading of norm 0.354, and X = 0
// (320 x 320, k = 320) took about 10 s through canonical-vector retries.
for (const [n, kind] of [[4, "identity"], [8, "identity"], [320, "identity"], [320, "zero"],
                         [320, "half"]]) {
    const x = new Float64Array(n * n);
    let rank = 0;
    if (kind === "identity") { for (let i = 0; i < n; ++i) x[i * n + i] = 1; rank = n - 1; }
    if (kind === "half") {
        for (let i = 0; i < n / 2; ++i) { x[i * n + i] = 1; x[(n / 2 + i) * n + i] = -1; }
        rank = n / 2;
    }
    const started = performance.now();
    const r = preflightPca({data: x, rows: n, cols: n}, n);
    const elapsed = performance.now() - started;
    assert.ok(elapsed < 5000, `${kind} ${n}: ${elapsed} ms`);
    const v = kind === "identity" ? 1 / (n - 1) : 2 / (n - 1);
    for (let a = 0; a < n; ++a) {
        if (a < rank) {
            assert.equal(r.unresolved[a], 0, `${kind} ${n} component ${a}`);
            relativeNearTol(r.eigenvalues[a], v);
        } else {
            assert.equal(r.unresolved[a], 1, `${kind} ${n} component ${a}`);
            assert.equal(r.eigenvalues[a], 0);
        }
    }
    for (let a = 0; a < n; ++a) for (let b = 0; b <= a; ++b) {
        let dot = 0;
        for (let j = 0; j < n; ++j) dot += r.loadings.data[a * n + j] * r.loadings.data[b * n + j];
        assert.ok(Math.abs(dot - (a === b ? 1 : 0)) <= 1e-12, `${kind} ${n} rows ${a},${b}`);
    }
}
// Graded wide input (12 x 16, column j scaled by 10^(-d j / 11)): the rows
// Xc^T u / sigma of the weakest resolved components drifted up to 5e-5 from
// orthonormality before their re-orthonormalisation.
for (const d of [3, 5, 6, 7]) {
    const n = 12, p = 16;
    const x = new Array(n * p).fill(0);
    for (let i = 0; i < n; ++i) for (let j = 0; j < n; ++j) {
        x[i * p + j] = (1.5 + Math.sin(1 + 0.7 * i * j + 0.3 * i * i + 1.3 * j)) * 10 ** (-d * j / 11);
    }
    const r = pcaOf(x, n, p, n);
    for (let a = 0; a < n; ++a) for (let b = 0; b <= a; ++b) {
        let dot = 0;
        for (let j = 0; j < p; ++j) dot += r.loadings.data[a * p + j] * r.loadings.data[b * p + j];
        assert.ok(Math.abs(dot - (a === b ? 1 : 0)) <= 1e-12, `graded ${d}: rows ${a},${b}`);
    }
}
// The budget is a guaranteed bound over every pass and live buffer, with the
// QL iteration at its ceiling (mirrors pca_budget_tests): 700 x 700 with
// k = 700 (3.2e9 long-double product equivalents) is refused before
// allocating.
assert.equal(statusOf(new Array(700 * 700).fill(1), 700, 700, 700), UNSUPPORTED);
// Non-dyadic constant columns centre to exact zeros at every scale; a
// varying column i * s next to them keeps its variance s^2 n (n+1) / 12.
for (const s of [1, 3, 1e-7, 1e7, 0.1]) {
    for (const n of [3, 10, 101]) {
        const x = [];
        for (let i = 0; i < n; ++i) x.push(0.1 * s, s / 3, -2.7e-3 * s, 12345.678 * s);
        const r = pcaOf(x, n, 4, 3);
        assert.deepEqual(Array.from(r.eigenvalues), [0, 0, 0]);
        assert.deepEqual(Array.from(r.fullMoments), [0, 0]);
        assert.ok(r.scores.data.every((v) => v === 0));
        for (let i = 0; i < n; ++i) x[4 * i + 2] = i * s;
        const m = pcaOf(x, n, 4, 3);
        relativeNearTol(m.eigenvalues[0], s * s * n * (n + 1) / 12);
        assert.equal(m.eigenvalues[1], 0);
        assert.equal(m.eigenvalues[2], 0);
        assert.ok(Math.abs(m.loadings.data[2] - 1) <= 1e-15);
        for (const j of [0, 1, 3]) assert.equal(m.loadings.data[j], 0);
    }
}

// Review b8fix4 P0-2 (mirrors test_hotelling_t2_constant_columns): a
// non-dyadic constant column centres to exact zeros (ten values 0.1, k = 1
// used to give rank 1, T² = 0.9 and UCL 5.63).
for (const n of [3, 10, 101, 1000]) {
    for (const c of [0.1, 1 / 3, -2.7e-7, 12345.678, 314159.26535]) {
        for (const s of [1, 3, 1e-7, 1e7, 2 ** 60]) {
            const out = new HotellingT2({n_components: 1, alpha: 0.05})
                .run({data: new Float64Array(n).fill(c * s), rows: n, cols: 1});
            assert.equal(out.n_components_used, 0, `${n} x ${c * s}`);
            assert.equal(out.ucl, 0);
            assert.ok(out.t2.data.every((v) => v === 0));
        }
    }
}
for (const n of [10, 101]) {
    for (const s of [1, 3, 1e-7, 1e7, 2 ** 60]) {
        const data = new Float64Array(3 * n);
        for (let i = 0; i < n; ++i) data.set([0.1 * s, i * s, -s / 3], 3 * i);
        const out = new HotellingT2({n_components: 2, alpha: 0.05}).run({data, rows: n, cols: 3});
        assert.equal(out.n_components_used, 1);
        for (let i = 0; i < n; ++i) {
            const z = i - (n - 1) / 2;
            const ref = 12 * z * z / (n * (n + 1));
            assert.ok(Math.abs(out.t2.data[i] - ref) <= 1e-12 * Math.max(1, ref), `${n} ${s} ${i}`);
        }
    }
}
// Review b8fix5 (mirrors test_q_residuals_scaling and
// test_q_residuals_constant_columns in cpp/tests/test_metrics.cpp). The
// orthogonal centred columns of [[2,1],[2,-1],[-2,1],[-2,-1]] with k = 1 leave
// the second one as residual: Q_i = 4^s, one residual eigenvalue
// 4^(s+1) / 3, Jackson-Mudholkar UCL lambda (7/9 + ca sqrt(2) / 3)^3 (ca from
// SciPy norm.isf). The moments used to overflow at 2^200 and underflow at
// 2^-200, publishing UCL 0 with status OK.
const NUMERICAL_FAILURE = 7;
const qOf = (data, rows, cols, k, alpha = 0.05) =>
    new QResiduals({n_components: k, alpha}).run({data: new Float64Array(data), rows, cols});
const qStatus = (data, rows, cols, k, alpha = 0.05) => {
    try { qOf(data, rows, cols, k, alpha); return 0; } catch (e) { return e.status; }
};
const jmSingle = (lambda, ca) => lambda * (7 / 9 + ca * Math.SQRT2 / 3) ** 3;
const relClose = (got, want, tol) => assert.ok(Math.abs(got - want) <= tol * Math.abs(want),
    `${got} != ${want}`);
const qBase = [2, 1, 2, -1, -2, 1, -2, -1];
const q0 = qOf(qBase, 4, 2, 1);
assert.deepEqual(Array.from(q0.q.data), [1, 1, 1, 1]);
relClose(q0.ucl, jmSingle(4 / 3, 1.6448536269514729), 1e-8);
for (const s of [-511, -400, -256, -200, -100, 1, 100, 200, 256, 400, 510]) {
    const f = pow2(s);
    const r = qOf(qBase.map((v) => v * f), 4, 2, 1);
    r.q.data.forEach((v) => assert.equal(v, pow2(2 * s)));
    assert.equal(r.ucl, q0.ucl * pow2(2 * s), `scale 2^${s}`);
}
relClose(qOf(qBase.map((v) => v * pow2(200)), 4, 2, 1).ucl, 1.29001e121, 1e-5);
for (const s of [511, 600, -512, -600]) {
    assert.equal(qStatus(qBase.map((v) => v * pow2(s)), 4, 2, 1), NUMERICAL_FAILURE, `2^${s}`);
}
for (const [alpha, ca, tol] of [[0.5, 0, 1e-15], [0.9, -1.2815515655446004, 5e-8],
                                [1e-20, 9.262340089798409, 1e-8]]) {
    relClose(qOf(qBase, 4, 2, 1, alpha).ucl, jmSingle(4 / 3, ca), tol);
}
assert.equal(qStatus(qBase, 4, 2, 1, 0.99), NUMERICAL_FAILURE);
// Constant columns centre to exact zeros; an unresolved residual subspace
// gives Q = 0 and UCL = 0 exactly (ten rows of 0.1 used to give Q = 1.9e-65).
for (const n of [3, 10, 101]) {
    for (const p of [1, 2]) {
        for (const c of [0.1, 1 / 3, -2.7e-7, 12345.678]) {
            for (const s of [1, 3, 1e-7, 1e7, 2 ** 60, pow2(-200), pow2(200)]) {
                const r = qOf(new Array(n * p).fill(c * s), n, p, 1);
                assert.equal(r.ucl, 0, `${n} x ${p} of ${c * s}`);
                assert.ok(r.q.data.every((v) => v === 0));
            }
        }
    }
}
{
    const a = [3, 1, -1, -3], b = [1, -1, -1, 1];
    for (const s of [1, 3, 1e-7, 1e7, 2 ** 60, pow2(-200), pow2(200)]) {
        for (const offset of [0, 0.1, 12345.678]) {
            const x = [];
            for (let i = 0; i < 4; ++i) x.push(0.1 * s, (a[i] + offset) * s, -s / 3, (b[i] + offset) * s);
            const r = qOf(x, 4, 4, 1);
            const tol = offset === 0 ? 1e-15 : 1e-9;
            r.q.data.forEach((v) => relClose(v, s * s, tol));
            relClose(r.ucl, jmSingle(4 * s * s / 3, 1.6448536269514729), 1e-8);
            const full = qOf(x, 4, 4, 2);
            assert.equal(full.ucl, 0);
            assert.ok(full.q.data.every((v) => v === 0));
        }
    }
}
// Hotelling T² is exactly scale-invariant (no sigma^2: NaN beyond 2^512
// before) and its F quantile has no fixed bracket: n = 3, r = 1 gives
// UCL = 4/3 * 2 (1-alpha)^2 / (alpha (2-alpha)) in closed form (the former
// [1e-10, 1e12] bracket returned about 1.1e12 for alpha = 1e-14).
{
    const hBase = [1, 0.5, 2, -1, 4, 3];
    const h0 = new HotellingT2({n_components: 1, alpha: 0.05}).run({data: new Float64Array(hBase), rows: 3, cols: 2});
    for (const s of [-1000, -600, -200, 200, 600, 1000]) {
        const h = new HotellingT2({n_components: 1, alpha: 0.05})
            .run({data: new Float64Array(hBase.map((v) => v * pow2(s))), rows: 3, cols: 2});
        assert.deepEqual(Array.from(h.t2.data), Array.from(h0.t2.data), `2^${s}`);
        assert.equal(h.ucl, h0.ucl);
    }
    for (const alpha of [0.05, 1e-6, 1e-14, 1e-300]) {
        const h = new HotellingT2({n_components: 1, alpha}).run({data: new Float64Array([1, 2, 3]), rows: 3, cols: 1});
        relClose(h.ucl, 4 / 3 * 2 * (1 - alpha) ** 2 / (alpha * (2 - alpha)), 1e-8);
        relClose(h.t2.data[0], 1, 1e-15);
        assert.equal(h.t2.data[1], 0);
        relClose(h.t2.data[2], 1, 1e-15);
    }
}
// Review b8fix6 P0-1 (mirrors test_q_residuals_negative_h0): for h0 < 0 the
// upper quantile of Q maps to the lower quantile of (Q / theta_1)^h0. Columns
// 1..14 of the Sylvester H_16 times [2, 1, 1/4 x 12], k = 1 (h0 = -0.0676)
// gave UCL 0.365 and flagged every Q_i = 1.75. mpmath references of the
// signed-h0 Jackson-Mudholkar limit on the exact residual eigenvalues.
{
    const hadamard = (n) => {
        let h = [[1]];
        while (h.length < n) h = [...h.map((r) => [...r, ...r]), ...h.map((r) => [...r, ...r.map((v) => -v)])];
        return h;
    };
    const scaledH = (n, scales) => {
        const h = hadamard(n), x = [];
        for (let i = 0; i < n; ++i) scales.forEach((c, j) => x.push(h[i][j + 1] * c));
        return x;
    };
    const astra = [2, 1, ...new Array(12).fill(0.25)];
    for (const [alpha, ref] of [[0.05, 5.2294583820033547], [0.01, 9.6906288186272182],
                                [0.9, 0.47906393402987125]]) {
        const r = qOf(scaledH(16, astra), 16, 14, 1, alpha);
        relClose(r.ucl, ref, 5e-9);
        r.q.data.forEach((v) => relClose(v, 1.75, 1e-14));
    }
    let previous = 0;
    for (const [c, ref] of [[4.5, 6.3099963890565998], [4.53125, 6.3814368019494924],
                            [4.5390625, 6.3993694611641446], [5, 7.5091436613414903]]) {
        const ucl = qOf(scaledH(16, [2, c / 4, ...new Array(12).fill(0.25)]), 16, 14, 1).ucl;
        relClose(ucl, ref, 5e-9);
        assert.ok(ucl > previous);
        previous = ucl;
    }
    const wide = scaledH(32, [8, 3.25, ...new Array(22).fill(1)]);
    assert.equal(qStatus(wide, 32, 24, 1, 1e-6), NUMERICAL_FAILURE);
    relClose(qOf(wide, 32, 24, 1, 0.05).ucl, 71.914193025598727, 5e-9);
    relClose(qOf(wide, 32, 24, 1, 0.1).ucl, 56.312036502701673, 5e-9);
}
// Review b8fix6 P0-2 / P1 (mirrors test_hotelling_t2_quantile_tails): the
// small tail is inverted, so alpha = 1 - 2^-30 gives the closed-form F_{1,2}
// UCL 2.312964635e-18 (2.96e-16 before) and flags the central T²; alpha =
// 1e-308 gives 1.333e308 (refused before); a UCL above DBL_MAX is refused.
{
    const hOf = (data, rows, cols, k, alpha) =>
        new HotellingT2({n_components: k, alpha}).run({data: new Float64Array(data), rows, cols});
    const hStatus = (data, rows, cols, k, alpha) => {
        try { hOf(data, rows, cols, k, alpha); return 0; } catch (e) { return e.status; }
    };
    const tiny = [0.05, 0.3, 1e-6, 1e-100, 1e-300, 1e-308];
    const oneMinus = [0.5, 0.1, 2 ** -30, 2 ** -40, 2 ** -53];
    for (const alpha of tiny) {
        const b = 1 - alpha;
        relClose(hOf([1, 2, 3], 3, 1, 1, alpha).ucl, 4 / 3 * (2 * b * b / (alpha * (1 + b))), 1e-12);
    }
    for (const beta of oneMinus) {
        const h = hOf([-1, 2 ** -28, 1], 3, 1, 1, 1 - beta);
        relClose(h.ucl, 4 / 3 * (2 * beta * beta / ((1 - beta) * (1 + beta))), 1e-12);
        if (beta === 2 ** -30) {
            relClose(h.ucl, 2.312964634635743e-18, 1e-12);
            relClose(h.t2.data[1], 6.167905692361981e-18, 1e-9);
            assert.ok(h.t2.data[1] > h.ucl);
        }
    }
    for (const alpha of [7e-309, 5e-324]) assert.equal(hStatus([1, 2, 3], 3, 1, 1, alpha), NUMERICAL_FAILURE);
    const x24 = [1, 1, 1, -1, 1, 0, -1, 1, -1, -1, -1, 0];
    const f24 = (logAlpha) => 2 * Math.expm1(-0.5 * logAlpha);
    for (const alpha of tiny) relClose(hOf(x24, 6, 2, 2, alpha).ucl, 35 / 12 * f24(Math.log(alpha)), 1e-12);
    for (const beta of oneMinus) relClose(hOf(x24, 6, 2, 2, 1 - beta).ucl, 35 / 12 * f24(Math.log1p(-beta)), 1e-12);
}
console.log("JS/WASM Preflight diagnostics passed");
