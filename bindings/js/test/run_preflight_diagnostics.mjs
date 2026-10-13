// SPDX-License-Identifier: CECILL-2.1
// Verify the public WASM diagnostic wrappers against hand-computed results.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
    loadModule, abiVersion, preflightPca, preflightRepeatability,
    preflightFactorAssociation, preflightHolmAdjust,
    preflightNoiseProxy, preflightPairsWithinTolerance,
    preflightSummary, preflightNearestNeighbors, preflightLocalContrast,
    clusterHdbscan, preflightAdjustedRandIndex, ppCreate, ppTransform, ppDestroy, preflightSnv,
} from "../dist/index.js";

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9,
    `${actual} != ${expected}`);
const relativeNear = (actual, expected) => assert.ok(
    Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)),
    `${actual} != ${expected}`);
await loadModule();
assert.deepEqual(abiVersion(), [2, 20, 0]);
const pca = preflightPca({
    data: new Float64Array([2, 1, 2, -1, -2, 1, -2, -1]), rows: 4, cols: 2,
}, 1);
near(pca.eigenvalues[0], 16 / 3);
near(pca.fullMoments[0], 20 / 3);
near(pca.fullMoments[1], 272 / 9);
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
const extremeSummary = preflightSummary(new Float64Array([-1e308, 0, 1e308]));
near(extremeSummary[6], 0);
relativeNear(extremeSummary[4], 1e308);
relativeNear(extremeSummary[5], -5e307);
relativeNear(extremeSummary[7], 5e307);
assert.throws(() => preflightSummary(new Float64Array([Number.MAX_VALUE, -Number.MAX_VALUE])),
    (error) => error.status === 7);
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
    {data: new Float64Array([1e308, 1e308]), rows: 1, cols: 2},
    {data: new Float64Array([0, 0]), rows: 1, cols: 2}, 1);
relativeNear(extremeNeighbors.distances[0], Math.SQRT2 * 1e308);
assert.throws(() => preflightNearestNeighbors(
    {data: new Float64Array([Number.MAX_VALUE]), rows: 1, cols: 1},
    {data: new Float64Array([-Number.MAX_VALUE]), rows: 1, cols: 1}, 1),
    (error) => error.status === 7);
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
const numericalFailure = (fn) => assert.throws(fn, (e) => e.status === 7);  // N4M_ERR_NUMERICAL_FAILURE
// Review b1 #3: q - r overflows, yet the RMS distance sqrt(2) * 1e308 <= 1.5e308.
const huge = pair([1e308, 0], [-1e308, 0], 1.5e308);
assert.equal(huge.totalPairs, 1);
rel(huge.distances[0], Math.SQRT2 * 1e308, 4e-16);
assert.equal(pair([1e308, 0], [-1e308, 0], 1.414e308).totalPairs, 0);
// Review b1fix #2 / b14fix #8: tolerance-invariant tiny distance, both sides of the threshold.
for (const tol of [7.0711e-171, 1e-100, 1, 1e200, 1e308]) {
    const tiny = pair([1e-170, 0], [0, 0], tol);
    assert.equal(tiny.totalPairs, 1);
    rel(tiny.distances[0], 1e-170 / Math.SQRT2, 4e-16, `tol ${tol}`);
}
assert.equal(pair([1e-170, 0], [0, 0], 7.0710e-171).totalPairs, 0);
// Review b14fix #2: [2a,0] vs [0,0] has RMS sqrt(2) a > a at a = 2^-1074.
const tinyA = 2 ** -1074;
assert.equal(pair([2 * tinyA, 0], [0, 0], tinyA).totalPairs, 0);
const subnormal = pair([tinyA, 0], [0, 0], tinyA);
assert.ok(subnormal.totalPairs === 1 && subnormal.distances[0] === tinyA);
// Magnitude sweep (same invariants as cpp/tests/test_preflight_robustness.cpp):
// q - r = [3, 4, 0, 12] has RMS 6.5 at every power-of-two scale.
for (const k of [-1074, -1022, -500, -100, 0, 100, 500, 1000]) {
    const s = 2 ** k, q = [5, 4, 9, 12].map((v) => v * s), r = [2, 0, 9, 0].map((v) => v * s);
    const exact = (6.5 * s) / s === 6.5;
    const result = pair(q, r, (exact ? 6.5 : 7) * s);
    assert.equal(result.totalPairs, 1, `k ${k}`);
    if (6.5 * s >= 2 ** -1022) assert.equal(result.distances[0], 6.5 * s);
    else assert.ok(result.distances[0] > 0 && Math.abs(result.distances[0] - 6.5 * s) <= tinyA);
    assert.equal(pair(q, r, 6 * s).totalPairs, 0, `k ${k}`);
}
// Test-only goldens: exact rational decisions (Python 3.10 fractions) and
// 400-digit decimal square roots, rounded to binary64.
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
    [[4.149515568880993e+180, 2.409919865102884e-181, -2.7997908555096566e-301], [4.149515568880993e+180, 0.0, 9.332636185032189e-302], 1.3913678828425764e-181, false, 1.3913678828425767e-181],
    [[4.149515568880993e+180, 2.409919865102884e-181, -2.7997908555096566e-301], [4.149515568880993e+180, 0.0, 9.332636185032189e-302], 1.3913678828425767e-181, false, 1.3913678828425767e-181],
    [[4.149515568880993e+180, 2.409919865102884e-181, -2.7997908555096566e-301], [4.149515568880993e+180, 0.0, 9.332636185032189e-302], 1.391367882842577e-181, true, 1.3913678828425767e-181],
    [[1e+300, 1.0, -1e-300], [-1e+300, 1.0, 1e-300], 1.1547005383792515e+300, false, 1.1547005383792516e+300],
    [[1e+300, 1.0, -1e-300], [-1e+300, 1.0, 1e-300], 1.1547005383792516e+300, true, 1.1547005383792516e+300],
    [[1e+300, 1.0, -1e-300], [-1e+300, 1.0, 1e-300], 1.1547005383792518e+300, true, 1.1547005383792516e+300],
    [[3.0, -4.0, 8.673617379884035e-19], [-3.0, 4.0, 0.0], 5.773502691896256, false, 5.773502691896257],
    [[3.0, -4.0, 8.673617379884035e-19], [-3.0, 4.0, 0.0], 5.773502691896257, false, 5.773502691896257],
    [[3.0, -4.0, 8.673617379884035e-19], [-3.0, 4.0, 0.0], 5.773502691896258, true, 5.773502691896257],
    [[1e+308, 5e-324], [1e+308, 0.0], 5e-324, true, 5e-324],
    [[1e+308, 5e-324], [1e+308, 0.0], 1e-323, true, 5e-324],
    [[-1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.271161006153646e+308, false, 1.2711610061536462e+308],
    [[-1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.2711610061536462e+308, true, 1.2711610061536462e+308],
    [[-1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.7976931348623157e+308, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.2711610061536464e+308, true, 1.2711610061536462e+308],
    [[1e-323, 0.0], [0.0, 0.0], 5e-324, false, 5e-324],
    [[1e-323, 0.0], [0.0, 0.0], 1e-323, true, 5e-324],
    [[1e+16, 1.0], [9999999999999998.0, 0.0], 1.5811388300841895, false, 1.5811388300841898],
    [[1e+16, 1.0], [9999999999999998.0, 0.0], 1.5811388300841898, true, 1.5811388300841898],
    [[1e+16, 1.0], [9999999999999998.0, 0.0], 1.58113883008419, true, 1.5811388300841898],
];
const snvGoldens = [
    [[1.0715086071862673e+301, 9.332636185032189e-302, -9.332636185032189e-302, 0.0], [1.7320508075688772, -0.5773502691896257, -0.5773502691896257, -0.5773502691896257]],
    [[1e+300, 1.0, 2.0, 3.0], [1.7320508075688772, -0.5773502691896257, -0.5773502691896257, -0.5773502691896257]],
    [[-1e-300, 1e-310, 5e-324, 0.0], [-1.7320508075688772, 0.5773502693435858, 0.5773502691126458, 0.5773502691126458]],
    [[5e-324, 1e-323, 1.5e-323, 0.0], [-0.4472135954999579, 0.4472135954999579, 1.3416407864998738, -1.3416407864998738]],
    [[1e+16, 1.0, -1.0, 1.0000000000000002e+16], [0.9999999999999998, -0.9999999999999998, -1.0000000000000002, 1.0000000000000002]],
    [[1.7976931348623157e+308, -1.7976931348623157e+308, 1e-300, 0.0], [1.4142135623730951, -1.4142135623730951, 0.0, -0.0]],
    [[1.0, 1.0000000000000002, 1.0, 1.0], [-0.5773502691896257, 1.7320508075688772, -0.5773502691896257, -0.5773502691896257]],
];
// Review b14fix2 #1: exact threshold decisions on mixed-exponent rows.
for (const [q, r, tol, within, rms] of pairGoldens) {
    const got = pair(q, r, tol);
    assert.equal(got.totalPairs === 1, within, `${q} ${r} ${tol}`);
    if (!within) continue;
    assert.ok(got.distances[0] > 0 && got.distances[0] <= tol);
    if (rms >= 2 ** -1022) assert.ok(Math.abs(got.distances[0] - rms) <= 2 ** -50 * rms);
    else assert.ok(Math.abs(got.distances[0] - rms) <= tinyA);
}
// SNV of mixed-exponent rows within the reported bound of 400-digit references.
for (const [x, z] of snvGoldens) {
    const out = preflightSnv(row(x));
    assert.ok(out.errorBounds[0] < 1e-14);
    z.forEach((v, j) => assert.ok(Math.abs(out.matrix.data[j] - v) <= out.errorBounds[0] + 2 ** -53 * Math.abs(v), `${x} ${j}`));
}
// Review b14fix2 #2: a value outside every evaluable window cannot change them.
for (const big of [0, 1, 1e16, 1e300, Number.MAX_VALUE, -1e300]) {
    const x = row([big, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
    const mask = new Uint8Array(11).fill(1);
    mask[1] = 0;
    const out = preflightNoiseProxy(x, 3, 1, mask);
    rel(out.bandNoise[5], Math.sqrt(2 / 3), 1e-15, `H ${big}`);
    rel(out.pooled, 1 / Math.sqrt(7), 1e-15, `H ${big}`);
}
{
    const mixed = preflightNoiseProxy(row([0, 0, -(2 ** 500), 2 ** -500, 2 ** 500, 0, 0]), 3, 1);
    rel(mixed.bandNoise[3], (2 / 3) * 2 ** -500 / Math.sqrt(2 / 3), 1e-15);
}
// Review b1fix #1/#3, b14fix #3: SNV at the smallest subnormal, bounds and
// aggregation of very unbalanced repeats (a [1,2,4], a = 1, 2, 4, 8; 1, 10000, 1, 1 copies).
const snvRows = preflightSnv({data: new Float64Array([tinyA, 0, -tinyA, 0.1, 0.1, 0.1]), rows: 2, cols: 3});
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
// Noise proxy: power-of-two equivariance, exact-offset invariance, explicit overflow error.
{
    const base = [];
    for (let j = 0; j < 11; j++) base.push(j * j);
    for (let j = 0; j < 11; j++) base.push(1 + 2 * j + (j === 5 ? 35 : 0));
    const ref = preflightNoiseProxy({data: new Float64Array(base), rows: 2, cols: 11}, 5, 2);
    for (const k of [-1074, -1022, -500, -100, 0, 100, 500, 1000]) {
        const s = 2 ** k;
        const out = preflightNoiseProxy({data: new Float64Array(base.map((v) => v * s)), rows: 2, cols: 11}, 5, 2);
        if (ref.pooled * s >= 2 ** -1022) assert.equal(out.pooled, ref.pooled * s, `k ${k}`);
        else assert.ok(out.pooled > 0 && Math.abs(out.pooled - ref.pooled * s) <= tinyA, `k ${k}`);
    }
    const shifted = preflightNoiseProxy({data: new Float64Array(base.map((v) => v * 2 ** 944 + 2 ** 996)), rows: 2, cols: 11}, 5, 2);
    rel(shifted.pooled, ref.pooled * 2 ** 944, 1e-15);
    const alt = Array.from({length: 9}, (_, j) => (j % 2 ? -1 : 1) * Number.MAX_VALUE);
    numericalFailure(() => preflightNoiseProxy({data: new Float64Array(alt), rows: 1, cols: 9}, 3, 1));
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
console.log("JS/WASM Preflight diagnostics passed");
