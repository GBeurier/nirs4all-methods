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
// Domain edge 2^256 (rule 13): [-a, 0, a] has SD a and quartiles -a/2, a/2.
const edge = 2 ** 256;
const extremeSummary = preflightSummary(new Float64Array([-edge, 0, edge]));
assert.ok(extremeSummary[6] === 0 && extremeSummary[4] === edge);
assert.ok(extremeSummary[5] === -edge / 2 && extremeSummary[7] === edge / 2);
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
    {data: new Float64Array([edge, edge]), rows: 1, cols: 2},
    {data: new Float64Array([0, 0]), rows: 1, cols: 2}, 1);
assert.ok(Math.abs(extremeNeighbors.distances[0] - Math.SQRT2 * edge) <= 4e-16 * Math.SQRT2 * edge);
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
console.log("JS/WASM Preflight diagnostics passed");
