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
// Exact status codes (types.ts Status): 10 = N4M_ERR_UNSUPPORTED (outside the
// supported numerical domain), never a trap or a silently wrong value.
const unsupported = (fn) => assert.throws(fn, (e) => e.status === 10 &&
    /outside the supported numerical domain/.test(e.message));
// Supported domain D (rule 13): nonzero |x| in [2^-256, 2^256], per-row spread <= 2^64.
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
assert.equal(pair([1, 2 ** 64], [0, 0], 1).totalPairs, 0);
unsupported(() => pair([1, 2 ** 64 + 2 ** 12], [0, 0], 1));
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
    [[1.152921504606847e+18, 0.125, -0.1875], [1.152921504606847e+18, 0.0, 0.0625], 0.16137430609197567, false, 0.1613743060919757],
    [[1.152921504606847e+18, 0.125, -0.1875], [1.152921504606847e+18, 0.0, 0.0625], 0.1613743060919757, false, 0.1613743060919757],
    [[1.152921504606847e+18, 0.125, -0.1875], [1.152921504606847e+18, 0.0, 0.0625], 0.16137430609197573, true, 0.1613743060919757],
    [[1.8092513943330656e+75, 1.6069380442589903e+60, 2.4104070663884854e+60], [-1.8092513943330656e+75, 1.6069380442589903e+60, 0.0], 2.0891435590998018e+75, false, 2.089143559099802e+75],
    [[1.8092513943330656e+75, 1.6069380442589903e+60, 2.4104070663884854e+60], [-1.8092513943330656e+75, 1.6069380442589903e+60, 0.0], 2.089143559099802e+75, false, 2.089143559099802e+75],
    [[1.8092513943330656e+75, 1.6069380442589903e+60, 2.4104070663884854e+60], [-1.8092513943330656e+75, 1.6069380442589903e+60, 0.0], 2.0891435590998026e+75, true, 2.089143559099802e+75],
    [[3.0, -4.0, 8.673617379884035e-19], [-3.0, 4.0, 0.0], 5.773502691896256, false, 5.773502691896257],
    [[3.0, -4.0, 8.673617379884035e-19], [-3.0, 4.0, 0.0], 5.773502691896257, false, 5.773502691896257],
    [[3.0, -4.0, 8.673617379884035e-19], [-3.0, 4.0, 0.0], 5.773502691896258, true, 5.773502691896257],
    [[6.223015277861142e-61, 5.527147875260445e-76], [0.0, -5.527147875260445e-76], 4.4003363024031e-61, false, 4.400336302403101e-61],
    [[6.223015277861142e-61, 5.527147875260445e-76], [0.0, -5.527147875260445e-76], 4.400336302403101e-61, true, 4.400336302403101e-61],
    [[6.223015277861142e-61, 5.527147875260445e-76], [0.0, -5.527147875260445e-76], 4.4003363024031016e-61, true, 4.400336302403101e-61],
    [[1e+16, 1.0], [9999999999999998.0, 0.0], 1.5811388300841895, false, 1.5811388300841898],
    [[1e+16, 1.0], [9999999999999998.0, 0.0], 1.5811388300841898, true, 1.5811388300841898],
    [[1e+16, 1.0], [9999999999999998.0, 0.0], 1.58113883008419, true, 1.5811388300841898],
    [[1.157920892373162e+77, 6.277101735386681e+57], [-1.157920892373162e+77, 6.277101735386681e+57], 1.6375474301492824e+77, false, 1.6375474301492827e+77],
    [[1.157920892373162e+77, 6.277101735386681e+57], [-1.157920892373162e+77, 6.277101735386681e+57], 1.6375474301492827e+77, true, 1.6375474301492827e+77],
    [[1.157920892373162e+77, 6.277101735386681e+57], [-1.157920892373162e+77, 6.277101735386681e+57], 1.637547430149283e+77, true, 1.6375474301492827e+77],
    [[8.636168555094445e-78, 1.5930919111324523e-58], [0.0, -1.5930919111324523e-58], 2.2529721868303873e-58, false, 2.2529721868303876e-58],
    [[8.636168555094445e-78, 1.5930919111324523e-58], [0.0, -1.5930919111324523e-58], 2.2529721868303876e-58, true, 2.2529721868303876e-58],
    [[8.636168555094445e-78, 1.5930919111324523e-58], [0.0, -1.5930919111324523e-58], 2.252972186830388e-58, true, 2.2529721868303876e-58],
];
const snvGoldens = [
    [[1.152921504606847e+18, 0.125, -0.125, 0.0], [1.7320508075688772, -0.5773502691896257, -0.5773502691896257, -0.5773502691896257]],
    [[1.157920892373162e+77, 1.6069380442589903e+60, 6.277101735386681e+57, 0.0], [1.7320508075688772, -0.5773502691896257, -0.5773502691896257, -0.5773502691896257]],
    [[8.636168555094445e-78, 6.223015277861142e-61, 2.5908505665283334e-77, 0.0], [-0.5773502691896257, 1.7320508075688772, -0.5773502691896257, -0.5773502691896258]],
    [[1e+16, 1.0, -1.0, 1.0000000000000002e+16], [0.9999999999999998, -0.9999999999999998, -1.0000000000000002, 1.0000000000000002]],
    [[9.223372036854776e+18, -9.223372036854776e+18, 0.5, 0.0], [1.4142135623730951, -1.4142135623730951, 5.749850312562652e-20, -1.9166167708542177e-20]],
    [[1.0, 1.0000000000000002, 1.0, 1.0], [-0.5773502691896257, 1.7320508075688772, -0.5773502691896257, -0.5773502691896257]],
    [[-1.5930919111324523e-58, 8.636168555094445e-78, 1.727233711018889e-77, 0.0], [-1.7320508075688772, 0.5773502691896257, 0.5773502691896257, 0.5773502691896257]],
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
unsupported(() => preflightSnv(row([1, 0, 0, 2 ** 64 + 2 ** 12])));
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
    for (const big of [0, 1, 1e16, 2 ** 64, -(2 ** 60)]) {
        const out = preflightNoiseProxy(row([big, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 3, 1, mask);
        rel(out.bandNoise[5], Math.sqrt(2 / 3), 1e-15, `H ${big}`);
        rel(out.pooled, 1 / Math.sqrt(7), 1e-15, `H ${big}`);
    }
    unsupported(() => preflightNoiseProxy(row([2 ** 65, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 3, 1, mask));
    // Spread 2^60 inside one window: [-2^60, 2^60, 1, 0, 0], window 5 / order 1 -> sqrt(4/5).
    const spread = preflightNoiseProxy(row([-(2 ** 60), 2 ** 60, 1, 0, 0]), 5, 1);
    rel(spread.bandNoise[2], Math.sqrt(0.8), 1e-14);
    // Outside D (review b14fix3 #3/#4): explicit refusal.
    unsupported(() => preflightNoiseProxy(row([-(2 ** 100), 2 ** 100, 1, 0, 0]), 5, 1));
    unsupported(() => preflightNoiseProxy(row([-Number.MAX_VALUE, 2 ** -1074, Number.MAX_VALUE]), 3, 1));
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
