// SPDX-License-Identifier: CECILL-2.1
// Verify the public WASM diagnostic wrappers against hand-computed results.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
    loadModule, abiVersion, preflightPca, preflightRepeatability,
    preflightFactorAssociation, preflightHolmAdjust,
    preflightNoiseProxy, preflightPairsWithinTolerance,
    preflightSummary, preflightNearestNeighbors, preflightLocalContrast,
    clusterHdbscan, preflightAdjustedRandIndex, HotellingT2,
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
const shifted = pcaOf(pcaBase.map((v) => 16 * v + 1e16));
for (let a = 0; a < 3; ++a) relativeNearTol(shifted.eigenvalues[a], 256 * base.eigenvalues[a]);
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
console.log("JS/WASM Preflight diagnostics passed");
