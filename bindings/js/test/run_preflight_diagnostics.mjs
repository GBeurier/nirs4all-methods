// SPDX-License-Identifier: CECILL-2.1
// Verify the public WASM diagnostic wrappers against hand-computed results.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
    loadModule, abiVersion, preflightPca, preflightRepeatability,
    preflightFactorAssociation, preflightHolmAdjust,
    preflightNoiseProxy, preflightPairsWithinTolerance,
    preflightSummary, preflightNearestNeighbors, preflightLocalContrast,
    clusterHdbscan, preflightAdjustedRandIndex, clusterJaccardMatrix, linearSumAssignment,
    randomMultivariateNormal, randomPermutations, getModule, makeMatrixView, Context,
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
// Overflowing squared distances once produced 7 tree nodes for 4 rows.
const huge = clusterHdbscan({data: new Float64Array([0, 1e155, 2e155, 3e155]), rows: 4, cols: 1},
    {minClusterSize: 2, minSamples: 2});
const unit = clusterHdbscan({data: new Float64Array([0, 1, 2, 3]), rows: 4, cols: 1},
    {minClusterSize: 2, minSamples: 2});
assert.ok(huge.tree.length <= 4);
assert.deepEqual(huge.labels, unit.labels);
assert.throws(() => clusterHdbscan({data: new Float64Array([-1e308, 1e308]), rows: 2, cols: 1},
    {minClusterSize: 2, minSamples: 2}));
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
// Magnitude robustness (same rules as n4m_cluster_robustness_tests): every
// result is either invariant or an explicit N4M_ERR_UNSUPPORTED (10).
{
    const status = (fn) => { try { fn(); return 0; } catch (e) { return e.status ?? -1; } };
    // Purely relative: the expected values span 2^-1000 to 2^1000.
    const relEq = (a, e) => assert.ok(Math.abs(a - e) <= 1e-12 * Math.abs(e), `${a} != ${e}`);
    const line = [0, 1, 3, 10, 11, 14];
    const unit = clusterHdbscan({data: new Float64Array(line), rows: 6, cols: 1}, {minClusterSize: 2, minSamples: 2});
    for (const k of [-500, -100, 100, 500, 1000]) {
        const got = clusterHdbscan({data: new Float64Array(line.map((v) => v * 2 ** k)), rows: 6, cols: 1},
            {minClusterSize: 2, minSamples: 2});
        assert.deepEqual(got.labels, unit.labels);
        got.persistence.forEach((v, i) => relEq(v, unit.persistence[i]));
        got.tree.forEach((node, i) => relEq(node.stability, unit.tree[i].stability * 2 ** -k));
    }
    // Review 2026-10-13: 1e-308 overflowed the stability sums (persistence NaN)
    // and 1e-310 gave an infinite density for a positive distance, with OK.
    for (const factor of [1e-308, 1e-310]) {
        const code = status(() => clusterHdbscan({data: new Float64Array(line.map((v) => v * factor)), rows: 6, cols: 1},
            {minClusterSize: 2, minSamples: 2}));
        assert.equal(code, 10, `hdbscan x ${factor}`);
    }
    const base = Array.from({length: 35}, (_, i) => 1 + ((i * 7.31) % 9));
    const want = linearSumAssignment({data: new Float64Array(base), rows: 5, cols: 7});
    for (const scale of [2 ** -1014, 2 ** -500, 2 ** 500, 2 ** 1019]) {
        assert.deepEqual(linearSumAssignment({data: new Float64Array(base.map((v) => v * scale / 16)), rows: 5, cols: 7}).colInd,
            want.colInd);
    }
    // Costs near DBL_MAX next to subnormal costs cannot be scaled exactly.
    const span = base.map((v) => v * 1e-310);
    span[0] = 1.7e308;
    assert.equal(status(() => linearSumAssignment({data: new Float64Array(span), rows: 5, cols: 7})), 10);
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
    const optimal = (cost, rows, cols) => {
        const got = linearSumAssignment({data: new Float64Array(cost), rows, cols}).colInd;
        const total = (columns) => columns.reduce((acc, j, i) => acc + exactBig(cost[i * cols + j]), 0n);
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
    // Review 2026-10-13 P0: global centring returned [0, 1, 2] here and
    // declared the +-1e308 matrix infeasible.
    assert.deepEqual(optimal([-1e20, 1e20, 1e20, 1e20, 1, 0, 1e20, 0, 1], 3, 3), [0, 2, 1]);
    optimal([-1e308, 1e308, -1e308, 1e308], 2, 2);
    let state = 12345;
    const random = () => { state = (state * 1103515245 + 12345) % 2147483648; return state / 2147483648; };
    for (let trial = 0; trial < 150; ++trial) {
        const rows = 2 + Math.floor(random() * 3), cols = rows + Math.floor(random() * 3);
        const spread = [60, 300, 1000][trial % 3];
        const cost = Array.from({length: rows * cols}, () => {
            const exponent = Math.min(1022, Math.floor(random() * 2 * spread) - spread);
            return (random() < 0.5 ? -1 : 1) * (1 + Math.floor(random() * 4096) / 4096) * 2 ** exponent;
        });
        optimal(cost, rows, cols);
    }
    const cov = [2.0, 0.3, 0.1, 0.3, 1.0, -0.2, 0.1, -0.2, 0.5];
    const unitDraw = randomMultivariateNormal(new Float64Array(3), {data: new Float64Array(cov), rows: 3, cols: 3}, 6, 42n);
    for (const k of [-509, 511]) {
        const got = randomMultivariateNormal(new Float64Array(3),
            {data: new Float64Array(cov.map((v) => v * 4 ** k)), rows: 3, cols: 3}, 6, 42n);
        got.data.forEach((v, i) => assert.equal(v, unitDraw.data[i] * 2 ** k));
    }
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
console.log("JS/WASM Preflight diagnostics passed");
