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
    clusterHdbscan, preflightAdjustedRandIndex, statsPermanova, statsPermdisp, statsPermanovaFreedmanLane, N4mError,
    statsEnergyDistance, statsPcaMonitor,
    statsVarianceComponentsReml, statsVarianceComponentsIntervals,
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
const lo = 2 ** -256, hi = 2 ** 256, sp = 2 ** 32;
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
console.log("JS/WASM Preflight diagnostics passed");

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
console.log("variance components: ok");
