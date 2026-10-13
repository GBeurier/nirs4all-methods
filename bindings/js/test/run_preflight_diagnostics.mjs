// SPDX-License-Identifier: CECILL-2.1
// Verify the public WASM diagnostic wrappers against hand-computed results.
import assert from "node:assert/strict";
import {
    loadModule, abiVersion, preflightPca, preflightRepeatability,
    preflightSummary, preflightNearestNeighbors, preflightLocalContrast,
    preflightFactorAssociation, preflightHolmAdjust,
} from "../dist/index.js";

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9,
    `${actual} != ${expected}`);
const relativeNear = (actual, expected) => assert.ok(
    Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)),
    `${actual} != ${expected}`);
await loadModule();
assert.deepEqual(abiVersion(), [2, 19, 0]);
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
near(categorical.contrast, 0.4);
near(categorical.observedSameClassFraction, 0.5);
near(categorical.expectedSameClassFraction, 1 / 6);
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
console.log("JS/WASM Preflight diagnostics passed");
