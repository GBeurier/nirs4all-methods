// SPDX-License-Identifier: CECILL-2.1
// Verify the public WASM diagnostic wrappers against hand-computed results.
import assert from "node:assert/strict";
import {
    loadModule, abiVersion, preflightPca, preflightRepeatability,
    preflightSummary, preflightNearestNeighbors, preflightLocalContrast,
    preflightNoiseProxy, preflightPairsWithinTolerance,
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
console.log("JS/WASM Preflight diagnostics passed");
