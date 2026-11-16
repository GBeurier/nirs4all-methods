// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Explainability bindings over the built WASM module (n4m/explain.h):
// ask/tell sessions with sync/async JS models, exact references, TreeSHAP on a
// hand-built ensemble, grouping helpers and boundary validation.

import assert from "node:assert/strict";
import {
    explainModel, groupsFixedWidth, groupsFromRanges, linearShap, loadModule,
    N4mError, summarizeAttributions, treeShap,
} from "../dist/index.js";

await loadModule();

const matrix = (rows, cols, values) => ({ rows, cols, data: Float64Array.from(values) });
const close = (actual, expected, tol, label) => {
    assert.equal(actual.length, expected.length, `${label}: length`);
    for (let i = 0; i < expected.length; i += 1) {
        assert.ok(Math.abs(actual[i] - expected[i]) <= tol,
            `${label}[${i}]: ${actual[i]} != ${expected[i]}`);
    }
};

// ---- linear model: permutation SHAP is exact for additive functions ---------
const nFeatures = 6;
const w = Float64Array.from([0.5, -1.25, 2, 0, 0.75, -0.3]);
const bias = 0.4;
const linearPredict = (rows, nRows, p) => {
    const out = new Float64Array(nRows);
    for (let r = 0; r < nRows; r += 1) {
        let s = bias;
        for (let j = 0; j < p; j += 1) s += rows[r * p + j] * w[j];
        out[r] = s;
    }
    return out;
};
const X = matrix(3, nFeatures, Array.from({ length: 18 }, (_, i) => Math.sin(i + 1)));
const background = matrix(4, nFeatures, Array.from({ length: 24 }, (_, i) => Math.cos(0.7 * i)));
const reference = new Float64Array(nFeatures);
for (let b = 0; b < background.rows; b += 1) {
    for (let j = 0; j < nFeatures; j += 1) reference[j] += background.data[b * nFeatures + j] / background.rows;
}

let lastProgress = null;
const perm = await explainModel({
    X, background, predictBatch: linearPredict,
    options: { method: "permutation_shap", nPermutations: 3, seed: 7, maxBatchRows: 50 },
    onProgress: (done, total) => { lastProgress = [done, total]; },
});
assert.equal(perm.kind, "permutation_shap");
assert.equal(perm.exactness, "sampled");
assert.equal(perm.additive, true);
assert.equal(perm.groupAggregation, "group_game");
assert.equal(perm.baseline, "background");
assert.deepEqual([perm.nRows, perm.nGroups, perm.nOutputs, perm.nFeatures], [3, nFeatures, 1, nFeatures]);
assert.ok(perm.additivityMaxAbsError < 1e-10, String(perm.additivityMaxAbsError));
assert.ok(perm.stdError instanceof Float64Array && perm.stdError.length === perm.attributions.length);
assert.equal(perm.engine.name, "nirs4all-methods");
assert.match(perm.engine.abi, /^\d+\.\d+\.\d+$/);
assert.equal(perm.settings.n_outputs_inferred, true);
assert.ok(lastProgress !== null && lastProgress[0] === lastProgress[1]);
assert.equal(perm.evaluations, lastProgress[1] + 1); // + the nOutputs probe row

const lin = linearShap({ X, coefficients: matrix(nFeatures, 1, w), intercept: Float64Array.of(bias), reference });
assert.equal(lin.kind, "linear_shap");
assert.equal(lin.exactness, "exact");
assert.equal(lin.groupAggregation, "none");
assert.equal(lin.baseline, null);
assert.equal(lin.evaluations, 0);
close(perm.attributions, lin.attributions, 1e-10, "permutation vs linear");
close(perm.expected, lin.expected, 1e-10, "expected");
close(perm.predictions, lin.predictions, 1e-12, "predictions");
for (let r = 0; r < X.rows; r += 1) {
    for (let j = 0; j < nFeatures; j += 1) {
        const analytic = w[j] * (X.data[r * nFeatures + j] - reference[j]);
        assert.ok(Math.abs(lin.attributions[r * nFeatures + j] - analytic) < 1e-12);
    }
}

// Linear group sums are exact group Shapley values (group game).
const fixed = groupsFixedWidth(nFeatures, 4);
const linGrouped = linearShap({ X, coefficients: matrix(nFeatures, 1, w), intercept: Float64Array.of(bias),
    reference, grouping: fixed });
assert.equal(linGrouped.groupAggregation, "group_game");
assert.equal(linGrouped.nGroups, 2);
for (let r = 0; r < X.rows; r += 1) {
    const g0 = lin.attributions.slice(r * 6, r * 6 + 4).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(linGrouped.attributions[r * 2] - g0) < 1e-12);
}

// Async predictBatch reproduces the synchronous result (deterministic plan).
const asyncPerm = await explainModel({
    X, background, nOutputs: 1,
    predictBatch: async (rows, n, p) => { await new Promise(r => setTimeout(r, 1)); return Array.from(linearPredict(rows, n, p)); },
    options: { method: "permutation_shap", nPermutations: 3, seed: 7, maxBatchRows: 50 },
});
close(asyncPerm.attributions, perm.attributions, 0, "async vs sync");
assert.equal(asyncPerm.settings.n_outputs_inferred, false);

// ---- exact SHAP vs analytic Shapley of f = x0*x1 + 2*x2 --------------------
const interact = (rows, n, p) => Float64Array.from({ length: n },
    (_, r) => rows[r * p] * rows[r * p + 1] + 2 * rows[r * p + 2]);
const X3 = matrix(2, 3, [1.5, -2, 0.5, 0.25, 3, -1]);
const bg3 = matrix(2, 3, [0.5, 1, 0, -1, 0.5, 2]);
const exact = await explainModel({ X: X3, background: bg3, predictBatch: interact, options: { method: "exact_shap" } });
assert.equal(exact.exactness, "exact");
const analytic = [], expectedValue = [];
for (let r = 0; r < 2; r += 1) {
    const [x0, x1, x2] = X3.data.slice(r * 3, r * 3 + 3);
    const phi = [0, 0, 0];
    let e = 0;
    for (let b = 0; b < 2; b += 1) {
        const [b0, b1, b2] = bg3.data.slice(b * 3, b * 3 + 3);
        phi[0] += 0.5 * (x0 - b0) * (x1 + b1) / 2;
        phi[1] += 0.5 * (x1 - b1) * (x0 + b0) / 2;
        phi[2] += 2 * (x2 - b2) / 2;
        e += (b0 * b1 + 2 * b2) / 2;
    }
    analytic.push(...phi);
    expectedValue.push(e);
}
close(exact.attributions, analytic, 1e-12, "exact vs analytic");
close(exact.expected, expectedValue, 1e-12, "exact expected");

// Kernel SHAP enumerating every coalition (2^3 - 2 <= auto budget) is exact.
const kernel = await explainModel({ X: X3, background: bg3, predictBatch: interact, options: { method: "kernel_shap" } });
assert.equal(kernel.exactness, "exact");
assert.equal(kernel.stdError, null);
close(kernel.attributions, analytic, 1e-8, "kernel (enumerated) vs analytic");

// Non-additive explainers.
const lime = await explainModel({ X: X3, background: bg3, predictBatch: interact,
    options: { method: "lime", seed: 3 } });
assert.deepEqual([lime.exactness, lime.additive, lime.additivityMaxAbsError], ["surrogate", false, null]);
const occ = await explainModel({ X: X3, predictBatch: interact,
    options: { method: "occlusion", baseline: "interpolate" } });
assert.deepEqual([occ.exactness, occ.additive, occ.baseline], ["perturbation", false, "interpolate"]);
assert.equal(occ.settings.n_background, 0);

// Multi-output inference from a single probe row.
const twoOut = await explainModel({ X: X3, background: bg3, options: { method: "exact_shap" },
    predictBatch: (rows, n, p) => {
        const one = interact(rows, n, p), out = new Float64Array(2 * n);
        for (let r = 0; r < n; r += 1) { out[2 * r] = one[r]; out[2 * r + 1] = -one[r]; }
        return out;
    } });
assert.equal(twoOut.nOutputs, 2);
for (let i = 0; i < 6; i += 1) {
    assert.ok(Math.abs(twoOut.attributions[2 * i] - analytic[i]) < 1e-12);
    assert.ok(Math.abs(twoOut.attributions[2 * i + 1] + analytic[i]) < 1e-12);
}

// ---- abort and failing models ------------------------------------------------
const controller = new AbortController();
await assert.rejects(explainModel({ X, background, nOutputs: 1, signal: controller.signal,
    predictBatch: (rows, n, p) => { controller.abort(); return linearPredict(rows, n, p); } }),
    err => err.name === "AbortError");
const pre = new AbortController();
pre.abort();
await assert.rejects(explainModel({ X, background, nOutputs: 1, signal: pre.signal, predictBatch: linearPredict }),
    err => err.name === "AbortError");
const boom = new Error("model failed");
await assert.rejects(explainModel({ X, background, nOutputs: 1, predictBatch: () => { throw boom; } }),
    err => err === boom);
await assert.rejects(explainModel({ X, background, nOutputs: 1, predictBatch: async () => { throw boom; } }),
    err => err === boom);
// A session that failed mid-plan was destroyed; the module keeps working.
const again = await explainModel({ X, background, nOutputs: 1, predictBatch: linearPredict,
    options: { nPermutations: 3, seed: 7, maxBatchRows: 50 } });
close(again.attributions, perm.attributions, 0, "after failures");

// ---- TreeSHAP on a hand-built two-tree ensemble ------------------------------
// Tree 0: x0 <= .5 ? (x1 <= .5 ? 1 : 2) : 5, covers 4 / 2 / 1 / 1 / 2.
// Tree 1: x2 <= .5 ? 0 : 4, covers 4 / 1 / 3.
const trees = {
    treeOffsets: [0, 5, 8],
    feature: Int32Array.of(0, 1, -1, -1, -1, 2, -1, -1),
    threshold: Float64Array.of(0.5, 0.5, 0, 0, 0, 0.5, 0, 0),
    left: Int32Array.of(1, 2, -1, -1, -1, 1, -1, -1),
    right: Int32Array.of(4, 3, -1, -1, -1, 2, -1, -1),
    cover: Float64Array.of(4, 2, 1, 1, 2, 4, 1, 3),
    value: Float64Array.of(0, 0, 1, 2, 5, 0, 0, 4),
    nOutputs: 1,
    baseOffset: Float64Array.of(0.5),
};
const Xt = matrix(2, 3, [0, 1, 0, 1, 0, 1]);
const tree = treeShap({ X: Xt, trees });
assert.deepEqual([tree.kind, tree.exactness, tree.groupAggregation], ["tree_shap", "exact", "none"]);
// Path-dependent values derived by hand from the cover-weighted conditional expectations.
close(tree.attributions, [-1.625, 0.375, -3, 1.875, -0.125, 1], 1e-12, "tree attributions");
close(tree.expected, [0.5 + 3.25 + 3, 0.5 + 3.25 + 3], 1e-12, "tree expected");
close(tree.predictions, [0.5 + 2 + 0, 0.5 + 5 + 4], 1e-12, "tree predictions");
assert.ok(tree.additivityMaxAbsError < 1e-12);
const treeGrouped = treeShap({ X: Xt, trees, grouping: groupsFromRanges(3, [[0, 2]]) });
assert.equal(treeGrouped.groupAggregation, "sum_of_feature_values");
close(treeGrouped.attributions, [-1.25, -3, 1.75, 1], 1e-12, "tree grouped");
const forest = treeShap({ X: Xt, trees: { ...trees, baseOffset: undefined, treeScale: 0.5 } });
close(forest.attributions, Array.from(tree.attributions, v => v / 2), 1e-12, "tree scale");

// ---- grouping and summary helpers --------------------------------------------
assert.deepEqual(Array.from(fixed.featureGroup), [0, 0, 0, 0, 1, 1]);
const fw = groupsFixedWidth(5, 2);
assert.deepEqual([Array.from(fw.featureGroup), fw.nGroups], [[0, 0, 1, 1, 2], 3]);
const ranged = groupsFromRanges(6, [[3, 5], [0, 1]]);
assert.deepEqual(Array.from(ranged.featureGroup), [1, 2, 2, 0, 0, 3]);
assert.equal(ranged.nGroups, 4);
assert.deepEqual(Array.from(ranged.rangeOfGroup), [0, 1, -1, -1]);
assert.throws(() => groupsFromRanges(6, [[0, 3], [2, 4]]), N4mError);
assert.throws(() => groupsFixedWidth(5, 0), /positive integers/);

const summary = summarizeAttributions(exact);
const meanAbs = [0, 1, 2].map(g => (Math.abs(exact.attributions[g]) + Math.abs(exact.attributions[3 + g])) / 2);
const meanSigned = [0, 1, 2].map(g => (exact.attributions[g] + exact.attributions[3 + g]) / 2);
close(summary.meanAbs, meanAbs, 1e-12, "meanAbs");
close(summary.meanSigned, meanSigned, 1e-12, "meanSigned");
assert.throws(() => summarizeAttributions(exact, 1), /outputIndex/);

// Grouped session: the group is the player of the coalition game.
const grouped = await explainModel({ X: X3, background: bg3, predictBatch: interact,
    grouping: groupsFromRanges(3, [[0, 2]]), options: { method: "exact_shap" } });
assert.equal(grouped.nGroups, 2);
for (let r = 0; r < 2; r += 1) {
    assert.ok(Math.abs(grouped.attributions[r * 2 + 1] - analytic[r * 3 + 2]) < 1e-12);
}

// ---- boundary validation -----------------------------------------------------
const base = { X, background, nOutputs: 1, predictBatch: linearPredict };
await assert.rejects(explainModel({ ...base, options: { nPermutation: 3 } }), /unknown explain option/);
await assert.rejects(explainModel({ ...base, options: { method: "deep_shap" } }), /unknown explain method/);
await assert.rejects(explainModel({ ...base, options: { baseline: "zero" } }), /unknown explain baseline/);
await assert.rejects(explainModel({ ...base, background: undefined }), /requires a background/);
await assert.rejects(explainModel({ ...base, background: matrix(2, 3, [0, 0, 0, 0, 0, 0]) }), /X.cols columns/);
await assert.rejects(explainModel({ ...base, options: { seed: -1 } }), /seed/);
await assert.rejects(explainModel({ ...base, grouping: { featureGroup: Int32Array.of(0, 1), nGroups: 2 } }),
    /grouping/);
await assert.rejects(explainModel({ ...base, grouping: { featureGroup: Int32Array.of(0, 0, 0, 2, 2, 2), nGroups: 3 } }),
    N4mError); // empty group 1: not a partition
await assert.rejects(explainModel({ ...base, predictBatch: (rows, n) => new Float64Array(n + 1) }), /expected/);
await assert.rejects(explainModel({ ...base, predictBatch: (rows, n) => new Float64Array(n).fill(NaN) }),
    /nonfinite/);
await assert.rejects(explainModel({ ...base, X: { rows: 2, cols: 6, data: new Float64Array(5) } }), /X must be/);
// Exact SHAP enumerates 2^G coalitions and is limited to G <= 16 (native refusal).
await assert.rejects(explainModel({ X: matrix(1, 17, new Array(17).fill(1)), background: matrix(1, 17, new Array(17).fill(0)),
    nOutputs: 1, predictBatch: (rows, n) => new Float64Array(n), options: { method: "exact_shap" } }), N4mError);
assert.throws(() => linearShap({ X, coefficients: matrix(5, 1, [1, 1, 1, 1, 1]), reference }), /coefficients/);
assert.throws(() => linearShap({ X, coefficients: matrix(6, 1, w), reference: new Float64Array(5) }), /reference/);
assert.throws(() => treeShap({ X: Xt, trees: { ...trees, treeOffsets: [0, 5, 9] } }), /node count/);
assert.throws(() => treeShap({ X: Xt, trees: { ...trees, value: new Float64Array(7) } }), /trees.value/);
assert.throws(() => treeShap({ X: Xt, trees: { ...trees, decisionType: 2 } }), /decisionType/);

console.log("JS explainability: sessions (sync/async/abort/failure), linear, tree, grouping and validation OK");
