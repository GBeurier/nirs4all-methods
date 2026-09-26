// SPDX-License-Identifier: CECILL-2.1
//
// Smoke the compiled public JS API. The raw C splitter mask is kept for
// compatibility, while computeSplitIndices exposes libn4m's ordered indices.

import assert from "node:assert/strict";
import {
    computeSplit,
    computeSplitIndices,
    loadModule,
    selectSpa,
} from "../dist/index.js";

await loadModule();

const rows = 10, cols = 3;
const data = new Float64Array(rows * cols);
for (let r = 0; r < rows; ++r) {
    for (let c = 0; c < cols; ++c) {
        data[r * cols + c] = Math.sin((r + 1) * (c + 2) * 0.19) + r * 0.03;
    }
}

const X = { data, rows, cols };
const mask = computeSplit("KennardStone", X, null, { testSize: 0.3 });
const split = computeSplitIndices("KennardStone", X, null, { testSize: 0.3 });

assert.equal(mask.length, rows);
assert.equal(split.trainIndices.length + split.testIndices.length, rows);
assert.deepEqual(
    Array.from(split.testIndices).sort((a, b) => a - b),
    Array.from(mask.entries()).filter(([, v]) => v === 1).map(([i]) => i),
);
assert.deepEqual(
    Array.from(split.trainIndices).sort((a, b) => a - b),
    Array.from(mask.entries()).filter(([, v]) => v === 0).map(([i]) => i),
);

console.log("JS API split indices smoke OK");

const y = Float64Array.from({ length: rows }, (_, row) =>
    2 * data[row * cols] - data[row * cols + 2]);
const selected = selectSpa(X, { data: y, rows, cols: 1 }, 2, 1);
assert.ok(selected instanceof BigInt64Array);
assert.equal(selected.length, 2);
assert.equal(new Set(selected).size, 2);
assert.ok(Array.from(selected).every((index) => index >= 0n && index < BigInt(cols)));
assert.throws(() => selectSpa(X, { data: y, rows, cols: 1 }, cols + 1), /topK/);
console.log("JS API SPA int64 selection smoke OK");
