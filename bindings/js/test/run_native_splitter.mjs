// SPDX-License-Identifier: CECILL-2.1
import assert from "node:assert/strict";
import { computeSplitIndices, loadModule, splitNative } from "../dist/index.js";

await loadModule();
const X = { rows: 30, cols: 2, data: Float64Array.from(
    Array.from({ length: 30 }, (_, i) => [i + 1, ((i + 1) * 7) % 13]).flat()) };
const Y = { rows: 30, cols: 1, data: Float64Array.from(
    Array.from({ length: 30 }, (_, i) => (i + 1) % 11)) };
const groups = Array.from({ length: 30 }, (_, i) => BigInt(Math.floor(i / 2)));
const cases = [
    ["KennardStone", X, null, {}],
    ["SPXY", X, Y, {}],
    ["SPXYFold", X, Y, { nSplits: 3 }],
    ["SPXYGroupFold", X, Y, { nSplits: 3, groups }],
    ["KMeans", X, null, { seed: 42 }],
    ["KBinsStratified", null, Y, { seed: 42, nBins: 2 }],
    ["BinnedStratGroupFold", null, Y,
        { nSplits: 3, nBins: 2, groups, seed: 42 }],
    ["SystematicCircular", null, Y, { seed: 42 }],
    ["DataTwinning", X, null, { seed: 42 }],
];
for (const [kind, x, y, options] of cases) {
    const first = splitNative(kind, x, y, options);
    const second = splitNative(kind, x, y, options);
    assert.deepEqual(first, second, kind);
    assert.equal(first.trainIndices.length + first.testIndices.length, 30, kind);
    assert.deepEqual(Array.from([...first.trainIndices, ...first.testIndices]).sort((a, b) => a - b),
        Array.from({ length: 30 }, (_, i) => i), kind);
    if (options.groups) {
        const trainGroups = new Set(Array.from(first.trainIndices, i => groups[i]));
        for (const row of first.testIndices) assert.ok(!trainGroups.has(groups[row]));
    }
    if (["SPXYFold", "SPXYGroupFold", "BinnedStratGroupFold"].includes(kind)) {
        const other = splitNative(kind, x, y, { ...options, foldIndex: 1 });
        assert.notDeepEqual(other.testIndices, first.testIndices);
    }
}
for (const kind of ["KennardStone", "SPXY", "KMeans", "DataTwinning",
                    "SystematicCircular"]) {
    const selected = cases.find(row => row[0] === kind);
    const [_, x, y, options] = selected;
    const old = computeSplitIndices(kind, x ?? X, y,
        { testSize: 0.25, seed: Number(options.seed ?? 0) });
    const current = splitNative(kind, x, y, options);
    assert.deepEqual(current, old, `${kind} legacy C route parity`);
}
assert.throws(() => splitNative("SPXY", X, null), /X\/Y\/groups/);
assert.throws(() => splitNative("SPXYGroupFold", X, Y,
    { groups: groups.slice(1) }), /groups length/);
assert.throws(() => splitNative("SPXYFold", X, Y,
    { nSplits: 3, foldIndex: 3 }), /fold/);
console.log("JS generic native splitter: 9 kinds, ordering, seed, folds and groups OK");
