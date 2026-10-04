// SPDX-License-Identifier: CECILL-2.1
// Consume a freshly generated Python/oracle fixture; no saved success output.
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadModule, MultimodalClassifierPipeline, MultimodalPipeline } from "../dist/index.js";

if (!process.argv[2]) throw new Error("actual raw classifier fixture JSON path required");
const fixture = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
await loadModule();
const blocks = (raw) => Object.fromEntries(Object.entries(raw).map(([name, value]) => {
    if (name === "metadata") return [name, structuredClone(value)];
    const Type = fixture.source_schemas[name].dtype === "float32" ? Float32Array : Float64Array;
    return [name, { data: new Type(value.data), shape: value.shape }];
}));
const train = blocks(fixture.train), heldout = blocks(fixture.heldout);
const check = (model) => {
    assert.deepEqual(model.classes(), fixture.label_names);
    assert.deepEqual(model.predict(heldout), fixture.expected_labels);
    const matrix = model.predictProba(heldout), expected = fixture.expected_probabilities.flat();
    assert.equal(matrix.rows, fixture.expected_labels.length);
    assert.equal(matrix.cols, fixture.label_names.length);
    assert.equal(matrix.data.length, expected.length);
    matrix.data.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) <= 1e-8 * (1 + Math.abs(expected[i]))));
    for (let i = 0; i < matrix.rows; ++i) {
        const row = matrix.data.slice(i * matrix.cols, (i + 1) * matrix.cols);
        assert.ok(row.every((value) => Number.isFinite(value) && value >= 0 && value <= 1));
        assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    }
};
const state = Uint8Array.from(Buffer.from(fixture.state, "base64"));
assert.equal(Buffer.from(state.slice(0, 4)).toString(), "N4MC");
const options = { classNames: fixture.label_names };
const replay = MultimodalClassifierPipeline.fromState(state, fixture.recipe, fixture.source_schemas, options);
try {
    check(replay);
    const rawIds = MultimodalClassifierPipeline.fromState(state, fixture.recipe, fixture.source_schemas);
    try {
        assert.deepEqual(rawIds.classes(), fixture.label_names.map((_, i) => i));
        assert.deepEqual(rawIds.predict(heldout), fixture.expected_labels.map((name) => fixture.label_names.indexOf(name)));
    } finally { rawIds.dispose(); }
    const broken = state.slice(); broken[60] ^= 1;
    assert.throws(() => MultimodalClassifierPipeline.fromState(broken, fixture.recipe, fixture.source_schemas, options));
    const wrong = structuredClone(fixture.source_schemas), first = fixture.recipe.source_order[0];
    wrong[first].identity += ":foreign-source";
    assert.throws(() => replay.predict(heldout, wrong));
    assert.throws(() => MultimodalClassifierPipeline.fromState(state, fixture.recipe, wrong, options));
    const recipe = structuredClone(fixture.recipe); recipe.model.params.n_components += 1;
    assert.throws(() => MultimodalClassifierPipeline.fromState(state, recipe, fixture.source_schemas, options));
    assert.throws(() => MultimodalClassifierPipeline.fromState(state, fixture.recipe, fixture.source_schemas,
        { classNames: fixture.label_names.slice(1) }));
    assert.throws(() => MultimodalClassifierPipeline.fromState(state, fixture.recipe, fixture.source_schemas,
        { classNames: Array(fixture.label_names.length).fill(fixture.label_names[0]) }));
    assert.throws(() => new MultimodalPipeline(fixture.recipe, fixture.source_schemas));
    const fresh = new MultimodalClassifierPipeline(fixture.recipe, fixture.source_schemas);
    try {
        fresh.fit(train, fixture.y); check(fresh);
        const exported = fresh.exportState();
        const hydrated = MultimodalClassifierPipeline.fromState(exported, fixture.recipe, fixture.source_schemas,
            { classNames: fresh.labelNames() });
        try { check(hydrated); } finally { hydrated.dispose(); }
        const beforeLabels = fresh.predict(heldout), beforeProbabilities = fresh.predictProba(heldout).data;
        const numeric = fixture.recipe.source_order.find((name) => name !== "metadata");
        if (numeric) train[numeric].data[0] = NaN; else train.metadata[0][0] = NaN;
        assert.throws(() => fresh.fit(train, fixture.y));
        assert.deepEqual(fresh.predict(heldout), beforeLabels);
        assert.deepEqual(fresh.predictProba(heldout).data, beforeProbabilities);
        assert.throws(() => fresh.fit(blocks(fixture.train), fixture.y.map(() => fixture.label_names[0])));
        assert.deepEqual(fresh.predict(heldout), beforeLabels);
    } finally { fresh.dispose(); fresh.dispose(); }
    check(replay);
} finally { replay.dispose(); replay.dispose(); }
console.log("native WASM classifier fit/import/labels/probability-order/schema/transactional-cleanup PASS");
