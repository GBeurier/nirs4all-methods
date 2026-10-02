// SPDX-License-Identifier: CECILL-2.1
// Actual native WASM fit and fresh Python-state import against sklearn oracle.
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadModule, MultimodalPipeline } from "../dist/index.js";

if (!process.argv[2]) throw new Error("raw diagnostic fixture JSON path required");
const fixture = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
await loadModule();
const blocks = (raw) => Object.fromEntries(Object.entries(raw).map(([name, value]) =>
    [name, name === "metadata" ? value : { data: new Float64Array(value.data), shape: value.shape }]));
const heldout = blocks(fixture.heldout), train = blocks(fixture.train);
const check = (prediction) => {
    assert.equal(prediction.rows, fixture.expected.length); assert.equal(prediction.cols, 1);
    prediction.data.forEach((value, i) => assert.ok(Math.abs(value - fixture.expected[i]) <= 1e-8 * (1 + Math.abs(fixture.expected[i]))));
};
const state = Uint8Array.from(Buffer.from(fixture.state, "base64"));
const replay = MultimodalPipeline.fromState(state, fixture.recipe, fixture.source_schemas);
try {
    check(replay.predict(heldout));
    const z = replay.transform(heldout);
    assert.equal(z.cols, 14); assert.deepEqual(Array.from(z.data.slice(11, 14)), [0, 0, 0]);
    const broken = state.slice(); broken[60] ^= 1;
    assert.throws(() => MultimodalPipeline.fromState(broken, fixture.recipe, fixture.source_schemas));
    const schema = structuredClone(fixture.source_schemas); schema.image.identity += ":wrong-axis";
    assert.throws(() => replay.predict(heldout, schema));
    assert.throws(() => MultimodalPipeline.fromState(state, fixture.recipe, schema));
    const fresh = new MultimodalPipeline(fixture.recipe, fixture.source_schemas);
    try {
        fresh.fit(train, new Float64Array(fixture.y)); check(fresh.predict(heldout));
        const exported = fresh.exportState(); assert.equal(Buffer.from(exported.slice(0, 4)).toString(), "N4MF");
        const hydrated = MultimodalPipeline.fromState(exported, fixture.recipe, fixture.source_schemas);
        try { assert.deepEqual(hydrated.predict(heldout), fresh.predict(heldout)); } finally { hydrated.dispose(); }
        const before = fresh.predict(heldout).data; train.series.data[0] = NaN;
        assert.throws(() => fresh.fit(train, new Float64Array(fixture.y)));
        assert.deepEqual(fresh.predict(heldout).data, before);
    } finally { fresh.dispose(); fresh.dispose(); }
} finally { replay.dispose(); }
console.log("native WASM raw multimodal fit/import/schema/unknown-category/cleanup PASS");
