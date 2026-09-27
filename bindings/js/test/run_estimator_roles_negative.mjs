// SPDX-License-Identifier: CECILL-2.1
// Refusals of the generic estimator roles: the shared negative fixture (also
// replayed by Python, R and Rust) gives the same refusals and the same
// surviving state in JS/WASM.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as n4m from "../dist/index.js";

await n4m.loadModule();
const fixture = JSON.parse(readFileSync(
    new URL("../../../parity/fixtures/estimator_roles_negative.json", import.meta.url), "utf-8"));

const matrix = (rows) => ({
    data: Float64Array.from(rows.flat()), rows: rows.length, cols: rows[0].length,
});
const xTrain = matrix(fixture.x_train);
const xTest = matrix(fixture.x_test);
const bytes = (b64) => Uint8Array.from(Buffer.from(b64, "base64"));
const target = (c) => c.y_matrix ? matrix(c.y_matrix) : c.labels ?? Float64Array.from(c.y);
// JS names inputs in camelCase (sampleWeight), the core in snake_case.
const camel = (name) => name.replace(/_(\w)/g, (_, ch) => ch.toUpperCase());
const mentions = (c) => new RegExp(`${c.mentions}|${camel(c.mentions)}`);

// A caller's output view of another width is refused before any write,
// through the raw ABI (the facade always allocates the native width).
function checkOutputViews(c) {
    const m = n4m.getModule();
    const payload = bytes(c.n4me_base64);
    const est = n4m.NativeEstimator.fromN4me(payload);
    assert.equal(est.transform(xTest).cols, c.transform_cols, c.id);
    est.dispose();
    const scratch = m._malloc(8);
    const data = m._malloc(payload.byteLength);
    m.HEAPU8.set(payload, data);
    m.ccall("n4m_context_create", "number", ["number"], [scratch]);
    const ctx = m.getValue(scratch, "i32");
    m.ccall("n4m_estimator_import_from_buffer", "number", ["number", "number", "number", "number"],
        [ctx, data, payload.byteLength, scratch]);
    const handle = m.getValue(scratch, "i32");
    const xv = n4m.makeMatrixView(xTest.data, xTest.rows, xTest.cols);
    for (const [symbol, cols] of [["n4m_estimator_transform", c.transform_cols - 1],
        ["n4m_estimator_transform", c.transform_cols + 1], ["n4m_estimator_predict", c.predict_cols + 1]]) {
        const ov = n4m.makeMatrixView(new Float64Array(xTest.rows * cols).fill(-7), xTest.rows, cols);
        const status = m.ccall(symbol, "number", ["number", "number", "number", "number"],
            [ctx, handle, xv.viewPtr, ov.viewPtr]);
        assert.equal(status, 3, `${c.id} ${symbol} x ${cols}: SHAPE_MISMATCH`);
        assert.match(m.UTF8ToString(m.ccall("n4m_context_last_error", "number", ["number"], [ctx])),
            mentions(c));
        const written = m.HEAPF64.subarray(ov.dataPtr / 8, ov.dataPtr / 8 + xTest.rows * cols);
        assert.ok(written.every((v) => v === -7), `${c.id}: nothing written`);
        ov.free();
    }
    xv.free();
    m.ccall("n4m_estimator_destroy", null, ["number"], [handle]);
    m.ccall("n4m_context_destroy", null, ["number"], [ctx]);
    m._free(data);
    m._free(scratch);
}

for (const c of fixture.cases) {
    const Cls = n4m.methodClass(c.method_id);
    if (c.step === "fit") {
        const inputs = {
            ...(c.sample_weight ? { sampleWeight: c.sample_weight } : {}),
            ...(c.fold_ids ? { foldIds: c.fold_ids } : {}),
        };
        assert.throws(() => new Cls(c.params ?? {}).fit(xTrain, target(c), inputs), mentions(c), c.id);
    } else if (c.step === "import") {
        assert.throws(() => n4m.NativeEstimator.fromN4me(bytes(c.n4me_base64)), mentions(c), c.id);
        const control = n4m.NativeEstimator.fromN4me(bytes(c.control_base64));
        const pred = control.predict(xTest).data;
        c.control_predict.forEach((v, i) => assert.ok(Math.abs(pred[i] - v) <= 1e-12 * (1 + Math.abs(v)), c.id));
        control.dispose();
    } else if (c.step === "output_view") {
        checkOutputViews(c);
    } else if (c.step === "export") {
        const est = new Cls(c.params).fit(xTrain, target(c));
        assert.equal(est.containsTrainingRows(), c.contains_training_rows, c.id);
        assert.throws(() => est.toN4me(), mentions(c), c.id);
        assert.equal(est.toN4me({ allowTrainingRows: true }).length, c.export_size_with_opt_in, c.id);
        est.dispose();
    } else {
        assert.equal(c.step, "refit");
        const est = new Cls(c.params).fit(xTrain, c.labels);
        assert.throws(() => est.fit(xTrain, c.refit_labels), mentions(c), c.id);
        assert.deepEqual(est.classes(), c.classes, c.id);
        assert.deepEqual(est.predictLabels(xTest), c.predict_labels, c.id);
        est.dispose();
    }
}

// JS-side conversions name the argument before any native call.
const y = Float64Array.from(fixture.x_train.map((r) => r[0]));
assert.throws(() => new n4m.Ridge().fit(xTrain, y.subarray(0, 20)), /y must have length 40/);
assert.throws(() => new n4m.Ridge().fit({ data: xTrain.data, rows: 40, cols: 7 }, y), /X must be/);
assert.throws(() => new n4m.PLSLDA().fit(xTrain, new Array(40).fill(0.5)), /labels must contain integers/);
assert.throws(() => new n4m.WeightedPLS().fit(xTrain, y, { sampleWeight: new Array(39).fill(1) }),
              /sampleWeight must have length 40/);
const filter = new n4m.YOutlierFilter().fit(xTrain, y);
assert.throws(() => filter.getMask(xTrain, y.subarray(0, 20)), /y must have length 40/);
filter.dispose();
// Seeds are optional: unset runs as seed 0.
const noise = (params) => new n4m.GaussianNoise(params).augment(xTrain).data;
assert.deepEqual(noise({}), noise({ seed: 0 }));

console.log(`estimator roles negative fixture: ${fixture.cases.length} cases refused identically in JS/WASM`);
