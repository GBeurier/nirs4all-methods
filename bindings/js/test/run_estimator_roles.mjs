// SPDX-License-Identifier: CECILL-2.1
// Generic estimator roles: the shared N4ME fixture (written by the Python
// binding, also replayed by R) predicts identically in JS/WASM, and JS fits
// reproduce the Python fits.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as n4m from "../dist/index.js";

await n4m.loadModule();
const fixture = JSON.parse(readFileSync(
    new URL("../../../parity/fixtures/estimator_roles_n4me.json", import.meta.url), "utf-8"));

const matrix = (rows) => ({
    data: Float64Array.from(rows.flat()), rows: rows.length, cols: rows[0].length,
});
const close = (actual, expected, tol, label) => {
    assert.equal(actual.length, expected.length, label);
    for (let i = 0; i < expected.length; ++i) {
        const scale = 1 + Math.abs(expected[i]);
        assert.ok(Math.abs(actual[i] - expected[i]) <= tol * scale,
                  `${label}[${i}]: ${actual[i]} vs ${expected[i]}`);
    }
};

const xTest = matrix(fixture.x_test);
const xTrain = matrix(fixture.x_train);
const yTrain = Float64Array.from(fixture.y_train);
const inputsFor = (names) => ({
    ...(names.includes("feature_groups") ? { featureGroups: fixture.feature_groups } : {}),
    ...(names.includes("blocks") ? { blocks: fixture.blocks } : {}),
    ...(names.includes("X_target") ? { XTarget: matrix(fixture.x_target) } : {}),
});

const classes = Object.values(n4m).filter(
    (c) => typeof c === "function" && c.prototype instanceof n4m.NativeEstimator);
const byMethod = new Map(classes.map((c) => [new c().methodId, c]));
assert.deepEqual([...byMethod.keys()].sort(), fixture.cases.map((c) => c.method_id).sort());

for (const c of fixture.cases) {
    const payload = Uint8Array.from(Buffer.from(c.n4me_base64, "base64"));
    const est = n4m.NativeEstimator.fromN4me(payload);
    assert.equal(est.methodId, c.method_id);
    if (c.predict) {
        close(est.predict(xTest).data, c.predict, 1e-12, `${c.method_id} predict`);
    } else {
        assert.equal(typeof est.predict, "undefined", `${c.method_id} must not predict`);
    }
    if (c.selected_indices) assert.deepEqual(est.selectedIndices(), c.selected_indices);
    if (c.transform) {
        close(est.transform(xTest).data, c.transform.flat(), 1e-12, `${c.method_id} transform`);
    } else {
        assert.equal(typeof est.transform, "undefined", `${c.method_id} must not transform`);
    }
    assert.deepEqual(est.toN4me(), payload, `${c.method_id} re-export`);
    est.dispose();

    const Cls = byMethod.get(c.method_id);
    const fitted = new Cls(c.params).fit(xTrain, yTrain, inputsFor(c.fit_inputs));
    if (c.predict) close(fitted.predict(xTest).data, c.predict, 1e-9, `${c.method_id} JS fit`);
    if (c.selected_indices) {
        assert.deepEqual(fitted.selectedIndices(), c.selected_indices, `${c.method_id} JS fit`);
    }
    if (c.transform) close(fitted.transform(xTest).data, c.transform.flat(), 1e-9, `${c.method_id} JS fit`);
    fitted.dispose();
}

assert.throws(() => new n4m.GroupSparsePLS().fit(xTrain, yTrain), /feature_groups/);
assert.throws(() => new n4m.CPPLS().fit(xTrain, yTrain, { groups: new Array(xTrain.rows).fill(1) }),
              /not used/);
assert.throws(() => new n4m.PLSRegression({ solver: "bogus" }).fit(xTrain, yTrain), /solver/);

console.log(`estimator roles: ${fixture.cases.length} N4ME states replayed and refitted in JS/WASM`);
