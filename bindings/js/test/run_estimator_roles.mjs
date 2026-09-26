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
const checkClassifier = (est, c, tol, label) => {
    assert.deepEqual(est.classes(), c.classes, `${label} classes`);
    assert.deepEqual(est.predictLabels(xTest), c.predict_labels, `${label} labels`);
    close(est.decisionFunction(xTest).data, c.decision_function.flat(), tol, `${label} decision`);
    if (c.predict_proba) {
        close(est.predictProba(xTest).data, c.predict_proba.flat(), tol, `${label} proba`);
    } else {
        assert.equal(typeof est.predictProba, "undefined", `${label} must not define probabilities`);
    }
};
const inputsFor = (names) => ({
    ...(names.includes("feature_groups") ? { featureGroups: fixture.feature_groups } : {}),
    ...(names.includes("blocks") ? { blocks: fixture.blocks } : {}),
    ...(names.includes("axis") ? { axis: fixture.axis } : {}),
    ...(names.includes("X_target") ? { XTarget: matrix(fixture.x_target) } : {}),
});

const classes = Object.values(n4m).filter(
    (c) => typeof c === "function" && c.prototype instanceof n4m.NativeEstimator);
const byMethod = new Map(classes.map((c) => [new c().methodId, c]));
assert.deepEqual([...byMethod.keys()].sort(), fixture.cases.map((c) => c.method_id).sort());

const yTest = Float64Array.from(fixture.y_test);
const checkMask = (est, c, label) => {
    if (c.mask) assert.deepEqual(est.getMask(xTest, yTest), c.mask.map((v) => v === 1), `${label} mask`);
};

for (const c of fixture.cases) {
    const Cls = byMethod.get(c.method_id);
    if (c.n4me_base64 === null) {
        // Train-only filter without a serializable state: refit only.
        const fitted = new Cls(c.params).fit(xTrain, yTrain, inputsFor(c.fit_inputs));
        checkMask(fitted, c, `${c.method_id} JS fit`);
        fitted.dispose();
        continue;
    }
    const payload = Uint8Array.from(Buffer.from(c.n4me_base64, "base64"));
    const est = n4m.NativeEstimator.fromN4me(payload);
    assert.equal(est.methodId, c.method_id);
    if (c.predict) {
        close(est.predict(xTest).data, c.predict, 1e-12, `${c.method_id} predict`);
    } else {
        assert.equal(typeof est.predict, "undefined", `${c.method_id} must not predict`);
    }
    if (c.selected_indices) assert.deepEqual(est.selectedIndices(), c.selected_indices);
    if (c.classes) checkClassifier(est, c, 1e-12, c.method_id);
    checkMask(est, c, c.method_id);
    if (c.transform) {
        close(est.transform(xTest).data, c.transform.flat(), 1e-12, `${c.method_id} transform`);
    } else {
        assert.equal(typeof est.transform, "undefined", `${c.method_id} must not transform`);
    }
    assert.deepEqual(est.toN4me(), payload, `${c.method_id} re-export`);
    est.dispose();

    const target = c.classes ? fixture.labels_train : yTrain;
    const fitted = new Cls(c.params).fit(xTrain, target, inputsFor(c.fit_inputs));
    if (c.classes) checkClassifier(fitted, c, 1e-9, `${c.method_id} JS fit`);
    checkMask(fitted, c, `${c.method_id} JS fit`);
    if (c.predict) close(fitted.predict(xTest).data, c.predict, 1e-9, `${c.method_id} JS fit`);
    if (c.selected_indices) {
        assert.deepEqual(fitted.selectedIndices(), c.selected_indices, `${c.method_id} JS fit`);
    }
    if (c.transform) close(fitted.transform(xTest).data, c.transform.flat(), 1e-9, `${c.method_id} JS fit`);
    fitted.dispose();
}

// Procedures: the same parameters give the Python run's folds, augmented rows
// and named outputs.
const data = {
    x_train: xTrain, x_predictions: matrix(fixture.x_predictions),
};
for (const c of fixture.procedures) {
    const proc = new (n4m.methodClass(c.method_id))(c.params);
    assert.ok(proc instanceof n4m.NativeProcedure, c.method_id);
    const X = data[c.x];
    const y = c.inputs.includes("y") ? yTrain : undefined;
    if (c.folds) {
        const folds = proc.split(X, y, c.inputs.includes("groups") ? fixture.groups : undefined);
        assert.deepEqual(folds.map((f) => [f.train, f.test]), c.folds, `${c.method_id} folds`);
    }
    if (c.X) {
        const axis = c.inputs.includes("axis") ? fixture.axis : undefined;
        close(proc.augment(X, axis).data, c.X.flat(), 1e-12, `${c.method_id} augment`);
    }
    if (c.outputs) {
        const inputs = c.inputs.includes("X_target") ? { XTarget: matrix(fixture.x_target) } : {};
        const out = proc.run(X, y, inputs);
        assert.deepEqual(Object.keys(out).sort(), Object.keys(c.outputs).sort(), c.method_id);
        for (const [name, expected] of Object.entries(c.outputs)) {
            const got = out[name];
            if (typeof expected === "number") close([got], [expected], 1e-9, `${c.method_id}.${name}`);
            else if (Array.isArray(expected[0])) close(got.data, expected.flat(), 1e-9, `${c.method_id}.${name}`);
            else close(got, expected, 1e-9, `${c.method_id}.${name}`);
        }
    }
}
assert.throws(() => n4m.methodClass("models.pls.missing"), /no n4m role class/);
const native = n4m.manifest();
assert.equal(native.methods.length, fixture.cases.length + fixture.procedures.length);
for (const m of native.methods) assert.ok(n4m.methodClass(m.method_id), m.method_id);

assert.throws(() => new n4m.GroupSparsePLS().fit(xTrain, yTrain), /feature_groups/);
assert.throws(() => new n4m.CPPLS().fit(xTrain, yTrain, { groups: new Array(xTrain.rows).fill(1) }),
              /not used/);
assert.throws(() => new n4m.PLSRegression({ solver: "bogus" }).fit(xTrain, yTrain), /solver/);
assert.throws(() => new n4m.PLSLDA().fit(xTrain), /labels/);

console.log(`estimator roles: ${fixture.cases.length} estimators and ${fixture.procedures.length} procedures reproduced in JS/WASM`);
