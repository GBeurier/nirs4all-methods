// SPDX-License-Identifier: CECILL-2.1
// Native role pipeline (ABI 2.14): the shared fixture written by the Python
// binding (also replayed by R and Rust). Its pipelines predict identically in
// JS/WASM, JS refits reproduce the Python fits, and the negative cases are
// refused with the same native status and message.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as n4m from "../dist/index.js";

await n4m.loadModule();
const fx = JSON.parse(readFileSync(
    new URL("../../../parity/fixtures/role_pipeline_negative.json", import.meta.url), "utf-8"));

const matrix = (rows) => ({
    data: Float64Array.from(rows.flat()), rows: rows.length, cols: rows[0].length,
});
const decode = (b64) => Uint8Array.from(Buffer.from(b64, "base64"));
// Fixture outputs come from Linux x86-64; WASM kernels may differ by a few ulps.
const REPLAY_TOL = 1e-9;
const close = (actual, expected, tol, label) => {
    assert.equal(actual.length, expected.length, label);
    for (let i = 0; i < expected.length; ++i) {
        assert.ok(Math.abs(actual[i] - expected[i]) <= tol * (1 + Math.abs(expected[i])),
                  `${label}[${i}]: ${actual[i]} vs ${expected[i]}`);
    }
};

const xTrain = matrix(fx.x_train);
const xTest = matrix(fx.x_test);
const names = fx.feature_names;

// Positive regression pipeline: filter on y, SNV, PLS scores, ridge.
{
    const reg = fx.regression;
    const pipeline = n4m.RolePipeline.fromStates(reg.steps, reg.states.map((s) => decode(s.n4me_base64)),
                                                 { featureNames: names });
    close(pipeline.predict(xTest, names).data, reg.predict, REPLAY_TOL, "regression predict");
    close(pipeline.transform(xTest).data, reg.transform.flat(), REPLAY_TOL, "regression transform");
    const exported = pipeline.exportStates();
    assert.deepEqual(exported.map((s) => s.methodId), reg.states.map((s) => s.method_id));
    exported.forEach((s, i) => assert.deepEqual(s.n4me, decode(reg.states[i].n4me_base64), `state ${i} bytes`));
    assert.deepEqual(pipeline.stepsInfo().map((s) => s.role),
                     ["sample_filter", "transformer", "transformer", "regressor"]);
    const refit = n4m.RolePipeline.fromSteps(reg.steps).fit(xTrain, fx.y_train, { featureNames: names });
    close(refit.predict(xTest).data, reg.predict, REPLAY_TOL, "regression refit");
    pipeline.dispose();
    refit.dispose();
}

// Positive classification pipeline with string labels.
{
    const cls = fx.classification;
    const pipeline = n4m.RolePipeline.fromStates(cls.steps, cls.states.map((s) => decode(s.n4me_base64)),
                                                 { featureNames: names, classNames: cls.class_names });
    assert.deepEqual(pipeline.predictLabels(xTest), cls.predict);
    assert.deepEqual(pipeline.classes(), cls.class_names);
    assert.deepEqual(pipeline.labelNames(), cls.class_names);
    close(pipeline.decisionFunction(xTest).data, cls.decision_function.flat(), REPLAY_TOL, "decision");
    assert.throws(() => pipeline.predict(xTest), (e) => e.status === n4m.Status.ERR_UNSUPPORTED);
    const refit = n4m.RolePipeline.fromSteps(cls.steps).fit(xTrain, fx.labels_train);
    assert.deepEqual(refit.predictLabels(xTest), cls.predict);
    pipeline.dispose();
    refit.dispose();
}

// Negative cases (and the multi-target fit).
const fitted = n4m.RolePipeline.fromStates(fx.regression.steps,
    fx.regression.states.map((s) => decode(s.n4me_base64)), { featureNames: names });
const target = (key) => (Array.isArray(fx[key][0]) ? matrix(fx[key]) : fx[key]);
for (const c of fx.cases) {
    if (c.stage === "fit") {
        const pipeline = n4m.RolePipeline.fromSteps(c.steps).fit(xTrain, target(c.y));
        close(pipeline.predict(xTest).data, c.predict.flat(), REPLAY_TOL, c.name);
        pipeline.dispose();
        continue;
    }
    const run = {
        create: () => n4m.RolePipeline.fromSteps(c.steps),
        import: () => n4m.RolePipeline.fromStates(c.steps, c.states.map(decode)),
        predict: () => {
            const columns = c.drop_last_column ? names.slice(0, -1) : c.feature_names;
            const X = matrix(fx.x_test.map((row) => row.slice(0, columns.length)));
            return fitted.predict(X, columns);
        },
        export: () => n4m.RolePipeline.fromSteps(c.steps).fit(xTrain, target(c.y)).exportStates(),
    }[c.stage];
    assert.throws(run, (e) => e.status === c.status && e.message.includes(c.message),
                  `${c.name}: expected status ${c.status} with '${c.message}'`);
}
fitted.dispose();

// Label tables and labels (shared label contract): refused with the fixture
// message, no native status. JSON has no NaN / infinity: `inject` adds one.
const injected = (values, c) => {
    const out = [...values];
    if (c.inject) out[c.inject.at] = c.inject.value === "inf" ? Infinity : NaN;
    return out;
};
for (const c of fx.label_cases) {
    if (c.stage === "import") {
        const source = fx[c.pipeline];
        const load = () => n4m.RolePipeline.fromStates(source.steps,
            source.states.map((s) => decode(s.n4me_base64)), { classNames: injected(c.class_names, c) });
        if (c.accept) {
            const pipeline = load();
            assert.deepEqual(pipeline.predictLabels(xTest), source.predict, c.name);
            pipeline.dispose();
            continue;
        }
        assert.throws(load, (e) => e.message.includes(c.message), `${c.name}: expected '${c.message}'`);
        continue;
    }
    assert.throws(() => n4m.RolePipeline.fromSteps(fx.classification.steps).fit(xTrain, injected(c.labels, c)),
                  (e) => e.message.includes(c.message), `${c.name}: expected '${c.message}'`);
}
// Numbers that are not integers become a label table, as in Python and R.
{
    const labels = fx.labels_train.map((v) => ({ high: 2.5, low: 0.5, mid: 1.5 })[v]);
    const pipeline = n4m.RolePipeline.fromSteps(fx.classification.steps).fit(xTrain, labels);
    assert.deepEqual(pipeline.labelNames(), [0.5, 1.5, 2.5]);
    const restored = n4m.RolePipeline.fromStates(fx.classification.steps, pipeline.exportStates(),
                                                 { classNames: pipeline.labelNames() });
    assert.deepEqual(restored.predictLabels(xTest), pipeline.predictLabels(xTest));
    pipeline.dispose();
    restored.dispose();
}

// Column names holding NUL are refused before a C string would truncate them.
{
    const reg = fx.regression;
    const states = reg.states.map((s) => decode(s.n4me_base64));
    const imported = n4m.RolePipeline.fromStates(reg.steps, states, { featureNames: names });
    for (const c of fx.name_cases) {
        const run = {
            fit: () => n4m.RolePipeline.fromSteps(c.steps).fit(xTrain, fx[c.y], { featureNames: c.feature_names }),
            import: () => n4m.RolePipeline.fromStates(reg.steps, states, { featureNames: c.feature_names }),
            predict: () => imported.predict(xTest, c.feature_names),
        }[c.stage];
        assert.throws(run, (e) => e.message.includes(c.message), `${c.name}: expected '${c.message}'`);
    }
    imported.dispose();
}

// Zero new rows: an empty output of the right width, whatever the steps.
for (const steps of [["models.regularized.ridge"], ["preprocessing.scatter.snv", "models.regularized.ridge"]]) {
    const pipeline = n4m.RolePipeline.fromSteps(steps).fit(xTrain, fx.y_train);
    const empty = { data: new Float64Array(0), rows: 0, cols: xTrain.cols };
    const out = pipeline.predict(empty);
    assert.equal(out.rows, 0);
    assert.equal(out.data.length, 0);
    pipeline.dispose();
}

// Training rows need an explicit opt-in; an unused input is refused.
{
    const pipeline = n4m.RolePipeline.fromSteps(["preprocessing.scatter.snv", "models.pls.kernel"])
        .fit(xTrain, fx.y_train);
    assert.deepEqual(pipeline.stepsInfo().map((s) => s.containsTrainingRows), [false, true]);
    assert.throws(() => pipeline.exportStates(), /state retains training rows/);
    assert.equal(pipeline.exportStates({ allowTrainingRows: true }).length, 2);
    pipeline.dispose();
    assert.throws(() => n4m.RolePipeline.fromSteps(["models.regularized.ridge"])
        .fit(xTrain, fx.y_train, { groups: fx.y_train.map((_, i) => i) }),
    /not used by any step of the pipeline 'groups'/);
}

console.log(`role pipeline: ${fx.cases.length} shared cases, ${fx.label_cases.length} label cases, ` +
            `${fx.name_cases.length} name cases, 2 cross-language pipelines OK`);
