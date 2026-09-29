// SPDX-License-Identifier: CECILL-2.1
import assert from "node:assert/strict";
import { loadModule, Optimizer } from "../dist/index.js";

await loadModule();

const space = {
    components: { kind: "int", low: 1n, high: 6n },
    rate: { kind: "log_float", low: 0.001, high: 0.1 },
    mode: { kind: "categorical", type: "string", choices: ["simple", "advanced"] },
    penalty: { kind: "float", low: 0, high: 1 },
    family: { kind: "categorical", type: "integer", choices: [1n, 2n, 4294967299n] },
    approved: { kind: "categorical", type: "boolean", choices: [false, true] },
    cutoff: { kind: "categorical", type: "float", choices: [0.25, 0.5] },
    window: { kind: "log_int", low: 1n, high: 8n },
    selected: { kind: "sorted_tuple", length: 2, low: 0, high: 8, integer: true },
    priority: { kind: "ordinal", choices: [0.2, 0.5, 0.8] },
};
const constraints = [{
    kind: "condition_in", refs: ["penalty", "mode"], labels: [null, "advanced"],
}];
const options = { sampler: "random", seed: 38n, metric: "rmse" };
const optimizer = Optimizer.create(space, options, constraints);
let restored;
let trace;
try {
    assert.deepEqual(optimizer.trialRecords(), []);
    const first = optimizer.ask();
    assert.equal(first.id, 0n);
    assert.equal(typeof first.parameters.components, "bigint");
    assert.equal(typeof first.parameters.rate, "number");
    assert.equal(typeof first.parameters.family, "bigint");
    assert.equal(first.parameters.selected.length, 2);
    assert.equal(Object.hasOwn(first.parameters, "penalty"), first.parameters.mode === "advanced");
    assert.equal(optimizer.intermediate(first, 1, 0.8), false);
    optimizer.tell(first, "completed", 0.8);

    const batch = optimizer.askBatch(2);
    assert.equal(batch.nativeStatus, 0);
    assert.deepEqual(batch.trials.map(trial => trial.id), [1n, 2n]);
    optimizer.tell(batch.trials[0], "failed", undefined, "SYNTHETIC_FAILURE");
    optimizer.tell(batch.trials[1], "completed", 0.3);
    assert.equal(optimizer.best().trial.id, 2n);
    assert.equal(optimizer.best().score, 0.3);
    const records = optimizer.trialRecords();
    assert.deepEqual(records.map(record => record.status), ["completed", "failed", "completed"]);
    assert.deepEqual(records.map(record => record.id), [0n, 1n, 2n]);
    assert.equal(records[0].intermediates[0].score, 0.8);
    assert.equal(records[1].error.code, "OBJECTIVE_ERROR");
    assert.equal(records[1].error.message, "SYNTHETIC_FAILURE");
    assert.equal(typeof records[0].parameters.family, "bigint");
    assert.equal(typeof records[0].parameters.approved, "boolean");
    assert.equal(typeof records[0].parameters.cutoff, "number");
    assert.equal(typeof records[0].parameters.window, "bigint");
    assert.equal(Object.hasOwn(records[0].parameters, "penalty"), records[0].parameters.mode === "advanced");
    assert.equal(records[0].parameters["selected#0"] <= records[0].parameters["selected#1"], true);
    assert.equal(optimizer.trialRecords(2n).length, 1);

    trace = optimizer.trials();
    assert.deepEqual(Array.from(trace.vectorInt64("trial_ids_i64")), [0n, 1n, 2n]);
    assert.deepEqual(Array.from(trace.matrix("trial_status").data), [1, 3, 1]);
    trace.destroy();
    trace = undefined;

    const checkpoint = optimizer.save();
    assert.ok(checkpoint.length > 64);
    restored = Optimizer.load(checkpoint, space);
    const next = optimizer.ask();
    const resumed = restored.ask();
    assert.deepEqual(resumed, next);
    optimizer.tell(next, "completed", 0.4);
    restored.tell(resumed, "completed", 0.4);
    assert.deepEqual(restored.best(), optimizer.best());
    // Wall-clock durations are checkpointed and can differ across the two
    // live handles; compare the native decision stream instead of raw bytes.
    const originalTrace = optimizer.trials();
    const resumedTrace = restored.trials();
    try {
        assert.deepEqual(resumedTrace.vectorInt64("trial_ids_i64"),
                         originalTrace.vectorInt64("trial_ids_i64"));
        assert.deepEqual(resumedTrace.matrix("trial_scores").data,
                         originalTrace.matrix("trial_scores").data);
    } finally {
        resumedTrace.destroy();
        originalTrace.destroy();
    }

    const damaged = checkpoint.slice();
    damaged[damaged.length - 1] ^= 1;
    assert.throws(() => Optimizer.load(damaged, space));
    assert.throws(() => Optimizer.create({ broken: { kind: "int", low: 2 ** 54, high: 2 ** 55 } }));
    assert.throws(() => optimizer.askBatch(10001));
    assert.deepEqual(optimizer.askBatch(0), { trials: [], nativeStatus: 0 });
} finally {
    trace?.destroy();
    restored?.dispose();
    optimizer.dispose();
}
assert.throws(() => optimizer.ask());
optimizer.dispose();

const warm = Optimizer.create({ x: { kind: "int", low: 1n, high: 4n } }, { seed: 7n });
try {
    warm.enqueue({ x: 3n });
    const trial = warm.ask();
    assert.equal(trial.parameters.x, 3n);
    warm.tell(trial, "completed", 0.2);
} finally { warm.dispose(); }

const population = Optimizer.create({
    x: { kind: "float", low: -2, high: 2 },
    y: { kind: "float", low: -2, high: 2 },
}, { sampler: "ga", seed: 5n });
try {
    const firstGeneration = population.askBatch(17);
    assert.equal(firstGeneration.nativeStatus, 0);
    assert.equal(firstGeneration.trials.length, 16);
    for (const trial of firstGeneration.trials) {
        population.tell(trial, "completed", trial.parameters.x ** 2 + trial.parameters.y ** 2);
    }
    assert.equal(population.ask().id, 16n);
} finally { population.dispose(); }

console.log("JS/WASM native optimizer lifecycle and N4MOPT continuation OK");
