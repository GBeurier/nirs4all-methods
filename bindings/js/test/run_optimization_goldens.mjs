// SPDX-License-Identifier: CECILL-2.1
// Execute the authoritative 14 HPO specs through JS/WASM and compare the
// native rich trace with the published Python/native golden tapes.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { loadModule, Optimizer } from "../dist/index.js";

const [specsFile, goldenDir] = process.argv.slice(2);
if (!specsFile || !goldenDir) throw new Error("usage: run_optimization_goldens.mjs SPECS.json GOLDEN_DIR");
await loadModule();
const specs = JSON.parse(fs.readFileSync(specsFile, "utf8"));
assert.equal(specs.length, 14);

function searchSpace(spec) {
    return Object.fromEntries(spec.space.map(parameter => {
        const { kind, name, args, length } = parameter;
        if (kind === "int" || kind === "log_int") {
            return [name, { kind, low: args[0], high: args[1], step: args[2] ?? 1 }];
        }
        if (kind === "float" || kind === "log_float") {
            return [name, { kind, low: args[0], high: args[1], step: args[2] ?? 0 }];
        }
        if (kind === "categorical") return [name, { kind, type: "string", choices: args }];
        if (kind === "ordinal") return [name, { kind, choices: args }];
        if (kind === "sorted_tuple") {
            return [name, { kind, length, low: args[0], high: args[1], integer: false }];
        }
        throw new Error(`Unknown HPO spec axis: ${kind}`);
    }));
}

function sphere(parameters) {
    let total = 0;
    for (const [name, value] of Object.entries(parameters)) {
        if (typeof value === "string") total += value ? Math.max(0, value.codePointAt(0) - 97) : 0;
        else if (typeof value === "boolean") total += value ? 0 : 1;
        else {
            const target = name.startsWith("x") ? 2 : 0;
            const delta = Number(value) - target;
            total += delta * delta;
        }
    }
    return total;
}

function intermediate(kind, parameters, step) {
    if (kind === "learning_curve") return sphere(parameters) + 5 / (step + 1);
    if (kind === "racing_observation") {
        return sphere(parameters) + [-0.21, 0.08, 0.17, -0.04, 0.13, -0.19, 0.02, 0.04][step % 8];
    }
    throw new Error(`Unknown intermediate objective: ${kind}`);
}

const STATUS = { running: 0, completed: 1, pruned: 2, failed: 3, cancelled: 4 };
function goldenRecord(record, withIntermediate) {
    const parameters = Object.fromEntries(Object.entries(record.parameters).map(([key, value]) =>
        [key, typeof value === "bigint" ? Number(value) : value]));
    const out = {
        id: Number(record.id),
        params: parameters,
        asked_at: Number(record.askSequence),
        status: STATUS[record.status],
    };
    if (record.terminalSequence !== null) out.terminal_at = Number(record.terminalSequence);
    if (record.status === "completed") out.score = record.score;
    if (withIntermediate) {
        out.intermediates = record.intermediates.map(value => ({
            sequence: Number(value.sequence), step: value.step,
            score: value.score, should_prune: value.shouldPrune,
        }));
        out.pruned_at = record.intermediates.find(value => value.shouldPrune)?.step ?? -1;
    }
    if (record.error) {
        out.error = {
            category: "objective", code: record.error.code,
            details: { phase: "evaluate", trial_id: Number(record.id) },
            message: record.error.message, retryable: record.error.retryable,
        };
    }
    return out;
}

for (const spec of specs) {
    const golden = JSON.parse(fs.readFileSync(path.join(goldenDir, `${spec.id}.json`), "utf8"));
    let optimizer = Optimizer.create(searchSpace(spec), {
        sampler: spec.sampler, pruner: spec.pruner, direction: spec.direction,
        startupTrials: spec.n_startup_trials, seed: spec.seed,
        maxResource: spec.max_resource, reductionFactor: spec.reduction_factor,
    });
    try {
        const batchSize = spec.max_in_flight;
        for (let start = 0; start < spec.n_trials; start += batchSize) {
            const size = Math.min(batchSize, spec.n_trials - start);
            const batch = Array.from({ length: size }, () => optimizer.ask());
            for (const trial of batch) {
                const expected = golden[Number(trial.id)];
                const params = Object.fromEntries(Object.entries(trial.parameters).map(([name, value]) =>
                    [name, typeof value === "bigint" ? Number(value) : value]));
                assert.deepEqual(params, expected.params, `${spec.id} trial ${trial.id} proposal`);
            }
            const order = (spec.tell_order.length ? spec.tell_order : Array.from({ length: size }, (_, i) => i))
                .filter(index => index < size);
            const pruned = new Set();
            if (spec.intermediate) {
                for (let step = 0; step < spec.intermediate_steps; step += 1) {
                    for (const position of order) {
                        const trial = batch[position];
                        if (pruned.has(position)) continue;
                        const expected = golden[Number(trial.id)].intermediates.find(value => value.step === step);
                        assert.ok(expected, `${spec.id} trial ${trial.id} missing step ${step}`);
                        const computed = intermediate(spec.intermediate, trial.parameters, step);
                        assert.ok(Math.abs(computed - expected.score) <=
                            1e-12 * Math.max(1, Math.abs(expected.score)),
                            `${spec.id} trial ${trial.id} intermediate objective drift`);
                        const shouldPrune = optimizer.intermediate(trial, step, expected.score);
                        assert.equal(shouldPrune, expected.should_prune,
                            `${spec.id} trial ${trial.id} prune decision`);
                        if (shouldPrune) pruned.add(position);
                    }
                }
            }
            for (const position of order) {
                const trial = batch[position];
                if (pruned.has(position)) continue;
                if (spec.failed_trial_ids.includes(Number(trial.id))) {
                    optimizer.tell(trial, "failed", undefined,
                        "n4m.error.v1|OBJECTIVE_EVALUATION_FAILED|0|deterministic objective failure fixture");
                } else {
                    const expected = golden[Number(trial.id)];
                    const computed = sphere(trial.parameters);
                    assert.ok(Math.abs(computed - expected.score) <=
                        1e-12 * Math.max(1, Math.abs(expected.score)),
                        `${spec.id} trial ${trial.id} terminal objective drift`);
                    // Fixed native score tape: Python's `**2` and JS multiplication
                    // can differ by one ULP although proposals are bit-identical.
                    optimizer.tell(trial, "completed", expected.score);
                }
            }
        }
        const actual = optimizer.trialRecords().map(record => goldenRecord(record, Boolean(spec.intermediate)));
        assert.deepEqual(actual, golden, `${spec.id} native JS/WASM trace diverges from golden`);
        console.log(`HPO JS/WASM golden: ${spec.id} (${actual.length} trials)`);
    } finally {
        optimizer.dispose();
    }
}
console.log("All 14 HPO JS/WASM golden traces match Python/native");
