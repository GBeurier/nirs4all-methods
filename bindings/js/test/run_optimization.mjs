// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
import assert from "node:assert/strict";
import { loadModule, Optimizer } from "../dist/index.js";

const t4fModule = await loadModule();

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

// Log-domain proposals must remain valid when checkpointed at either stage.
for (const [low, high] of [[1e-5, 1e-2], [5, 5000], [7, 7000], [50, 50000],
    [1000, 1e6], [1e18, 9e18], [2e19, 9e20], [1e21, 9e21], [1e15, 1e16],
    [0.5, 2.5], [1, 1000]]) {
    const logSpace = { scale: { kind: "log_float", low, high }, aux: { kind: "float", low: 0, high: 1 } };
    const original = Optimizer.create(logSpace, { sampler: "sobol", seed: 19n });
    try {
        const first = original.ask();
        assert.ok(Number.isFinite(first.parameters.scale) && first.parameters.scale >= low && first.parameters.scale <= high);
        const running = Optimizer.load(original.save(), logSpace);
        try {
            assert.deepEqual(running.trialRecords().map(record => record.status), ["running"]);
            original.tell(first, "completed", 0.5);
            running.tell(first, "completed", 0.5);
            assert.deepEqual(original.trialRecords().map(record => record.status), ["completed"]);
            const completed = Optimizer.load(original.save(), logSpace);
            try {
                assert.deepEqual(completed.trialRecords().map(record => record.status), ["completed"]);
                const next = original.ask();
                const fromRunning = running.ask();
                const fromCompleted = completed.ask();
                assert.deepEqual(fromRunning, next);
                assert.deepEqual(fromCompleted, next);
                const floatBits = value => {
                    const view = new DataView(new ArrayBuffer(8));
                    view.setFloat64(0, value);
                    return view.getBigUint64(0);
                };
                for (const restored of [fromRunning, fromCompleted]) {
                    assert.equal(restored.id, next.id);
                    for (const name of ["scale", "aux"]) {
                        assert.equal(floatBits(restored.parameters[name]), floatBits(next.parameters[name]));
                    }
                }
            } finally { completed.dispose(); }
        } finally { running.dispose(); }
    } finally { original.dispose(); }
}
function checkStagedContinuation(original, space, trial, score) {
    const running = Optimizer.load(original.save(), space);
    try {
        const statuses = running.trialRecords().map(record => record.status);
        assert.equal(statuses.at(-1), "running");
        assert.ok(statuses.slice(0, -1).every(status => status === "completed"));
        original.tell(trial, "completed", score);
        running.tell(trial.id, "completed", score);
        const completed = Optimizer.load(original.save(), space);
        try {
            assert.ok(completed.trialRecords().every(record => record.status === "completed"));
            const expected = original.ask();
            const view = new DataView(new ArrayBuffer(8));
            const bits = value => { view.setFloat64(0, value); return view.getBigUint64(0); };
            for (const copy of [running, completed]) {
                const actual = copy.ask();
                assert.equal(actual.id, expected.id);
                for (const name of Object.keys(expected.parameters)) {
                    const value = expected.parameters[name];
                    if (typeof value === "number") {
                        assert.equal(bits(actual.parameters[name]), bits(value));
                    } else {
                        assert.equal(actual.parameters[name], value);
                    }
                }
            }
        } finally { completed.dispose(); }
    } finally { running.dispose(); }
}
for (const [low, high] of [[1e15, 1e16], [2e19, 9e20]]) {
    for (const sampler of ["cmaes", "tpe"]) {
        const logSpace = { scale: { kind: "log_float", low, high }, aux: { kind: "float", low: 0, high: 1 } };
        const original = Optimizer.create(logSpace, { sampler, seed: 19n, direction: "minimize" });
        try {
            for (let i = 0; i < 60; i++) {
                const trial = original.ask();
                const value = trial.parameters.scale;
                assert.ok(Number.isFinite(value) && value >= low && value <= high);
                if (i === 59) checkStagedContinuation(original, logSpace, trial, -Math.log(value));
                else original.tell(trial, "completed", -Math.log(value));
            }
        } finally { original.dispose(); }
    }
}
{
    const logSpace = { scale: { kind: "log_int", low: 1000000000000000n, high: 9007199254740991n } };
    const original = Optimizer.create(logSpace, { sampler: "sobol", seed: 19n });
    try {
        const trial = original.ask();
        const value = trial.parameters.scale;
        assert.ok(value >= logSpace.scale.low && value <= logSpace.scale.high);
        checkStagedContinuation(original, logSpace, trial, 0.5);
    } finally { original.dispose(); }
}

// Genuine pre-fix EMPTY checkpoints must be rejected after span validation.
{
    const checkpoint = Uint8Array.from(Buffer.from("4e344d4f50540d0a0100000020000000e800000000000000c0000000000000002a4561ca5cd613b13c0000000000000001000000010000007801000000763b7730d142eeff763b7730d142ee7f000000000000000000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000eb9af1ffa822dc3e00000000000000000000000000000000000000000000000000000000e41aeb13812ec382", "hex"));
    assert.throws(() => Optimizer.load(checkpoint, { x: { kind: 'float', low: -1.7e308, high: 1.7e308 } }), /n4m error 14:.*optimizer checkpoint option\/search-space contract mismatch/);
}
{
    const checkpoint = Uint8Array.from(Buffer.from("4e344d4f50540d0a0100000020000000e800000000000000c00000000000000061667f6d1e6e72713c0000000000000001000000010000007806000000763b7730d142eeff763b7730d142ee7f000000000000000000000000000000000000000000000200000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000fb3f8cb2ac6bdc3e000000000000000000000000000000000000000000000000000000008f9e37096cfc04b1", "hex"));
    assert.throws(() => Optimizer.load(checkpoint, { x: { kind: 'sorted_tuple', length: 2, low: -1.7e308, high: 1.7e308 } }), /n4m error 14:.*optimizer checkpoint option\/search-space contract mismatch/);
}

for (const axis of [
    { kind: "float", low: -1.7e308, high: 1.7e308 },
    { kind: "sorted_tuple", length: 2, low: -1.7e308, high: 1.7e308 },
]) {
    assert.throws(() => Optimizer.create({ x: axis }, { seed: 19n }));
}

// Each stage is captured from a fresh live optimizer, including mid-run states.
function assertNativeConfigurationMatches(left, right) {
    const out = t4fModule._malloc(4);
    try {
        assert.equal(t4fModule.ccall("n4m_optimizer_configuration_matches", "number",
            ["number", "number", "number"], [left._ptr, right._ptr, out]), 0);
        assert.equal(t4fModule.getValue(out, "i32"), 1);
    } finally { t4fModule._free(out); }
}
function domainStages(space, sampler, seed, validate) {
    for (const [count, running] of [[0, false], [1, true], [1, false], [12, true], [12, false], [13, true]]) {
        const original = Optimizer.create(space, { sampler, seed: BigInt(seed), direction: "minimize" });
        try {
            let pending;
            for (let i = 0; i < count; ++i) {
                pending = original.ask();
                validate(pending);
                if (!(running && i === count - 1)) original.tell(pending, "completed", (i % 7) / 7);
            }
            const captured = original.trialRecords();
            const restored = Optimizer.load(original.save(), space);
            try {
                assertNativeConfigurationMatches(original, restored);
                assert.deepEqual(restored.trialRecords(), captured);
                if (running) {
                    original.tell(pending, "completed", 0.5);
                    restored.tell(pending.id, "completed", 0.5);
                }
                for (let i = 0; i < 6; ++i) {
                    const a = original.ask(), b = restored.ask();
                    validate(a); validate(b);
                    assert.equal(a.id, b.id);
                    assert.deepEqual(a.parameters, b.parameters);
                    original.tell(a, "completed", (i % 7) / 7);
                    restored.tell(b, "completed", (i % 7) / 7);
                }
            } finally { restored.dispose(); }
        } finally { original.dispose(); }
    }
}
const domainView = new DataView(new ArrayBuffer(8));
function domainNext(value, count = 1n) {
    domainView.setFloat64(0, value);
    domainView.setBigUint64(0, domainView.getBigUint64(0) + count);
    return domainView.getFloat64(0);
}
const logCases = [
    ["log_float", 0.1, domainNext(0.1)], ["log_float", 1e300, 1e300 * (1 + 1e-14)],
    ["log_float", 1e-5, domainNext(1e-5, 2n)],
    ["log_int", 1000000000000000n, 1000000000000001n],
    ["log_int", 1000000000000000n, 1000000000000002n],
    ["log_int", 9007199254740990n, 9007199254740992n],
    ["log_int", 1000000000000000n, 1000000000000003n],
    ["log_float", 1, domainNext(1)], ["log_float", 5, 5],
    ["log_float", 0.5, 2.5], ["log_float", 1, 1000],
];
for (const [kind, low, high] of logCases) for (const sampler of ["gp_ei", "tpe"]) for (const seed of [19, 3]) {
    const sp = { x: { kind, low, high }, aux: { kind: "float", low: 0, high: 1 } };
    domainStages(sp, sampler, seed, trial => {
        assert.ok(Number.isFinite(Number(trial.parameters.x)) && trial.parameters.x >= low && trial.parameters.x <= high);
        assert.ok(trial.parameters.aux >= 0 && trial.parameters.aux <= 1);
    });
}
const domainP = 2 ** 52, domainQ = 2 ** 53;
const tupleCases = [[0,10],[-5,5],[1000,1002],[-1000,-998],[2**40,2**40+2],
    [2**50,2**50+2],[2**51,2**51+2],[-(2**51)-2,-(2**51)],
    [domainP,domainP],[domainP,domainP+1],[domainP,domainP+2],[domainP+1000,domainP+1003],
    [domainQ-3,domainQ-1],[domainQ-2,domainQ],[-domainQ,-domainQ+2],[-domainQ,-domainQ],
    [-domainP-2,-domainP],[-domainQ,domainQ],[-domainQ+1,domainQ-1]];
for (const [low, high] of tupleCases) for (const sampler of ["random","sobol","lhs","ternary","ga","pso","cmaes","tpe","gp_ei"]) for (const length of [1, 3]) {
    const sp = { x: { kind: "sorted_tuple", length, low, high, integer: true }, aux: { kind: "float", low: 0, high: 1 } };
    domainStages(sp, sampler, 19, trial => {
        const values = Array.isArray(trial.parameters.x) ? trial.parameters.x : Array.from({ length }, (_, j) => trial.parameters[`x#${j}`]);
        assert.equal(values.length, length);
        const numbers = values.map(Number);
        assert.ok(numbers.every((v, j) => Number.isFinite(v) && Number.isInteger(v) && v >= low && v <= high && (j === 0 || numbers[j-1] <= v)));
    });
}
const spanPower = 2 ** 1023, spanMax = Number.MAX_VALUE;
const spanBad = [[-1.7e308,1.7e308],[-spanMax,spanMax],[-spanPower,spanMax/2],[-spanPower,spanPower]];
for (const [low, high] of spanBad) for (const tuple of [false, true]) for (const aux of [false, true]) {
    const x = tuple ? {kind: "sorted_tuple", length: 2, low, high} : {kind: "float", low, high};
    const sp = aux ? {x, aux: {kind: "float", low: 0, high: 1}} : {x};
    assert.throws(() => Optimizer.create(sp, {seed: 19n}), error => error.status === 1);
}
for (const [low, high] of [[-spanMax/2,spanMax/2],[0,spanMax],[-8e307,8e307],[-spanPower,domainNext(spanMax/2,-1n)]]) for (const tuple of [false,true]) {
    const x = tuple ? {kind: "sorted_tuple", length: 2, low, high} : {kind: "float", low, high};
    const sp = {x}; const opt = Optimizer.create(sp, {seed: 19n});
    try { const restored = Optimizer.load(opt.save(), sp); restored.dispose(); } finally { opt.dispose(); }
}
for (const sampler of ["random", "tpe", "gp_ei"]) {
    const low = 1000000000000000n, high = low + 2n;
    const sp = {x: {kind: "log_int", low, high}, aux: {kind: "float", low: 0, high: 1}};
    const opt = Optimizer.create(sp, {sampler, seed: 19n});
    try {
        if (sampler === "gp_ei") {
            assert.throws(() => opt.enqueue({x: low, aux: 0.25}), error => error.status === 10);
        } else {
            for (const x of [low, high, low + 1n]) opt.enqueue({x, aux: 0.25});
            for (const x of [low, high, low + 1n]) {
                const trial = opt.ask(); assert.equal(trial.parameters.x, x); opt.tell(trial, "completed", 0.5);
            }
        }
    } finally { opt.dispose(); }
}
for (const [low, high] of [[-spanMax/2,spanMax/2],[0,spanMax],[-8e307,8e307],[-spanPower,domainNext(spanMax/2,-1n)]])
    for (const tuple of [false,true]) for (const sampler of ["random","sobol","lhs","ternary","ga","pso","cmaes","tpe","gp_ei"]) {
        const axis = tuple ? {kind: "sorted_tuple", length: 2, low, high} : {kind: "float", low, high};
        const sp = {x: axis, aux: {kind: "float", low: 0, high: 1}};
        domainStages(sp, sampler, 19, trial => {
            const values = tuple ? trial.parameters.x : [trial.parameters.x];
            assert.ok(Array.isArray(values));
            assert.ok(values.every((value, index) => Number.isFinite(value) && value >= low && value <= high && (index === 0 || values[index-1] <= value)));
        });
    }
{
    const low = 2 ** 52, high = low + 1, count = 6000;
    function stream(tupleLength, asks) {
        const axis = tupleLength ? {kind: "sorted_tuple", length: tupleLength, low, high, integer: true}
            : {kind: "int", low: BigInt(low), high: BigInt(high)};
        const opt = Optimizer.create({x: axis}, {sampler: "random", seed: 19n});
        try {
            const result = [];
            for (let i = 0; i < asks; ++i) {
                const trial = opt.ask(); result.push(trial.parameters.x); opt.tell(trial, "completed", 0);
            }
            return result;
        } finally { opt.dispose(); }
    }
    const scalar = stream(0, count), singles = stream(1, count), triples = stream(3, count/3);
    assert.deepEqual(singles.map(v => v[0]), scalar);
    assert.deepEqual(triples, Array.from({length: count/3}, (_, i) => scalar.slice(3*i, 3*i+3).sort((a,b) => a < b ? -1 : a > b ? 1 : 0)));
    assert.ok(Math.abs(singles.filter(v => v[0] === BigInt(high)).length / count - 0.5) <= 0.03);
}
{
    const space = {x: {kind: "float", low: 0, high: 1}};
    const opt = Optimizer.create(space, {sampler: "gp_ei", seed: 19n});
    try {
        const trial = opt.ask(), raw = opt.save(), dv = new DataView(new ArrayBuffer(8));
        dv.setFloat64(0, trial.parameters.x, true);
        const pattern = new Uint8Array(dv.buffer);
        const hits = [];
        for (let i = 0; i + 8 <= raw.length - 8; ++i) if (pattern.every((v, j) => raw[i+j] === v)) hits.push(i);
        assert.equal(hits.length, 2);
        for (const bad of [NaN, -0.1, 1.5]) {
            const bytes = raw.slice(); dv.setFloat64(0, bad, true); bytes.set(new Uint8Array(dv.buffer), hits[1]);
            let hash = 14695981039346656037n;
            for (const byte of bytes.subarray(0, bytes.length - 8)) hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 1099511628211n);
            new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).setBigUint64(bytes.length - 8, hash, true);
            assert.throws(() => Optimizer.load(bytes, space), error => error.status === 14);
        }
    } finally { opt.dispose(); }
}
{
    const space = {x: {kind: "log_float", low: 0.1, high: domainNext(0.1)}};
    const opt = Optimizer.create(space, {sampler: "gp_ei", seed: 19n});
    try {
        const marker = [255,255,255,255,255,255,239,63];
        for (let count = 1; count <= 3; ++count) {
            const trial = opt.ask(); opt.tell(trial, "completed", 0.5);
            const blob = opt.save(); let hits = 0;
            for (let i = 0; i + 8 <= blob.length; ++i) if (marker.every((byte, j) => blob[i+j] === byte)) ++hits;
            assert.equal(hits, count);
            const restored = Optimizer.load(blob, space); restored.dispose();
        }
    } finally { opt.dispose(); }
}
for (const sampler of ["random", "tpe", "gp_ei"]) {
    const low = 1e300, high = 1e300 * (1 + 1e-14), middle = low + (high-low)/2;
    assert.ok(middle > low && middle < high);
    const sp = {x: {kind: "log_float", low, high}, aux: {kind: "float", low: 0, high: 1}};
    const opt = Optimizer.create(sp, {sampler, seed: 19n});
    try {
        if (sampler === "gp_ei") assert.throws(() => opt.enqueue({x: low, aux: 0.25}), error => error.status === 10);
        else {
            for (const x of [low, high, middle]) opt.enqueue({x, aux: 0.25});
            for (const x of [low, high, middle]) { const t = opt.ask(); assert.equal(t.parameters.x, x); opt.tell(t, "completed", 0.5); }
        }
    } finally { opt.dispose(); }
}

// T4e: seed-19 reference vectors were calculated from a raw unit stream with
// Python integer arithmetic (see implementation-preparation/raw-u-vectors.json).
{
    const h = 2n ** 53n;
    const expected = [h,h-2n,h-1n,h-2n,h,h,h-1n,h-1n,h-2n,h-1n,h,h,h,h,h,h];
    for (const kind of ["int", "float"]) {
        const axis = kind === "int" ? {kind, low:h-2n, high:h, step:1n}
            : {kind, low:Number(h-2n), high:Number(h), step:1};
        const opt = Optimizer.create({x:axis}, {sampler:"random", seed:19n});
        const counts = [0,0,0];
        try {
            for (let i=0; i<6000; ++i) {
                const trial=opt.ask(), x=BigInt(trial.parameters.x);
                assert.ok(x>=h-2n && x<=h);
                if (i<expected.length) assert.equal(x,expected[i]);
                counts[Number(x-(h-2n))]++;
                opt.tell(trial,"completed",0);
            }
            assert.deepEqual(counts,[2015,2005,1980]);
        } finally { opt.dispose(); }
    }
}
for (const [low,high,step,cells] of [
    [1-2**-52,1,2**-53,3], [0.5-2**-53,0.5,2**-54,3],
    [2**60-256,2**60,128,3], [1-3*2**-53,1,2**-53,4], [2**52,2**52+4,1.5,3],
]) {
    const opt=Optimizer.create({x:{kind:"float",low,high,step}}, {sampler:"random",seed:19n});
    const counts=new Map();
    try {
        for(let i=0;i<6000;++i) { const t=opt.ask(), x=t.parameters.x;
            assert.ok(x>=low && x<=high); counts.set(x,(counts.get(x)||0)+1); opt.tell(t,"completed",0); }
        assert.equal(counts.size,cells);
        for(const count of counts.values()) assert.ok(Math.abs(count/6000-1/cells)<=0.03);
    } finally { opt.dispose(); }
}
for (const [low,high,step] of [[2n**53n-3n,2n**53n,2n],[-(2n**53n),2n**53n,3n]])
    for(const kind of ["int","float"]) for(const sampler of ["random","sobol","lhs","ternary","ga","pso","cmaes","tpe","gp_ei"]) {
        const axis=kind==="int" ? {kind,low,high,step} : {kind,low:Number(low),high:Number(high),step:Number(step)};
        const sp={x:axis,aux:{kind:"float",low:0,high:1}};
        domainStages(sp,sampler,19,trial=> { const x=BigInt(trial.parameters.x);
            assert.ok(x>=low && x<=high && (x-low)%step===0n); });
    }
{
    const h=2**53;
    for(const [low,high,step,accept,reject] of [
        [-h,h,3,[1,7,13,h-1],[]], [2**52,2**52+16,8,[2**52+8],[2**52+1,2**52+7,2**52+9]],
        [h-3,h,2,[],[h]],
    ]) {
        const opt=Optimizer.create({x:{kind:"float",low,high,step}},{sampler:"random",seed:19n});
        try {
            for(const x of reject) assert.throws(()=>opt.enqueue({x}));
            for(const x of accept) {opt.enqueue({x});const t=opt.ask();assert.equal(t.parameters.x,x);opt.tell(t,"completed",0);}
        } finally {opt.dispose();}
    }
    const opt=Optimizer.create({x:{kind:"int",low:2n**53n-3n,high:2n**53n,step:2n}},{sampler:"random",seed:19n});
    try {assert.throws(()=>opt.enqueue({x:2n**53n}));} finally {opt.dispose();}
}

// Python bigint oracle for a wide step-3 integer grid.
{
    const expected = [4239843483439303n,-6906221722996298n,-2457079771300949n,-7517534437616789n,5116916499969259n,8257295123674171n,-417271150248371n,1254295467694255n,-4530896056998680n,452284686773314n,7284296237484454n,7676065770718294n,5637129696437785n,6717941993400823n,5031873289515835n,4454328880214974n]; const h=2n**53n;
    for(const kind of ["int","float"]) { const axis=kind==="int" ? {kind,low:-h,high:h,step:3n} : {kind,low:-Number(h),high:Number(h),step:3};
        const opt=Optimizer.create({x:axis},{sampler:"random",seed:19n});
        try {for(const x of expected){const t=opt.ask();assert.equal(BigInt(t.parameters.x),x);opt.tell(t,"completed",0);}} finally {opt.dispose();}
    }
}

// Complete required integer-grid matrix from the independent Python bigint oracle.
for (const [low,high,step,expected] of [
    [9007199254740990n,9007199254740992n,1n,[9007199254740992n,9007199254740990n,9007199254740991n,9007199254740990n,9007199254740992n,9007199254740992n,9007199254740991n,9007199254740991n,9007199254740990n,9007199254740991n,9007199254740992n,9007199254740992n,9007199254740992n,9007199254740992n,9007199254740992n,9007199254740992n]],
    [9007199254740989n,9007199254740992n,2n,[9007199254740991n,9007199254740989n,9007199254740989n,9007199254740989n,9007199254740991n,9007199254740991n,9007199254740989n,9007199254740991n,9007199254740989n,9007199254740991n,9007199254740991n,9007199254740991n,9007199254740991n,9007199254740991n,9007199254740991n,9007199254740991n]],
    [9007199254740987n,9007199254740992n,3n,[9007199254740990n,9007199254740987n,9007199254740987n,9007199254740987n,9007199254740990n,9007199254740990n,9007199254740987n,9007199254740990n,9007199254740987n,9007199254740990n,9007199254740990n,9007199254740990n,9007199254740990n,9007199254740990n,9007199254740990n,9007199254740990n]],
    [4503599627370496n,4503599627370503n,8n,[4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n,4503599627370496n]],
    [562949953421312n,562949953421315n,4n,[562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n,562949953421312n]],
    [281474976710656n,281474976710659n,4n,[281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n,281474976710656n]],
    [-9007199254740992n,9007199254740992n,1n,[4239843483439302n,-6906221722996298n,-2457079771300950n,-7517534437616788n,5116916499969258n,8257295123674170n,-417271150248372n,1254295467694256n,-4530896056998678n,452284686773314n,7284296237484452n,7676065770718292n,5637129696437782n,6717941993400820n,5031873289515832n,4454328880214974n]],
    [-9007199254740992n,9007199254740992n,3n,[4239843483439303n,-6906221722996298n,-2457079771300949n,-7517534437616789n,5116916499969259n,8257295123674171n,-417271150248371n,1254295467694255n,-4530896056998680n,452284686773314n,7284296237484454n,7676065770718294n,5637129696437785n,6717941993400823n,5031873289515835n,4454328880214974n]],
    [-9007199254740992n,9007199254740992n,7n,[4239843483439301n,-6906221722996299n,-2457079771300949n,-7517534437616788n,5116916499969263n,8257295123674173n,-417271150248369n,1254295467694258n,-4530896056998679n,452284686773314n,7284296237484458n,7676065770718299n,5637129696437781n,6717941993400823n,5031873289515831n,4454328880214979n]],
    [-1n,9007199254740992n,1n,[6623521369090147n,1050488765872346n,3275059741720020n,744832408562101n,7062057877355125n,8632247189207581n,4294964052246309n,5130747361217624n,2238151598871156n,4729741970757153n,8145747746112722n,8341632512729642n,7322164475589387n,7862570624070906n,7019536272128412n,6730764067477983n]],
    [-3n,9007199254740992n,1n,[6623521369090146n,1050488765872344n,3275059741720019n,744832408562099n,7062057877355125n,8632247189207581n,4294964052246308n,5130747361217623n,2238151598871154n,4729741970757152n,8145747746112722n,8341632512729642n,7322164475589387n,7862570624070906n,7019536272128412n,6730764067477982n]],
    [0n,9007199254740992n,1n,[6623521369090147n,1050488765872347n,3275059741720021n,744832408562102n,7062057877355125n,8632247189207581n,4294964052246310n,5130747361217624n,2238151598871157n,4729741970757153n,8145747746112722n,8341632512729642n,7322164475589387n,7862570624070906n,7019536272128412n,6730764067477983n]],
    [0n,9007199254740990n,1n,[6623521369090146n,1050488765872346n,3275059741720020n,744832408562101n,7062057877355124n,8632247189207580n,4294964052246309n,5130747361217623n,2238151598871156n,4729741970757152n,8145747746112721n,8341632512729641n,7322164475589386n,7862570624070905n,7019536272128411n,6730764067477982n]],
    [-9007199254740992n,9007199254740992n,5n,[4239843483439303n,-6906221722996302n,-2457079771300952n,-7517534437616792n,5116916499969258n,8257295123674168n,-417271150248372n,1254295467694253n,-4530896056998682n,452284686773313n,7284296237484453n,7676065770718293n,5637129696437783n,6717941993400818n,5031873289515833n,4454328880214973n]],
    [-9007199254740992n,9007199254740992n,9n,[4239843483439309n,-6906221722996304n,-2457079771300952n,-7517534437616792n,5116916499969262n,8257295123674177n,-417271150248368n,1254295467694255n,-4530896056998680n,452284686773311n,7284296237484460n,7676065770718297n,5637129696437782n,6717941993400826n,5031873289515838n,4454328880214977n]],
    [-9007199254740992n,9007199254740992n,11n,[4239843483439303n,-6906221722996307n,-2457079771300954n,-7517534437616794n,5116916499969255n,8257295123674171n,-417271150248378n,1254295467694250n,-4530896056998682n,452284686773307n,7284296237484456n,7676065770718293n,5637129696437781n,6717941993400817n,5031873289515834n,4454328880214975n]]
]) for (const kind of ["int","float"]) {
    const axis=kind==="int" ? {kind,low,high,step} : {kind,low:Number(low),high:Number(high),step:Number(step)};
    const opt=Optimizer.create({x:axis},{sampler:"random",seed:19n});
    try {for(const x of expected) {const t=opt.ask();assert.equal(BigInt(t.parameters.x),x);opt.tell(t,"completed",0);}}
    finally {opt.dispose();}
}

// T4f R1/R2: adaptive boundedness and load of the resulting checkpoint.
const t4fR1=[[-1,1-2**-52,3*2**-53,400],[-.5,.5-2**-53,1.5*2**-53,400]];
const t4fR2=[[1.3094115832905704,9.007642868376131e21,3.020767460720702,240],[1.3552258541085935,3.1816648682723785e21,9.932851196532727,240]];
const t4fR4=[[-1,1,3*2**-53],[-(2**43),2**43,3/1024],[-(2**54),2**54,6]];
const t4fR3=[...t4fR4,[-1,1,1.25*2**-53],[-1,1,1.5*2**-53]];
for(const [low,high,step,count] of [...t4fR1,...t4fR2]) for(const sampler of ["cmaes","pso","ga","tpe"]) {
 const sp={x:{kind:"float",low,high,step},aux:{kind:"float",low:0,high:1}}; const opt=Optimizer.create(sp,{sampler,seed:19n,direction:"minimize"});let outside=0;
 try {for(let i=0;i<count;i++){const t=opt.ask(),x=t.parameters.x;if(x<low||x>high)outside++;opt.tell(t,"completed",-x);}assert.equal(outside,0);const copy=Optimizer.load(opt.save(),sp);copy.dispose();}finally{opt.dispose();}
}
// R3: each generated FLOAT is enqueued and returned without changing bits.
for(const [low,high,step] of t4fR3) {const sp={x:{kind:"float",low,high,step}};const a=Optimizer.create(sp,{sampler:"random",seed:19n}),b=Optimizer.create(sp,{sampler:"random",seed:3n});
 try {for(let i=0;i<300;i++){const t=a.ask(),x=t.parameters.x;b.enqueue({x});const u=b.ask();assert.equal(u.parameters.x,x);a.tell(t,"completed",0);b.tell(u,"completed",0);}}finally{a.dispose();b.dispose();}}
// R4: first 16 values pinned from the independent rational one-round oracle.
const t4fVectors = [
  [0x3fde203af977a18en,0xbfe8892c0c8a0e4an,0xbfd17566ede808aan,0xbfeab52861bdac95n,0x3fe22dcef65374ebn,0x3fed55f756a6f43bn,0xbfa7b8182af81b30n,0x3fc1d319af2e62bcn,0xbfe018d371fddb18n,0x3fa9b59bc17ac420n,0x3fe9e10761ac89a6n,0x3feb455755894456n,0x3fe406f0891be619n,0x3fe7ddeec22369f7n,0x3fe1e0764bf9f33bn,0x3fdfa660853c677cn],
  [0x428e203af977a18en,0xc298892c0c8a0e4an,0xc2817566ede808aan,0xc29ab52861bdac95n,0x42922dcef65374ebn,0x429d55f756a6f43bn,0xc257b8182af81b30n,0x4271d319af2e62bcn,0xc29018d371fddb18n,0x4259b59bc17ac420n,0x4299e10761ac89a6n,0x429b455755894456n,0x429406f0891be619n,0x4297ddeec22369f7n,0x4291e0764bf9f33bn,0x428fa660853c677cn],
  [0x433e203af977a18en,0xc348892c0c8a0e4an,0xc3317566ede808aan,0xc34ab52861bdac95n,0x43422dcef65374ebn,0x434d55f756a6f43bn,0xc307b8182af81b30n,0x4321d319af2e62bcn,0xc34018d371fddb18n,0x4309b59bc17ac420n,0x4349e10761ac89a6n,0x434b455755894456n,0x434406f0891be619n,0x4347ddeec22369f7n,0x4341e0764bf9f33bn,0x433fa660853c677cn]
];
for(let j=0;j<t4fR4.length;j++){const [low,high,step]=t4fR4[j],sp={x:{kind:"float",low,high,step}},o=Optimizer.create(sp,{sampler:"random",seed:19n});const dv=new DataView(new ArrayBuffer(8));
 try{for(const expected of t4fVectors[j]){const t=o.ask();dv.setFloat64(0,t.parameters.x);assert.equal(dv.getBigUint64(0),expected);o.tell(t,"completed",0);}}finally{o.dispose();}}
for(const [low,high,step] of [...t4fR1,...t4fR3]) for(const sampler of ["random","sobol","lhs","ternary","ga","pso","cmaes","tpe","gp_ei"]){
 const sp={x:{kind:"float",low,high,step},aux:{kind:"float",low:0,high:1}};domainStages(sp,sampler,19,t=>assert.ok(t.parameters.x>=low&&t.parameters.x<=high));}
for(const [low,high,step] of [...t4fR1,...t4fR3]) for(const sampler of ["random","sobol","lhs","ternary","ga","pso","cmaes","tpe","gp_ei"]) {
 const sp={x:{kind:"float",low,high,step},aux:{kind:"float",low:0,high:1}},o=Optimizer.create(sp,{sampler,seed:19n});
 try {if(["ga","pso","cmaes","gp_ei"].includes(sampler)){assert.throws(()=>o.enqueue({x:low,aux:.25}),e=>e.status===10);continue;}
  o.enqueue({x:low,aux:.25});const copy=Optimizer.load(o.save(),sp);try{assertNativeConfigurationMatches(o,copy);for(let i=0;i<6;++i){const a=o.ask(),b=copy.ask();assert.equal(a.id,b.id);assert.deepEqual(a.parameters,b.parameters);if(i===0)assert.equal(a.parameters.x,low);o.tell(a,"completed",.5);copy.tell(b,"completed",.5);}}finally{copy.dispose();}
 }finally{o.dispose();}
}

// Genuine name-and-rank targeted checkpoint mutations; see fixture provenance.
{
 const blobs = [
  "4e344d4f50540d0a010000002000000050010000000000002201000000000000fefd47e44137aed63c000000000000000100000001000000780000000000000000000000000000000000002440000000000000084001000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000db5912a55f50163f010000000000000002000000000000000100000000000000000000000100000001000000780000000000001c40ffffffff000000000101000000000000000000d03f01f44d02eab9c4f23e00000000000000000000000000000100000000000000f87c41c9ec45f73e00000000000000000000000000000000000000000000000000000079f206fca54cbe86",
  "4e344d4f50540d0a010000002000000050010000000000002201000000000000fefd47e44137aed63c000000000000000100000001000000780000000000000000000000000000000000002440000000000000084001000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a00000013000000000000000000000000000000000000000000000000000000130000000000000049d81abe4359f13e010000000000000001000000000000000100000000000000000000000100000001000000780000000000001c40ffffffff00000000010000000000000000000000000000000000000000000000000000000000000000000000ffffffffffffffff2dd5cd116e3cdc3e00000000000000000000000000000000000000000000000000000055db2ba5873406ea",
  "4e344d4f50540d0a0100000020000000500100000000000022010000000000007279c6e12b96c67b3c00000000000000010000000100000078010000000000000000000000000000000000f03f9a9999999999b93f00000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a00000013000000000000000000000000000000000000000000000000000000130000000000000086c613419c87f33e01000000000000000200000000000000010000000000000000000000010000000100000078333333333333d33fffffffff000000000101000000000000000000d03f0155a485752ab3dd3e00000000000000000000000000000100000000000000f008a9065ab4e23e0000000000000000000000000000000000000000000000000000008453efa0f3c528c4",
  "4e344d4f50540d0a01000000200000005001000000000000220100000000000086412335e4f789433c0000000000000001000000010000007801000000000000000000f0bf000000000000f03f000000000000b83c00000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a00000013000000000000000000000000000000000000000000000000000000130000000000000060062128d08aec3e01000000000000000200000000000000010000000000000000000000010000000100000078ffffffffffffefbfffffffff000000000101000000000000000000d03f01aff75f0ab6e1d53e0000000000000000000000000000010000000000000068c24a93bf6cdc3e000000000000000000000000000000000000000000000000000000b80d6507cf98d4c0",
  "4e344d4f50540d0a0100000020000000e001000000000000b1010000000000009e900a20946c3bd2af0000000000000002000000040000006d6f64650400000000000000000000000000000000000000000000000000000000000000000002000000030000006f6666020000006f6e020000000000000000000000000000000000f03f0000000000010000007801000000000000000000f03f0000000000001440000000000000004000000000000000000000000000000000000000010000000300000002000000010000007800000000040000006d6f6465020000006f6e0d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000ddfd5cfab8b2fb3e0100000000000000020000000000000001000000000000000000000002000000040000006d6f6465000000000000000000000000030000006f66660101000000780000000000001040ffffffff000000000001000000000000000000d03f0101153dd995e2df3e000000000000000000000000000001000000000000003f384ef356a9e53e000000000000000000000000000000000000000000000000000000006c2ea3ac43e1c8c2",
  "4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000840ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000bd0d7ae11a381dfd",
  "4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000840ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000750c03c702595a90",
];
 const spaces = [
  {x:{kind:"int",low:0n,high:10n,step:3n}},
  {x:{kind:"int",low:0n,high:10n,step:3n}},
  {x:{kind:"float",low:0,high:1,step:0.1}},
  {x:{kind:"float",low:-1,high:1,step:3*2**-53}},
  {mode:{kind:"categorical",type:"string",choices:["off","on"]},x:{kind:"float",low:1,high:5,step:2}},
  {x:{kind:"float",low:0,high:4,step:2}},
  {x:{kind:"float",low:0,high:4,step:2}},
 ];
 const raw = h => Uint8Array.from(Buffer.from(h, "hex"));
 for(let i=0;i<blobs.length;i++) assert.throws(()=>Optimizer.load(raw(blobs[i]),spaces[i]),e=>e.status===14);
 const good = raw("4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000040ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000750c03c702595a90");
 const a=Optimizer.load(good,spaces[5]), b=Optimizer.load(good,spaces[5]);
 try { assertNativeConfigurationMatches(a,b); for(let i=0;i<6;i++) {const x=a.ask(),y=b.ask();assert.equal(x.id,y.id);assert.deepEqual(x.parameters,y.parameters);a.tell(x,"completed",0.5);b.tell(y,"completed",0.5);} } finally {a.dispose();b.dispose();}
}
