// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Targeted real-WASM StandardScale cases, also called by the existing roles runner.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as n4m from "../dist/index.js";

const matrix = rows => ({ data: Float64Array.from(rows.flat()), rows: rows.length, cols: rows[0].length });
const close = (actual, expected, label) => {
    assert.equal(actual.length, expected.length, label);
    expected.forEach((value, i) => assert.ok(Math.abs(actual[i] - value) <= 1e-12 * (1 + Math.abs(value)), `${label}[${i}]`));
};
const bytes = values => Buffer.from(values.buffer, values.byteOffset, values.byteLength);

export function checkStandardScale() {
    // Preserved analytic population case from run_estimator_roles.mjs.
    const offset = 1e12;
    const train = matrix([[offset, 7, 0], [offset + 2, 7, 2],
        [offset + 4, 7, 4], [offset + 6, 7, 6]]);
    const heldout = matrix([[offset + 1, 7, 1], [offset + 5, 7, 5]]);
    const expected = [-2 / Math.sqrt(5), 0, -2 / Math.sqrt(5),
        2 / Math.sqrt(5), 0, 2 / Math.sqrt(5)];
    const scaler = new n4m.StandardScale();
    let restored;
    try {
        scaler.fit(train);
        close(scaler.transform(heldout).data, expected, "population StandardScale");
        const state = scaler.toN4me();
        restored = n4m.NativeEstimator.fromN4me(state);
        close(restored.transform(heldout).data, expected, "population StandardScale replay");
        assert.deepEqual(restored.toN4me(), state);
    } finally {
        scaler.dispose();
        restored?.dispose();
    }
    for (const ddof of [0, 1]) for (const with_mean of [false, true]) for (const with_std of [false, true]) {
        const est = new n4m.StandardScale({ ddof, with_mean, with_std });
        let replay, defaults;
        try {
            est.fit(train);
            const out = est.transform(heldout).data;
            const mean = [offset + 3, 7, 3], sd = Math.sqrt(20 / (4 - ddof));
            close(out, Array.from(heldout.data, (value, i) => (value - (with_mean ? mean[i % 3] : 0)) / (with_std && i % 3 !== 1 ? sd : 1)), "analytic ddof");
            assert.equal(out[1], with_mean ? 0 : 7);
            const state = est.toN4me();
            replay = n4m.NativeEstimator.fromN4me(state);
            assert.deepEqual(bytes(replay.transform(heldout).data), bytes(out));
            assert.deepEqual(replay.toN4me(), state);
            const together = matrix([[offset + 1, 7, 1], [offset + 5, 7, 5], [offset + 999, 8, -1000]]);
            assert.deepEqual(bytes(est.transform(together).data.slice(0, 6)), bytes(out));
            if (ddof === 0) {
                defaults = new n4m.StandardScale({ with_mean, with_std }).fit(train);
                assert.deepEqual(bytes(defaults.transform(heldout).data), bytes(out));
                assert.deepEqual(defaults.toN4me(), state);
            }
        } finally { est.dispose(); replay?.dispose(); defaults?.dispose(); }
    }
    for (const with_std of [false, true]) {
        const singleton = matrix([[0.1, 7]]);
        const est = new n4m.StandardScale({ with_std });
        try { est.fit(singleton); assert.deepEqual(Array.from(est.transform(singleton).data), [0, 0]); }
        finally { est.dispose(); }
        for (const ddof of [-1, 1, 2, 0.5]) {
            const invalid = new n4m.StandardScale({ ddof, with_std });
            try { assert.throws(() => invalid.fit(singleton)); }
            finally { invalid.dispose(); }
        }
    }
}

export function checkPrechangeStandardScale() {
    const dir = process.env.N4M_STANDARD_SCALE_WITNESS_DIR;
    if (!dir) {
        console.log("SKIP genuine pre-change scaler/N4MF/N4MC witnesses: Root must set N4M_STANDARD_SCALE_WITNESS_DIR");
        return;
    }
    const supplement = resolve(dir, "supplement-population0");
    const arrays = JSON.parse(readFileSync(resolve(supplement, "RECEIPT.json"), "utf8")).arrays;
    const readArray = name => {
        const spec = arrays[name];
        assert.equal(spec.dtype, "float64.little_endian");
        const raw = readFileSync(resolve(supplement, spec.file));
        assert.equal(raw.byteLength, spec.shape[0] * spec.shape[1] * 8);
        const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
        return { data: Float64Array.from({ length: raw.byteLength / 8 }, (_, i) => view.getFloat64(8 * i, true)),
            rows: spec.shape[0], cols: spec.shape[1] };
    };
    for (const [training, transform, expected] of [
        ["normal-training-input", "normal-heldout-input", "normal-heldout-output"],
        ["large-offset-input", "large-offset-input", "large-offset-output"],
    ]) {
        const expectedBytes = readFileSync(resolve(supplement, arrays[expected].file));
        const est = new n4m.StandardScale();
        try {
            est.fit(readArray(training));
            const actual = est.transform(readArray(transform));
            assert.deepEqual([actual.rows, actual.cols], arrays[expected].shape);
            assert.deepEqual(bytes(actual.data), expectedBytes);
        } finally { est.dispose(); }
        assert.deepEqual(readFileSync(resolve(supplement, arrays[expected].file)), expectedBytes);
    }
    const refused = error => error.status === n4m.Status.ERR_CORRUPT_BUFFER && /parameter count/.test(error.message);
    const direct = new Uint8Array(readFileSync(resolve(dir, "standard-scale.n4me")));
    assert.throws(() => n4m.NativeEstimator.fromN4me(direct), refused);
    assert.deepEqual(direct, new Uint8Array(readFileSync(resolve(dir, "standard-scale.n4me"))));
    for (const [kind, file, cls] of [
        ["regression", "standard-scale.n4mf", n4m.MultimodalPipeline],
        ["classification", "standard-scale.n4mc", n4m.MultimodalClassifierPipeline],
    ]) {
        const metadata = JSON.parse(readFileSync(resolve(dir, `${kind}.json`), "utf8"));
        const payload = new Uint8Array(readFileSync(resolve(dir, file)));
        assert.throws(() => cls.fromState(payload, metadata.recipe, metadata.source_schemas), refused);
        assert.deepEqual(payload, new Uint8Array(readFileSync(resolve(dir, file))));
    }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    await n4m.loadModule();
    checkStandardScale();
    checkPrechangeStandardScale();
    console.log("StandardScale numerical/current-state cases PASS; witness skips, if any, are reported separately");
}
