// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// MI32_JS_V1: raw public C ABI tests. An abort is never a typed refusal.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const here = path.dirname(fileURLToPath(import.meta.url));
const keys = ["HOME", "home", "CODEX_HOME"];
const inherited = keys.map(k => [Object.hasOwn(process.env, k), process.env[k]]);
function envGuard(env = process.env) {
    assert(keys.every((k, j) => Object.hasOwn(env, k) === inherited[j][0]
        && env[k] === inherited[j][1]), "protected environment changed (values private)");
}
const digest = bytes => crypto.createHash("sha256").update(bytes).digest("hex");
const variables = ["N4M_WASM_PATH", "N4M_WASM_GLUE_PATH",
    "N4M_EXPECTED_WASM_SHA256", "N4M_EXPECTED_GLUE_SHA256"];
const configured = variables.filter(k => process.env[k] !== undefined);
assert(configured.length === 0 || configured.length === 4,
    "qualification requires all four WASM/glue paths and hashes");
const wasmPath = process.env.N4M_WASM_PATH ?? path.join(here, "../dist/n4m.wasm");
const gluePath = process.env.N4M_WASM_GLUE_PATH ?? path.join(here, "../dist/n4m.js");
const selection = process.argv.indexOf("--case");
const selected = selection < 0 ? null : process.argv[selection + 1];
const childIndex = process.argv.indexOf("--child");
const cases = ["positive", "snv", "selector", "linear", "procedure", "wide-seed",
    "feature-groups-refit", "repeated-refusal"];

if (childIndex < 0 && !process.argv.includes("--parity")) {
    if (selected !== null) assert(cases.includes(selected), "unknown case");
    for (const name of selected === null ? cases : [selected]) {
        envGuard();
        const result = spawnSync(process.execPath, [fileURLToPath(import.meta.url), "--child", name],
            { env: { ...process.env }, encoding: "utf8", timeout: 60000, maxBuffer: 4 * 1024 * 1024 });
        envGuard();
        process.stdout.write(result.stdout ?? "");
        process.stderr.write(result.stderr ?? "");
        assert.ifError(result.error);
        assert.equal(result.signal, null, `${name}: terminated instead of returning statuses`);
        assert.equal(result.status, 0, `${name}: raw child failed (abort is not a refusal)`);
    }
} else {
    envGuard();
    const wasmBytes = fs.readFileSync(wasmPath), glueBytes = fs.readFileSync(gluePath);
    if (configured.length) {
        assert.equal(digest(wasmBytes), process.env.N4M_EXPECTED_WASM_SHA256);
        assert.equal(digest(glueBytes), process.env.N4M_EXPECTED_GLUE_SHA256);
    }
    console.log("MI32_JS_V1_ORIGIN", JSON.stringify({ node: process.version,
        wasm: fs.realpathSync(wasmPath), wasm_sha256: digest(wasmBytes),
        glue: fs.realpathSync(gluePath), glue_sha256: digest(glueBytes),
        qualification_hashes_enforced: configured.length === 4 }));
    const factory = (await import(pathToFileURL(gluePath).href)).default;
    const m = await factory({ wasmBinary: wasmBytes });
    envGuard();
    const allocations = [], handles = [];
    let aborted = false;
    const allocate = size => {
        const p = m._malloc(Math.max(1, size));
        assert(p !== 0, "malloc failed");
        m.HEAPU8.fill(0, p, p + size);
        allocations.push(p);
        return p;
    };
    const string = s => {
        const p = allocate(m.lengthBytesUTF8(s) + 1);
        m.stringToUTF8(s, p, m.lengthBytesUTF8(s) + 1);
        return p;
    };
    const ok = (status, label) => assert.equal(status, 0, label);
    const slot = allocate(4);
    ok(m._n4m_context_create(slot), "context_create");
    const ctx = m.HEAP32[slot / 4];
    handles.push(["context", ctx]);
    const error = () => m.UTF8ToString(m._n4m_context_last_error(ctx));
    function params(id, values) {
        const idp = string(id);
        ok(m._n4m_method_find(idp, slot), "method_find");
        const index = m.HEAP32[slot / 4];
        ok(m._n4m_params_create(ctx, index, slot), "params_create");
        const p = m.HEAP32[slot / 4];
        handles.push(["params", p]);
        for (const [name, value] of Object.entries(values)) {
            const key = string(name);
            if (typeof value === "boolean") ok(m._n4m_params_set_bool(p, key, +value), name);
            else {
                ok(m._n4m_params_set_int(p, key, BigInt(value)), name);
                const v = allocate(8), count = allocate(8);
                ok(m._n4m_params_get_int(p, key, v, 1n, count), "exact int64 readback");
                assert.equal(new DataView(m.HEAPU8.buffer).getBigInt64(v, true), BigInt(value));
            }
        }
        ok(m._n4m_params_validate(ctx, p), "params_validate");
        return { idp, p, index };
    }
    function estimator(id, values) {
        const p = params(id, values);
        ok(m._n4m_estimator_create(ctx, p.idp, p.p, slot), "estimator_create");
        const est = m.HEAP32[slot / 4];
        handles.push(["estimator", est]);
        return est;
    }
    function matrix(values, rows, cols) {
        const data = allocate(values.length * 8), view = allocate(48);
        m.HEAPF64.set(values, data / 8);
        ok(m._n4m_matrix_view_init_rowmajor(view, data, BigInt(rows), BigInt(cols), 1), "view");
        return { data, view, rows, cols, length: values.length };
    }
    const bytes = v => Buffer.from(m.HEAPU8.subarray(v.data, v.data + v.length * 8)).toString("hex");
    function inputs(x, y = null, groups = null) {
        const p = allocate(120);
        m.HEAP32[p / 4] = 120;
        m.HEAP32[(p + 4) / 4] = x.view;
        m.HEAP32[(p + 8) / 4] = y?.view ?? 0;
        if (groups) {
            const gp = allocate(groups.length * 8);
            const dv = new DataView(m.HEAPU8.buffer);
            groups.forEach((g, j) => dv.setBigInt64(gp + j * 8, BigInt(g), true));
            m.HEAP32[(p + 56) / 4] = gp;
            new DataView(m.HEAPU8.buffer).setBigInt64(p + 64, BigInt(groups.length), true);
        }
        return p;
    }
    const X = matrix([1, 3, 5, 2, 2, 4, 6, 9, 3, 1, 0, 4,
        9, 8, 1, 2, 4, 4, 7, 1, 0, 2, 3, 8], 6, 4);
    const Y = matrix([-4, -4, 3, 8, -3, -3], 6, 1);
    const max = (1n << 63n) - 1n;
    const overflows = [2147483648n, 4294967296n, max];
    const fit = (est, inPtr) => m._n4m_estimator_fit(ctx, est, inPtr);
    function refused(est, inPtr) {
        assert.equal(fit(est, inPtr), 1, "narrowing must return INVALID_ARGUMENT1");
        assert.equal(error(), "integer parameter exceeds int32");
    }
    function positive(includeAdapters = true) {
        const fixture = JSON.parse(fs.readFileSync(new URL("../../../parity/fixtures/snv_parameter_contract_v1.json", import.meta.url)));
        const x = matrix(fixture.matrix.flat(), 4, 4);
        const valid = [];
        for (const ref of fixture.references) {
            const est = estimator("preprocessing.scatter.snv", ref.params);
            ok(fit(est, inputs(x)), "valid SNV fit");
            const out = matrix(Array(16).fill(-777.25), 4, 4);
            ok(m._n4m_estimator_transform(ctx, est, x.view, out.view), "valid SNV transform");
            const actual = Array.from(m.HEAPF64.subarray(out.data / 8, out.data / 8 + 16));
            ref.values.flat().forEach((v, j) => assert(Math.abs(actual[j] - v) < 1e-12));
            if (ref.id === "mean1-std1-ddof0") assert.equal(bytes(out), fixture.default_bits.wasm_8b0);
            valid.push({ id: ref.id, bytes: bytes(out) });
        }
        if (includeAdapters) {
            // Retained reviewer positive controls: the same fixed6x4 X/y.
            const robust = estimator("models.regularized.robust_pls", { max_irls_iter: 5 });
            ok(fit(robust, inputs(X, Y)), "valid robust PLS fit");
            const predictions = matrix(Array(6).fill(-777.25), 6, 1);
            ok(m._n4m_estimator_predict(ctx, robust, X.view, predictions.view), "valid robust PLS predict");
            assert(Array.from(m.HEAPF64.subarray(predictions.data / 8, predictions.data / 8 + 6)).every(Number.isFinite));
            valid.push({ id: "review-robust-pls-5", bytes: bytes(predictions) });
            const selector = estimator("selection.vip_spa", { top_k: 2, n_components: 2 });
            ok(fit(selector, inputs(X, Y)), "valid vip_spa fit");
            const columns = allocate(8);
            ok(m._n4m_estimator_transform_cols(selector, columns), "valid selector width");
            const width = Number(new DataView(m.HEAPU8.buffer).getBigInt64(columns, true));
            assert.equal(width, 2);
            const selected = matrix(Array(6 * width).fill(-777.25), 6, width);
            ok(m._n4m_estimator_transform(ctx, selector, X.view, selected.view), "valid selector transform");
            valid.push({ id: "review-vip-spa-2", bytes: bytes(selected) });
        }
        return valid;
    }
    try {
        const name = childIndex < 0 ? "parity" : process.argv[childIndex + 1];
        console.log("MI32_JS_V1_CASE", name);
        if (name === "positive" || name === "parity") {
            const valid = positive();
            console.log("MI32_VALID_BYTES", JSON.stringify(valid));
            if (name === "parity") {
                // Fixed existing9-reference SNV workload; no search or new data.
                const snvOnly = valid.slice(0, 9);
                const start = performance.now();
                for (let repeat = 0; repeat < 64; ++repeat) assert.deepEqual(positive(false), snvOnly);
                console.log("MI32_FIXED_WORKLOAD", JSON.stringify({ repetitions: 64,
                    references: snvOnly.length, milliseconds: performance.now() - start }));
            }
        }
        else if (["snv", "selector", "linear"].includes(name)) {
            const routes = { snv: ["preprocessing.scatter.snv", "ddof"],
                selector: ["selection.vip_spa", "top_k"],
                linear: ["models.regularized.robust_pls", "max_irls_iter"] };
            const [id, parameter] = routes[name];
            for (const v of overflows) {
                const est = estimator(id, { [parameter]: v });
                const out = matrix(Array(24).fill(-777.25), 6, name === "snv" ? 4 : 1);
                const sentinel = bytes(out), source = bytes(X);
                refused(est, inputs(X, name === "snv" ? null : Y));
                ok(m._n4m_estimator_is_fitted(est, slot), "is_fitted");
                assert.equal(m.HEAP32[slot / 4], 0);
                const status = name === "snv" ? m._n4m_estimator_transform(ctx, est, X.view, out.view)
                    : m._n4m_estimator_predict(ctx, est, X.view, out.view);
                assert.equal(status, 6);
                assert.equal(bytes(out), sentinel);
                assert.equal(bytes(X), source);
            }
        } else if (name === "procedure") {
            for (const v of overflows) {
                const p = params("splitters.kmeans", { max_iter: v });
                m.HEAP32[slot / 4] = -1;
                assert.equal(m._n4m_procedure_run(ctx, p.index, p.p, inputs(X), slot), 1);
                assert.equal(error(), "integer parameter exceeds int32");
                assert.equal(m.HEAP32[slot / 4], 0);
            }
        } else if (name === "wide-seed") {
            const p = params("augmentation.noise.gaussian_noise", { seed: (1n << 40n) + 17n });
            ok(m._n4m_procedure_run(ctx, p.index, p.p, inputs(X), slot), "wide seed procedure");
            const result = m.HEAP32[slot / 4];
            assert.notEqual(result, 0);
            m._n4m_method_result_destroy(result);
        } else if (name === "feature-groups-refit") {
            const est = estimator("models.sparse.group_sparse_pls", {});
            ok(fit(est, inputs(X, Y, [0, 0, 1, 1])), "initial group-sparse fit");
            const old = matrix(Array(6).fill(-777.25), 6, 1);
            ok(m._n4m_estimator_predict(ctx, est, X.view, old.view), "prior prediction");
            const prior = bytes(old);
            for (const bad of [-2147483649n, -(1n << 63n), 2147483648n]) {
                refused(est, inputs(X, Y, [bad, 0, 1, 1]));
                ok(m._n4m_estimator_is_fitted(est, slot), "still fitted");
                assert.equal(m.HEAP32[slot / 4], 1);
                m.HEAPF64.fill(-777.25, old.data / 8, old.data / 8 + 6);
                ok(m._n4m_estimator_predict(ctx, est, X.view, old.view), "preserved prediction");
                assert.equal(bytes(old), prior);
            }
        } else if (name === "repeated-refusal") {
            const est = estimator("preprocessing.scatter.snv", { ddof: 2147483648n });
            const inPtr = inputs(X);
            refused(est, inPtr); // Warm up before the stable memory observation.
            const heapLength = () => new Int8Array(m.HEAPU8.buffer).length;
            const before = heapLength();
            for (let k = 0; k < 1000; ++k) refused(est, inPtr);
            assert.equal(heapLength(), before, "HEAP8.length grew over1000 refusals");
            positive(); // The same module must remain usable.
            console.log("MI32_1000_REFUSALS", JSON.stringify({ count: 1000, heap8_length: before }));
        } else throw Error("unknown raw i32 test case");
    } catch (e) {
        // Diagnostic-only cleanup protection. Never swallow or reclassify an abort.
        if (e instanceof WebAssembly.RuntimeError) aborted = true;
        throw e;
    } finally {
        if (!aborted) {
            const handlesCreated = handles.length, allocationsCreated = allocations.length;
            for (const [kind, ptr] of handles.reverse()) m[`_n4m_${kind}_destroy`](ptr);
            for (const ptr of allocations.reverse()) m._free(ptr);
            console.log("MI32_CLEANUP", JSON.stringify({ handlesCreated, handlesDestroyed: handles.length,
                allocationsCreated, allocationsFreed: allocations.length }));
        }
        envGuard();
    }
}
