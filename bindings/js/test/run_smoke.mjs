// SPDX-License-Identifier: CECILL-2.1
//
// Node smoke + parity:
// 1. Loads the WASM build, prints version / ABI.
// 2. Fits a SIMPLS PLS regression on deterministic data via the
//    `n4m_wasm_pls_fit` helper.
// 3. Predicts in-sample.
// 4. Loads the parity reference. By default this is the frozen fixture saved by
//    `bindings/js/test/generate_parity_fixture.py`; ecosystem gates can pass
//    `N4M_WASM_PARITY_FIXTURE` to make WASM consume an orchestrator dataset
//    ledger generated during the same cross-binding run.
// 5. Compares coefficients + predictions; fails if RMSE-rel > 1e-3.

import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { readFileSync, existsSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = resolve(here, "..", "..", "..", "build", "emscripten",
                        "bindings", "js");
const factory = (await import(resolve(wasmDir, "n4m.js"))).default;
const M = await factory({locateFile: (p) => resolve(wasmDir, p)});
const cc = (name, result, types, args) => M.ccall(name, result, types, args);
const F64 = 1;
const MATRIX_VIEW_BYTES = 48;

// ----- Version + ABI -----
const versionPtr = M._n4m_get_version_string();
const version = M.UTF8ToString(versionPtr);
console.log("libn4m version:", version);
const abi = [
    M._n4m_get_abi_version_major(),
    M._n4m_get_abi_version_minor(),
    M._n4m_get_abi_version_patch(),
];
console.log("ABI:", abi.join("."));

// ----- Parity fixture -----
const fixturePath = process.env.N4M_WASM_PARITY_FIXTURE
    ? resolve(process.cwd(), process.env.N4M_WASM_PARITY_FIXTURE)
    : resolve(here, "parity_fixture.json");
if (!existsSync(fixturePath)) {
    throw new Error(
        `Required parity fixture missing at ${fixturePath}.\n` +
        `Regenerate the default fixture via:\n` +
        `PYTHONPATH=bindings/python/src parity/python_generator/.venv/bin/python \\\n` +
        `  bindings/js/test/generate_parity_fixture.py`);
}
const ref = JSON.parse(readFileSync(fixturePath, "utf-8"));

function asFloat64(values, expectedLength, name) {
    if (!Array.isArray(values)) {
        throw new Error(`Fixture field ${name} must be an array`);
    }
    if (values.length !== expectedLength) {
        throw new Error(
            `Fixture field ${name} length ${values.length} != ${expectedLength}`);
    }
    return Float64Array.from(values);
}

let n = Number(ref.n);
let p = Number(ref.p);
let q = Number(ref.q ?? 1);
const nComponents = Number(ref.n_components ?? 3);
let X;
let Y;
if (Array.isArray(ref.X) && Array.isArray(ref.Y)) {
    X = asFloat64(ref.X, n * p, "X");
    Y = asFloat64(ref.Y, n * q, "Y");
} else {
    // Deterministic default input matched to the checked-in Python fixture.
    n = 50; p = 5; q = 1;
    X = new Float64Array(n * p);
    Y = new Float64Array(n * q);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) {
            X[i * p + j] = Math.sin((i + 1) * (j + 1) * 0.3);
        }
        Y[i] = X[i * p] + 0.5 * X[i * p + 1] - 0.3 * X[i * p + 2];
    }
}
if (!Number.isInteger(n) || !Number.isInteger(p) || !Number.isInteger(q) ||
        !Number.isInteger(nComponents)) {
    throw new Error("Fixture n/p/q/n_components must be integers");
}
console.log("Parity fixture:", fixturePath);
if (ref.schema) console.log("Fixture schema:", ref.schema);

function checkStatus(status, label) {
    if (status !== 0) throw new Error(`${label} failed: ${status}`);
}

function makeView(data, rows, cols) {
    const dataPtr = M._malloc(data.length * 8);
    M.HEAPF64.set(data, dataPtr >>> 3);
    const viewPtr = M._malloc(MATRIX_VIEW_BYTES);
    checkStatus(
        cc("n4m_matrix_view_init_rowmajor", "number",
            ["number", "number", "i64", "i64", "number"],
            [viewPtr, dataPtr, BigInt(rows), BigInt(cols), F64]),
        "n4m_matrix_view_init_rowmajor");
    return {
        dataPtr,
        viewPtr,
        free() {
            M._free(viewPtr);
            M._free(dataPtr);
        },
    };
}

function modelArray(ctx, model, kind) {
    const arrPtrPtr = M._malloc(4);
    M.setValue(arrPtrPtr, 0, "i32");
    checkStatus(
        cc("n4m_model_get_array", "number",
            ["number", "number", "number", "number"],
            [ctx, model, kind, arrPtrPtr]),
        "n4m_model_get_array");
    const arr = M.getValue(arrPtrPtr, "i32");
    M._free(arrPtrPtr);
    const viewPtr = M._malloc(MATRIX_VIEW_BYTES);
    checkStatus(
        cc("n4m_array_view", "number", ["number", "number"], [arr, viewPtr]),
        "n4m_array_view");
    const dataPtr = M.getValue(viewPtr, "i32");
    const rows = Number(M.getValue(viewPtr + 8, "i64"));
    const cols = Number(M.getValue(viewPtr + 16, "i64"));
    const out = new Float64Array(M.HEAPU8.buffer, dataPtr, rows * cols).slice();
    M._free(viewPtr);
    cc("n4m_array_free", null, ["number"], [arr]);
    return out;
}

function ownedArray(arr) {
    const viewPtr = M._malloc(MATRIX_VIEW_BYTES);
    try {
        checkStatus(
            cc("n4m_array_view", "number", ["number", "number"], [arr, viewPtr]),
            "n4m_array_view");
        const dataPtr = M.getValue(viewPtr, "i32");
        const rows = Number(M.getValue(viewPtr + 8, "i64"));
        const cols = Number(M.getValue(viewPtr + 16, "i64"));
        return new Float64Array(M.HEAPU8.buffer, dataPtr, rows * cols).slice();
    } finally {
        M._free(viewPtr);
        cc("n4m_array_free", null, ["number"], [arr]);
    }
}

function fitRawEstimator() {
    const xPtr = M._malloc(X.byteLength);
    const yPtr = M._malloc(Y.byteLength);
    const coefsPtr = M._malloc(p * q * 8);
    const xmPtr = M._malloc(p * 8);
    const ymPtr = M._malloc(q * 8);
    const predsPtr = M._malloc(n * q * 8);
    M.HEAPF64.set(X, xPtr >>> 3);
    M.HEAPF64.set(Y, yPtr >>> 3);

    const status = M._n4m_estimators_pls_fit(
        xPtr, yPtr, n, p, q, nComponents,
        coefsPtr, xmPtr, ymPtr, predsPtr);
    if (status !== 0) throw new Error(`n4m_estimators_pls_fit failed: ${status}`);

    const out = {
        coefs: new Float64Array(M.HEAPU8.buffer, coefsPtr, p * q).slice(),
        xMean: new Float64Array(M.HEAPU8.buffer, xmPtr, p).slice(),
        yMean: new Float64Array(M.HEAPU8.buffer, ymPtr, q).slice(),
        preds: new Float64Array(M.HEAPU8.buffer, predsPtr, n * q).slice(),
    };
    M._free(xPtr); M._free(yPtr);
    M._free(coefsPtr); M._free(xmPtr); M._free(ymPtr); M._free(predsPtr);
    return out;
}

function fitModelApiUnscaled() {
    const ctxPtrPtr = M._malloc(4);
    checkStatus(cc("n4m_context_create", "number", ["number"], [ctxPtrPtr]),
        "n4m_context_create");
    const ctx = M.getValue(ctxPtrPtr, "i32");
    M._free(ctxPtrPtr);
    const cfgPtrPtr = M._malloc(4);
    checkStatus(cc("n4m_config_create", "number", ["number"], [cfgPtrPtr]),
        "n4m_config_create");
    const cfg = M.getValue(cfgPtrPtr, "i32");
    M._free(cfgPtrPtr);
    const xView = makeView(X, n, p);
    const yView = makeView(Y, n, q);
    let model = 0;
    try {
        checkStatus(cc("n4m_config_set_algorithm", "number", ["number", "number"], [cfg, 0]),
            "n4m_config_set_algorithm");
        checkStatus(cc("n4m_config_set_solver", "number", ["number", "number"], [cfg, 1]),
            "n4m_config_set_solver");
        checkStatus(cc("n4m_config_set_deflation", "number", ["number", "number"], [cfg, 0]),
            "n4m_config_set_deflation");
        checkStatus(cc("n4m_config_set_n_components", "number", ["number", "number"], [cfg, nComponents]),
            "n4m_config_set_n_components");
        checkStatus(cc("n4m_config_set_center_x", "number", ["number", "number"], [cfg, 1]),
            "n4m_config_set_center_x");
        checkStatus(cc("n4m_config_set_center_y", "number", ["number", "number"], [cfg, 1]),
            "n4m_config_set_center_y");
        checkStatus(cc("n4m_config_set_scale_x", "number", ["number", "number"], [cfg, 0]),
            "n4m_config_set_scale_x");
        checkStatus(cc("n4m_config_set_scale_y", "number", ["number", "number"], [cfg, 0]),
            "n4m_config_set_scale_y");
        const modelPtrPtr = M._malloc(4);
        M.setValue(modelPtrPtr, 0, "i32");
        checkStatus(
            cc("n4m_model_fit", "number",
                ["number", "number", "number", "number", "number"],
                [ctx, cfg, xView.viewPtr, yView.viewPtr, modelPtrPtr]),
            "n4m_model_fit");
        model = M.getValue(modelPtrPtr, "i32");
        M._free(modelPtrPtr);
        const predArrPtrPtr = M._malloc(4);
        M.setValue(predArrPtrPtr, 0, "i32");
        try {
            checkStatus(
                cc("n4m_model_predict_alloc", "number",
                    ["number", "number", "number", "number"],
                    [ctx, model, xView.viewPtr, predArrPtrPtr]),
                "n4m_model_predict_alloc");
            const predArr = M.getValue(predArrPtrPtr, "i32");
            return {
                coefs: modelArray(ctx, model, 0),
                xMean: modelArray(ctx, model, 2),
                yMean: modelArray(ctx, model, 4),
                preds: ownedArray(predArr),
            };
        } finally {
            M._free(predArrPtrPtr);
        }
    } finally {
        if (model) cc("n4m_model_destroy", null, ["number"], [model]);
        xView.free();
        yView.free();
        cc("n4m_config_destroy", null, ["number"], [cfg]);
        cc("n4m_context_destroy", null, ["number"], [ctx]);
    }
}

// ----- WASM fit -----
const fit = ref.schema === "n4a.methods.wasm_orchestrator_fixture.v1"
    ? fitModelApiUnscaled()
    : fitRawEstimator();
const coefs = fit.coefs;
const xMean = fit.xMean;
const yMean = fit.yMean;
const preds = fit.preds;

console.log("WASM coefficients:",
    Array.from(coefs).map(v => v.toFixed(6)));
console.log("WASM x_mean:",
    Array.from(xMean).map(v => v.toFixed(6)));
console.log("WASM y_mean:",
    Array.from(yMean).map(v => v.toFixed(6)));

// In-sample RMSE check.
let sumsq = 0;
for (let i = 0; i < n; i++) {
    const d = preds[i] - Y[i];
    sumsq += d * d;
}
const rmseInSample = Math.sqrt(sumsq / n);
console.log("WASM in-sample RMSE:", rmseInSample.toFixed(6));
if (!ref.schema && rmseInSample > 1e-3) {
    throw new Error(`In-sample RMSE too high: ${rmseInSample}`);
}

function rmseRel(actual, expected) {
    let sq = 0, sqExp = 0;
    for (let i = 0; i < actual.length; i++) {
        const d = actual[i] - expected[i];
        sq += d * d;
        sqExp += expected[i] * expected[i];
    }
    return Math.sqrt(sq / actual.length) /
        Math.max(1e-12, Math.sqrt(sqExp / actual.length));
}

function metric(label, actual, expected) {
    if (!Array.isArray(expected)) return null;
    const rel = rmseRel(actual, expected);
    console.log(`  ${label.padEnd(12)} rmse_rel: ${rel.toExponential(3)}`);
    return rel;
}

console.log("\nParity vs native/cross-binding fixture:");
const coefsRel = metric("coefficients", coefs, ref.coefficients);
const xMeanRel = metric("x_mean", xMean, ref.x_mean);
const yMeanRel = metric("y_mean", yMean, ref.y_mean);
const expectedPredictions = ref.predictions ?? ref.reference_predictions;
const predsRel = metric("predictions", preds, expectedPredictions);
if (predsRel === null) {
    throw new Error("Fixture must provide predictions or reference_predictions");
}

// WASM binding vs the C++ engine: same algorithm, but Emscripten fp gets a
// documented 1e-9 isolated band (achieved ~2.1e-16). Native bindings gate at 1e-12.
const tol = 1e-9;
for (const [label, value] of [
    ["coefficients", coefsRel],
    ["x_mean", xMeanRel],
    ["y_mean", yMeanRel],
    ["predictions", predsRel],
]) {
    if (value !== null && value > tol) {
        throw new Error(`Parity failure for ${label}: tolerance ${tol} exceeded`);
    }
}
if (predsRel > tol) {
    throw new Error(`Parity failure: tolerance ${tol} exceeded`);
}
console.log("\nWASM smoke + parity OK");
