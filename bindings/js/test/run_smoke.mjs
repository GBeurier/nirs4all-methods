// SPDX-License-Identifier: CECILL-2.1
//
// Node smoke + parity:
// 1. Loads the WASM build, prints version / ABI.
// 2. Fits a SIMPLS PLS regression on deterministic data via the
//    `n4m_wasm_pls_fit` helper.
// 3. Predicts in-sample.
// 4. Loads the parity reference (fixture saved by
//    `bindings/js/test/generate_parity_fixture.py`, which calls the
//    native Python binding on the same X / Y).
// 5. Compares coefficients + predictions; fails if RMSE-rel > 1e-3.

import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { readFileSync, existsSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = resolve(here, "..", "..", "..", "build", "emscripten",
                        "bindings", "js");
const factory = (await import(resolve(wasmDir, "n4m.js"))).default;
const M = await factory({locateFile: (p) => resolve(wasmDir, p)});

// ----- Version + ABI -----
const versionPtr = M._n4m_get_version_string();
const version = M.UTF8ToString(versionPtr);
console.log("pls4all version:", version);
const abi = [
    M._n4m_get_abi_version_major(),
    M._n4m_get_abi_version_minor(),
    M._n4m_get_abi_version_patch(),
];
console.log("ABI:", abi.join("."));

// ----- Deterministic input (matched to the Python fixture) -----
const n = 50, p = 5, q = 1;
const X = new Float64Array(n * p);
const Y = new Float64Array(n * q);
for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
        X[i * p + j] = Math.sin((i + 1) * (j + 1) * 0.3);
    }
    Y[i] = X[i * p] + 0.5 * X[i * p + 1] - 0.3 * X[i * p + 2];
}

// ----- WASM fit -----
const xPtr = M._malloc(X.byteLength);
const yPtr = M._malloc(Y.byteLength);
const coefsPtr = M._malloc(p * q * 8);
const xmPtr = M._malloc(p * 8);
const ymPtr = M._malloc(q * 8);
const predsPtr = M._malloc(n * q * 8);
M.HEAPF64.set(X, xPtr >>> 3);
M.HEAPF64.set(Y, yPtr >>> 3);

const status = M._n4m_pls_fit_simple(
    xPtr, yPtr, n, p, q, 3,
    coefsPtr, xmPtr, ymPtr, predsPtr);
if (status !== 0) throw new Error(`n4m_pls_fit_simple failed: ${status}`);

const coefs = new Float64Array(M.HEAPU8.buffer, coefsPtr, p * q).slice();
const xMean = new Float64Array(M.HEAPU8.buffer, xmPtr, p).slice();
const yMean = new Float64Array(M.HEAPU8.buffer, ymPtr, q).slice();
const preds = new Float64Array(M.HEAPU8.buffer, predsPtr, n * q).slice();
M._free(xPtr); M._free(yPtr);
M._free(coefsPtr); M._free(xmPtr); M._free(ymPtr); M._free(predsPtr);

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
if (rmseInSample > 1e-3) {
    throw new Error(`In-sample RMSE too high: ${rmseInSample}`);
}

// ----- Parity vs native Python -----
const fixturePath = resolve(here, "parity_fixture.json");
if (!existsSync(fixturePath)) {
    console.warn(
        `[WARN] parity fixture missing at ${fixturePath}.\n` +
        `      Regenerate via:\n` +
        `      PYTHONPATH=bindings/python/src parity/python_generator/.venv/bin/python \\\n` +
        `        bindings/js/test/generate_parity_fixture.py`);
    console.log("WASM smoke OK (parity skipped)");
    process.exit(0);
}
const ref = JSON.parse(readFileSync(fixturePath, "utf-8"));

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

const coefsRel = rmseRel(coefs, ref.coefficients);
const xMeanRel = rmseRel(xMean, ref.x_mean);
const yMeanRel = rmseRel(yMean, ref.y_mean);
const predsRel = rmseRel(preds, ref.predictions);
console.log("\nParity vs native Python pls4all:");
console.log(`  coefficients rmse_rel: ${coefsRel.toExponential(3)}`);
console.log(`  x_mean       rmse_rel: ${xMeanRel.toExponential(3)}`);
console.log(`  y_mean       rmse_rel: ${yMeanRel.toExponential(3)}`);
console.log(`  predictions  rmse_rel: ${predsRel.toExponential(3)}`);

// WASM binding vs the C++ engine: same algorithm, but Emscripten fp gets a
// documented 1e-9 isolated band (achieved ~2.1e-16). Native bindings gate at 1e-12.
const tol = 1e-9;
if (coefsRel > tol || xMeanRel > tol || yMeanRel > tol || predsRel > tol) {
    throw new Error(`Parity failure: tolerance ${tol} exceeded`);
}
console.log("\nWASM smoke + parity OK");
