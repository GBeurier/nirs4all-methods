// SPDX-License-Identifier: CECILL-2.1
//
// Node smoke for the operator-adaptive PLS surface:
//   1. POP-PLS (per-component AOM) via n4m_wasm_pop_fit → finite input-space
//      coeffs + intercept + per-component selected operators (in bank range).
//   2. AOM-PLS via n4m_wasm_aom_fit with the DEFAULT bank and with a CUSTOM
//      bank ([IDENTITY, SAVGOL_DERIVATIVE]); asserts predictions are finite and
//      that the custom bank actually CHANGES the result vs the default bank.
//
// Runs against the freshly-built emscripten artifact (no staging required).

import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = resolve(here, "..", "..", "..", "build", "emscripten",
                        "bindings", "js");
const factory = (await import(resolve(wasmDir, "n4m.js"))).default;
const M = await factory({ locateFile: (p) => resolve(wasmDir, p) });

let failed = 0;
function ok(cond, msg) {
    if (cond) console.log("  ✓ " + msg);
    else { console.error("  ✗ " + msg); failed = 1; }
}

// strict-linear operator kinds (from n4m_operator_kind_t)
const OP_IDENTITY = 0, OP_SAVGOL_DERIVATIVE = 9;

// ---- deterministic data: spectra-like rows (varying baseline + peaks +
// per-row noise so the operator screen has real signal to discriminate) and a
// target driven by the peak amplitudes (which a derivative/detrend can recover
// better than identity), all under a deterministic LCG so runs are stable. ----
let _seed = 12345;
function rnd() { _seed = (_seed * 1103515245 + 12345) & 0x7fffffff; return _seed / 0x7fffffff; }
const n = 60, p = 32, q = 1;
const X = new Float64Array(n * p);
const Y = new Float64Array(n * q);
for (let i = 0; i < n; ++i) {
    const a = 1.0 + 2.0 * rnd();          // peak-1 amplitude (drives the target)
    const b = 0.5 + 1.0 * rnd();          // peak-2 amplitude
    const slope = 0.5 * rnd();            // per-row sloping baseline (scatter)
    const offset = 0.8 * rnd();           // per-row additive offset (scatter)
    for (let j = 0; j < p; ++j) {
        const peak1 = a * Math.exp(-((j - 10) ** 2) / 8);
        const peak2 = b * Math.exp(-((j - 22) ** 2) / 6);
        const baseline = offset + slope * (j / p);
        X[i * p + j] = peak1 + peak2 + baseline + 0.01 * (rnd() - 0.5);
    }
    // target depends on peak amplitudes; the baseline (offset/slope) is nuisance
    // that detrend/derivative removes — so the operator screen has a real winner.
    Y[i] = 3.0 + 1.5 * a + 0.8 * b + 0.02 * (rnd() - 0.5);
}

function malloc_f64(arr) {
    const ptr = M._malloc(arr.length * 8);
    M.HEAPF64.set(arr, ptr >> 3);
    return ptr;
}

function predictAffine(coeffs, intercept, Xnew, nNew) {
    const out = new Float64Array(nNew * q);
    for (let i = 0; i < nNew; ++i) {
        for (let t = 0; t < q; ++t) {
            let s = intercept[t];
            for (let f = 0; f < p; ++f) s += Xnew[i * p + f] * coeffs[f * q + t];
            out[i * q + t] = s;
        }
    }
    return out;
}

function finite(arr) { return arr.every((v) => Number.isFinite(v)); }
function rmse(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; ++i) s += (a[i] - b[i]) ** 2;
    return Math.sqrt(s / a.length);
}

const maxComp = 6, nFolds = 5;

// ============================ POP-PLS =====================================
console.log("POP-PLS (per-component AOM):");
{
    const xPtr = malloc_f64(X), yPtr = malloc_f64(Y);
    const coefs = M._malloc(p * q * 8);
    const inter = M._malloc(q * 8);
    const opsOut = M._malloc(maxComp * 4);
    const nSel = M._malloc(4);
    const score = M._malloc(8);
    const status = M.ccall(
        "n4m_wasm_pop_fit", "number",
        ["number", "number", "number", "number", "number",
         "number", "number", "number", "number", "number",
         "number", "number", "number", "number", "number"],
        [xPtr, yPtr, n, p, q, maxComp, nFolds, 0, 0, 0,
         coefs, inter, opsOut, nSel, score]);
    ok(status === 0, `n4m_wasm_pop_fit returned 0 (got ${status})`);
    const B = new Float64Array(M.HEAPF64.buffer, coefs, p * q).slice();
    const I = new Float64Array(M.HEAPF64.buffer, inter, q).slice();
    const k = M.HEAP32[nSel >> 2];
    const selOps = new Int32Array(M.HEAP32.buffer, opsOut, maxComp).slice();
    const sc = M.HEAPF64[score >> 3];
    ok(finite(B) && finite(I), "POP coeffs + intercept are finite");
    ok(k >= 1 && k <= maxComp, `POP selected ${k} component(s) in [1, ${maxComp}]`);
    const sel = Array.from(selOps.slice(0, k));
    // default bank has 5 operators (indices 0..4)
    ok(sel.every((o) => o >= 0 && o < 5), `POP per-component ops in bank range: [${sel.join(", ")}]`);
    ok(Number.isFinite(sc), `POP best score finite (${sc.toFixed(4)})`);
    const preds = predictAffine(B, I, X, n);
    ok(finite(preds), "POP in-sample predictions are finite");
    const target = Array.from(Y);
    const ymean = target.reduce((a, v) => a + v, 0) / target.length;
    const rmseFit = rmse(Array.from(preds), target);
    const rmseBaseline = rmse(target.map(() => ymean), target);
    ok(rmseFit < rmseBaseline,
       `POP fit beats the constant baseline (fit=${rmseFit.toFixed(4)} < base=${rmseBaseline.toFixed(4)})`);
    for (const ptr of [xPtr, yPtr, coefs, inter, opsOut, nSel, score]) M._free(ptr);
}

// ============================ AOM-PLS: default vs custom bank ==============
console.log("AOM-PLS (default bank vs custom bank):");
function fitAom(opsArr) {
    const xPtr = malloc_f64(X), yPtr = malloc_f64(Y);
    const coefs = M._malloc(p * q * 8);
    const inter = M._malloc(q * 8);
    const sel = M._malloc(4);
    const score = M._malloc(8);
    let opsPtr = 0;
    if (opsArr.length > 0) {
        opsPtr = M._malloc(opsArr.length * 4);
        M.HEAP32.set(Int32Array.from(opsArr), opsPtr >> 2);
    }
    const status = M.ccall(
        "n4m_wasm_aom_fit", "number",
        ["number", "number", "number", "number", "number",
         "number", "number", "number", "number", "number",
         "number", "number", "number", "number"],
        [xPtr, yPtr, n, p, q, maxComp, nFolds, 0, opsPtr, opsArr.length,
         coefs, inter, sel, score]);
    const B = new Float64Array(M.HEAPF64.buffer, coefs, p * q).slice();
    const I = new Float64Array(M.HEAPF64.buffer, inter, q).slice();
    const selOp = M.HEAP32[sel >> 2];
    for (const ptr of [xPtr, yPtr, coefs, inter, sel, score]) M._free(ptr);
    if (opsPtr) M._free(opsPtr);
    return { status, B, I, selOp };
}

const def = fitAom([]);
ok(def.status === 0, `default-bank AOM returned 0 (got ${def.status})`);
ok(finite(def.B) && finite(def.I), "default-bank coeffs + intercept finite");
const defPreds = predictAffine(def.B, def.I, X, n);
ok(finite(defPreds), "default-bank predictions finite");

const custom = fitAom([OP_IDENTITY, OP_SAVGOL_DERIVATIVE]);
ok(custom.status === 0, `custom-bank AOM returned 0 (got ${custom.status})`);
ok(finite(custom.B) && finite(custom.I), "custom-bank coeffs + intercept finite");
ok(custom.selOp >= 0 && custom.selOp < 2, `custom-bank selected op in [0,1] (got ${custom.selOp})`);
const customPreds = predictAffine(custom.B, custom.I, X, n);
ok(finite(customPreds), "custom-bank predictions finite");

// PROOF: the custom 2-op bank must actually change the fitted model. The
// default bank screens 5 operators and is overwhelmingly likely to pick a
// derivative/detrend winner different from the [IDENTITY, SG-deriv] restriction.
const coefDelta = rmse(Array.from(def.B), Array.from(custom.B));
ok(coefDelta > 1e-9,
   `custom bank CHANGES the coefficients vs default (||ΔB||rmse=${coefDelta.toExponential(3)})`);

console.log(failed ? "POP/AOM SMOKE FAILED" : "POP/AOM SMOKE PASSED");
process.exit(failed);
