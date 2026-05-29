// SPDX-License-Identifier: CECILL-2.1
//
// Item #19 regression guard: a JS-built n4m_matrix_view_t* reaches the deep
// entrypoints correctly once i64 dims are marshalled as BigInt under
// WASM_BIGINT=1. Proves the generic method_result path (n4m_sparse_simpls_fit)
// and the generic model path (n4m_model_fit -> get_array -> array_view) match
// the raw-pointer oracle (n4m_pls_fit_simple) byte-for-byte.

import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const wasmDir = resolve(here, "..", "..", "..", "build", "emscripten",
                        "bindings", "js");
const factory = (await import(resolve(wasmDir, "n4m.js"))).default;
const M = await factory({});
const cc = (n, r, t, a) => M.ccall(n, r, t, a);
const F64 = 1, MV = 48, ncomp = 2;

const n = 6, p = 4, q = 1;
const X = new Float64Array([1,2,3,4, 2,1,0,3, 3,0,1,2, 0,3,2,1, 4,2,1,0, 1,1,1,1]);
const Y = new Float64Array([10,7,6,6,9,4]);

function makeView(data, rows, cols) {
    const dp = M._malloc(data.length * 8); M.HEAPF64.set(data, dp / 8);
    const vp = M._malloc(MV);
    const st = cc("n4m_matrix_view_init_rowmajor", "number",
        ["number", "number", "i64", "i64", "number"],
        [vp, dp, BigInt(rows), BigInt(cols), F64]);
    if (st !== 0) throw new Error("view init " + st);
    return { dp, vp, free() { M._free(vp); M._free(dp); } };
}
function resultMatrix(res, name) {
    const nb = M.lengthBytesUTF8(name); const np = M._malloc(nb + 1);
    M.stringToUTF8(name, np, nb + 1);
    const dpp = M._malloc(4), rp = M._malloc(8), cp = M._malloc(8);
    const st = cc("n4m_method_result_get_double_matrix", "number",
        ["number", "number", "number", "number", "number"], [res, np, dpp, rp, cp]);
    let out = null;
    if (st === 0) {
        const dp = M.getValue(dpp, "i32");
        const rows = Number(M.getValue(rp, "i64")), cols = Number(M.getValue(cp, "i64"));
        out = Array.from(M.HEAPF64.subarray(dp / 8, dp / 8 + rows * cols));
    }
    M._free(np); M._free(dpp); M._free(rp); M._free(cp); return out;
}
function cfg() {
    const pp = M._malloc(4); cc("n4m_config_create", "number", ["number"], [pp]);
    const c = M.getValue(pp, "i32"); M._free(pp);
    cc("n4m_config_set_algorithm", "number", ["number", "number"], [c, 0]);
    cc("n4m_config_set_solver", "number", ["number", "number"], [c, 1]);
    cc("n4m_config_set_deflation", "number", ["number", "number"], [c, 0]);
    cc("n4m_config_set_n_components", "number", ["number", "number"], [c, ncomp]);
    cc("n4m_config_set_center_x", "number", ["number", "number"], [c, 1]);
    cc("n4m_config_set_center_y", "number", ["number", "number"], [c, 1]);
    return c;
}
function approx(a, b, tol = 1e-12) {
    return a && b && a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) < tol);
}

// Oracle: raw-pointer path (always worked).
let oracle;
{
    const xp = M._malloc(n*p*8); M.HEAPF64.set(X, xp/8);
    const yp = M._malloc(n*q*8); M.HEAPF64.set(Y, yp/8);
    const cp = M._malloc(p*q*8), xm = M._malloc(p*8), ym = M._malloc(q*8);
    cc("n4m_pls_fit_simple", "number",
        ["number","number","number","number","number","number","number","number","number","number"],
        [xp, yp, n, p, q, ncomp, cp, xm, ym, 0]);
    oracle = Array.from(M.HEAPF64.subarray(cp/8, cp/8 + p*q));
    M._free(xp); M._free(yp); M._free(cp); M._free(xm); M._free(ym);
}

const ctxPP = M._malloc(4); cc("n4m_context_create", "number", ["number"], [ctxPP]);
const ctx = M.getValue(ctxPP, "i32"); M._free(ctxPP);
let ok = true;

// (1) generic method_result producer
{
    const c = cfg(); const xv = makeView(X, n, p), yv = makeView(Y, n, q);
    const resPP = M._malloc(4); M.setValue(resPP, 0, "i32");
    const st = cc("n4m_sparse_simpls_fit", "number",
        ["number","number","number","number","number","number"],
        [ctx, c, xv.vp, yv.vp, 0.0, resPP]);
    const res = M.getValue(resPP, "i32"); M._free(resPP);
    const coef = resultMatrix(res, "coefficients");
    const pass = st === 0 && approx(coef, oracle);
    console.log("method_result path (n4m_sparse_simpls_fit):", pass ? "OK" : "FAIL", coef);
    ok = ok && pass;
    cc("n4m_method_result_destroy", null, ["number"], [res]);
    xv.free(); yv.free(); cc("n4m_config_destroy", null, ["number"], [c]);
}
// (2) generic model path
{
    const c = cfg(); const xv = makeView(X, n, p), yv = makeView(Y, n, q);
    const mpp = M._malloc(4); M.setValue(mpp, 0, "i32");
    const st = cc("n4m_model_fit", "number",
        ["number","number","number","number","number"], [ctx, c, xv.vp, yv.vp, mpp]);
    const model = M.getValue(mpp, "i32"); M._free(mpp);
    const app = M._malloc(4); M.setValue(app, 0, "i32");
    cc("n4m_model_get_array", "number", ["number","number","number","number"], [ctx, model, 0, app]);
    const arr = M.getValue(app, "i32"); M._free(app);
    const vp = M._malloc(MV); cc("n4m_array_view", "number", ["number","number"], [arr, vp]);
    const dp = M.getValue(vp, "i32");
    const rows = Number(M.getValue(vp+8, "i64")), cols = Number(M.getValue(vp+16, "i64"));
    const coef = Array.from(M.HEAPF64.subarray(dp/8, dp/8 + rows*cols)); M._free(vp);
    const pass = st === 0 && approx(coef, oracle);
    console.log("model path (n4m_model_fit -> get_array):", pass ? "OK" : "FAIL", coef);
    ok = ok && pass;
    cc("n4m_array_free", null, ["number"], [arr]);
    cc("n4m_model_destroy", null, ["number"], [model]);
    xv.free(); yv.free(); cc("n4m_config_destroy", null, ["number"], [c]);
}
cc("n4m_context_destroy", null, ["number"], [ctx]);
console.log(ok ? "\nITEM#19 PASS" : "\nITEM#19 FAIL");
process.exit(ok ? 0 : 1);
