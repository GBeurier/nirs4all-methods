// SPDX-License-Identifier: CECILL-2.1
// Unit invariance of the consumers of fitted PLS models, through the raw C
// ABI on WASM (mirrors the scale-invariance cases of
// cpp/tests/test_pls_null_components.cpp). With the centred, orthogonal
// a = (-3,-1,1,3), b = (1,-3,3,-1) (a'a = b'b = 20):
//  - VIP, X = 2^e [a, b], Y = 2^f a, one component: (sqrt 2, 0); Y = 2^f [2a, b],
//    two components: (sqrt 1.6, sqrt 0.4); X = [a, a], Y = a, two components
//    (null second component): (1, 1). SIMPLS returned (0, 0) from e = 30 and
//    NIPALS refused Y = 2^-30 a.
//  - selectivity ratio, X = 2^e [a + b, a - b], Y = a, one component: (1, 1)
//    (was 0.078125 at 2^-30); NIPALS on X = [a, 0], Y = a: (+inf, 0).
//  - DModX against X_ref = 2^e [a, 0] (Y = a, one component, every reference
//    row reconstructed): NaN with dmodx_reference_degenerate = 1 instead of
//    the unit-dependent 2^e (1, 3, 3, 1); Q = 2^2e b^2, T2 = 3 a^2 / 20 kept.
//    X_ref = 2^e [a, 2^-20 b]: DModX = |b| 2^20 / sqrt 10, flag 0.

import assert from "node:assert/strict";
import * as n4m from "../dist/index.js";
import { MethodResult } from "../dist/index.js";

await n4m.loadModule();
const M = n4m.getModule();
const cc = (name, args, types = args.map(() => "number")) => M.ccall(name, "number", types, args);
const ok = (status, what) => assert.equal(status, 0, `${what} failed with status ${status}`);
const TOL = 1e-12;
const p2 = (e) => 2 ** e;

function withPtr(fn) {
    const out = M._malloc(8);
    try {
        M.setValue(out, 0, "i32");
        fn(out);
        return M.getValue(out, "i32");
    } finally {
        M._free(out);
    }
}

function close(where, got, want, floor = 0) {
    assert.equal(got.length, want.length, `${where}: size`);
    want.forEach((w, i) => assert.ok(Math.abs(got[i] - w) <= TOL * Math.abs(w) + floor,
                                     `${where}[${i}] = ${got[i]}, expected ${w}`));
}

const a = [-3, -1, 1, 3], b = [1, -3, 3, -1];
const SOLVERS = [0, 1, 2, 3, 4, 5, 6, 7];  // NIPALS ... randomized SVD
const RANDOMIZED_SVD = 7;

function config(settings) {
    const cfg = withPtr((o) => ok(cc("n4m_config_create", [o]), "config_create"));
    for (const [setter, value] of settings) ok(cc(`n4m_config_set_${setter}`, [cfg, value]), setter);
    return cfg;
}

// Fits a PLS model on n x p X and n x q Y (centred, unscaled, store_scores).
function fitModel(x, p, y, q, components, solver = 0) {
    const n = x.length / p;
    const ctx = withPtr((o) => ok(cc("n4m_context_create", [o]), "context_create"));
    const cfg = config([["n_components", components], ["solver", solver], ["scale_x", 0],
                        ["scale_y", 0], ["store_scores", 1]]);
    const xv = n4m.makeMatrixView(Float64Array.from(x), n, p);
    const yv = n4m.makeMatrixView(Float64Array.from(y), n, q);
    try {
        const model = withPtr((o) => ok(cc("n4m_model_fit", [ctx, cfg, xv.viewPtr, yv.viewPtr, o]), "fit"));
        return { ctx, model };
    } finally {
        xv.free();
        yv.free();
        M.ccall("n4m_config_destroy", null, ["number"], [cfg]);
    }
}

function release({ ctx, model }) {
    M.ccall("n4m_model_destroy", null, ["number"], [model]);
    M.ccall("n4m_context_destroy", null, ["number"], [ctx]);
}

// Variable-importance scores (method 0 = VIP, 2 = selectivity ratio).
function variableScores(fit, x, p, method) {
    const xv = n4m.makeMatrixView(Float64Array.from(x), x.length / p, p);
    try {
        const res = withPtr((o) => ok(cc("n4m_feature_selection_variable_select_rank",
            [fit.ctx, fit.model, xv.viewPtr, method, 1, o]), `variable_select_rank ${method}`));
        const r = new MethodResult(res);
        try {
            return Array.from(r.matrix("scores").data);
        } finally {
            r.destroy();
        }
    } finally {
        xv.free();
    }
}

let checks = 0;
// VIP is invariant to the units of X and Y.
for (const solver of SOLVERS) {
    const floor = solver === RANDOMIZED_SVD ? 1e-8 : TOL;
    for (const e of [0, 30, 254, -200]) {
        for (const f of [0, -30, -200, 200]) {
            const x = a.flatMap((v, i) => [v * p2(e), b[i] * p2(e)]);
            const one = fitModel(x, 2, a.map((v) => v * p2(f)), 1, 1, solver);
            const two = fitModel(x, 2, a.flatMap((v, i) => [2 * v * p2(f), b[i] * p2(f)]), 2, 2, solver);
            try {
                const where = `VIP solver ${solver} X = 2^${e} [a, b], Y = 2^${f}`;
                close(`${where} a`, variableScores(one, x, 2, 0), [Math.SQRT2, 0], floor);
                close(`${where} [2a, b]`, variableScores(two, x, 2, 0), [Math.sqrt(1.6), Math.sqrt(0.4)], floor);
                checks += 2;
            } finally {
                release(one);
                release(two);
            }
        }
    }
    const x = a.flatMap((v) => [v, v]);
    const fit = fitModel(x, 2, a, 1, 2, solver);
    try {
        close(`VIP solver ${solver} X = [a, a]`, variableScores(fit, x, 2, 0), [1, 1]);
        ++checks;
    } finally {
        release(fit);
    }
}
// The selectivity ratio is invariant to the units of X.
for (const solver of SOLVERS) {
    for (const e of [0, -30, -256, 30, 250]) {
        const x = a.flatMap((v, i) => [(v + b[i]) * p2(e), (v - b[i]) * p2(e)]);
        const fit = fitModel(x, 2, a, 1, 1, solver);
        try {
            close(`SR solver ${solver} X = 2^${e} [a + b, a - b]`, variableScores(fit, x, 2, 2), [1, 1], 1e-11);
            ++checks;
        } finally {
            release(fit);
        }
    }
}
{
    const x = a.flatMap((v) => [v, 0]);
    const fit = fitModel(x, 2, a, 1, 1, 0);
    try {
        assert.deepEqual(variableScores(fit, x, 2, 2), [Infinity, 0], "SR X = [a, 0]: (+inf, 0)");
        ++checks;
    } finally {
        release(fit);
    }
}
// DModX against a reference whose residual vanishes.
for (const solver of SOLVERS) {
    for (const e of [0, -30, 200]) {
        for (const degenerate of [true, false]) {
            const xRef = a.flatMap((v, i) => [v * p2(e), degenerate ? 0 : b[i] * p2(e - 20)]);
            const x = a.flatMap((v, i) => [v * p2(e), b[i] * p2(e)]);
            const fit = fitModel(xRef, 2, a, 1, 1, solver);
            const xv = n4m.makeMatrixView(Float64Array.from(x), 4, 2);
            const rv = n4m.makeMatrixView(Float64Array.from(xRef), 4, 2);
            try {
                const where = `DModX solver ${solver} X_ref = 2^${e} ${degenerate ? "[a, 0]" : "[a, 2^-20 b]"}`;
                const res = withPtr((o) => ok(cc("n4m_metrics_pls_diagnostics_compute",
                    [fit.ctx, fit.model, xv.viewPtr, rv.viewPtr, o]), "diagnostics"));
                const r = new MethodResult(res);
                try {
                    close(`${where} q`, Array.from(r.matrix("q").data), b.map((v) => v * v * p2(2 * e)));
                    close(`${where} t2`, Array.from(r.matrix("t2").data), a.map((v) => 3 * v * v / 20));
                    const dmodx = Array.from(r.matrix("dmodx").data);
                    if (degenerate) {
                        assert.equal(r.scalar("dmodx_reference_degenerate"), 1, `${where}: flag`);
                        assert.ok(dmodx.every(Number.isNaN), `${where}: DModX is NaN`);
                    } else {
                        assert.equal(r.scalar("dmodx_reference_degenerate"), 0, `${where}: flag`);
                        close(`${where} dmodx`, dmodx, b.map((v) => Math.abs(v) * p2(20) / Math.sqrt(10)));
                    }
                    ++checks;
                } finally {
                    r.destroy();
                }
            } finally {
                xv.free();
                rv.free();
                release(fit);
            }
        }
    }
}
console.log(`PLS scale-invariant consumers OK (${checks} checks)`);
