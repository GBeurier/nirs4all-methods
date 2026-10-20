// SPDX-License-Identifier: CECILL-2.1
// Consumers of PLS models with null (unresolved) components, through the raw
// C ABI on WASM (mirrors cpp/tests/test_pls_null_components.cpp). With the
// centred, orthogonal a = (-3,-1,1,3), b = (1,-3,3,-1) (a'a = b'b = 20):
//  - diagnostics, X = [a, a, b], Y = a, two components (the second null):
//    T2 = 3 a^2 / 20, Q = b^2, DModX = |b| / sqrt 2 (|b| / sqrt 10 with X as
//    reference), n_latent_directions = 1, equal to the one-component model;
//  - diagnostics of an autoscaled model, X = [a, 2 b], Y = a, one component:
//    T2 = 3 a^2 / 20, Q = 3 b^2 / 20;
//  - PLS-LDA, X = [a, a], classes (0, 0, 1, 1), two components: decision
//    log(1/2) - 1 -+ a_i, predictions (0, 0, 1, 1), equal to one component;
//  - T2 selection, X = [a, 2a, 3a, 4a], Y = a, two components: T2 = (1.35,
//    0.15, 0.15, 1.35), limits 2.25 (1 - alpha)^2 rounded: 2.03 / 1.82 at
//    alpha 0.05 / 0.1, equal to one component.
// Each of these was refused before (zero reference variance, singular
// covariance) or, for the autoscaled diagnostics, wrong.

import assert from "node:assert/strict";
import * as n4m from "../dist/index.js";
import { MethodResult } from "../dist/index.js";

await n4m.loadModule();
const M = n4m.getModule();
const cc = (name, args, types = args.map(() => "number")) => M.ccall(name, "number", types, args);
const ok = (status, what) => assert.equal(status, 0, `${what} failed with status ${status}`);
const TOL = 1e-12;

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

function close(where, got, want) {
    assert.equal(got.length, want.length, `${where}: size`);
    want.forEach((w, i) => assert.ok(Math.abs(got[i] - w) <= TOL * Math.abs(w) + 1e-300,
                                     `${where}[${i}] = ${got[i]}, expected ${w}`));
}

const a = [-3, -1, 1, 3], b = [1, -3, 3, -1];

function config(settings) {
    const cfg = withPtr((o) => ok(cc("n4m_config_create", [o]), "config_create"));
    for (const [setter, value] of settings) ok(cc(`n4m_config_set_${setter}`, [cfg, value]), setter);
    return cfg;
}

// Fits a PLS model (centred, store_scores) and returns { ctx, model }.
function fitModel(x, p, y, components, solver = 0, scale = 0) {
    const ctx = withPtr((o) => ok(cc("n4m_context_create", [o]), "context_create"));
    const cfg = config([["n_components", components], ["solver", solver], ["scale_x", scale],
                        ["scale_y", scale], ["store_scores", 1]]);
    const xv = n4m.makeMatrixView(Float64Array.from(x), 4, p);
    const yv = n4m.makeMatrixView(Float64Array.from(y), 4, 1);
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

function diagnostics(fit, x, p, withReference) {
    const xv = n4m.makeMatrixView(Float64Array.from(x), 4, p);
    try {
        const res = withPtr((o) => ok(cc("n4m_metrics_pls_diagnostics_compute",
            [fit.ctx, fit.model, xv.viewPtr, withReference ? xv.viewPtr : 0, o]), "diagnostics"));
        const r = new MethodResult(res);
        try {
            return { t2: Array.from(r.matrix("t2").data), q: Array.from(r.matrix("q").data),
                     dmodx: Array.from(r.matrix("dmodx").data), k: r.scalar("n_latent_directions") };
        } finally {
            r.destroy();
        }
    } finally {
        xv.free();
    }
}

let checks = 0;
// Diagnostics with a null component, every PLS solver.
{
    const x = a.flatMap((v, i) => [v, v, b[i]]);
    for (const solver of [0, 1, 2, 3, 4, 5, 6, 7]) {
        const two = fitModel(x, 3, a, 2, solver), one = fitModel(x, 3, a, 1, solver);
        try {
            for (const ref of [false, true]) {
                const where = `diagnostics solver ${solver}${ref ? " (reference)" : ""}`;
                const d = diagnostics(two, x, 3, ref), d1 = diagnostics(one, x, 3, ref);
                close(`${where} t2`, d.t2, a.map((v) => 3 * v * v / 20));
                close(`${where} q`, d.q, b.map((v) => v * v));
                close(`${where} dmodx`, d.dmodx, b.map((v) => Math.abs(v) / Math.sqrt(ref ? 10 : 2)));
                assert.equal(d.k, 1, `${where}: n_latent_directions`);
                assert.deepEqual(d, { ...d1, k: d.k }, `${where}: differs from the one-component model`);
                ++checks;
            }
        } finally {
            release(two);
            release(one);
        }
    }
}
// Autoscaled diagnostics.
{
    const x = a.flatMap((v, i) => [v, 2 * b[i]]);
    const fit = fitModel(x, 2, a, 1, 0, 1);
    try {
        const d = diagnostics(fit, x, 2, false);
        close("autoscaled t2", d.t2, a.map((v) => 3 * v * v / 20));
        close("autoscaled q", d.q, b.map((v) => 3 * v * v / 20));
        ++checks;
    } finally {
        release(fit);
    }
}
// PLS-LDA with a null component.
function lda(components, solver) {
    const ctx = withPtr((o) => ok(cc("n4m_context_create", [o]), "context_create"));
    const cfg = config([["n_components", components], ["solver", solver], ["scale_x", 0], ["scale_y", 0]]);
    const xv = n4m.makeMatrixView(Float64Array.from(a.flatMap((v) => [v, v])), 4, 2);
    const labels = M._malloc(16);
    M.HEAP32.set([0, 0, 1, 1], labels / 4);
    try {
        const res = withPtr((o) => ok(cc("n4m_estimators_pls_lda_fit",
            [ctx, cfg, xv.viewPtr, labels, 4n, 2, o],
            ["number", "number", "number", "number", "i64", "number", "number"]), "pls_lda_fit"));
        const r = new MethodResult(res);
        try {
            return { decision: Array.from(r.matrix("decision_scores").data),
                     predictions: Array.from(r.vectorInt("predictions")) };
        } finally {
            r.destroy();
        }
    } finally {
        M._free(labels);
        xv.free();
        M.ccall("n4m_config_destroy", null, ["number"], [cfg]);
        M.ccall("n4m_context_destroy", null, ["number"], [ctx]);
    }
}
for (const solver of [0, 1]) {
    const two = lda(2, solver), one = lda(1, solver);
    close(`pls-lda solver ${solver} decision`, two.decision,
          a.flatMap((v) => [Math.log(0.5) - 1 - v, Math.log(0.5) - 1 + v]));
    assert.deepEqual(two.predictions, [0, 0, 1, 1], "pls-lda predictions");
    assert.deepEqual(two, one, "pls-lda differs from the one-component head");
    ++checks;
}
// T2 selection with a null component.
function t2Select(components) {
    const ctx = withPtr((o) => ok(cc("n4m_context_create", [o]), "context_create"));
    const cfg = config([["n_components", components], ["scale_x", 0], ["scale_y", 0]]);
    const plan = withPtr((o) => ok(cc("n4m_validation_plan_create", [o]), "plan_create"));
    const xv = n4m.makeMatrixView(Float64Array.from(a.flatMap((v) => [v, 2 * v, 3 * v, 4 * v])), 4, 4);
    const yv = n4m.makeMatrixView(Float64Array.from(a), 4, 1);
    const alphas = M._malloc(16);
    M.HEAPF64.set([0.05, 0.1], alphas / 8);
    const idx = M._malloc(32);
    try {
        ok(cc("n4m_validation_plan_set_n_samples", [plan, 4n], ["number", "i64"]), "n_samples");
        for (let fold = 0; fold < 4; ++fold) {
            const train = [0, 1, 2, 3].filter((i) => i !== fold);
            M.HEAP64.set([...train, fold].map(BigInt), idx / 8);
            ok(cc("n4m_validation_plan_add_fold", [plan, idx, 3n, idx + 24, 1n],
                  ["number", "number", "i64", "number", "i64"]), "add_fold");
        }
        const res = withPtr((o) => ok(cc("n4m_feature_selection_t2_select",
            [ctx, cfg, xv.viewPtr, yv.viewPtr, plan, alphas, 2n, 2, o],
            ["number", "number", "number", "number", "number", "number", "i64", "number", "number"]),
            "t2_select"));
        const r = new MethodResult(res);
        try {
            return { t2: Array.from(r.matrix("t2_scores").data), ucl: Array.from(r.matrix("ucl_by_alpha").data),
                     rmse: Array.from(r.matrix("rmse_by_alpha").data), k: r.scalar("n_latent_directions") };
        } finally {
            r.destroy();
        }
    } finally {
        M._free(idx);
        M._free(alphas);
        xv.free();
        yv.free();
        M.ccall("n4m_validation_plan_destroy", null, ["number"], [plan]);
        M.ccall("n4m_config_destroy", null, ["number"], [cfg]);
        M.ccall("n4m_context_destroy", null, ["number"], [ctx]);
    }
}
{
    const two = t2Select(2), one = t2Select(1);
    close("t2 selection scores", two.t2, [1.35, 0.15, 0.15, 1.35]);
    close("t2 selection limits", two.ucl, [2.03, 1.82]);
    assert.equal(two.k, 1, "t2 selection n_latent_directions");
    assert.deepEqual({ ...two, k: 1 }, { ...one, k: 1 }, "t2 selection differs from one component");
    ++checks;
}
console.log(`PLS null-component consumers OK (${checks} checks)`);
