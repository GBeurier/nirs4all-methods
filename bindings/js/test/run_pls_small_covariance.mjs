// SPDX-License-Identifier: CECILL-2.1
// Analytic small-covariance cases of the PLSRegression solvers and PCR
// without autoscaling, checked through the raw C ABI on W, P, Q, R, T, U,
// coefficients and predictions, on the fitted model and on its
// exported/imported state (mirrors models_pls/small_covariance_analytic in
// cpp/tests/test_models_pls.cpp). a = (-1,-1,1,1) and b = (-1,1,-1,1) are
// centred and orthogonal; every expected array is written down from the
// solver conventions:
//  (1) X = 2^15 a, Y = 2^e b + a (e = 25, 30), one component. Unit X weights
//      (NIPALS, PCR, orthogonal scores, SVD, power, randomized SVD):
//      W = P = R = 1, T = 2^15 a, Q = 2^-15, coefficient 2^-15, U = 2^15 Y
//      (NIPALS, PCR) or U = Y (unit Y weight). Unit-norm X scores (SIMPLS,
//      kernel): T = a / 2, W = R = 2^-16, P = 2^16, Q = 2, U = Y / 2.
//  (2) X = [2^30 a, b], Y = a + b, two components: coefficients (2^-30, 1),
//      predictions = Y; PCR: W = P = R = I, T = X, Q = (2^-30, 1),
//      U = [2^30 Y, Y].
//  (4) X = [a, 2^-32 b], Y = b, one component: PLS coefficients (0, 2^32),
//      predictions = Y; PCR keeps direction a (Q = 0, coefficients 0, U = 0:
//      u = Y q / q'q is undefined).
//  (3) X = [a, a]: the second component does not exist and is refused with
//      N4M_ERR_NUMERICAL_FAILURE.

import assert from "node:assert/strict";
import * as n4m from "../dist/index.js";

await n4m.loadModule();
const M = n4m.getModule();
const cc = (name, args) => M.ccall(name, "number", args.map(() => "number"), args);
const ok = (status, what) => assert.equal(status, 0, `${what} failed with status ${status}`);

const NUMERICAL_FAILURE = 7;
const PLS = 0, PCR = 10;
const SOLVERS = [
    // tag, algorithm, solver, unit-norm X scores, U = y for one target
    ["nipals", PLS, 0, false, false],
    ["pcr", PCR, 5, false, false],
    ["simpls", PLS, 1, true, false],
    ["kernel", PLS, 3, true, false],
    ["wide kernel", PLS, 4, true, false],
    ["orthogonal scores", PLS, 2, false, true],
    ["svd", PLS, 5, false, true],
    ["power", PLS, 6, false, true],
    ["randomized svd", PLS, 7, false, true],
];
// Coefficients, W, P, Q, R, T, U (n4m_model_array_t).
const ARRAYS = [[0, "coefficients"], [6, "weights_w"], [7, "loadings_p"], [8, "y_loadings_q"],
                [9, "rotations_r"], [10, "scores_t"], [11, "y_scores_u"]];
const TOL = 1e-12;

const a = [-1, -1, 1, 1], b = [-1, 1, -1, 1];
const combine = (ca, cb) => a.map((v, i) => ca * v + cb * b[i]);
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

function arrays(ctx, model, xView) {
    const out = ARRAYS.map(([kind]) => {
        const arr = withPtr((o) => ok(cc("n4m_model_get_array", [ctx, model, kind, o]), "get_array"));
        try {
            return Array.from(n4m.readArrayView(arr).data);
        } finally {
            M.ccall("n4m_array_free", null, ["number"], [arr]);
        }
    });
    const pred = withPtr((o) => ok(cc("n4m_model_predict_alloc", [ctx, model, xView, o]), "predict"));
    try {
        out.push(Array.from(n4m.readArrayView(pred).data));
    } finally {
        M.ccall("n4m_array_free", null, ["number"], [pred]);
    }
    return out;
}

// Returns { status, fitted, restored } (arrays in ARRAYS order + predictions).
function fit([, algorithm, solver], x, p, y, components) {
    const ctx = withPtr((o) => ok(cc("n4m_context_create", [o]), "context_create"));
    const cfg = withPtr((o) => ok(cc("n4m_config_create", [o]), "config_create"));
    const xv = n4m.makeMatrixView(Float64Array.from(x), 4, p);
    const yv = n4m.makeMatrixView(Float64Array.from(y), 4, 1);
    try {
        ok(M.ccall("n4m_context_set_seed", "number", ["number", "i64"], [ctx, 123456789n]), "seed");
        for (const [setter, value] of [["n_components", components], ["algorithm", algorithm],
            ["solver", solver], ["center_x", 1], ["scale_x", 0], ["center_y", 1],
            ["scale_y", 0], ["store_scores", 1]]) {
            ok(cc(`n4m_config_set_${setter}`, [cfg, value]), setter);
        }
        let status = 0;
        const model = withPtr((o) => { status = cc("n4m_model_fit", [ctx, cfg, xv.viewPtr, yv.viewPtr, o]); });
        if (status !== 0) return { status };
        try {
            const fitted = arrays(ctx, model, xv.viewPtr);
            const size = withPtr((o) => ok(cc("n4m_model_export_size", [model, o]), "export_size"));
            const buf = M._malloc(Math.max(1, size));
            try {
                const written = withPtr((o) => ok(cc("n4m_model_export_to_buffer", [model, buf, size, o]), "export"));
                const imported = withPtr((o) => ok(cc("n4m_model_import_from_buffer", [ctx, buf, written, o]), "import"));
                try {
                    return { status, fitted, restored: arrays(ctx, imported, xv.viewPtr) };
                } finally {
                    M.ccall("n4m_model_destroy", null, ["number"], [imported]);
                }
            } finally {
                M._free(buf);
            }
        } finally {
            M.ccall("n4m_model_destroy", null, ["number"], [model]);
        }
    } finally {
        xv.free();
        yv.free();
        M.ccall("n4m_config_destroy", null, ["number"], [cfg]);
        M.ccall("n4m_context_destroy", null, ["number"], [ctx]);
    }
}

function expectClose(where, name, got, want) {
    if (want === undefined) return;
    assert.equal(got.length, want.length, `${where}: ${name} size`);
    const largest = Math.max(...want.map(Math.abs));
    want.forEach((w, i) => {
        const scale = w !== 0 ? Math.abs(w) : largest;
        assert.ok(Math.abs(got[i] - w) <= TOL * scale,
            `${where}: ${name}[${i}] = ${got[i]}, expected ${w}`);
    });
}

// want: [coefficients, W, P, Q, R, T, U, predictions]; undefined = unchecked.
function check(where, result, want) {
    assert.equal(result.status, 0, `${where}: fit refused with status ${result.status}`);
    const names = [...ARRAYS.map(([, name]) => name), "predictions"];
    names.forEach((name, i) => {
        expectClose(where, name, result.fitted[i], want[i]);
        assert.deepEqual(result.restored[i], result.fitted[i], `${where}: ${name} changed through export/import`);
    });
}

let cases = 0;
// (1) one feature, Y nearly orthogonal to X.
for (const e of [25, 30]) {
    const x = combine(p2(15), 0), y = combine(1, p2(e));
    for (const s of SOLVERS) {
        const [tag, , , unitScores, rawY] = s;
        const want = unitScores
            ? [[p2(-15)], [p2(-16)], [p2(16)], [2], [p2(-16)], combine(0.5, 0), combine(0.5, p2(e - 1))]
            : [[p2(-15)], [1], [1], [p2(-15)], [1], x, rawY ? y : combine(p2(15), p2(e + 15))];
        want.push(combine(1, 0));
        check(`${tag} X = 2^15 a, Y = 2^${e} b + a`, fit(s, x, 1, y, 1), want);
        ++cases;
    }
}
// (2) singular values 2^31 and 2.
{
    const x = a.flatMap((v, i) => [p2(30) * v, b[i]]), y = combine(1, 1);
    for (const s of SOLVERS) {
        const want = [[p2(-30), 1]];
        if (s[1] === PCR) {
            const identity = [1, 0, 0, 1];
            want.push(identity, identity, [p2(-30), 1], identity, x, y.flatMap((v) => [p2(30) * v, v]));
        } else {
            want.push(undefined, undefined, undefined, undefined, undefined, undefined);
        }
        want.push(y);
        check(`${s[0]} X = [2^30 a, b], Y = a + b`, fit(s, x, 2, y, 2), want);
        ++cases;
    }
}
// (4) spread 2^32 inside X, Y follows the tiny column.
{
    const x = a.flatMap((v, i) => [v, p2(-32) * b[i]]), y = combine(0, 1);
    for (const s of SOLVERS) {
        const [tag, algorithm, , unitScores, rawY] = s;
        let want;
        if (algorithm === PCR) {
            want = [[0, 0], [1, 0], [1, 0], [0], [1, 0], combine(1, 0), [0, 0, 0, 0], [0, 0, 0, 0]];
        } else if (unitScores) {
            want = [[0, p2(32)], [0, p2(31)], [0, p2(-31)], [2], [0, p2(31)], combine(0, 0.5), combine(0, 0.5), y];
        } else {
            want = [[0, p2(32)], [0, 1], [0, 1], [p2(32)], [0, 1], combine(0, p2(-32)),
                    rawY ? y : combine(0, p2(-32)), y];
        }
        check(`${tag} X = [a, 2^-32 b], Y = b`, fit(s, x, 2, y, 1), want);
        ++cases;
    }
}
// (3) rank-one X: the second component is refused explicitly.
{
    const x = a.flatMap((v) => [v, v]);
    for (const s of SOLVERS) {
        assert.equal(fit(s, x, 2, combine(1, 1), 2).status, NUMERICAL_FAILURE,
            `${s[0]} X = [a, a]: second component not refused`);
        ++cases;
    }
}
console.log(`PLS/PCR small-covariance analytic cases OK (${cases} fits)`);
