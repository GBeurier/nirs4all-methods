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
//  (3) X = [a, a], Y = a + b, two components: the second component does not
//      exist; every solver publishes it as a null component flagged
//      unresolved (zero W, P, Q, R, T, U columns), coefficients (1/2, 1/2),
//      predictions = a.
// Every fit also publishes its component resolution ((n + a) eps ||X|| ||Y||
// for PLS, (n + p) eps sigma_max ||Y|| for PCR) and per-component unresolved
// flags, checked on the fitted model, on the imported one and through
// NativeModel.componentResolution().
// Unresolved components and multi-target resolution, with
// a3 = (-3,-1,1,3), b3 = (1,-3,3,-1) (a3'a3 = b3'b3 = 20):
//  (5) X = a3, Y = b3 + 2^-50 a3: exactly Q = 2^-50 and U = 2^50 Y, but the
//      cross-covariance 20 2^-50 is below the PCR resolution: PCR flags the
//      component unresolved (U column zero, Q = 2^-50, coefficient 2^-50);
//      every PLS solver (and PLS-DA) publishes a null component flagged
//      unresolved: all latent arrays 0, coefficient 0, predictions = Y mean.
//  (6) X = [2 a3, b3], Y = [a3 + 2^-48 b3, 3 a3 - 2^-48 b3], PCR, two
//      components: Q = [[1/2, 2^-48], [3/2, -2^-48]], the second unresolved,
//      U = [Y q1 / q1'q1, 0].
//  (7) X = [a3, b3], Y = [a3, 64 x (2^-29 a3 + 2^-51 b3)]: the residual
//      columns fall below scikit-learn's 10 eps mask but carry a resolved
//      covariance together (NIPALS refused); coefficients
//      [[1, 2^-29 ...], [0, 2^-51 ...]], predictions = Y, every solver.
//  (8) X = [a, b], Y = [a, 100 x (2^-29 a + 2^-52 b)]: no residual entry
//      above eps (NIPALS and kernel refused); same coefficients pattern.
//  (9) X = [a3, b3], Y = [a3 + d b3, d a3 + b3], d = 2^-12, 2^-20, 2^-30:
//      close leading singular values 20 (1 +- d); every solver and PLS-DA
//      returns the dominant direction (1, 1) / sqrt 2 (see below).
//  (10) the same with X = [a3, b3, c3] and 64 targets (see below).
//  (11) PCR with equal singular values: the documented deterministic basis.

import assert from "node:assert/strict";
import * as n4m from "../dist/index.js";
import { Context, ModelArray, NativeModel } from "../dist/index.js";

await n4m.loadModule();
const M = n4m.getModule();
const cc = (name, args) => M.ccall(name, "number", args.map(() => "number"), args);
const ok = (status, what) => assert.equal(status, 0, `${what} failed with status ${status}`);

const UNSUPPORTED = 10;
const PLS = 0, PLS_DA = 3, PCR = 10;
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
const EPS = 2 ** -52;
const norm = (v) => Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));

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
    const out = [...ARRAYS, [ModelArray.COMPONENT_RESOLUTION], [ModelArray.UNRESOLVED_COMPONENTS]].map(([kind]) => {
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

// Returns { status, fitted, restored, native } (arrays in ARRAYS order,
// resolution, flags, predictions; native = NativeModel.componentResolution()
// of the imported N4MM bytes).
function fit([, algorithm, solver], x, p, y, components, q = 1) {
    const ctx = withPtr((o) => ok(cc("n4m_context_create", [o]), "context_create"));
    const cfg = withPtr((o) => ok(cc("n4m_config_create", [o]), "config_create"));
    const xv = n4m.makeMatrixView(Float64Array.from(x), 4, p);
    const yv = n4m.makeMatrixView(Float64Array.from(y), 4, q);
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
                const bytes = new Uint8Array(M.HEAPU8.subarray(buf, buf + written));
                const jsCtx = Context.create();
                const native = NativeModel.fromN4mm(jsCtx, bytes);
                try {
                    assert.equal(new DataView(bytes.buffer, bytes.byteOffset).getUint32(4, true), 3,
                                 "PLS/PCR models export N4MM format 3");
                    return { status, fitted, restored: arrays(ctx, imported, xv.viewPtr),
                             native: native.componentResolution() };
                } finally {
                    native.destroy();
                    jsCtx.destroy();
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

// want: [coefficients, W, P, Q, R, T, U, resolution, unresolved flags,
// predictions]; undefined = unchecked (the flags are compared exactly).
function check(where, result, want) {
    assert.equal(result.status, 0, `${where}: fit refused with status ${result.status}`);
    const names = [...ARRAYS.map(([, name]) => name), "resolution", "unresolved", "predictions"];
    names.forEach((name, i) => {
        if (name === "unresolved") {
            if (want[i] !== undefined) assert.deepEqual(result.fitted[i], want[i], `${where}: unresolved flags`);
        } else {
            expectClose(where, name, result.fitted[i], want[i]);
        }
        assert.deepEqual(result.restored[i], result.fitted[i], `${where}: ${name} changed through export/import`);
    });
    assert.equal(result.native.resolution, result.fitted[7][0], `${where}: NativeModel resolution`);
    assert.deepEqual(result.native.unresolved, result.fitted[8].map((v) => v !== 0),
                     `${where}: NativeModel unresolved flags`);
}

// Every column `comp` of W, P, R (p x a), Q (q x a), T, U (4 x a) is +0.
function expectNullComponent(where, result, components, comp, p, q = 1) {
    [p, p, q, p, 4, 4].forEach((rows, k) => {
        const values = result.fitted[k + 1];
        assert.equal(values.length, rows * components, `${where}: ${ARRAYS[k + 1][1]} size`);
        for (let row = 0; row < rows; ++row) {
            assert.ok(Object.is(values[row * components + comp], 0),
                      `${where}: ${ARRAYS[k + 1][1]} column ${comp} of the null component`);
        }
    });
}

// Expected resolution (n = 4): PLS (n + a) eps ||X|| ||Y||, PCR
// (n + p) eps sigma_max ||Y||.
const resolution = (algorithm, x, p, y, components, sigmaMax) =>
    [algorithm === PCR ? (4 + p) * EPS * sigmaMax * norm(y) : (4 + components) * EPS * norm(x) * norm(y)];

let cases = 0;
// (1) one feature, Y nearly orthogonal to X.
for (const e of [25, 30]) {
    const x = combine(p2(15), 0), y = combine(1, p2(e));
    for (const s of SOLVERS) {
        const [tag, , , unitScores, rawY] = s;
        const want = unitScores
            ? [[p2(-15)], [p2(-16)], [p2(16)], [2], [p2(-16)], combine(0.5, 0), combine(0.5, p2(e - 1))]
            : [[p2(-15)], [1], [1], [p2(-15)], [1], x, rawY ? y : combine(p2(15), p2(e + 15))];
        want.push(resolution(s[1], x, 1, y, 1, p2(16)), [0], combine(1, 0));
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
        want.push(resolution(s[1], x, 2, y, 2, p2(31)), [0, 0], y);
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
        const res = resolution(algorithm, x, 2, y, 1, 2);
        if (algorithm === PCR) {
            // q = 0: u = Y q / q'q is mathematically undefined, flagged unresolved.
            want = [[0, 0], [1, 0], [1, 0], [0], [1, 0], combine(1, 0), [0, 0, 0, 0], res, [1], [0, 0, 0, 0]];
        } else if (unitScores) {
            want = [[0, p2(32)], [0, p2(31)], [0, p2(-31)], [2], [0, p2(31)], combine(0, 0.5), combine(0, 0.5),
                    res, [0], y];
        } else {
            want = [[0, p2(32)], [0, 1], [0, 1], [p2(32)], [0, 1], combine(0, p2(-32)),
                    rawY ? y : combine(0, p2(-32)), res, [0], y];
        }
        check(`${tag} X = [a, 2^-32 b], Y = b`, fit(s, x, 2, y, 1), want);
        ++cases;
    }
}
// (3) rank-one X: the second component is a flagged null component.
{
    const x = a.flatMap((v) => [v, v]), y = combine(1, 1);
    for (const s of SOLVERS) {
        const where = `${s[0]} X = [a, a], Y = a + b`;
        const result = fit(s, x, 2, y, 2);
        check(where, result, [[0.5, 0.5], undefined, undefined, undefined, undefined, undefined, undefined,
                              resolution(s[1], x, 2, y, 2, Math.sqrt(8)), [0, 1], combine(1, 0)]);
        expectNullComponent(where, result, 2, 1, 2);
        ++cases;
    }
}
const a3 = [-3, -1, 1, 3], b3 = [1, -3, 3, -1];
const combine3 = (ca, cb) => a3.map((v, i) => ca * v + cb * b3[i]);
const sqrt20 = Math.sqrt(20);
// (5) unresolved PCR component (kept); PLS and PLS-DA publish a null component.
{
    const x = combine3(1, 0), y = combine3(p2(-50), 1);
    for (const s of SOLVERS) {
        const [tag, algorithm] = s;
        const result = fit(s, x, 1, y, 1);
        if (algorithm !== PCR) {
            const zeros = [0, 0, 0, 0];
            const want = [[0], [0], [0], [0], [0], zeros, zeros, resolution(PLS, x, 1, y, 1), [1], zeros];
            for (const [label, run] of [["", result], [" (PLS-DA)", fit([tag, PLS_DA, s[2]], x, 1, y, 1)]]) {
                check(`${tag} X = a3, Y = b3 + 2^-50 a3${label}`, run, want);
                expectNullComponent(`${tag}${label}`, run, 1, 0, 1);
            }
        } else {
            check(`${tag} X = a3, Y = b3 + 2^-50 a3`, result,
                  [[p2(-50)], [1], [1], [p2(-50)], [1], x, [0, 0, 0, 0],
                   resolution(PCR, x, 1, y, 1, sqrt20), [1], combine3(p2(-50), 0)]);
        }
        ++cases;
    }
}
// (6) two-target PCR with an unresolved second component.
{
    const x = a3.flatMap((v, i) => [2 * v, b3[i]]);
    const y = a3.flatMap((v, i) => [v + p2(-48) * b3[i], 3 * v - p2(-48) * b3[i]]);
    const identity = [1, 0, 0, 1];
    const u = [];
    for (let i = 0; i < 4; ++i) u.push((0.5 * y[2 * i] + 1.5 * y[2 * i + 1]) / 2.5, 0);
    check("pcr X = [2 a3, b3], Y = [a3 + 2^-48 b3, 3 a3 - 2^-48 b3]", fit(SOLVERS[1], x, 2, y, 2, 2),
          [[0.5, 1.5, p2(-48), -p2(-48)], identity, identity, [0.5, p2(-48), 1.5, -p2(-48)], identity, x, u,
           resolution(PCR, x, 2, y, 2, 2 * sqrt20), [0, 1], y]);
    ++cases;
}
// (7), (8) many targets whose residuals are individually negligible.
for (const [va, vb, copies, e, sigmaMax] of [[a3, b3, 64, 51, sqrt20], [a, b, 100, 52, 2]]) {
    const q = copies + 1;
    const x = va.flatMap((v, i) => [v, vb[i]]);
    const y = va.flatMap((v, i) => [v, ...Array(copies).fill(p2(-29) * v + p2(-e) * vb[i])]);
    const coefficients = [1, ...Array(copies).fill(p2(-29)), 0, ...Array(copies).fill(p2(-e))];
    for (const s of SOLVERS) {
        const [tag, algorithm, solver] = s;
        const where = `${tag} X = [a, b], Y = [a, ${copies} x (2^-29 a + 2^-${e} b)]`;
        const result = fit(s, x, 2, y, 2, q);
        // NIPALS reproduces every coefficient; the other solvers resolve the
        // 2^-e entries to rounding of their column (checked below).
        check(where, result, [solver === 0 && algorithm === PLS ? coefficients : undefined,
                              undefined, undefined, undefined, undefined, undefined, undefined,
                              resolution(algorithm, x, 2, y, 2, sigmaMax), [0, 0], y]);
        const got = result.fitted[0];
        for (let t = 0; t < q; ++t) {
            const column = Math.max(Math.abs(coefficients[t]), Math.abs(coefficients[q + t]));
            for (const f of [0, 1]) {
                assert.ok(Math.abs(got[f * q + t] - coefficients[f * q + t]) <= TOL * column,
                          `${where}: coefficient (${f}, ${t}) = ${got[f * q + t]}`);
            }
        }
        ++cases;
    }
}
// (9) close leading singular values: X = [a3, b3], Y = [a3 + d b3, d a3 + b3],
// d = 2^-e, one component; X'Y = 20 [[1, d], [d, 1]], singular values
// 20 (1 +- d). SIMPLS failed to converge and NIPALS, kernel, power and
// randomized SVD stopped on the weights (1, 0). With s = 1/sqrt 2,
// h = (1 + d) / 2: coefficients h, predictions h (a3 + b3); unit X weights:
// W = P = R = (s, s), T = s (a3 + b3), Q = (1 + d) s (1, 1), U = s (a3 + b3)
// (u = Y q / q'q) or (1 + d) s (a3 + b3) (unit Y weights); unit-norm X
// scores: W = R = (1, 1)/sqrt 40, T = U = (a3 + b3)/sqrt 40,
// P = sqrt 10 (1, 1), Q = (1 + d) sqrt 10 (1, 1). PLS-DA too.
for (const e of [12, 20, 30]) {
    const d = p2(-e), h = (1 + d) / 2, s = Math.SQRT1_2, r40 = 1 / Math.sqrt(40), r10 = Math.sqrt(10);
    const x = a3.flatMap((v, i) => [v, b3[i]]);
    const y = a3.flatMap((v, i) => [v + d * b3[i], d * v + b3[i]]);
    const ab = combine3(1, 1);
    for (const s0 of SOLVERS) {
        const [tag, algorithm, solver, unitScores, rawY] = s0;
        if (algorithm === PCR) continue;  // X'X = 20 I: PCR is undetermined
        const want = unitScores
            ? [[h, h, h, h], [r40, r40], [r10, r10], [(1 + d) * r10, (1 + d) * r10], [r40, r40],
               ab.map((v) => r40 * v), ab.map((v) => r40 * v)]
            : [[h, h, h, h], [s, s], [s, s], [(1 + d) * s, (1 + d) * s], [s, s], ab.map((v) => s * v),
               ab.map((v) => (rawY && solver !== 2 ? (1 + d) * s : s) * v)];
        want.push(resolution(PLS, x, 2, y, 1), [0], ab.flatMap((v) => [h * v, h * v]));
        for (const [label, alg] of [["", PLS], [" (PLS-DA)", PLS_DA]]) {
            check(`${tag}${label} X = [a3, b3], Y = [a3 + 2^-${e} b3, 2^-${e} a3 + b3]`,
                  fit([tag, alg, solver], x, 2, y, 1, 2), want);
            ++cases;
        }
    }
}
// (10) close leading singular values with 64 targets (reference dominant pair
// on a nontrivial Gram matrix): c3 = (1, -1, -1, 1), X = [a3, b3, c3],
// Y = [a3 + d b3, d a3 + b3, s_j c3] with s_j = (64 + j) 2^-10, j = 1..62.
// X'Y = [[20, 20 d, 0], [20 d, 20, 0], [0, 0, 4 s']]: dominant left singular
// vector (1, 1, 0) / sqrt 2, coefficients h = (1 + d) / 2 in the leading
// 2 x 2 block and 0 elsewhere, predictions h (a3 + b3) in the first two
// columns and 0 in the others.
for (const e of [12, 30]) {
    const d = p2(-e), h = (1 + d) / 2, q = 64;
    const c3 = [1, -1, -1, 1];
    const x = a3.flatMap((v, i) => [v, b3[i], c3[i]]);
    const y = a3.flatMap((v, i) => [v + d * b3[i], d * v + b3[i],
                                    ...Array.from({ length: q - 2 }, (_, j) => (65 + j) * p2(-10) * c3[i])]);
    const coefficients = Array.from({ length: 3 * q }, (_, k) => (k === 0 || k === 1 || k === q || k === q + 1 ? h : 0));
    const ab = combine3(1, 1);
    const predictions = ab.flatMap((v) => [h * v, h * v, ...new Array(q - 2).fill(0)]);
    for (const s0 of SOLVERS) {
        const [tag, algorithm, solver] = s0;
        if (algorithm === PCR) continue;  // X'X = diag(20, 20, 4): PCR is undetermined
        const want = [coefficients, undefined, undefined, undefined, undefined, undefined, undefined,
                      resolution(PLS, x, 3, y, 1), [0], predictions];
        check(`${tag} X = [a3, b3, c3], 64 targets, d = 2^-${e}`, fit([tag, algorithm, solver], x, 3, y, 1, q), want);
        ++cases;
    }
}
// (11) PCR with exactly equal singular values, X = [a3, b3] (X'X = 20 I):
// only the singular subspace is defined; PCR returns the deterministic Jacobi
// basis in column order: one component W = (1, 0), coefficients
// [[1, d], [0, 0]] ([[d, 1], [0, 0]] for X = [b3, a3]); two components W = I.
{
    const d = p2(-12);
    const y = a3.flatMap((v, i) => [v + d * b3[i], d * v + b3[i]]);
    for (const swapped of [false, true]) {
        const x = a3.flatMap((v, i) => (swapped ? [b3[i], v] : [v, b3[i]]));
        for (const components of [1, 2]) {
            const where = `pcr equal singular values ${swapped ? "[b3, a3]" : "[a3, b3]"}, ${components} component(s)`;
            const r = fit(SOLVERS[1], x, 2, y, components, 2);
            assert.equal(r.status, 0, `${where}: status`);
            expectClose(where, "weights_w", r.fitted[1], components === 1 ? [1, 0] : [1, 0, 0, 1]);
            const want = components === 1 ? (swapped ? [d, 1, 0, 0] : [1, d, 0, 0])
                                          : (swapped ? [d, 1, 1, d] : [1, d, d, 1]);
            expectClose(where, "coefficients", r.fitted[0], want);
            ++cases;
        }
    }
}
// Models without a component resolution (here canonical PLS) refuse the
// request explicitly with N4M_ERR_UNSUPPORTED.
{
    const ctx = withPtr((o) => ok(cc("n4m_context_create", [o]), "context_create"));
    const cfg = withPtr((o) => ok(cc("n4m_config_create", [o]), "config_create"));
    const xv = n4m.makeMatrixView(Float64Array.from(a.flatMap((v, i) => [v, b[i]])), 4, 2);
    const yv = n4m.makeMatrixView(Float64Array.from(combine(1, 2)), 4, 1);
    try {
        for (const [setter, value] of [["n_components", 1], ["algorithm", 1], ["solver", 0], ["deflation", 1]]) {
            ok(cc(`n4m_config_set_${setter}`, [cfg, value]), setter);
        }
        const model = withPtr((o) => ok(cc("n4m_model_fit", [ctx, cfg, xv.viewPtr, yv.viewPtr, o]), "canonical fit"));
        try {
            for (const kind of [ModelArray.COMPONENT_RESOLUTION, ModelArray.UNRESOLVED_COMPONENTS]) {
                withPtr((o) => assert.equal(cc("n4m_model_get_array", [ctx, model, kind, o]), UNSUPPORTED,
                                            "canonical PLS has no component resolution"));
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
console.log(`PLS/PCR small-covariance analytic cases OK (${cases} fits)`);
