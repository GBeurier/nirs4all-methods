// SPDX-License-Identifier: CECILL-2.1
// Scale equivariance of the NIPALS PLSRegression and PCR estimator roles
// without autoscaling (center on, scale off, the preflight probe settings).
// Fitting X 2^k, Y 2^m must predict exactly 2^m times the base predictions,
// which match an independent scikit-learn 1.7.2 reference: PLSRegression(
// n_components=3, scale=False) and PCA(n_components=3, svd_solver="full") +
// LinearRegression (PCR), fitted on the base data and rescaled analytically
// (scikit-learn's own absolute epsilons collapse its fit at Y 2^-60, so only
// its base-scale fit is a valid oracle). Mirrors cpp/tests/test_models_pls.cpp.

import assert from "node:assert/strict";
import * as n4m from "../dist/index.js";

await n4m.loadModule();

const n = 12, p = 6;
const X = [
    0.752, 1.091, 1.201, 0.996, 1.356, 0.611, 0.519, 1.08, 1.3, 1.521, 0.384, 1.386,
    0.223, 0.44, 0.998, 1.704, 1.783, 0.833, 0.872, 0.979, 0.606, 1.349, 1.489, 0.319,
    1.309, 1.043, 1.036, 1.106, 0.464, 1.287, 1.376, 1.578, 0.828, 0.32, 1.546, 1.048,
    0.838, 0.967, 1.47, 1.578, 0.227, 0.319, 1.736, 0.906, 1.633, 0.376, 0.349, 0.536,
    1.608, 1.397, 0.742, 0.225, 0.779, 0.254, 0.218, 0.432, 1.057, 0.403, 1.424, 1.701,
    1.571, 0.785, 0.743, 0.911, 1.43, 1.464, 1.058, 1.145, 0.668, 1.22, 0.4, 0.238];
const Y1 = [2.0172, 6.0458, 4.7421, 2.2906, 5.3699, 0.2102,
    4.7835, 2.5479, -0.3046, 3.1338, 4.6288, 2.7718];
const Y2 = [
    2.0172, 1.0212, 6.0458, 0.1926, 4.7421, -0.281, 2.2906, 1.8502,
    5.3699, 2.9417, 0.2102, 3.3368, 4.7835, 0.976, 2.5479, 3.5449,
    -0.3046, 4.0861, 3.1338, -0.3399, 4.6288, 4.0054, 2.7718, 2.5964];
// scikit-learn coefficients, p x q row-major.
const PLS_Y1 = [0.4957614061724583, -1.381677405563282, 0.514726893875039,
    2.580185279507641, -1.2169935455968648, 2.4340614676208925];
const PLS_Y2 = [
    0.4974189994656632, 2.5053049800839355, -1.280755699988794, 0.38069069715328074,
    0.38849543933311187, -0.6375453772329983, 2.6515591799941225, -0.5923586761548153,
    -1.2747648115664545, -0.13006086448726156, 2.4360684738591836, 0.3755339509000017];
const PCR_Y1 = [-1.2920592892405545, -0.8219629954140479, 1.0849870020018448,
    1.2297408628741708, -1.3908828952499102, 1.37521377329476];
const PCR_Y2 = [
    -1.2920592892405538, 1.4281767180739962, -0.8219629954140476, 0.7033326466062279,
    1.0849870020018442, -0.28684733153814906, 1.2297408628741704, -1.2781534393212208,
    -1.3908828952499093, -0.10487458205601959, 1.3752137732947596, -0.5728917833534646];

const scaled = (values, rows, cols, e) => ({
    data: Float64Array.from(values, (v) => v * 2 ** e), rows, cols,
});

function referencePredictions(y, q, coef) {
    const xMean = new Array(p).fill(0), yMean = new Array(q).fill(0);
    for (let i = 0; i < n; ++i) {
        for (let j = 0; j < p; ++j) xMean[j] += X[i * p + j] / n;
        for (let t = 0; t < q; ++t) yMean[t] += y[i * q + t] / n;
    }
    const out = [];
    for (let i = 0; i < n; ++i) {
        for (let t = 0; t < q; ++t) {
            let s = yMean[t];
            for (let j = 0; j < p; ++j) s += (X[i * p + j] - xMean[j]) * coef[j * q + t];
            out.push(s);
        }
    }
    return out;
}

function fitPredict(Cls, params, y, q, k, m) {
    const Xs = scaled(X, n, p, k);
    const est = new Cls({ n_components: 3, center_x: true, scale_x: false,
                          center_y: true, scale_y: false, ...params });
    try {
        est.fit(Xs, scaled(y, n, q, m));
        return Array.from(est.predict(Xs).data, (v) => v * 2 ** -m);
    } finally {
        est.dispose();
    }
}

const exponents = [0, 30, -30, 60, -60, 200, -200];
const cases = [
    ["nipals y1", n4m.PLSRegression, { solver: "nipals" }, Y1, 1, PLS_Y1],
    ["nipals y2", n4m.PLSRegression, { solver: "nipals" }, Y2, 2, PLS_Y2],
    ["pcr y1", n4m.PCR, {}, Y1, 1, PCR_Y1],
    ["pcr y2", n4m.PCR, {}, Y2, 2, PCR_Y2],
];
for (const [label, Cls, params, y, q, coef] of cases) {
    const ref = referencePredictions(y, q, coef);
    const norm = Math.max(...ref.map(Math.abs));
    const base = fitPredict(Cls, params, y, q, 0, 0);
    for (const k of exponents) {
        for (const m of exponents) {
            const got = fitPredict(Cls, params, y, q, k, m);
            const where = `${label} X 2^${k}, Y 2^${m}`;
            assert.deepEqual(got, base, `${where}: predictions not exactly equivariant`);
            got.forEach((v, i) => assert.ok(Math.abs(v - ref[i]) <= 1e-9 * norm,
                `${where}[${i}]: ${v} vs scikit-learn ${ref[i]}`));
        }
    }
}
console.log("PLS/PCR scale equivariance OK");
