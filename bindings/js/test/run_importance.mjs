// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Model-intrinsic importances (ABI 2.24) over the built WASM module: affine
// coefficients and latent structure after N4ME re-import, native importance
// profiles, the raw-input affine map of a role pipeline re-imported from its
// states (exact against its predictions) and the SNV refusal, step traits,
// the gradient explainer against linear SHAP, coalition rows, the standard
// error summary and profile aggregation / comparison.

import assert from "node:assert/strict";
import * as n4m from "../dist/index.js";

await n4m.loadModule();

const matrix = (rows, cols, values) => ({ rows, cols, data: Float64Array.from(values) });
const maxDiff = (a, b) => {
    assert.equal(a.length, b.length);
    let worst = 0;
    for (let i = 0; i < a.length; i += 1) worst = Math.max(worst, Math.abs(a[i] - b[i]));
    return worst;
};
const maxAbs = (a) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const applyAffine = ({ coefficients: B, intercept: b }, X) => {
    const out = new Float64Array(X.rows * B.cols);
    for (let i = 0; i < X.rows; i += 1) {
        for (let t = 0; t < B.cols; t += 1) {
            let s = b[t];
            for (let j = 0; j < X.cols; j += 1) s += X.data[i * X.cols + j] * B.data[j * B.cols + t];
            out[i * B.cols + t] = s;
        }
    }
    return out;
};

// Smooth synthetic spectra and a target.
const n = 40, p = 24;
const xs = [], ys = [];
for (let i = 0; i < n; i += 1) {
    const a = Math.sin(0.37 * i + 0.11), b = Math.cos(0.23 * ((i * i) % 17));
    let t = 0;
    for (let j = 0; j < p; j += 1) {
        const u = j / p;
        const v = 1 + a * Math.exp(-8 * (u - 0.3) ** 2) + b * Math.exp(-10 * (u - 0.7) ** 2)
            + 0.05 * Math.sin(1.7 * i + j);
        xs.push(v);
        t += (j % 3 === 0 ? 1.5 : -0.4) * v;
    }
    ys.push(t + 0.01 * Math.sin(13 * i));
}
const X = matrix(n, p, xs);
const y = Float64Array.from(ys);
const probe = matrix(5, p, xs.slice(0, 5 * p).map((v, k) => v * 1.1 + 0.01 * Math.cos(k)));

// ---- estimator: affine map + latent structure after N4ME import -------------
{
    const pls = new n4m.PLSRegression({ n_components: 3, scale_x: false, scale_y: false }).fit(X, y);
    const est = n4m.NativeEstimator.fromN4me(pls.toN4me());
    const affine = est.affineCoefficients();
    assert.equal(affine.coefficients.rows, p);
    assert.equal(affine.coefficients.cols, 1);
    const pred = est.predict(probe).data;
    assert.ok(maxDiff(applyAffine(affine, probe), pred) <= 1e-10 * (1 + maxAbs(pred)));

    const latent = est.latentStructure();
    assert.equal(latent.nComponents, 3);
    assert.equal(latent.xRotations.rows, p);
    assert.equal(latent.yLoadings.rows, 1);
    assert.ok(latent.xScale.every((s) => s === 1));
    // Centred, unscaled: B = R Q'.
    const B = new Float64Array(p);
    for (let j = 0; j < p; j += 1) {
        for (let c = 0; c < 3; c += 1) B[j] += latent.xRotations.data[j * 3 + c] * latent.yLoadings.data[c];
    }
    assert.ok(maxDiff(B, affine.coefficients.data) <= 1e-9 * (1 + maxAbs(B)));

    const vip = n4m.modelImportance(est, X, { kind: "vip" });
    assert.equal(vip.length, p);
    const meanSquare = vip.reduce((s, v) => s + v * v, 0) / p;
    assert.ok(Math.abs(meanSquare - 1) < 1e-10, String(meanSquare));
    const sr = est.modelImportance(X, { kind: "selectivity_ratio", outputIndex: 0 });
    assert.ok(sr.every((v) => v >= 0));
    const sc = n4m.modelImportance(est, X, { kind: "standardized_coefficients" });
    assert.equal(sc.length, p);
    assert.throws(() => n4m.modelImportance(est, X, { kind: "bogus" }), /unknown importance kind/);

    const ridge = new n4m.Ridge().fit(X, y);
    const ridgeAffine = n4m.NativeEstimator.fromN4me(ridge.toN4me()).affineCoefficients();
    assert.ok(maxDiff(applyAffine(ridgeAffine, probe), ridge.predict(probe).data) <= 1e-9);
    assert.throws(() => ridge.latentStructure(), (e) => e instanceof n4m.N4mError && /latent/.test(e.message));
    assert.throws(() => n4m.modelImportance(ridge, X, { kind: "vip" }), n4m.N4mError);
    const snv = new n4m.SNV().fit(X);
    assert.throws(() => snv.affineCoefficients(), n4m.N4mError);
    pls.dispose(); est.dispose(); ridge.dispose(); snv.dispose();
}

// ---- role pipeline: exact raw-input coefficients after state import --------
{
    const steps = [
        ["preprocessing.derivatives.savitzky_golay", { window_length: 7, polyorder: 2, deriv: 0 }],
        ["preprocessing.scaling.standard_scale", {}],
        ["models.pls.pls_regression", { n_components: 4 }],
    ];
    const fitted = n4m.RolePipeline.fromSteps(steps).fit(X, y);
    const pipeline = n4m.RolePipeline.fromStates(steps, fitted.exportStates());
    const raw = pipeline.inputAffineCoefficients();
    assert.equal(raw.nFeaturesIn, p);
    const pred = pipeline.predict(probe).data;
    assert.ok(maxDiff(applyAffine(raw, probe), pred) <= 1e-10 * (1 + maxAbs(pred)));
    const info = pipeline.stepsInfo();
    assert.deepEqual(info.map((s) => s.affine), [true, true, true]);
    assert.deepEqual(info.map((s) => s.columnAligned), [true, true, false]);
    assert.deepEqual(pipeline.stepInfo(1), info[1]);

    const snv = n4m.RolePipeline.fromSteps([
        ["preprocessing.derivatives.savitzky_golay", {}],
        ["preprocessing.scatter.snv", {}],
        ["models.pls.pls_regression", { n_components: 2 }],
    ]).fit(X, y);
    assert.throws(() => snv.inputAffineCoefficients(),
        (e) => e instanceof n4m.N4mError && /step 1 \(preprocessing\.scatter\.snv\)/.test(e.message));
    assert.deepEqual(snv.stepsInfo().map((s) => [s.affine, s.columnAligned]),
        [[true, true], [false, true], [true, false]]);
    fitted.dispose(); pipeline.dispose(); snv.dispose();
}

// ---- review case: mirror SG derivative + default autoscaled PLS -----------
{
    const steps = [
        ["preprocessing.derivatives.savitzky_golay", { window_length: 11, polyorder: 2, deriv: 1 }],
        ["models.pls.pls_regression", { n_components: 5 }],
    ];
    const pipeline = n4m.RolePipeline.fromStates(steps, n4m.RolePipeline.fromSteps(steps).fit(X, y).exportStates());
    const Z = pipeline.transform(X);
    for (let i = 0; i < Z.rows; i += 1) {
        assert.equal(Z.data[i * Z.cols], 0);
        assert.equal(Z.data[i * Z.cols + Z.cols - 1], 0);
    }
    const raw = pipeline.inputAffineCoefficients();
    assert.ok(raw.coefficients.data.every((v) => Number.isFinite(v) && Math.abs(v) < 1e3));
    const pred = pipeline.predict(probe).data;
    assert.ok(maxDiff(applyAffine(raw, probe), pred) <= 1e-10 * (1 + maxAbs(pred)));
    pipeline.dispose();
}

// ---- gradient explainer equals linear SHAP on an affine model --------------
{
    const F = 8, O = 2;
    const W = Float64Array.from({ length: F * O }, (_, k) => 3 * Math.sin(1.3 * k));
    const c = Float64Array.from([0.7, -1.9]);
    const Xe = matrix(4, F, Array.from({ length: 4 * F }, (_, i) => 2 + Math.sin(0.7 * i)));
    const bg = matrix(6, F, Array.from({ length: 6 * F }, (_, i) => 2 + 0.4 * Math.cos(0.3 * i)));
    const predictBatch = async (rows, nRows, nf) => {
        const out = new Float64Array(nRows * O);
        for (let r = 0; r < nRows; r += 1) {
            for (let o = 0; o < O; o += 1) {
                let s = c[o];
                for (let j = 0; j < nf; j += 1) s += rows[r * nf + j] * W[j * O + o];
                out[r * O + o] = s;
            }
        }
        return out;
    };
    const grouping = n4m.groupsFixedWidth(F, 3);
    let progress = null;
    const g = await n4m.explainModel({
        X: Xe, background: bg, grouping, nOutputs: O, predictBatch,
        options: { method: "gradient", baseline: "mean", maxBatchRows: 9, gradientStep: 1e-3 },
        onProgress: (done, total) => { progress = [done, total]; },
    });
    assert.equal(g.kind, "gradient");
    assert.equal(g.exactness, "perturbation");
    assert.equal(g.additive, false);
    assert.equal(g.groupAggregation, "sum_of_feature_values");
    assert.deepEqual(progress, [4 * (2 * F + 1) + 1, 4 * (2 * F + 1) + 1]);
    assert.equal(g.gradients.length, 4 * F * O);
    assert.ok(maxDiff(g.meanGradient, W) < 1e-8);
    assert.ok(g.additivityMaxAbsError < 1e-8);
    const reference = new Float64Array(F);
    for (let b = 0; b < bg.rows; b += 1) for (let j = 0; j < F; j += 1) reference[j] += bg.data[b * F + j] / bg.rows;
    const lin = n4m.linearShap({ X: Xe, coefficients: matrix(F, O, W), intercept: c, reference, grouping });
    assert.ok(maxDiff(g.attributions, lin.attributions) < 1e-8);
    assert.ok(maxDiff(g.expected, lin.expected) < 1e-12);
    await assert.rejects(n4m.explainModel({
        X: Xe, background: bg, nOutputs: O, predictBatch,
        options: { method: "gradient", baseline: "interpolate" },
    }), /background mean/);
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(n4m.explainModel({
        X: Xe, background: bg, nOutputs: O, predictBatch, signal: controller.signal,
        options: { method: "gradient" },
    }), { name: "AbortError" });
}

// ---- coalition rows, standard error summary, profiles ----------------------
{
    const Xr = matrix(2, 6, [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60]);
    const bg = matrix(3, 6, [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 3, 3, 3, 3, 3, 3]);
    const grouping = { featureGroup: Int32Array.from([0, 0, 1, 1, 2, 2]), nGroups: 3 };
    const rows = n4m.perturbRows({ X: Xr, background: bg, grouping, absentGroups: [1], baseline: "background" });
    assert.equal(rows.rows, 6);
    assert.deepEqual(Array.from(rows.data.slice(6, 12)), [1, 2, -1, -1, 5, 6]);
    const mean = n4m.perturbRows({ X: Xr, background: bg, grouping, absentGroups: [1], baseline: "mean" });
    assert.equal(mean.rows, 2);
    assert.ok(Math.abs(mean.data[2] - 2 / 3) < 1e-15);
    const bridge = n4m.perturbRows({ X: Xr, grouping, absentGroups: [1], baseline: "interpolate" });
    assert.deepEqual(Array.from(bridge.data.slice(0, 6)), [1, 2, 3, 4, 5, 6]);
    assert.throws(() => n4m.perturbRows({ X: Xr, background: bg, grouping, absentGroups: [3], baseline: "mean" }),
        n4m.N4mError);

    const perm = await n4m.explainModel({
        X: Xr, background: bg, grouping, nOutputs: 1,
        predictBatch: (r, nr, nf) => Float64Array.from({ length: nr }, (_, i) => r[i * nf] * r[i * nf + 3]),
        options: { method: "permutation_shap", nPermutations: 4 },
    });
    const summary = n4m.summarizeAttributions(perm);
    assert.equal(summary.meanSignedStdError.length, 3);
    for (let g = 0; g < 3; g += 1) {
        let s = 0;
        for (let r = 0; r < 2; r += 1) s += perm.stdError[r * 3 + g] ** 2;
        assert.ok(Math.abs(summary.meanSignedStdError[g] - Math.sqrt(s) / 2) < 1e-15);
    }
    const lin = n4m.linearShap({ X: Xr, coefficients: matrix(6, 1, [1, 1, 1, 1, 1, 1]), reference: new Float64Array(6) });
    assert.equal(n4m.summarizeAttributions(lin).meanSignedStdError, null);

    const values = Float64Array.from([1, -3, 2, -0.5, 4]);
    const g2 = { featureGroup: Int32Array.from([0, 0, 1, 1, 1]), nGroups: 2 };
    assert.deepEqual(Array.from(n4m.aggregateProfile(values, g2, "sum")), [-2, 5.5]);
    assert.deepEqual(Array.from(n4m.aggregateProfile(values, g2, "max_abs")), [3, 4]);
    assert.throws(() => n4m.aggregateProfile(values, g2, "median"), /unknown profile aggregation/);

    // scipy.stats.spearmanr 1.15.3 on tied profiles; top-4 Jaccard.
    const profiles = matrix(4, 8, [3, 1, 2, 2, 5, 0.5, 4, 4, 1, 2, 2, 3, 4, 0.2, 5, 6,
        0, 0, 1, 1, 2, 2, 3, 3, 5, 4, 3, 2, 1, 0, -1, -2]);
    const cmp = n4m.compareProfiles(profiles, { topK: 4 });
    const scipy = [1.0, 0.7575896711961079, 0.4938647983247948, -0.36147201953390995,
        0.7575896711961079, 1.0, 0.6626889364596515, -0.6347419188140273,
        0.4938647983247948, 0.6626889364596515, 1.0, -0.9759000729485332,
        -0.36147201953390995, -0.6347419188140273, -0.9759000729485332, 1.0];
    assert.ok(maxDiff(cmp.spearman.data, scipy) < 1e-12);
    assert.ok(maxDiff(cmp.topKJaccard.data, [1, 0.6, 0.6, 1 / 7, 0.6, 1, 0.6, 1 / 7,
        0.6, 0.6, 1, 0, 1 / 7, 1 / 7, 0, 1]) < 1e-15);
    assert.equal(n4m.compareProfiles(profiles).topKJaccard.data[1], 1);  // K = min(10, G)
}

console.log("run_importance: ok");
