// SPDX-License-Identifier: CECILL-2.1
// The JS binding promotes a native GroupSparse result without host arithmetic.

import assert from "node:assert/strict";
import {
    Algorithm, Config, Context, Deflation, MethodResult, NativeModel,
    SERIALIZED_MODEL_CAPABILITY_AFFINE, SERIALIZED_MODEL_CAPABILITY_PREDICT,
    Solver, inspectN4mm, loadModule, makeMatrixView,
} from "../dist/index.js";

const wasm = await loadModule();
const context = Context.create();
const config = Config.create();
config.setAlgorithm(Algorithm.PLS_REGRESSION);
config.setSolver(Solver.SIMPLS);
config.setDeflation(Deflation.REGRESSION);
config.setNComponents(2);
config.setCenterX(true);
config.setCenterY(true);
for (const symbol of ["n4m_config_set_scale_x", "n4m_config_set_scale_y"]) {
    assert.equal(wasm.ccall(symbol, "number", ["number", "number"],
                            [config.handle, 0]), 0);
}

const rows = 21, features = 12;
const x = new Float64Array(rows * features);
const y = new Float64Array(rows);
for (let i = 1; i <= rows; ++i) {
    for (let j = 1; j <= features; ++j) {
        x[(i - 1) * features + j - 1] =
            Math.sin(i * j / 9) + Math.cos(i + j / 7) + i * j / 100;
    }
    y[i - 1] = 1.3 + 0.7 * x[(i - 1) * features + 1]
                     - 0.4 * x[(i - 1) * features + 5];
}
const testX = new Float64Array(3 * features);
for (const [row, source] of [1, 7, 16].entries()) {
    for (let j = 0; j < features; ++j) {
        testX[row * features + j] = x[source * features + j] + 0.031;
    }
}

const xView = makeMatrixView(x, rows, features);
const yView = makeMatrixView(y, rows, 1);
const groups = Int32Array.from([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
const groupsPtr = wasm._malloc(groups.byteLength);
const resultPtr = wasm._malloc(4);
let result, model, restored;
try {
    wasm.HEAP32.set(groups, groupsPtr / 4);
    wasm.setValue(resultPtr, 0, "i32");
    assert.equal(wasm.ccall(
        "n4m_estimators_group_sparse_pls_fit", "number",
        ["number", "number", "number", "number", "number", "i64", "number", "number"],
        [context.handle, config.handle, xView.viewPtr, yView.viewPtr,
         groupsPtr, BigInt(groups.length), 0.05, resultPtr]), 0);
    result = new MethodResult(wasm.getValue(resultPtr, "i32"));
    model = NativeModel.fromMethodResult(context, result);
    result.destroy();
    result = undefined;

    const heldOut = { data: testX, rows: 3, cols: features };
    const expected = [1.3302312403142085, 1.9038372601712736, 0.8103946779875588];
    const prediction = model.predict(heldOut);
    assert.equal(prediction.rows, 3);
    assert.equal(prediction.cols, 1);
    for (let i = 0; i < expected.length; ++i) {
        assert.ok(Math.abs(prediction.data[i] - expected[i]) < 1e-10,
                  `R GroupSparse oracle row ${i}: ${prediction.data[i]} vs ${expected[i]}`);
    }

    const payload = model.toN4mm();
    const info = inspectN4mm(payload);
    assert.equal(info.algorithm, Algorithm.IMPORTED_LINEAR_PREDICTOR);
    assert.equal(info.trainingSamples, 0n);
    assert.equal(info.nFeatures, features);
    assert.equal(info.nTargets, 1);
    assert.equal(info.capabilities,
                 SERIALIZED_MODEL_CAPABILITY_PREDICT | SERIALIZED_MODEL_CAPABILITY_AFFINE);
    restored = NativeModel.fromN4mm(context, payload);
    const replay = restored.predict(heldOut);
    for (let i = 0; i < expected.length; ++i) {
        assert.ok(Math.abs(replay.data[i] - expected[i]) < 1e-10);
    }
    console.log("JS native MethodResult model held-out and N4MM replay OK");
} finally {
    restored?.destroy();
    model?.destroy();
    result?.destroy();
    wasm._free(resultPtr);
    wasm._free(groupsPtr);
    xView.free();
    yView.free();
    config.destroy();
    context.destroy();
}
