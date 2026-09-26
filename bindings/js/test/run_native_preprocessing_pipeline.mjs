// SPDX-License-Identifier: CECILL-2.1
// N4MP native WASM lifecycle, held-out R/Python oracle, and ordered-plan tests.

import assert from "node:assert/strict";
import {
    Context, NativePreprocessingPipeline, PipelineOperatorKind, loadModule,
} from "../dist/index.js";

await loadModule();
const ctx = Context.create();
const signal = (samples, features, divisor = 31) => ({
    rows: samples.length,
    cols: features,
    data: Float64Array.from(samples.flatMap(i =>
        Array.from({ length: features }, (_, index) => {
            const j = index + 1;
            return Math.sin(i * j / divisor) + Math.cos(i + j / 11) + i * j / 1000;
        }))),
});
const steps = (...kinds) => kinds.map(kind => ({ kind, params: [] }));
const close = (...pipelines) => pipelines.forEach(pipeline => pipeline?.destroy());

try {
    const train = signal(Array.from({ length: 30 }, (_, i) => i + 1), 24);
    const heldOut = signal([3.5, 17.5], 24);
    const response = { rows: 30, cols: 1,
        data: Float64Array.from({ length: 30 }, (_, i) => (i + 1) / 30) };

    for (let kind = 0; kind <= 14; ++kind) {
        let fitted, imported;
        try {
            fitted = NativePreprocessingPipeline.fit(ctx, steps(kind), train,
                kind === PipelineOperatorKind.OSC || kind === PipelineOperatorKind.EPO
                    ? response : undefined);
            const transformed = fitted.transform(heldOut);
            assert.equal(transformed.rows, heldOut.rows);
            assert.equal(transformed.cols, heldOut.cols);
            assert.ok(transformed.data.every(Number.isFinite), `kind ${kind}`);
            const blob = fitted.toBytes();
            assert.equal(Buffer.from(blob.subarray(0, 4)).toString(), "N4MP");
            imported = NativePreprocessingPipeline.fromBytes(ctx, blob, steps(kind));
            assert.deepEqual(imported.steps, steps(kind));
            assert.deepEqual(imported.transform(heldOut).data, transformed.data);
            assert.deepEqual(imported.toBytes(), blob);
        } finally {
            close(imported, fitted);
        }
    }

    // Independent held-out R/Python SNV→MSC oracle (not training-row replay).
    const oracleSignal = samples => ({
        rows: samples.length, cols: 8,
        data: Float64Array.from(samples.flatMap(i =>
            Array.from({ length: 8 }, (_, index) => {
                const j = index + 1;
                return Math.sin(i * j / 9) + Math.cos(i + j / 7) + i * j / 100;
            }))),
    });
    const oracle = [
        -4.001193774003808, -2.2897281335515838, -0.7449819940745884,
        0.5119733321851917, 1.3908885370727357, 1.8394358338967038,
        1.8472190824668655, 1.446387116008484,
        -2.2015814295950307, -2.6519026524658353, -1.5440849758961113,
        0.4949255610220601, 2.225790408871716, 2.604408534032623,
        1.461767129717428, -0.3893225756868496,
    ];
    let fitted, imported;
    try {
        fitted = NativePreprocessingPipeline.fit(ctx,
            steps(PipelineOperatorKind.SNV, PipelineOperatorKind.MSC),
            oracleSignal([1, 2, 3, 4, 5, 6]));
        const testX = oracleSignal([2.5, 7.5]);
        const prediction = fitted.transform(testX);
        for (let i = 0; i < oracle.length; ++i) {
            assert.ok(Math.abs(prediction.data[i] - oracle[i]) < 1e-10,
                `held-out oracle cell ${i}: ${prediction.data[i]} vs ${oracle[i]}`);
        }
        const blob = fitted.toBytes();
        imported = NativePreprocessingPipeline.fromBytes(ctx, blob,
            steps(PipelineOperatorKind.SNV, PipelineOperatorKind.MSC));
        assert.deepEqual(imported.transform(testX).data, prediction.data);
        assert.throws(() => NativePreprocessingPipeline.fromBytes(ctx, blob,
            steps(PipelineOperatorKind.MSC, PipelineOperatorKind.SNV)), /ordered plan/);
        assert.throws(() => imported.transform({
            rows: 2, cols: 7, data: Float64Array.from([
                ...testX.data.subarray(0, 7), ...testX.data.subarray(8, 15),
            ]),
        }), /feature width/);
        const corrupted = Uint8Array.from(blob);
        corrupted[0] ^= 0xff;
        assert.throws(() => NativePreprocessingPipeline.fromBytes(ctx, corrupted));
    } finally {
        close(imported, fitted);
    }

    // Native introspection retains original, unexpanded positional parameters.
    const derivative = [{ kind: PipelineOperatorKind.SAVGOL_DERIVATIVE,
        params: [7, 3, 1, 1] }, ...steps(PipelineOperatorKind.CENTER)];
    let paramModel, paramReplay;
    try {
        paramModel = NativePreprocessingPipeline.fit(ctx, derivative, train);
        paramReplay = NativePreprocessingPipeline.fromBytes(ctx, paramModel.toBytes(), derivative);
        assert.deepEqual(paramReplay.steps, derivative);
        assert.deepEqual(paramReplay.transform(heldOut).data,
            paramModel.transform(heldOut).data);
    } finally {
        close(paramReplay, paramModel);
    }
    assert.throws(() => NativePreprocessingPipeline.fit(ctx, steps(15), train), /unsupported/);
    assert.throws(() => NativePreprocessingPipeline.fit(ctx,
        steps(PipelineOperatorKind.OSC), train));
    console.log("JS N4MP: 15 native kinds, held-out oracle, plan and replay OK");
} finally {
    ctx.destroy();
}
