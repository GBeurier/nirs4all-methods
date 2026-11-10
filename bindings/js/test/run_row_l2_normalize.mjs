// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Real WASM only. Analytic rows qualify the owner kernel; no fixture-bank rewrite.
import assert from "node:assert/strict";
import * as n4m from "../dist/index.js";

const matrix = rows => ({data: Float64Array.from(rows.flat()), rows: rows.length, cols: rows[0].length});
const close = (actual, expected) => {
    assert.equal(actual.length, expected.length);
    expected.forEach((v, i) => assert.ok(Number.isFinite(actual[i]) && Math.abs(actual[i] - v) < 4e-15,
        `row L2 [${i}]: ${actual[i]} vs ${v}`));
};

export function checkRowL2Normalize() {
    const rows = [[3, 4], [0, 0], [-3, 4], [3e300, 4e300], [3e-300, 4e-300],
        [Number.MAX_VALUE, Number.MAX_VALUE], [Number.MIN_VALUE, Number.MIN_VALUE]];
    const X = matrix(rows);
    const u = 1 / Math.sqrt(2);
    const expected = [0.6, 0.8, 0, 0, -0.6, 0.8, 0.6, 0.8, 0.6, 0.8, u, u, u, u];
    const est = new n4m.RowL2Normalize();
    let replay;
    try {
        est.fit(matrix([[9, -12]]));
        const actual = est.transform(X).data;
        close(actual, expected);
        rows.forEach((row, i) => assert.deepEqual(est.transform(matrix([row])).data,
            actual.slice(i * 2, i * 2 + 2)));
        const state = est.toN4me();
        replay = n4m.NativeEstimator.fromN4me(state);
        assert.equal(replay.methodId, "preprocessing.scaling.row_l2_normalize");
        assert.deepEqual(replay.transform(X).data, actual);
        assert.deepEqual(replay.toN4me(), state);
        est.fit(X);
        assert.deepEqual(est.toN4me(), state);
        for (const v of [NaN, Infinity, -Infinity]) {
            assert.throws(() => est.transform(matrix([[v, 1]])), e => e.status === n4m.Status.ERR_INVALID_ARGUMENT);
            assert.throws(() => est.fit(matrix([[v, 1]])), e => e.status === n4m.Status.ERR_INVALID_ARGUMENT);
        }
    } finally { est.dispose(); replay?.dispose(); }
    // The Web host can compose the explicit method id without ppCreate or a method list.
    const steps = ["preprocessing.scaling.row_l2_normalize", ["models.regularized.ridge", {alpha: 1}]];
    const pipeline = n4m.RolePipeline.fromSteps(steps);
    let restored;
    try {
        pipeline.fit(matrix([[3, 4], [0, 0], [-3, 4], [4, 3]]), [1, 0, -1, 2]);
        close(pipeline.transform(X).data, expected);
        restored = n4m.RolePipeline.fromStates(steps, pipeline.exportStates());
        assert.deepEqual(restored.transform(X).data, pipeline.transform(X).data);
        assert.deepEqual(restored.predict(X).data, pipeline.predict(X).data);
    } finally { pipeline.dispose(); restored?.dispose(); }
}
