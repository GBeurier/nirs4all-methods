// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Focused real-WASM generic procedure/result contract, no host quantile fallback.
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as n4m from "../dist/index.js";

export function checkFiniteColumnQuantiles() {
    const probabilities = [0.1, 0.5, 0.9, 0, 1, 0.5];
    const procedure = new n4m.FiniteColumnQuantiles({ probabilities });
    assert.ok(procedure instanceof n4m.NativeProcedure);
    assert.equal(typeof procedure.fit, "undefined");
    const X = { data: Float64Array.from([
        3, 7, NaN, Infinity, 1, Infinity, NaN, 2,
        4, -Infinity, Infinity, NaN, 2, NaN, -Infinity, 6,
    ]), rows: 4, cols: 4 };
    const result = procedure.run(X);
    assert.deepEqual(Object.keys(result).sort(), ["finite_counts", "quantiles"]);
    assert.deepEqual(result.finite_counts, [4, 1, 0, 2]);
    assert.equal(result.quantiles.rows, 6);
    assert.equal(result.quantiles.cols, 4);
    const expected = [1.3, 7, NaN, 2.4, 2.5, 7, NaN, 4,
        3.7, 7, NaN, 5.6, 1, 7, NaN, 2, 4, 7, NaN, 6, 2.5, 7, NaN, 4];
    assert.equal(result.quantiles.data.length, expected.length);
    expected.forEach((value, i) => {
        if (Number.isNaN(value)) assert.ok(Number.isNaN(result.quantiles.data[i]));
        else assert.ok(Math.abs(result.quantiles.data[i] - value) < 1e-12, `quantile[${i}]`);
    });
    const singleton = procedure.run({ data: Float64Array.of(9, NaN), rows: 1, cols: 2 });
    assert.deepEqual(singleton.finite_counts, [1, 0]);
    assert.equal(singleton.quantiles.rows, 6);
    assert.equal(singleton.quantiles.cols, 2);
    for (let i = 0; i < 6; ++i) {
        assert.equal(singleton.quantiles.data[2 * i], 9);
        assert.ok(Number.isNaN(singleton.quantiles.data[2 * i + 1]));
    }
    const endpoints = new n4m.FiniteColumnQuantiles({ probabilities: [1, 0] }).run({
        data: Float64Array.of(-Number.MAX_VALUE, Number.MAX_VALUE), rows: 2, cols: 1,
    });
    assert.deepEqual(Array.from(endpoints.quantiles.data), [Number.MAX_VALUE, -Number.MAX_VALUE]);
    assert.deepEqual(endpoints.finite_counts, [2]);
    const invalid = error => error.status === n4m.Status.ERR_INVALID_ARGUMENT;
    for (const p of [[-0.1], [1.1], [NaN], [Infinity], [-Infinity], [0.5, NaN]]) {
        // Existing generic marshalling reports setter refusals as named errors.
        assert.throws(() => new n4m.FiniteColumnQuantiles({ probabilities: p }).run(X),
            /invalid value for parameter 'probabilities'/);
    }
    assert.throws(() => new n4m.FiniteColumnQuantiles({ probabilities: [] }).run(X), invalid);
    assert.throws(() => new n4m.FiniteColumnQuantiles().run(X), invalid);
    // Generic outputs own their copied buffers; later runs cannot overwrite them.
    assert.deepEqual(result.finite_counts, [4, 1, 0, 2]);
    assert.equal(result.quantiles.data[4], 2.5);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    await n4m.loadModule();
    checkFiniteColumnQuantiles();
    console.log("finite-column quantile JS/WASM contract PASS");
}
