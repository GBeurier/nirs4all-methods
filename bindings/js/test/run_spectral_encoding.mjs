// SPDX-License-Identifier: CECILL-2.1
import assert from "node:assert/strict";
import { loadModule, SpectralEncoder } from "../dist/index.js";
await loadModule();
const rows = 12, cols = 7;
const data = Float64Array.from({length: rows * cols}, (_, k) => Math.sin(k * .7) + (k % cols));
const x = {data, rows, cols};
const encoder = new SpectralEncoder({kind: "lvse", width: 4, rank: 2, overlap: .5}).fit(x);
const z = encoder.transform(x);
assert.equal(z.cols, 6);
const {operator, offset} = encoder.exportAffine();
for (let i = 0; i < rows; i++) {
    for (let a = 0; a < z.cols; a++) {
        let expected = offset[a];
        for (let j = 0; j < cols; j++) expected += data[i * cols + j] * operator.data[a * cols + j];
        assert.ok(Math.abs(expected - z.data[i * z.cols + a]) < 1e-10);
    }
}
encoder.dispose();
assert.throws(() => encoder.transform(x));
const gcu = new SpectralEncoder({kind: "gcu", rank: 2}).fit(x);
assert.ok(gcu.transform(x).data.every(v => Number.isFinite(v) && v >= 0));
assert.throws(() => gcu.exportAffine());
gcu.dispose();
console.log("LVSE/GCU WASM binding tests passed");
