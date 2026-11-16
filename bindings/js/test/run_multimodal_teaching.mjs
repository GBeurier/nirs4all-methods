// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Real WASM transport, learning and fitted replay against the native public-ABI probe.
import assert from 'node:assert/strict';
import * as n4m from '../dist/index.js';

await n4m.loadModule();
const rows = 24, cols = 69;
const X = { rows, cols, data: new Float64Array(rows * cols) };
const y = new Float64Array(rows);
for (let i = 0; i < rows; i++) {
  const height = 1 + 0.5 * Math.sin(0.7 * i);
  const center = 6 + 3 * Math.cos(0.3 * i);
  for (let l = 0; l < 16; l++) {
    const d = l - center;
    X.data[i * cols + l] = height * Math.exp(-0.25 * d * d) + 0.05 * Math.cos(1.3 * i + l);
  }
  const m0 = i % 3, m1 = (i * 7) % 3, m2 = 0.5 * Math.sin(1.1 * i);
  X.data.set([m0, m1, m2], i * cols + 16);
  const blob = 0.5 + 0.4 * Math.cos(0.45 * i);
  for (let h = 0; h < 5; h++) for (let w = 0; w < 5; w++) for (let c = 0; c < 2; c++) {
    const r2 = (h - 2) * (h - 2) + (w - 2) * (w - 2);
    X.data[i * cols + 19 + (h * 5 + w) * 2 + c] =
      (c === 0 ? blob : 1 - blob) * Math.exp(-0.5 * r2) + 0.02 * Math.sin(i + h * 3 + w + c);
  }
  y[i] = 2 * height + 0.3 * center + 0.8 * m0 - 0.5 * m1 + m2 + 1.5 * blob;
}

const params = {
  branch_kinds: [1, 0, 2], branch_shapes: [16, 1, 1, 3, 1, 1, 5, 5, 2],
  filters: 2, kernel_size: 3, embedding_dim: 3, head_hidden: 4,
  epochs: 40, batch_size: 8, seed: 5,
};
const net = new n4m.MultibranchRegressor(params);
let replay, scaler, scalerReplay, highSeed;
try {
  net.fit(X, y, { blocks: [16, 3, 50] });
  const diagnostics = net.fitResult();
  assert.equal(diagnostics.loss_history.rows, 41);
  assert.ok(diagnostics.final_loss < 0.5 * diagnostics.initial_loss);
  const prediction = net.predict(X), embedding = net.transform(X);
  assert.equal(embedding.cols, 9);
  const checksum = prediction.data.reduce((total, value, index) => total + (index + 1) * value, 0);
  const nativeChecksum = 1502.9024644352994;
  assert.ok(Math.abs(checksum - nativeChecksum) <= 1e-9 * Math.abs(nativeChecksum),
    `native/WASM predictions differ: ${checksum} vs ${nativeChecksum}`);
  const state = net.toN4me();
  replay = n4m.NativeEstimator.fromN4me(state);
  assert.deepEqual(replay.predict(X).data, prediction.data);
  assert.deepEqual(replay.transform(X).data, embedding.data);
  assert.deepEqual(replay.toN4me(), state);
  assert.equal(replay.fitResult(), null);

  const exactSeed = BigInt.asIntN(64, 18446744073709551615n).toString();
  highSeed = new n4m.MultibranchRegressor({ ...params, seed: exactSeed, epochs: 0 });
  highSeed.fit(X, y, { blocks: [16, 3, 50] });
  assert.ok(highSeed.predict(X).data.every(Number.isFinite));

  scaler = new (n4m.methodClass('preprocessing.scaling.block_rms'))();
  const train = { rows: 2, cols: 2, data: new Float64Array([3, 4, 0, 0]) };
  scaler.fit(train);
  const scaled = scaler.transform(train).data;
  const expectedScale = Math.sqrt(12.5);
  assert.ok(Math.abs(scaled[0] - 3 / expectedScale) < 1e-14);
  assert.ok(Math.abs(scaled[1] - 4 / expectedScale) < 1e-14);
  const fittedScale = scaler.toN4me();
  const unseen = { rows: 1, cols: 2, data: new Float64Array([3000, 4000]) };
  const transformed = scaler.transform(unseen).data;
  assert.ok(Math.abs(transformed[0] - 3000 / expectedScale) < 1e-10);
  assert.deepEqual(scaler.toN4me(), fittedScale, 'unseen values changed fitted statistics');
  scalerReplay = n4m.NativeEstimator.fromN4me(fittedScale);
  assert.deepEqual(scalerReplay.transform(unseen).data, transformed);
  scaler.fit({ rows: 1, cols: 2, data: new Float64Array(2) });
  assert.deepEqual(scaler.transform(unseen).data, unseen.data, 'zero block should retain scale one');
  console.log(JSON.stringify({ checksum, nativeChecksum, initialLoss: diagnostics.initial_loss,
    finalLoss: diagnostics.final_loss, learnedEmbeddings: embedding.cols,
    fittedReplay: 'exact', highBitSeed: 'exact signed transport', blockRms: 'analytic and replay PASS' }));
} finally {
  scalerReplay?.dispose(); scaler?.dispose(); highSeed?.dispose(); replay?.dispose(); net.dispose();
}
