// SPDX-License-Identifier: CECILL-2.1
// Raw WASM C ABI regression; explicit paths let this test qualify a changed build.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const moduleUrl = process.env.N4M_WASM_MODULE
  ? pathToFileURL(resolve(process.env.N4M_WASM_MODULE))
  : new URL('../dist/n4m.js', import.meta.url);
const binaryUrl = process.env.N4M_WASM_BINARY
  ? pathToFileURL(resolve(process.env.N4M_WASM_BINARY))
  : new URL('n4m.wasm', moduleUrl);
const { default: createModule } = await import(moduleUrl.href);
const m = await createModule({ wasmBinary: readFileSync(binaryUrl) });
const metrics = ['rmse', 'mae', 'bias', 'sep', 'rpd', 'rpiq', 'r2', 'nrmse'];
const truth = m._malloc(16);
const predicted = m._malloc(16);
const output = m._malloc(8);
let refusals = 0;
try {
  for (const metric of metrics) {
    const evaluate = m[`_n4m_metrics_regression_metrics_${metric}`];
    assert.equal(typeof evaluate, 'function', `missing raw ABI: ${metric}`);
    for (const field of ['y_true', 'y_pred']) {
      for (const bad of [NaN, Infinity, -Infinity]) {
        m.HEAPF64.set([1, 2], truth / 8);
        m.HEAPF64.set([1, 2], predicted / 8);
        m.HEAPF64[(field === 'y_true' ? truth : predicted) / 8 + 1] = bad;
        m.HEAPF64[output / 8] = 917.5;
        // N4M_ERR_INVALID_ARGUMENT = 1; n is int64_t in the WASM ABI.
        assert.equal(evaluate(truth, predicted, 2n, output), 1,
          `${metric}/${field}/${bad}: must reject nonfinite input`);
        assert.equal(m.HEAPF64[output / 8], 917.5,
          `${metric}/${field}/${bad}: output sentinel changed`);
        refusals++;
      }
    }
  }
  m.HEAPF64.set([1, 2], truth / 8);
  m.HEAPF64.set([1, 2], predicted / 8);
  assert.equal(m._n4m_metrics_regression_metrics_rmse(truth, predicted, 2n, output), 0);
  assert.equal(m.HEAPF64[output / 8], 0);
  assert.equal(m._n4m_metrics_regression_metrics_rpd(truth, predicted, 2n, output), 0);
  assert.equal(m.HEAPF64[output / 8], Infinity);
  assert.equal(refusals, 48);
  console.log(`shared scoring raw WASM C ABI: ${refusals} refusals, zero RMSE and intentional RPD +Inf PASS (${moduleUrl.pathname}; ${binaryUrl.pathname})`);
} finally {
  m._free(truth);
  m._free(predicted);
  m._free(output);
}
