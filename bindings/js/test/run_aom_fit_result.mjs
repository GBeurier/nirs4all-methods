// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// AOM fit results through the real WASM transport: the screened chain and
// operator descriptors and internal scores agree with the reported selection.
import assert from 'node:assert/strict';
import * as n4m from '../dist/index.js';

await n4m.loadModule();
const rows = 40, cols = 60;
const X = { rows, cols, data: new Float64Array(rows * cols) };
const y = new Float64Array(rows);
for (let i = 0; i < rows; i++) {
  const a = Math.sin(0.37 * i + 0.11), b = Math.cos(0.23 * ((i * i) % 17));
  for (let j = 0; j < cols; j++) {
    const t = j / cols;
    X.data[i * cols + j] = 1 + 0.2 * i / rows + a * Math.exp(-8 * (t - 0.3) ** 2)
      + b * Math.exp(-10 * (t - 0.7) ** 2) + 0.02 * Math.sin(1.7 * i + j);
  }
  y[i] = 2 * a - b + 0.05 * Math.sin(3.1 * i);
}

function argmin(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] < values[best]) best = i;
  return best;
}

function checkChains(result, nChains) {
  const { chain_offsets: chains, operator_kinds: kinds, operator_param_offsets: offsets } = result;
  assert.equal(chains.length, nChains + 1);
  assert.equal(chains[0], 0);
  assert.equal(chains.at(-1), kinds.length);
  assert.equal(offsets.length, kinds.length + 1);
  assert.equal(offsets.at(-1), result.operator_params.rows);
  const identity = chains.slice(0, -1)
    .filter((start, c) => chains[c + 1] - start === 1 && kinds[start] === 0).length;
  assert.equal(identity, 1);
}

const estimators = [];
try {
  const Calibration = n4m.methodClass('aom_pop.calibration');
  const full = new Calibration({ max_components: 4, max_depth: 2 });
  estimators.push(full);
  full.fit(X, y);
  const r = full.fitResult();
  checkChains(r, 62);
  assert.deepEqual(Array.from(r.branches), [0, 1, 2]);
  assert.equal(r.cv_scores.rows, 3 * 62);
  assert.equal(r.cv_scores.cols, 4);
  assert.equal(r.screen_scores, undefined);
  const best = argmin(r.cv_scores.data);
  assert.equal(r.branches[Math.floor(best / 4 / 62)], r.selected_branch);
  assert.equal(Math.floor(best / 4) % 62, r.selected_chain_index);
  assert.equal(best % 4, r.selected_parameter_index);

  const fast = new Calibration({ max_components: 4, fast: true });
  estimators.push(fast);
  fast.fit(X, y);
  const f = fast.fitResult();
  checkChains(f, 10);
  assert.equal(f.cv_scores.rows, 1);
  assert.equal(f.screen_scores.rows, 3 * 10);
  const screen = Array.from(f.screen_scores.data, (v) => (Number.isFinite(v) ? -v : Infinity));
  const selected = argmin(screen);
  assert.equal(f.branches[Math.floor(selected / 10)], f.selected_branch);
  assert.equal(selected % 10, f.selected_chain_index);

  const aom = new (n4m.methodClass('aom_pop.aom_pls'))();
  estimators.push(aom);
  aom.fit(X, y);
  const a = aom.fitResult();
  assert.equal(a.operator_scores.rows, a.operator_kinds.length);
  assert.equal(argmin(a.operator_scores.data), a.selected_operator_index);
  assert.equal(a.operator_scores.data[a.selected_operator_index], a.best_score);
  assert.equal(a.rmse_curves.rows, a.operator_kinds.length);
} finally {
  for (const estimator of estimators) estimator.dispose();
}
console.log('run_aom_fit_result: OK');
