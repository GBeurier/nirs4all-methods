// SPDX-License-Identifier: CECILL-2.1
// Run after `npm run build`: node --experimental-vm-modules test/selection_mock.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { SourceTextModule, SyntheticModule } from 'node:vm';
import test from 'node:test';

const source = readFileSync(new URL('../dist/selection.js', import.meta.url), 'utf8');

function harness() {
  const buffer = new ArrayBuffer(1 << 20);
  const heap = new Uint8Array(buffer);
  const data = new DataView(buffer);
  const calls = [];
  let next = 128;
  const m = {
    HEAPU8: heap,
    HEAPF64: new Float64Array(buffer),
    _malloc(size) { const p = next; next += Math.max(size, 8); next = (next + 7) & ~7; return p; },
    _free() {},
    getValue(ptr, kind) { return kind === 'i32' ? data.getInt32(ptr, true) : data.getBigInt64(ptr, true); },
    setValue(ptr, value) { data.setInt32(ptr, value, true); },
    ccall(name, _return, types, args) {
      calls.push({ name, types, args });
      if (name.endsWith('_create')) data.setInt32(args.at(-1), next + 16, true);
      if (name.startsWith('n4m_feature_selection_')) data.setInt32(args.at(-1), 4242, true);
      return 0;
    },
  };
  const mocks = {
    './config.js': { Config: class {
      static create() { return new this(); }
      get handle() { return 222; }
      setNComponents(k) { calls.push({ name: 'setNComponents', args: [k] }); }
      destroy() {}
    } },
    './context.js': { Context: class {
      static create() { return new this(); }
      get handle() { return 111; }
      destroy() {}
    } },
    './ffi.js': {
      getModule: () => m,
      checkStatus(status) { assert.equal(status, 0); },
      makeMatrixView() { return { viewPtr: next += 8, free() {} }; },
    },
    './methodResult.js': { MethodResult: class {
      constructor(ptr) { assert.equal(ptr, 4242); }
      vectorInt64(key) { assert.equal(key, 'selected_indices'); return BigInt64Array.of(8n, 3n, 1n); }
      destroy() {}
    } },
    './types.js': {},
  };
  const module = new SourceTextModule(source);
  return module.link(async (specifier) => {
    const exports = mocks[specifier];
    assert.ok(exports, specifier);
    return new SyntheticModule(Object.keys(exports), function () {
      for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
    });
  }).then(async () => {
    await module.evaluate();
    return { api: module.namespace, calls, data, heap };
  });
}

const X = { data: new Float64Array(48), rows: 6, cols: 8 };
const Y = { data: new Float64Array(6), rows: 6, cols: 1 };

test('all 25 selector ABI entries marshal ranked int64 and three signature groups', async () => {
  const { api, calls, data } = await harness();
  assert.equal(api.selectorMethods.length, 25);
  const vectorParams = { t2_select: { alpha_thresholds: [0.05, 0.2] },
    st_select: { thresholds: [0.1, 0.4] } };
  for (const name of api.selectorMethods) {
    const start = calls.length;
    const selected = api.selectVariables(name, X, Y, 2, vectorParams[name] ?? {});
    assert.deepEqual(Array.from(selected), [8n, 3n, 1n]);
    const own = calls.slice(start);
    const invoke = own.find((call) => call.name === `n4m_feature_selection_${name}`);
    assert.ok(invoke, name);
    assert.equal(invoke.args[0], 111);
    const noConfig = name === 'wvc_select' || name === 'wvc_threshold_select';
    assert.equal(own.some((call) => call.name === 'setNComponents'), !noConfig);
    if (noConfig) assert.deepEqual(invoke.args.slice(0, 4).filter((arg) => arg === 2), [2]);
    const plan = !['spa_select', 'wvc_select', 'wvc_threshold_select',
      'randomization_select', 'vip_spa_select'].includes(name);
    assert.equal(own.some((call) => call.name === 'n4m_validation_plan_create'), plan, name);
    if (plan) {
      const folds = own.filter((call) => call.name === 'n4m_validation_plan_add_fold');
      assert.equal(folds.length, 3, name);
      assert.deepEqual(folds.map(({ args }) => args.slice(2).filter((x) => typeof x === 'bigint')),
        [[4n, 2n], [4n, 2n], [4n, 2n]]);
      const first = folds[0].args;
      assert.deepEqual([0, 1, 2, 3].map((i) => Number(data.getBigInt64(first[1] + i * 8, true))), [2, 3, 4, 5]);
      assert.deepEqual([0, 1].map((i) => Number(data.getBigInt64(first[3] + i * 8, true))), [0, 1]);
    }
    if (name === 't2_select' || name === 'st_select') {
      const ptr = invoke.args.at(-4);
      assert.deepEqual([data.getFloat64(ptr, true), data.getFloat64(ptr + 8, true)],
        vectorParams[name][name === 't2_select' ? 'alpha_thresholds' : 'thresholds']);
      assert.equal(invoke.args.at(-3), 2n);
    }
  }
});

test('selector ABI uses BigInt seeds and rejects unsupported options', async () => {
  const { api, calls } = await harness();
  api.selectVariables('uve_select', X, Y, 2, { noise_seed: '18446744073709551615' });
  const invoke = calls.find((call) => call.name === 'n4m_feature_selection_uve_select');
  assert.equal(invoke.args.at(-2), 18446744073709551615n);
  assert.equal(invoke.types.at(-2), 'i64');
  assert.throws(() => api.selectVariables('uve_select', X, Y, 2, { noise_seed: -1 }), /uint64/);
  assert.throws(() => api.selectVariables('uve_select', X, Y, 2, { surprise: 1 }), /Unsupported/);
  assert.throws(() => api.selectVariables('t2_select', X, Y, 2, { alpha_thresholds: [] }), /non-empty/);
  assert.throws(() => api.selectVariables('spa_select', X, Y, 9, { top_k: 2 }), /rank/);
});
