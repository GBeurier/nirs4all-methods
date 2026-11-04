// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Run after TypeScript compilation with node --experimental-vm-modules.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { SourceTextModule, SyntheticModule } from 'node:vm';
import test from 'node:test';
const source = readFileSync(new URL('../dist/ranking.js', import.meta.url), 'utf8');
async function harness() {
  const buffer = new ArrayBuffer(8192), bytes = new Uint8Array(buffer), d = new DataView(buffer);
  const values = new Float64Array(buffer), calls = [], freed = [];
  let next = 128, fail = false;
  const m = { HEAPU8: bytes, HEAPF64: values,
    _malloc(size) { const p = next; next = (next + size + 7) & ~7; return p; },
    _free(p) { freed.push(p); },
    ccall(name, _return, types, args) {
      calls.push({ name, types, args });
      if (name === 'n4m_ranking_options_init_v1') {
        const p = args[0]; d.setBigUint64(p, 104n, true); d.setInt32(p + 12, 1, true);
        [0,25,32,400,25/3,25/6,0,0.74,1,1e-8].forEach((x,i) => d.setFloat64(p+16+i*8,x,true));
        d.setBigInt64(p+96,10000n,true);
      } else if (name === 'n4m_ranking_compute_v1') {
        assert.equal(args[4], 20n); assert.equal(args[6], 9n);
        assert.deepEqual(Array.from(bytes.slice(args[1], args[1] + 4)), [1,1,1,0]);
        values.set([1,NaN,1.5,2,1,0,1,0,0,1, 2,NaN,2,1,1,0,0,0,1,2], args[3]/8);
        values.set([2,1,1,1,NaN,1,1,0,NaN], args[5]/8);
        if (fail) return 1;
      }
      return 0;
    } };
  const ffi = { getModule: () => m, checkStatus(s) { if (s) throw Error('native failure'); },
    makeMatrixView() { return { viewPtr: 88, free() { freed.push(88); } }; } };
  const mod = new SourceTextModule(source);
  await mod.link(async spec => {
    assert.equal(spec, './ffi.js');
    return new SyntheticModule(Object.keys(ffi), function() { for (const [k,v] of Object.entries(ffi)) this.setExport(k,v); });
  });
  await mod.evaluate();
  return { rank: mod.namespace.rank, calls, d, freed, setFail() { fail = true; } };
}
const scores = { rows: 2, cols: 2, data: Float64Array.of(1,2,3,NaN) };
const mask = Uint8Array.of(1,1,1,0);
test('marshals options, masks, i64 sizes and ten-column native result', async () => {
  const { rank, calls, d, freed } = await harness();
  const result = rank(scores, mask, { method: 'openskill', higher_is_better: false, beta: 3, max_iterations: 23 });
  const init = calls[0].args[0];
  assert.equal(d.getInt32(init+8,true), 4); assert.equal(d.getInt32(init+12,true), 0);
  assert.equal(d.getFloat64(init+56,true), 3); assert.equal(d.getBigInt64(init+96,true), 23n);
  assert.equal(result.variant, 'weng_lin_bradley_terry'); assert.equal(result.uncertainty_kind, 'approximate_posterior_sd');
  assert.deepEqual(result.candidates[0], { index:0, score:1, uncertainty:null, average_rank:1.5,
    observations:2, matches:1, component:0, wins:1, ties:0, losses:0, rank:1 });
  assert.equal(result.inference.friedman_statistic, null);
  assert.equal(result.inference.friedman_p_value, null);
  assert.match(result.inference.reason, /chi_square_approximation/);
  assert.match(result.warnings[2], /composite approximation/);
  assert.equal(JSON.parse(JSON.stringify(result)).candidates[0].uncertainty, null);
  assert.equal(freed.length, 5);
});
test('rejects malformed mask and frees allocations after native failure', async () => {
  const h = await harness();
  assert.throws(() => h.rank(scores, Uint8Array.of(1,2,1,0)), /presence mask/);
  assert.throws(() => h.rank(scores, mask, { method: 'invalid' }), /unknown ranking/);
  assert.throws(() => h.rank(scores, mask, { unknown: 1 }), /unknown ranking option/);
  h.setFail(); assert.throws(() => h.rank(scores, mask), /native failure/);
  assert.equal(h.freed.length, 5);
});
