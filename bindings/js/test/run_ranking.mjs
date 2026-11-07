// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
import assert from 'node:assert/strict';
import { loadModule, rank } from '../dist/index.js';
const native = await loadModule();
const scores = { rows: 3, cols: 3, data: Float64Array.of(1,2,NaN, 2,1,NaN, NaN,NaN,4) };
const mask = Uint8Array.of(1,1,0, 1,1,0, 0,0,1);
for (const method of ['average_rank', 'elo', 'bradley_terry', 'trueskill', 'openskill']) {
    const result = rank(scores, mask, { method, higher_is_better: true });
    assert.equal(result.schema_version, 'n4m.ranking/1');
    assert.equal(result.method, method);
    assert.equal(result.coverage.paired_comparisons, 2);
    assert.equal(result.coverage.connected_components, 2);
    assert.deepEqual(result.candidates.map(x => x.observations), [2,2,1]);
    assert.deepEqual(result.candidates.map(x => x.rank), [null,null,null]);
    assert.equal(result.candidates[2].matches, 0);
    assert.equal(result.uncertainty_kind, ['trueskill','openskill'].includes(method) ? 'approximate_posterior_sd' : null);
    assert.equal(JSON.parse(JSON.stringify(result)).candidates.length, 3);
}
const complete = rank({ rows: 3, cols: 2, data: Float64Array.of(1,2, 1,2, 1,2) },
                      Uint8Array.of(1,1,1,1,1,1));
assert.equal(complete.coverage.complete_blocks, 3);
assert.equal(complete.inference.degrees_of_freedom, 1);
assert.ok(complete.inference.friedman_p_value > 0 && complete.inference.friedman_p_value < 1);
assert.deepEqual(complete.candidates.map(x => x.rank), [1,2]);
assert.throws(() => rank(scores, Uint8Array.of(1,2,0,1,1,0,0,0,1)), /presence mask/);
assert.throws(() => rank(scores, mask, { tie_tolerance: -1 }));
// Incomplete but connected design: A=2,B=1 ; B=2,C=1 ; nine rows with only A present.
// Mean ranks A=1.1, B=1.5, C=1.0 are descriptive; the average_rank global rank is withheld.
{
    const rows = 11, data = new Float64Array(rows * 3), present = new Uint8Array(rows * 3);
    data.set([2,1,0, 0,2,1]);
    present.set([1,1,0, 0,1,1]);
    for (let r = 2; r < rows; r++) { data[r * 3] = 2; present[r * 3] = 1; }
    const incomplete = rank({ rows, cols: 3, data }, present, { method: 'average_rank' });
    assert.equal(incomplete.coverage.complete_blocks, 0);
    assert.equal(incomplete.coverage.connected_components, 1);
    assert.deepEqual(incomplete.candidates.map(x => Math.round(x.average_rank * 1e9) / 1e9), [1.1, 1.5, 1]);
    assert.deepEqual(incomplete.candidates.map(x => x.rank), [null, null, null]);
    assert.ok(incomplete.warnings.some(w => w.includes('Incomplete design')));
    const bt = rank({ rows, cols: 3, data }, present, { method: 'bradley_terry' });
    assert.deepEqual(bt.candidates.map(x => x.rank), [3, 2, 1]);
    assert.ok(!bt.warnings.some(w => w.includes('Incomplete design')));
    const full = rank({ rows: 2, cols: 3, data: Float64Array.of(1,2,3, 1,2,3) }, Uint8Array.of(1,1,1,1,1,1),
                      { method: 'average_rank' });
    assert.deepEqual(full.candidates.map(x => x.rank), [1, 2, 3]);
    assert.ok(!full.warnings.some(w => w.includes('Incomplete design')));
}
console.log('JS native ranking: methods, missingness, disconnected graph and validation OK');

// Production C ABI counterexample: dominant inverse-variance weight must not
// cancel the DL denominator to zero. This also checks the WASM long-double path.
const ptrs = [native._malloc(16), native._malloc(16), native._malloc(2), native._malloc(16), native._malloc(64)];
try {
    native.HEAPF64.set([2,2], ptrs[0]/8);
    native.HEAPF64.set([1,1], ptrs[1]/8);
    native.HEAPU8.set([1,1], ptrs[2]);
    native.HEAPF64.set([1e-16,1], ptrs[3]/8);
    const status = native.ccall('n4m_ranking_relative_gain_v1', 'number',
        ['number','number','number','number','i64','number','i64'],
        [ptrs[0],ptrs[1],ptrs[2],ptrs[3],2n,ptrs[4],8n]);
    assert.equal(status, 0);
    const result = native.HEAPF64.subarray(ptrs[4]/8, ptrs[4]/8+8);
    assert.equal(result[3], .5);
    assert.ok(Math.abs(result[4]/1e-8-1) < 1e-14);
    assert.deepEqual(Array.from(result.subarray(5)), [0,0,0]);
} finally {
    ptrs.forEach(ptr => native._free(ptr));
}
console.log('JS native relative gain: dominant-weight DL regression OK');
