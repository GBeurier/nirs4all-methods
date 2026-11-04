// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
import assert from 'node:assert/strict';
import { loadModule, rank } from '../dist/index.js';
await loadModule();
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
console.log('JS native ranking: methods, missingness, disconnected graph and validation OK');
