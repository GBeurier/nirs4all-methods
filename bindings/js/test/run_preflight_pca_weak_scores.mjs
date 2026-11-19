// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Frozen Domains F4.3 input, with an independent 100-digit direct-SVD oracle.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadModule, preflightPca, statsCorrelationTest, preflightFactorAssociation } from "../dist/index.js";

await loadModule();
const fixture = JSON.parse(readFileSync(new URL("../../../parity/fixtures/preflight_pca_weak_scores_v1.json", import.meta.url)));
const decode = values => values.map(value => Buffer.from(value, "hex").readDoubleBE());
const input = decode(fixture.input_hex), reference = decode(fixture.reference_pc5_scores_hex);
const factor = decode(fixture.factor_hex), eigenvalues = decode(fixture.reference_eigenvalues_hex);
const pearson = decode(fixture.reference_pearson_hex)[0], eta = decode(fixture.reference_eta_squared_hex)[0];
const {rows: n, cols: p} = fixture;
let maxScoreError = 0, maxPearsonError = 0, maxEtaError = 0;
for (let permutation = 0; permutation < 4; ++permutation) {
    for (const exponent of [-30, 0, 30]) {
        const scale = 2 ** exponent;
        const row = i => permutation & 1 ? n - 1 - i : i;
        const col = j => permutation & 2 ? p - 1 - j : j;
        const data = Float64Array.from(input, (_, offset) => input[row(Math.floor(offset / p)) * p + col(offset % p)] * scale);
        const values = Float64Array.from(factor, (_, i) => factor[row(i)]);
        const groups = fixture.group_codes.map((_, i) => fixture.group_codes[row(i)]);
        const expected = reference.map((_, i) => reference[row(i)] * scale);
        let prefix;
        for (const k of [5, 20, 24]) {
            const result = preflightPca({data, rows: n, cols: p}, k);
            assert.equal(result.unresolved[4], 0);
            assert.ok(result.eigenvalues[4] > result.resolution);
            assert.ok(Math.abs(result.eigenvalues[4] - eigenvalues[4] * scale * scale) <= result.resolution);
            for (let a = 0; a < 5; ++a) {
                let variance = 0;
                for (let i = 0; i < n; ++i) {
                    const score = result.scores.data[i * k + a];
                    variance += score * score / (n - 1);
                    if (prefix) assert.equal(score, prefix[i * 5 + a]);
                }
                assert.ok(Math.abs(variance - result.eigenvalues[a]) <= result.resolution);
            }
            if (k === 5) prefix = result.scores.data;
            for (let a = 5; a < k; ++a) {
                assert.equal(result.unresolved[a], 1);
                assert.equal(result.eigenvalues[a], 0);
                for (let i = 0; i < n; ++i) assert.equal(result.scores.data[i * k + a], 0);
            }
            const raw = Array.from({length: n}, (_, i) => result.scores.data[i * k + 4]);
            const sign = raw.reduce((sum, value, i) => sum + value * expected[i], 0) < 0 ? -1 : 1;
            const scores = Float64Array.from(raw, value => sign * value);
            const scoreError = Math.sqrt(scores.reduce((sum, value, i) => sum + (value - expected[i]) ** 2, 0) /
                expected.reduce((sum, value) => sum + value ** 2, 0));
            assert.ok(scoreError <= fixture.score_relative_l2_tolerance, `PC5 relative L2 ${scoreError}`);
            const r = statsCorrelationTest(values, scores, "pearson", 0, 0).coefficient;
            const e = preflightFactorAssociation(groups, scores, false, 0, 0).effectSize;
            const tolerance = value => fixture.statistic_absolute_tolerance + fixture.statistic_relative_tolerance * Math.abs(value);
            assert.ok(Math.abs(r - pearson) <= tolerance(pearson), `Pearson ${r} vs ${pearson}`);
            assert.ok(Math.abs(e - eta) <= tolerance(eta), `eta squared ${e} vs ${eta}`);
            maxScoreError = Math.max(maxScoreError, scoreError);
            maxPearsonError = Math.max(maxPearsonError, Math.abs(r - pearson));
            maxEtaError = Math.max(maxEtaError, Math.abs(e - eta));
        }
    }
}
console.log(JSON.stringify({test: "PCA weak scores", cases: 36, maxScoreError, maxPearsonError, maxEtaError}));
