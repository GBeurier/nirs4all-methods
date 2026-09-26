// SPDX-License-Identifier: CECILL-2.1
import assert from "node:assert/strict";
import { loadModule, NativeSampleFilter, NativeFeatureFilter } from "../dist/index.js";

await loadModule();
const X = { rows: 10, cols: 3, data: Float64Array.from([
    1,2,4, 2,5,3, 3,1,5, 4,7,2, 5,3,8,
    6,8,6, 7,4,7, 8,9,10, 9,6,9, 10,10,11,
]) };
const Y = { rows: 10, cols: 1, data: Float64Array.from([1,2,3,4,5,6,7,8,9,100]) };
const heldX = { rows: 3, cols: 3, data: X.data.slice(21) };
const heldY = { rows: 3, cols: 1, data: Y.data.slice(7) };
const y = new NativeSampleFilter("YOutlier", [0], [1.5, 5, 95]);
assert.throws(() => y.apply(heldX, heldY), /not fitted/);
y.fit(X, Y);
const yResult = y.apply(heldX, heldY);
assert.deepEqual(Array.from(yResult.mask), [1, 1, 0]);
assert.equal(yResult.stats.nKept, 2);
assert.throws(() => y.apply(heldX, { rows: 3, cols: 2,
    data: new Float64Array(6) }), /one column/);
y.dispose();

const qualityX = { rows: 3, cols: 3, data: Float64Array.from([
    1,2,3, 0,0,0, 4,5,6,
]) };
const quality = new NativeSampleFilter("SpectralQuality", [0,0,1],
    [.1,.5,1e-8,0,0]);
quality.fit(qualityX);
assert.deepEqual(Array.from(quality.apply(qualityX).mask), [1,0,1]);
quality.dispose();

const composite = new NativeSampleFilter("Composite", [0], []);
assert.throws(() => composite.fit(X), /invalid argument/);
composite.addChild("HighLeverage", [0,0,0,1], [2,0]);
composite.addChild("SpectralQuality", [0,0,1], [.1,.5,1e-8,0,0]);
composite.fit(X);
assert.equal(composite.apply(X).mask.length, X.rows);
composite.dispose(); // recursively destroys owned children

for (const [kind, ints, doubles] of [
    ["XOutlier", [0,0,0,100,256], [0,.2]],
    ["HighLeverage", [0,0,0,1], [2,0]],
]) {
    const filter = new NativeSampleFilter(kind, ints, doubles, 17);
    filter.fit(X);
    assert.equal(filter.apply(X).mask.length, X.rows);
    filter.dispose();
}

const featureX = { rows: 5, cols: 3, data: Float64Array.from([
    1,4,2, 2,4,3, 3,4,4, 4,4,5, 5,4,6,
]) };
const featureY = { rows: 5, cols: 1, data: Float64Array.from([1,2,3,4,5]) };
const featureHeld = { rows: 2, cols: 3, data: Float64Array.from([6,9,7, 7,9,8]) };
for (const kind of ["Variance", "Correlation"]) {
    const filter = new NativeFeatureFilter(kind, .01);
    assert.throws(() => filter.selectedIndices(), /not fitted/);
    filter.fit(featureX, kind === "Correlation" ? featureY : null);
    assert.deepEqual(Array.from(filter.selectedIndices()), [0,2]);
    assert.deepEqual(Array.from(filter.transform(featureHeld).data), [6,7,7,8]);
    filter.dispose();
}
console.log("JS native filter roles: seven kinds, masks, indices, heldout and failures OK");
