// SPDX-License-Identifier: CECILL-2.1
import assert from "node:assert/strict";
import { augmentNative, loadModule } from "../dist/index.js";

await loadModule();
const X = { rows: 8, cols: 32, data: Float64Array.from(
    Array.from({ length: 8 * 32 }, (_, i) => 1 + 0.02 * (i % 32))) };
const originalX = Float64Array.from(X.data);
const cases = [
    ["GaussianNoise", [.03]], ["MultiplicativeNoise", [.03]],
    ["SpikeNoise", [1, 2, .1, .2]], ["HeteroNoise", [.01, .02]],
    ["LinearDrift", [.01, .02, .001, .002]], ["PathLength", [.05, .5]],
    ["BandPerturb", [1, 2, 4, .9, 1.1, -.01, .01]],
    ["BandMask", [1, 1, 2, 4, 0]], ["ChannelDropout", [.1, 0]],
    ["GaussJitter", [.5, 1, 5]], ["UnsharpMask", [.1, .2, 1, 5]],
    ["LocalClip", [1, 2, 4]], ["RotateTranslate", [.05, .05]],
    ["RandomXOp", [0, .9, 1.1]], ["ScatterSimMSC", [.9, 1.1, -.01, .01]],
    ["DeadBand", [1, 2, 4, .01, .5, 0]],
    ["BatchEffect", [.01, .01, .01, 0]], ["SplineSmoothing", []],
    ["SplineXPerturb", [3, .2, -.1, .1]],
    ["SplineYPerturb", [4, .1]], ["SplineXSimplify", [8, 1]],
    ["SplineCurveSimplify", [8, 1]],
];
for (const [kind, params] of cases) {
    const first = augmentNative(kind, X, params, 42);
    const second = augmentNative(kind, X, params, 42);
    assert.deepEqual(first, second, kind);
    assert.equal(first.data.length, X.data.length, kind);
    assert.ok(first.data.every(Number.isFinite), kind);
}
assert.deepEqual(X.data, originalX, "caller X remains unchanged");
// Frozen n4m-R oracle, same row-major X, PCG64 seed 42, four kernel families.
const rOracle = [
    ["GaussianNoise", 335.29953033891428, 447.93442481713896,
        [1.0016880886187998, 1.0142386382315953, 1.0441573912549518, 1.0652105927054174]],
    ["BandMask", 313.51999999999998, 417.7072,
        [1, 1.02, 1.04, 1.06]],
    ["ScatterSimMSC", 264.88033265406824, 274.76085353952521,
        [1.0473534823647035, 1.0472047278177739, 1.0470559732708440, 1.0469072187239143]],
    ["SplineXPerturb", 335.32596744464371, 447.94217484919568,
        [0.99888977536385415, 1.01915259930947943, 1.03941542325510450, 1.05967824720072956]],
];
for (const [kind, sum, sumsq, prefix] of rOracle) {
    const params = cases.find(row => row[0] === kind)[1];
    const values = augmentNative(kind, X, params, 42).data;
    const close = (a, b) => Math.abs(a - b) <= 1e-10 * Math.max(1, Math.abs(b));
    assert.ok(close(values.reduce((a, b) => a + b, 0), sum), `${kind} R sum`);
    assert.ok(close(values.reduce((a, b) => a + b * b, 0), sumsq), `${kind} R sumsq`);
    prefix.forEach((v, i) => assert.ok(close(values[i], v), `${kind} R row-major ${i}`));
}
assert.throws(() => augmentNative("Mixup", X, [.5], 42), /unknown/);
assert.throws(() => augmentNative("GaussianNoise", X, [], 42), /parameter/);
assert.throws(() => augmentNative("SpikeNoise", X, [1.5, 2, .1, .2], 42));
console.log("JS native train-only augmentation: 22 kinds, R oracle, seed and shape OK");
