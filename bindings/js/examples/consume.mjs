// SPDX-License-Identifier: CECILL-2.1
//
// Minimal consumption example for @nirs4all/methods-wasm.
// A downstream repo (nirs4all-lite) would do:
//   import * as n4m from "@nirs4all/methods-wasm";
// From inside this repo, import the built dist barrel instead:
import * as n4m from "../dist/index.js";

await n4m.loadModule();
console.log("libn4m", n4m.version(), "ABI", n4m.abiVersion().join("."));

// ----- Build raw row-major Float64Array inputs (the input contract) -----
const rows = 40, cols = 6;
const X = new Float64Array(rows * cols);
const y = new Float64Array(rows);
for (let i = 0; i < rows; i++) {
  for (let j = 0; j < cols; j++) X[i * cols + j] = Math.sin((i + 1) * (j + 1) * 0.3);
  y[i] = X[i * cols] + 0.5 * X[i * cols + 1] - 0.3 * X[i * cols + 2];
}

// ----- Fit a PLS regression (raw arrays in, model out) -----
const model = n4m.fitPls(
  { data: X, rows, cols },
  { data: y, rows, cols: 1 },
  3,                       // n_components
);
console.log("coefficients:", Array.from(model.coefficients).map((v) => v.toFixed(4)));

// ----- Predict in-sample -----
const preds = n4m.predictPls(model, { data: X, rows, cols });
let sse = 0;
for (let i = 0; i < rows; i++) { const d = preds.data[i] - y[i]; sse += d * d; }
console.log("in-sample RMSE:", Math.sqrt(sse / rows).toExponential(3));
