// SPDX-License-Identifier: CECILL-2.1
//
// Tier-2 idiomatic TypeScript wrapper around the pls4all WASM tier-1
// fitPls / predictPls surface.
//
// Design notes:
//   - `fit()` and `predict()` are synchronous; the only async step is
//     the one-time `loadModule()` call exported by the parent index.
//   - Predictions use the same coefficient/mean contract as the C ABI,
//     so the math is bit-exact across bindings.
//
// Example:
//   import { loadModule } from "@pls4all/wasm";
//   import { PLSRegression } from "@pls4all/wasm/sklearn";
//   await loadModule();
//   const m = new PLSRegression({ nComponents: 5 });
//   m.fit(X, y);                       // X: Float64Array | number[][]
//   const yhat = m.predict(Xnew);
//   const r2 = m.score(Xtest, ytest);
//   m.destroy();

import { Model } from "./model.js";
import { type Matrix } from "./types.js";

export interface PLSRegressionParams {
    /** Number of latent components. Default 2. */
    nComponents?: number;
}

// Detect whether a 2-D-like input is a flat Float64Array or a number[][]
// and coerce to a row-major Float64Array + dims tuple.
function _to_matrix(arr: number[][] | Float64Array | Matrix,
                     name: string): Matrix {
    if (arr instanceof Float64Array) {
        throw new Error(
            `${name} must be a 2-D array (number[][] or {data, rows, cols})`);
    }
    if (typeof (arr as Matrix).data !== "undefined") {
        return arr as Matrix;
    }
    const arr2 = arr as number[][];
    const rows = arr2.length;
    if (rows === 0) throw new Error(`${name} is empty`);
    const firstRow = arr2[0];
    if (firstRow === undefined) {
        throw new Error(`${name}[0] is undefined`);
    }
    const cols = firstRow.length;
    const data = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
        const row = arr2[i];
        if (row === undefined || row.length !== cols) {
            throw new Error(
                `${name}: row ${i} has wrong shape`);
        }
        for (let j = 0; j < cols; j++) data[i * cols + j] = row[j] ?? 0;
    }
    return { data, rows, cols };
}

function _matrix_to_rows(m: Matrix): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < m.rows; i++) {
        const row: number[] = new Array(m.cols);
        for (let j = 0; j < m.cols; j++) {
            row[j] = m.data[i * m.cols + j] ?? 0;
        }
        out.push(row);
    }
    return out;
}

function _to_y_matrix(y: number[] | number[][] | Float64Array | Matrix,
                       n_rows: number): { Y: Matrix; yWas1D: boolean } {
    if ((y as Matrix).data !== undefined) {
        return { Y: y as Matrix, yWas1D: false };
    }
    if (y instanceof Float64Array || typeof (y as number[])[0] === "number") {
        const vec = y instanceof Float64Array
            ? y
            : new Float64Array(y as number[]);
        if (vec.length !== n_rows) {
            throw new Error(
                `y length (${vec.length}) must match X rows (${n_rows})`);
        }
        return {
            Y: { data: vec, rows: n_rows, cols: 1 },
            yWas1D: true,
        };
    }
    return { Y: _to_matrix(y as number[][], "y"), yWas1D: false };
}

/**
 * sklearn-style PLS regression class for the pls4all WASM binding.
 *
 * Lifecycle:
 *   * Constructor only stores hyperparams (cloneable).
 *   * `fit(X, y)` stores a plain JS model snapshot returned by the WASM ABI.
 *   * `destroy()` clears the fitted state. It is idempotent.
 */
export class PLSRegression {
    private params: Required<PLSRegressionParams>;
    private _model: Model | null = null;
    public n_features_in_ = -1;
    public n_targets_ = -1;
    private _y_was_1d = false;

    constructor(params: PLSRegressionParams = {}) {
        this.params = {
            nComponents: params.nComponents ?? 2,
        };
        if (!Number.isInteger(this.params.nComponents) ||
                this.params.nComponents < 1) {
            throw new Error(
                `nComponents must be a positive integer; got ` +
                `${this.params.nComponents}`);
        }
    }

    /** Train the model on (X, y). Returns `this` for chaining. */
    fit(X: number[][] | Matrix, y: number[] | number[][] | Matrix): this {
        const Xm = _to_matrix(X, "X");
        const { Y, yWas1D } = _to_y_matrix(y, Xm.rows);
        // Tear down any prior fit (and its registry entry) so the WASM
        // memory is freed before we create new handles.
        this.destroy();
        this._model = Model.fit(null, null, Xm, Y, this.params.nComponents);
        this.n_features_in_ = Xm.cols;
        this.n_targets_ = Y.cols;
        this._y_was_1d = yWas1D;
        return this;
    }

    /** Predict from the fitted model. */
    predict(X: number[][] | Matrix): Float64Array | number[][] {
        if (!this._model) {
            throw new Error("PLSRegression is not fitted yet");
        }
        const Xm = _to_matrix(X, "X");
        if (Xm.cols !== this.n_features_in_) {
            throw new Error(
                `X has ${Xm.cols} features, but model was fitted with ` +
                `${this.n_features_in_}`);
        }
        const preds = this._model.predict(null, Xm);
        // 1-D y in => 1-D predictions out (sklearn convention).
        if (this._y_was_1d && preds.cols === 1) {
            return preds.data;
        }
        // Multi-target: return as number[][] for convenience.
        const out: number[][] = [];
        for (let i = 0; i < preds.rows; i++) {
            const row: number[] = new Array(preds.cols);
            for (let j = 0; j < preds.cols; j++) {
                row[j] = preds.data[i * preds.cols + j] ?? 0;
            }
            out.push(row);
        }
        return out;
    }

    /** R² coefficient of determination on (X, y).
     *
     * For single-target regression returns a scalar; for multi-target
     * returns the *uniformly-averaged* R² across targets (matches
     * sklearn's `multioutput="uniform_average"` default).
     */
    score(X: number[][] | Matrix,
          y: number[] | number[][] | Matrix): number {
        const yhat = this.predict(X);
        // Normalize both prediction and observed matrices to (n, q).
        const yhatMat: number[][] = (yhat instanceof Float64Array)
            ? Array.from(yhat).map(v => [v])
            : (yhat as number[][]);
        let yMat: number[][];
        if (y instanceof Float64Array) {
            yMat = Array.from(y).map(v => [v]);
        } else if (Array.isArray(y) && typeof (y as number[])[0] === "number") {
            yMat = (y as number[]).map(v => [v]);
        } else if ((y as Matrix).data !== undefined) {
            yMat = _matrix_to_rows(y as Matrix);
        } else {
            yMat = y as number[][];
        }
        if (yhatMat.length !== yMat.length) {
            throw new Error(
                `predict rows (${yhatMat.length}) != y rows (${yMat.length})`);
        }
        const n = yMat.length;
        if (n === 0) throw new Error("score: empty y");
        const firstRow = yMat[0]!;
        const q = firstRow.length;
        if (q === 0) throw new Error("score: zero-width y");
        // Per-target R², then mean.
        let r2_sum = 0;
        for (let t = 0; t < q; t++) {
            let mean = 0;
            for (let i = 0; i < n; i++) {
                const v = yMat[i]?.[t];
                if (v === undefined) {
                    throw new Error(`score: y[${i}][${t}] is undefined`);
                }
                mean += v;
            }
            mean /= n;
            let ss_res = 0, ss_tot = 0;
            for (let i = 0; i < n; i++) {
                const yi = yMat[i]?.[t];
                const yhi = yhatMat[i]?.[t];
                if (yi === undefined || yhi === undefined) {
                    throw new Error(
                        `score: shape mismatch at row ${i}, target ${t}`);
                }
                const d_res = yhi - yi;
                const d_tot = yi - mean;
                ss_res += d_res * d_res;
                ss_tot += d_tot * d_tot;
            }
            r2_sum += 1 - ss_res / ss_tot;
        }
        return r2_sum / q;
    }

    /** Clear the fitted model. Idempotent. */
    destroy(): void {
        if (this._model) {
            try { this._model.destroy(); } catch { /* swallow */ }
            this._model = null;
        }
    }
}
