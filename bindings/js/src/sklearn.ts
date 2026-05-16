// SPDX-License-Identifier: CeCILL-2.1
//
// Tier-2 idiomatic TypeScript wrapper around the pls4all WASM tier-1
// surface (Context / Config / Model / MethodResult).
//
// Design notes:
//   - WASM modules require explicit memory management. We use a
//     FinalizationRegistry so estimators released by the GC also
//     release their underlying C handles. Callers SHOULD still call
//     `destroy()` explicitly for predictable cleanup.
//   - `fit()` and `predict()` are synchronous; the only async step is
//     the one-time `loadModule()` call exported by the parent index.
//   - Predictions use the C ABI prediction path directly
//     (Model.predict) — same code path as Python's `pls4all.sklearn`,
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

import { Context } from "./context.js";
import { Config } from "./config.js";
import { Model } from "./model.js";
import {
    Algorithm,
    Deflation,
    Solver,
    type Matrix,
} from "./types.js";

export interface PLSRegressionParams {
    /** Number of latent components. Default 2. */
    nComponents?: number;
    /** Solver kind. Default 'simpls'. */
    solver?: "nipals" | "simpls" | "svd" | "power" | "kernel" |
              "wide-kernel" | "orthogonal-scores" | "randomized-svd";
    /** Center X columns to zero mean. Default true. */
    centerX?: boolean;
    /** Center Y columns. Default true. */
    centerY?: boolean;
}

const SOLVER_MAP: Record<NonNullable<PLSRegressionParams["solver"]>, Solver> = {
    "nipals": Solver.NIPALS,
    "simpls": Solver.SIMPLS,
    "svd": Solver.SVD,
    "power": Solver.POWER,
    "kernel": Solver.KERNEL_ALGORITHM,
    "wide-kernel": Solver.WIDE_KERNEL,
    "orthogonal-scores": Solver.ORTHOGONAL_SCORES,
    "randomized-svd": Solver.RANDOMIZED_SVD,
};

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

const _registry = new FinalizationRegistry((dispose: () => void) => {
    try { dispose(); } catch { /* swallow — GC must not throw */ }
});

/**
 * sklearn-style PLS regression class for the pls4all WASM binding.
 *
 * Lifecycle:
 *   * Constructor only stores hyperparams (cloneable).
 *   * `fit(X, y)` materialises a C Context + Config + Model and stores
 *     the Model + Context for `predict` hot-path use.
 *   * `destroy()` releases the C handles. A FinalizationRegistry is
 *     registered as a safety net so dropped instances eventually free
 *     their WASM memory.
 */
export class PLSRegression {
    private params: Required<PLSRegressionParams>;
    private _ctx: Context | null = null;
    private _model: Model | null = null;
    public n_features_in_ = -1;
    public n_targets_ = -1;
    private _y_was_1d = false;

    constructor(params: PLSRegressionParams = {}) {
        this.params = {
            nComponents: params.nComponents ?? 2,
            solver: params.solver ?? "simpls",
            centerX: params.centerX ?? true,
            centerY: params.centerY ?? true,
        };
    }

    /** Train the model on (X, y). Returns `this` for chaining. */
    fit(X: number[][] | Matrix, y: number[] | number[][] | Matrix): this {
        const Xm = _to_matrix(X, "X");
        const { Y, yWas1D } = _to_y_matrix(y, Xm.rows);
        // Tear down any prior fit so the WASM memory is freed before
        // we create new handles.
        this.destroy();
        const ctx = Context.create();
        const cfg = Config.create();
        try {
            cfg.setAlgorithm(Algorithm.PLS_REGRESSION);
            cfg.setSolver(SOLVER_MAP[this.params.solver]);
            cfg.setDeflation(Deflation.REGRESSION);
            cfg.setNComponents(this.params.nComponents);
            cfg.setCenterX(this.params.centerX);
            cfg.setCenterY(this.params.centerY);
            const model = Model.fit(ctx, cfg, Xm, Y);
            this._ctx = ctx;
            this._model = model;
            this.n_features_in_ = Xm.cols;
            this.n_targets_ = Y.cols;
            this._y_was_1d = yWas1D;
        } finally {
            cfg.destroy();
        }
        // Register WASM-handle finalizer as a safety net.
        const ctxRef = this._ctx;
        const modelRef = this._model;
        _registry.register(this, () => {
            try { modelRef!.destroy(); } catch { /* swallow */ }
            try { ctxRef!.destroy(); } catch { /* swallow */ }
        }, this);
        return this;
    }

    /** Predict from the fitted model. */
    predict(X: number[][] | Matrix): Float64Array | number[][] {
        if (!this._ctx || !this._model) {
            throw new Error("PLSRegression is not fitted yet");
        }
        const Xm = _to_matrix(X, "X");
        if (Xm.cols !== this.n_features_in_) {
            throw new Error(
                `X has ${Xm.cols} features, but model was fitted with ` +
                `${this.n_features_in_}`);
        }
        const preds = this._model.predict(this._ctx, Xm);
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

    /** R² score on (X, y). */
    score(X: number[][] | Matrix,
          y: number[] | number[][] | Matrix): number {
        const yhat = this.predict(X);
        const flat: number[] = (yhat instanceof Float64Array)
            ? Array.from(yhat)
            : (yhat as number[][]).map(r => r[0] ?? 0);
        let yVec: number[];
        if (y instanceof Float64Array) {
            yVec = Array.from(y);
        } else if (Array.isArray(y) && typeof (y as number[])[0] === "number") {
            yVec = y as number[];
        } else {
            yVec = (y as number[][]).map(r => r[0] ?? 0);
        }
        if (flat.length !== yVec.length) {
            throw new Error(
                `predict length (${flat.length}) != y length (${yVec.length})`);
        }
        const n = yVec.length;
        let mean = 0;
        for (let i = 0; i < n; i++) mean += yVec[i] ?? 0;
        mean /= n;
        let ss_res = 0, ss_tot = 0;
        for (let i = 0; i < n; i++) {
            const yi = yVec[i] ?? 0;
            const yhi = flat[i] ?? 0;
            const d_res = yhi - yi;
            const d_tot = yi - mean;
            ss_res += d_res * d_res;
            ss_tot += d_tot * d_tot;
        }
        return 1 - ss_res / ss_tot;
    }

    /** Release the underlying WASM handles. Idempotent. */
    destroy(): void {
        if (this._model) {
            try { this._model.destroy(); } catch { /* swallow */ }
            this._model = null;
        }
        if (this._ctx) {
            try { this._ctx.destroy(); } catch { /* swallow */ }
            this._ctx = null;
        }
    }
}
