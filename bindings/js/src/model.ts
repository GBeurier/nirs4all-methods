// SPDX-License-Identifier: CeCILL-2.1
//
// PLS regression model wrapper. Internally uses the `p4a_wasm_pls_fit`
// helper which takes raw double pointers and works around an Emscripten
// 5.0.7 codegen issue for matrix-view-pointer args (see README "Status").

import { checkStatus, getModule } from "./ffi.js";
import { Matrix } from "./types.js";

export interface PlsModel {
    /** Regression coefficients, row-major (n_features × n_targets). */
    coefficients: Float64Array;
    /** Per-feature mean used for centring. */
    xMean: Float64Array;
    /** Per-target mean used for centring. */
    yMean: Float64Array;
    /** Number of features `p` and targets `q`. */
    n_features: number;
    n_targets: number;
}

function _malloc_f64(M: ReturnType<typeof getModule>, len: number) {
    const ptr = M._malloc(len * 8);
    return { ptr, len };
}

function _copy_in(M: ReturnType<typeof getModule>, buf: Float64Array,
                   ptr: number) {
    if (buf.length === 0) return;
    M.HEAPF64.set(buf, ptr >>> 3);
}

function _read_out(M: ReturnType<typeof getModule>, ptr: number,
                    len: number): Float64Array {
    return new Float64Array(M.HEAPU8.buffer, ptr, len).slice();
}

/** Fit a SIMPLS PLS-regression model on (X, Y).
 *
 * @param X row-major (n × p) input matrix.
 * @param Y row-major (n × q) target matrix.
 * @param n_components number of latent components.
 */
export function fitPls(X: Matrix, Y: Matrix, n_components: number
                        ): PlsModel {
    if (X.rows !== Y.rows) {
        throw new Error(`X.rows (${X.rows}) must equal Y.rows (${Y.rows})`);
    }
    const M = getModule();
    const n = X.rows, p = X.cols, q = Y.cols;
    const xBuf = _malloc_f64(M, n * p);
    const yBuf = _malloc_f64(M, n * q);
    const coefsBuf = _malloc_f64(M, p * q);
    const xmBuf = _malloc_f64(M, p);
    const ymBuf = _malloc_f64(M, q);
    try {
        _copy_in(M, X.data, xBuf.ptr);
        _copy_in(M, Y.data, yBuf.ptr);
        // Uses the public ABI helper (1.13+): raw double pointers
        // + ints, no matrix-view structs in the JS↔WASM boundary.
        const status = M.ccall(
            "p4a_pls_fit_simple", "number",
            ["number", "number", "number", "number", "number",
             "number", "number", "number", "number", "number"],
            [xBuf.ptr, yBuf.ptr, n, p, q, n_components,
             coefsBuf.ptr, xmBuf.ptr, ymBuf.ptr, 0]) as number;
        checkStatus(status);
        return {
            coefficients: _read_out(M, coefsBuf.ptr, p * q),
            xMean: _read_out(M, xmBuf.ptr, p),
            yMean: _read_out(M, ymBuf.ptr, q),
            n_features: p,
            n_targets: q,
        };
    } finally {
        M._free(xBuf.ptr);
        M._free(yBuf.ptr);
        M._free(coefsBuf.ptr);
        M._free(xmBuf.ptr);
        M._free(ymBuf.ptr);
    }
}

/** Predict from a fitted PlsModel for new X (row-major n_new × p). */
export function predictPls(model: PlsModel, X_new: Matrix): Matrix {
    if (X_new.cols !== model.n_features) {
        throw new Error(
            `X_new.cols (${X_new.cols}) must equal n_features (` +
            `${model.n_features})`);
    }
    const M = getModule();
    const n_new = X_new.rows, p = model.n_features, q = model.n_targets;
    const xBuf = _malloc_f64(M, n_new * p);
    const coefsBuf = _malloc_f64(M, p * q);
    const xmBuf = _malloc_f64(M, p);
    const ymBuf = _malloc_f64(M, q);
    const predsBuf = _malloc_f64(M, n_new * q);
    try {
        _copy_in(M, X_new.data, xBuf.ptr);
        _copy_in(M, model.coefficients, coefsBuf.ptr);
        _copy_in(M, model.xMean, xmBuf.ptr);
        _copy_in(M, model.yMean, ymBuf.ptr);
        const status = M.ccall(
            "p4a_wasm_pls_predict_from_coeffs", "number",
            ["number", "number", "number", "number",
             "number", "number", "number", "number"],
            [xBuf.ptr, n_new, p, q,
             coefsBuf.ptr, xmBuf.ptr, ymBuf.ptr, predsBuf.ptr]) as number;
        checkStatus(status);
        return {
            data: _read_out(M, predsBuf.ptr, n_new * q),
            rows: n_new,
            cols: q,
        };
    } finally {
        M._free(xBuf.ptr);
        M._free(coefsBuf.ptr);
        M._free(xmBuf.ptr);
        M._free(ymBuf.ptr);
        M._free(predsBuf.ptr);
    }
}

/* Legacy class wrapper preserved for backwards compat with the
 * scaffold; not yet exposed via index.ts. */
export class Model {
    private _data: PlsModel;
    private constructor(data: PlsModel) { this._data = data; }

    static fit(_ctx: unknown, _cfg: unknown, X: Matrix, Y: Matrix,
                n_components: number = 3): Model {
        return new Model(fitPls(X, Y, n_components));
    }
    predict(_ctx: unknown, X_new: Matrix): Matrix {
        return predictPls(this._data, X_new);
    }
    get coefficients(): Float64Array { return this._data.coefficients; }
    get xMean(): Float64Array { return this._data.xMean; }
    get yMean(): Float64Array { return this._data.yMean; }
    destroy(): void { /* no-op: model is plain JS data */ }
}
