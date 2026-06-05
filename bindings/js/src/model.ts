// SPDX-License-Identifier: CECILL-2.1
//
// PLS regression model wrapper. Internally uses the `n4m_wasm_pls_fit`
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
            "n4m_pls_fit_simple", "number",
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
            "n4m_wasm_pls_predict_from_coeffs", "number",
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

/** A fitted coefficient-based model produced by {@link fitModel}. */
export interface FittedModel {
    /** Regression coefficients, row-major (n_features × n_targets). */
    coefficients: Float64Array;
    /** Per-feature mean used for centring. */
    xMean: Float64Array;
    /** Per-target mean used for centring. */
    yMean: Float64Array;
    /** Per-target intercept (null when the model centres without one). */
    intercept: Float64Array | null;
    n_features: number;
    n_targets: number;
}

/** Fit any coefficient-based libn4m model by its catalog `type` token.
 *
 * Tier A (PLS / PLSRegression / PCR / PLSCanonical / PLSSVD / PLSDA) routes
 * through the algorithm-enum model API; Tier B (Ridge, RidgePLS, CPPLS, ...)
 * through the matching standalone fit. The `params` vector is the documented
 * positional contract per model (see the studio-lite catalog). Unknown or
 * non-coefficient tokens throw (the C side returns N4M_ERR_NOT_IMPLEMENTED).
 *
 * @param model catalog `type` token, e.g. `'Ridge'`.
 * @param X row-major (n × p) input matrix.
 * @param Y row-major (n × q) target matrix.
 * @param n_components number of latent components (used by the PLS family).
 * @param params positional hyper-parameter vector for the model.
 */
export function fitModel(model: string, X: Matrix, Y: Matrix,
                          n_components: number, params: number[] = []
                          ): FittedModel {
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
    const interBuf = _malloc_f64(M, q);
    const hasInterBuf = M._malloc(4); // int32 flag: 1 iff a genuine intercept
    const pPtr = params.length > 0 ? _malloc_f64(M, params.length) : { ptr: 0, len: 0 };
    try {
        _copy_in(M, X.data, xBuf.ptr);
        _copy_in(M, Y.data, yBuf.ptr);
        if (params.length > 0) _copy_in(M, Float64Array.from(params), pPtr.ptr);
        const status = M.ccall(
            "n4m_wasm_model_fit", "number",
            ["string", "number", "number", "number", "number",
             "number", "number", "number", "number",
             "number", "number", "number", "number", "number", "number"],
            [model, pPtr.ptr, params.length, xBuf.ptr, yBuf.ptr,
             n, p, q, n_components,
             coefsBuf.ptr, xmBuf.ptr, ymBuf.ptr, interBuf.ptr,
             hasInterBuf, 0]) as number;
        checkStatus(status);
        // Only models with a genuine affine intercept (currently Ridge) report
        // has_intercept=1; the PLS/PCR family and the PLS-based Tier-B fits
        // predict via the centred form and carry no intercept (kept null so a
        // caller never adds a misleading zero/y_mean term to x.B).
        const hasIntercept = M.HEAP32[hasInterBuf >> 2] === 1;
        return {
            coefficients: _read_out(M, coefsBuf.ptr, p * q),
            xMean: _read_out(M, xmBuf.ptr, p),
            yMean: _read_out(M, ymBuf.ptr, q),
            intercept: hasIntercept ? _read_out(M, interBuf.ptr, q) : null,
            n_features: p,
            n_targets: q,
        };
    } finally {
        M._free(xBuf.ptr);
        M._free(yBuf.ptr);
        M._free(coefsBuf.ptr);
        M._free(xmBuf.ptr);
        M._free(ymBuf.ptr);
        M._free(interBuf.ptr);
        M._free(hasInterBuf);
        if (pPtr.ptr !== 0) M._free(pPtr.ptr);
    }
}

/** Predict from a fitted {@link FittedModel} for new X (row-major n_new × p). */
export function predictModel(model: FittedModel, X_new: Matrix): Matrix {
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
        // Centred form (intercept = NULL): pred = y_mean + (x - x_mean).B.
        // Every libn4m coeff model — PLS family and the Tier-B fits, including
        // Ridge — exposes a valid (coefficients, x_mean, y_mean) triple whose
        // in-sample predictions use exactly this convention, so one path covers
        // all of them and the stored `intercept` stays informational only.
        const status = M.ccall(
            "n4m_wasm_model_predict_from_coeffs", "number",
            ["number", "number", "number", "number",
             "number", "number", "number", "number", "number"],
            [coefsBuf.ptr, xmBuf.ptr, ymBuf.ptr, 0,
             xBuf.ptr, n_new, p, q, predsBuf.ptr]) as number;
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
