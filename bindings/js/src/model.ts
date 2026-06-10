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
    // AOM-PLS (and any future affine model) carries input-space coefficients +
    // a genuine intercept and zero means — it predicts on RAW X via the
    // explicit-intercept form  pred = intercept + x.B. Every centred model
    // (PLS family + the Tier-B fits, including Ridge) carries intercept = null
    // and predicts via  pred = y_mean + (x - x_mean).B. The C helper picks the
    // form from whether the intercept pointer is non-NULL.
    const useIntercept = model.intercept !== null;
    const interBuf = useIntercept ? _malloc_f64(M, q) : { ptr: 0, len: 0 };
    try {
        _copy_in(M, X_new.data, xBuf.ptr);
        _copy_in(M, model.coefficients, coefsBuf.ptr);
        _copy_in(M, model.xMean, xmBuf.ptr);
        _copy_in(M, model.yMean, ymBuf.ptr);
        if (useIntercept) _copy_in(M, model.intercept as Float64Array, interBuf.ptr);
        const status = M.ccall(
            "n4m_wasm_model_predict_from_coeffs", "number",
            ["number", "number", "number", "number",
             "number", "number", "number", "number", "number"],
            [coefsBuf.ptr, xmBuf.ptr, ymBuf.ptr, interBuf.ptr,
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
        if (interBuf.ptr !== 0) M._free(interBuf.ptr);
    }
}

/** A fitted AOM-PLS model — a {@link FittedModel} (so {@link predictModel}
 *  works unchanged) plus the screen result. Its `intercept` is a genuine
 *  input-space intercept and its `xMean` / `yMean` are zero, so prediction is
 *  the affine form  y = intercept + X.B  on RAW X. */
export interface AomModel extends FittedModel {
    /** Bank index of the operator the internal CV selected. */
    selectedOperator: number;
    /** Best internal-CV score of the selected operator. */
    score: number;
}

/** Fit AOM-PLS (operator-adaptive PLS) on (X, Y).
 *
 * Screens a bank of strict-linear preprocessing operators by internal k-fold CV
 * and fits SIMPLS on the winner, returning INPUT-SPACE coefficients so the model
 * predicts on RAW X — it is therefore used WITHOUT preceding preprocessing steps
 * (the screen does the preprocessing internally). Numerics are 100% libn4m
 * (`n4m_aom_global_select`); this only builds the bank + validation plan.
 *
 * @param X row-major (n × p) input matrix.
 * @param Y row-major (n × q) target matrix.
 * @param maxComponents max latent components for the internal SIMPLS fits.
 * @param nFolds internal-CV fold count for the operator screen.
 * @param seed reserved (the contiguous-fold partition is deterministic).
 * @param operatorKinds optional `n4m_operator_kind_t` bank override; when
 *   omitted a default strict bank (identity / detrend / SG smooth / SG
 *   derivative / finite-difference) is screened.
 */
export function fitAom(X: Matrix, Y: Matrix, maxComponents: number,
                        nFolds = 5, seed = 0,
                        operatorKinds: number[] = []): AomModel {
    if (X.rows !== Y.rows) {
        throw new Error(`X.rows (${X.rows}) must equal Y.rows (${Y.rows})`);
    }
    const M = getModule();
    const n = X.rows, p = X.cols, q = Y.cols;
    const xBuf = _malloc_f64(M, n * p);
    const yBuf = _malloc_f64(M, n * q);
    const coefsBuf = _malloc_f64(M, p * q);
    const interBuf = _malloc_f64(M, q);
    const selBuf = M._malloc(4); // int32 selected operator index
    const scoreBuf = _malloc_f64(M, 1);
    const opsPtr = operatorKinds.length > 0
        ? M._malloc(operatorKinds.length * 4)
        : 0;
    try {
        _copy_in(M, X.data, xBuf.ptr);
        _copy_in(M, Y.data, yBuf.ptr);
        if (opsPtr !== 0) M.HEAP32.set(Int32Array.from(operatorKinds), opsPtr >> 2);
        const status = M.ccall(
            "n4m_wasm_aom_fit", "number",
            ["number", "number", "number", "number", "number",
             "number", "number", "number", "number", "number",
             "number", "number", "number", "number"],
            [xBuf.ptr, yBuf.ptr, n, p, q,
             maxComponents, nFolds, seed, opsPtr, operatorKinds.length,
             coefsBuf.ptr, interBuf.ptr, selBuf, scoreBuf.ptr]) as number;
        checkStatus(status);
        return {
            coefficients: _read_out(M, coefsBuf.ptr, p * q),
            // zero means — AOM predicts on RAW X via the affine intercept form.
            xMean: new Float64Array(p),
            yMean: new Float64Array(q),
            intercept: _read_out(M, interBuf.ptr, q),
            n_features: p,
            n_targets: q,
            selectedOperator: M.HEAP32[selBuf >> 2] ?? -1,
            score: _read_out(M, scoreBuf.ptr, 1)[0] ?? NaN,
        };
    } finally {
        M._free(xBuf.ptr);
        M._free(yBuf.ptr);
        M._free(coefsBuf.ptr);
        M._free(interBuf.ptr);
        M._free(selBuf);
        M._free(scoreBuf.ptr);
        if (opsPtr !== 0) M._free(opsPtr);
    }
}

/** A fitted POP-PLS model — a {@link FittedModel} (so {@link predictModel}
 *  works unchanged) plus the per-component screen result. Its `intercept` is a
 *  genuine input-space intercept and its `xMean` / `yMean` are zero, so
 *  prediction is the affine form  y = intercept + X.B  on RAW X. */
export interface PopModel extends FittedModel {
    /** Bank index of the operator picked at each selected latent component
     *  (length = `selectedComponents`). */
    selectedOperators: number[];
    /** Number of latent components the per-component screen selected. */
    selectedComponents: number;
    /** Best internal-CV prefix score of the selected model. */
    score: number;
}

/** Fit POP-PLS (per-component operator-adaptive PLS) on (X, Y).
 *
 * Like AOM-PLS but picks one strict-linear operator PER latent component
 * (`n4m_aom_per_component_select`) rather than one for the whole model, then
 * returns INPUT-SPACE coefficients so it predicts on RAW X via the same affine
 * intercept path — so it is used WITHOUT preceding preprocessing steps (the
 * screen does the preprocessing internally). Numerics are 100% libn4m; this
 * only builds the bank + validation plan.
 *
 * @param X row-major (n × p) input matrix.
 * @param Y row-major (n × q) target matrix.
 * @param maxComponents max latent components for the internal SIMPLS fits.
 * @param nFolds internal-CV fold count for the operator screen.
 * @param seed reserved (the contiguous-fold partition is deterministic).
 * @param operatorKinds optional `n4m_operator_kind_t` bank override; when
 *   omitted a default strict bank (identity / detrend / SG smooth / SG
 *   derivative / finite-difference) is screened.
 */
export function fitPop(X: Matrix, Y: Matrix, maxComponents: number,
                        nFolds = 5, seed = 0,
                        operatorKinds: number[] = []): PopModel {
    if (X.rows !== Y.rows) {
        throw new Error(`X.rows (${X.rows}) must equal Y.rows (${Y.rows})`);
    }
    const M = getModule();
    const n = X.rows, p = X.cols, q = Y.cols;
    // The selector clamps max_components to min(maxComponents, p, n-1); allocate
    // the per-component op buffer to the un-clamped request (always >= clamp).
    const maxComp = Math.max(1, maxComponents);
    const xBuf = _malloc_f64(M, n * p);
    const yBuf = _malloc_f64(M, n * q);
    const coefsBuf = _malloc_f64(M, p * q);
    const interBuf = _malloc_f64(M, q);
    const opsOutBuf = M._malloc(maxComp * 4); // int32 per-component op indices
    const nSelBuf = M._malloc(4);             // int32 selected component count
    const scoreBuf = _malloc_f64(M, 1);
    const opsPtr = operatorKinds.length > 0
        ? M._malloc(operatorKinds.length * 4)
        : 0;
    try {
        _copy_in(M, X.data, xBuf.ptr);
        _copy_in(M, Y.data, yBuf.ptr);
        if (opsPtr !== 0) M.HEAP32.set(Int32Array.from(operatorKinds), opsPtr >> 2);
        const status = M.ccall(
            "n4m_wasm_pop_fit", "number",
            ["number", "number", "number", "number", "number",
             "number", "number", "number", "number", "number",
             "number", "number", "number", "number", "number"],
            [xBuf.ptr, yBuf.ptr, n, p, q,
             maxComp, nFolds, seed, opsPtr, operatorKinds.length,
             coefsBuf.ptr, interBuf.ptr, opsOutBuf, nSelBuf, scoreBuf.ptr]) as number;
        checkStatus(status);
        const nSel = Math.max(0, M.HEAP32[nSelBuf >> 2] ?? 0);
        const selectedOperators: number[] = [];
        for (let k = 0; k < nSel; ++k) {
            selectedOperators.push(M.HEAP32[(opsOutBuf >> 2) + k] ?? -1);
        }
        return {
            coefficients: _read_out(M, coefsBuf.ptr, p * q),
            // zero means — POP predicts on RAW X via the affine intercept form.
            xMean: new Float64Array(p),
            yMean: new Float64Array(q),
            intercept: _read_out(M, interBuf.ptr, q),
            n_features: p,
            n_targets: q,
            selectedOperators,
            selectedComponents: nSel,
            score: _read_out(M, scoreBuf.ptr, 1)[0] ?? NaN,
        };
    } finally {
        M._free(xBuf.ptr);
        M._free(yBuf.ptr);
        M._free(coefsBuf.ptr);
        M._free(interBuf.ptr);
        M._free(opsOutBuf);
        M._free(nSelBuf);
        M._free(scoreBuf.ptr);
        if (opsPtr !== 0) M._free(opsPtr);
    }
}

/** A train/test splitter kind for {@link computeSplit}. */
export type SplitKind = "KennardStone" | "SPXY" | "KMeans" | "KBinsStratified";

/** Options for {@link computeSplit}. `testSize` is a fraction in (0, 1). */
export interface SplitOptions {
    /** test fraction in (0, 1); default 0.25. */
    testSize?: number;
    /** seed for the stochastic splitters (KMeans, KBinsStratified). */
    seed?: number;
    /** KMeans: max iterations (default 100). */
    maxIter?: number;
    /** KBinsStratified: number of Y bins (default 5). */
    nBins?: number;
    /** KBinsStratified: 0 = uniform-width bins, 1 = quantile bins. */
    strategy?: number;
}

const SPLIT_KIND_CODE: Record<SplitKind, number> = {
    KennardStone: 0,
    SPXY: 1,
    KMeans: 2,
    KBinsStratified: 3,
};

/** Compute a single train/test split over the rows of X (and Y) via libn4m's
 * splitters, returning a `Uint8Array` mask of length n where 1 = test, 0 = train.
 *
 * Numerics are 100% libn4m (`n4m_wasm_split` → n4m_split_*). KennardStone and
 * SPXY are deterministic; KMeans and KBinsStratified use `opts.seed`. SPXY and
 * KBinsStratified need Y; KennardStone and KMeans use X only.
 *
 * @param kind splitter strategy.
 * @param X row-major (n × p) input matrix.
 * @param Y row-major (n × q) target matrix (required for SPXY / KBinsStratified).
 * @param opts split options (testSize / seed / maxIter / nBins / strategy).
 */
export function computeSplit(kind: SplitKind, X: Matrix, Y: Matrix | null,
                             opts: SplitOptions = {}): Uint8Array {
    const M = getModule();
    const n = X.rows, p = X.cols;
    const q = Y ? Y.cols : 1;
    const testSize = opts.testSize ?? 0.25;
    const seed = (opts.seed ?? 0) >>> 0;
    // p0/p1 generic int params: KMeans → maxIter; KBins → nBins, strategy.
    let p0 = 0, p1 = 0;
    if (kind === "KMeans") p0 = opts.maxIter ?? 100;
    if (kind === "KBinsStratified") { p0 = opts.nBins ?? 5; p1 = opts.strategy ?? 0; }

    const xBuf = _malloc_f64(M, n * p);
    const yBuf = Y ? _malloc_f64(M, n * q) : { ptr: 0, len: 0 };
    const maskBuf = M._malloc(n * 4); // int32[n]
    try {
        _copy_in(M, X.data, xBuf.ptr);
        if (Y) _copy_in(M, Y.data, yBuf.ptr);
        const status = M.ccall(
            "n4m_wasm_split", "number",
            ["number", "number", "number", "number", "number",
             "number", "number", "number", "number", "number", "number"],
            [SPLIT_KIND_CODE[kind], testSize, seed, p0, p1,
             xBuf.ptr, yBuf.ptr, n, p, q, maskBuf]) as number;
        checkStatus(status);
        const mask = new Uint8Array(n);
        for (let i = 0; i < n; i++) mask[i] = M.HEAP32[(maskBuf >> 2) + i] === 1 ? 1 : 0;
        return mask;
    } finally {
        M._free(xBuf.ptr);
        if (yBuf.ptr !== 0) M._free(yBuf.ptr);
        M._free(maskBuf);
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
