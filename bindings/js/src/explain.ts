// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Model explanation (attribution) bindings over cpp/include/n4m/explain.h.
//
// Every attribution is computed by libn4m. This module only marshals inputs,
// drives the ask/tell session (the host predicts the requested rows with any
// backend, synchronously or asynchronously) and reports the native results.

import { Context } from "./context.js";
import { checkStatus, getModule, makeMatrixView } from "./ffi.js";
import { N4mError, Status, type Matrix } from "./types.js";

export type ExplainMethod = "exact_shap" | "permutation_shap" | "kernel_shap" | "lime" | "occlusion";
export type ExplainBaseline = "background" | "mean" | "interpolate";
export type AttributionKind = ExplainMethod | "linear_shap" | "tree_shap";

export interface ExplainOptions {
    /** Default "permutation_shap". */
    method?: ExplainMethod;
    /** Default "background". */
    baseline?: ExplainBaseline;
    /** Antithetic permutation pairs per row (default 10). */
    nPermutations?: number;
    /** Sampled coalitions per row for kernel_shap/lime; 0 = native auto. */
    nCoalitions?: number;
    /** LIME kernel width; <= 0 = native auto. */
    limeKernelWidth?: number;
    /** LIME ridge penalty (default 1). */
    limeRidge?: number;
    /** Upper bound of rows per requested batch (default 4096). */
    maxBatchRows?: number;
    /** Sampling seed: a nonnegative integer < 2^53 (default 0). */
    seed?: number;
}

/** rows: row-major nRows x nFeatures in the explained feature space.
 *  Returns nRows x nOutputs row-major predictions. */
export type PredictBatch = (rows: Float64Array, nRows: number, nFeatures: number) =>
    Float64Array | number[] | Promise<Float64Array | number[]>;

/** Feature partition: featureGroup[j] in [0, nGroups) for every feature j. */
export interface FeatureGrouping {
    featureGroup: Int32Array;
    nGroups: number;
    /** groupsFromRanges only: source range index per group, -1 for uncovered runs. */
    rangeOfGroup?: Int32Array;
}

export interface ExplainModelInput {
    /** Rows to explain. */
    X: Matrix;
    /** Required for the "background" and "mean" baselines. */
    background?: Matrix;
    /** Omitted = one group per feature. */
    grouping?: FeatureGrouping;
    /** Omitted = inferred by predicting X row 0 once. */
    nOutputs?: number;
    predictBatch: PredictBatch;
    options?: ExplainOptions;
    onProgress?: (doneEvaluations: number, totalEvaluations: number) => void;
    /** Checked between batches; aborts with DOMException "AbortError". */
    signal?: AbortSignal;
}

export interface AttributionResult {
    kind: AttributionKind;
    /** null for linear/tree. */
    baseline: ExplainBaseline | null;
    exactness: "exact" | "sampled" | "surrogate" | "perturbation";
    additive: boolean;
    groupAggregation: "group_game" | "sum_of_feature_values" | "none";
    nRows: number;
    nGroups: number;
    nOutputs: number;
    nFeatures: number;
    /** [row][group][output] */
    attributions: Float64Array;
    /** [row][output] */
    expected: Float64Array;
    /** [row][output] */
    predictions: Float64Array;
    /** [row][group][output] (permutation only; NaN when a single pair was drawn). */
    stdError: Float64Array | null;
    /** max |prediction - expected - sum(attributions)|; null when not additive. */
    additivityMaxAbsError: number | null;
    /** Model rows evaluated by the host (0 for linear/tree). */
    evaluations: number;
    settings: Record<string, number | string | boolean | null>;
    engine: { name: "nirs4all-methods"; version: string; abi: string };
}

export interface FlatTreeEnsemble {
    /** nTrees + 1 offsets; tree t owns nodes [treeOffsets[t], treeOffsets[t+1]). */
    treeOffsets: number[] | Int32Array;
    feature: Int32Array | number[];
    threshold: Float64Array;
    /** Child indices LOCAL to the tree, -1 for a leaf. */
    left: Int32Array | number[];
    right: Int32Array | number[];
    defaultLeft?: Uint8Array;
    cover: Float64Array;
    /** nNodes * nOutputs leaf values. */
    value: Float64Array;
    nOutputs: number;
    /** 0: x <= threshold goes left (sklearn, LightGBM); 1: x < threshold (XGBoost). */
    decisionType?: 0 | 1;
    baseOffset?: Float64Array;
    /** 1/nTrees for averaging forests, 1 for boosting (default 1). */
    treeScale?: number;
}

// ---- enum codes (explain.h) ------------------------------------------------

const METHOD_CODES: Record<ExplainMethod, number> = {
    exact_shap: 0, permutation_shap: 1, kernel_shap: 2, lime: 3, occlusion: 4,
};
const BASELINE_CODES: Record<ExplainBaseline, number> = {
    background: 0, mean: 1, interpolate: 2,
};
const OPTION_KEYS = new Set<string>(["method", "baseline", "nPermutations", "nCoalitions",
    "limeKernelWidth", "limeRidge", "maxBatchRows", "seed"]);

// ---- n4m_explain_options_v1_t layout (identical on wasm32 and LP64) ---------
//   uint64 struct_size @0, int32 method @8, int32 baseline @12,
//   int64 n_outputs @16, n_permutations @24, n_coalitions @32,
//   double lime_kernel_width @40, lime_ridge @48, int64 max_batch_rows @56,
//   uint64 seed @64; total 72 bytes.
const OPTIONS_SIZE = 72;
const OPT_METHOD = 8, OPT_BASELINE = 12, OPT_N_OUTPUTS = 16, OPT_N_PERMUTATIONS = 24,
    OPT_N_COALITIONS = 32, OPT_LIME_WIDTH = 40, OPT_LIME_RIDGE = 48,
    OPT_MAX_BATCH_ROWS = 56, OPT_SEED = 64;
const DEFAULT_MAX_BATCH_ROWS = 4096;
type MatrixHandle = ReturnType<typeof makeMatrixView>;
const MAX_HEAP_BYTES = 0x7fffffff;

// ---- marshalling helpers ----------------------------------------------------

/** Tracks WASM heap allocations of one call; free() releases all of them. */
class Arena {
    private readonly ptrs: number[] = [];
    alloc(bytes: number): number {
        if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > MAX_HEAP_BYTES) {
            throw new RangeError("explanation buffer exceeds the WASM allocation limit");
        }
        const ptr = getModule()._malloc(Math.max(8, bytes));
        if (ptr === 0) throw new N4mError(Status.ERR_OUT_OF_MEMORY, "explanation allocation failed");
        this.ptrs.push(ptr);
        return ptr;
    }
    /** Copy an integer array into the heap as int64 (BigInt64 under WASM_BIGINT). */
    int64(values: ArrayLike<number>): number {
        const ptr = this.alloc(values.length * 8);
        const view = new BigInt64Array(getModule().HEAPU8.buffer, ptr, values.length);
        for (let i = 0; i < values.length; i += 1) view[i] = BigInt(values[i]!);
        return ptr;
    }
    f64(values: ArrayLike<number>): number {
        const ptr = this.alloc(values.length * 8);
        getModule().HEAPF64.set(values, ptr / 8);
        return ptr;
    }
    free(): void {
        const m = getModule();
        for (const ptr of this.ptrs) m._free(ptr);
        this.ptrs.length = 0;
    }
}

function readF64(ptr: number, n: number): Float64Array {
    // slice() copies, so the result never aliases the (growable) heap.
    return getModule().HEAPF64.slice(ptr / 8, ptr / 8 + n);
}

function readI64(ptr: number): number {
    return Number(new DataView(getModule().HEAPU8.buffer).getBigInt64(ptr, true));
}

function checkMatrix(name: string, x: Matrix, allowEmptyRows = false): void {
    if (!(x?.data instanceof Float64Array) || !Number.isSafeInteger(x.rows) ||
        !Number.isSafeInteger(x.cols) || x.rows < (allowEmptyRows ? 0 : 1) || x.cols < 1 ||
        x.data.length !== x.rows * x.cols) {
        throw new Error(`${name} must be a Float64Array row-major matrix with rows*cols entries`);
    }
}

function checkIntegers(name: string, values: ArrayLike<number>, length: number): void {
    if (values == null || values.length !== length) {
        throw new Error(`${name} must have ${length} entries`);
    }
    for (let i = 0; i < values.length; i += 1) {
        if (!Number.isSafeInteger(values[i])) throw new Error(`${name} must contain integers`);
    }
}

function checkGrouping(grouping: FeatureGrouping | undefined, nFeatures: number): number {
    if (grouping === undefined) return nFeatures;
    if (!(grouping.featureGroup instanceof Int32Array) || !Number.isSafeInteger(grouping.nGroups) ||
        grouping.nGroups < 1 || grouping.featureGroup.length !== nFeatures) {
        throw new Error("grouping.featureGroup must be an Int32Array of nFeatures group indices " +
            "and grouping.nGroups a positive integer");
    }
    return grouping.nGroups;
}

function engineInfo(): AttributionResult["engine"] {
    const m = getModule();
    const ptr = m.ccall("n4m_get_version_string", "number", [], []) as number;
    const abi = ["major", "minor", "patch"].map(part =>
        m.ccall(`n4m_get_abi_version_${part}`, "number", [], []) as number).join(".");
    return { name: "nirs4all-methods", version: ptr === 0 ? "" : m.UTF8ToString(ptr), abi };
}

function additivityError(attributions: Float64Array, expected: Float64Array,
                         predictions: Float64Array, nRows: number, nGroups: number,
                         nOutputs: number): number {
    // Reported diagnostic only: max |prediction - expected - sum_g attribution|.
    let worst = 0;
    for (let r = 0; r < nRows; r += 1) {
        for (let o = 0; o < nOutputs; o += 1) {
            let total = expected[r * nOutputs + o]!;
            for (let g = 0; g < nGroups; g += 1) total += attributions[(r * nGroups + g) * nOutputs + o]!;
            worst = Math.max(worst, Math.abs(predictions[r * nOutputs + o]! - total));
        }
    }
    return worst;
}

function abortError(): DOMException {
    return new DOMException("The explanation was aborted.", "AbortError");
}

function toPredictions(raw: unknown, rows: number, nOutputs: number): Float64Array {
    if (!(raw instanceof Float64Array) && !Array.isArray(raw)) {
        throw new TypeError("predictBatch must return a Float64Array or number[]");
    }
    if (raw.length !== rows * nOutputs) {
        throw new Error(`predictBatch returned ${raw.length} values, expected ${rows} x ${nOutputs}`);
    }
    const out = raw instanceof Float64Array ? raw : Float64Array.from(raw as number[]);
    for (let i = 0; i < out.length; i += 1) {
        if (!Number.isFinite(out[i])) throw new Error("predictBatch returned a nonfinite prediction");
    }
    return out;
}

/** True when kernel SHAP evaluates every proper nonempty coalition, i.e. the
 *  native coalition budget (explain.h: 0 = min(2^G - 2, 2G + 2048)) covers
 *  all 2^G - 2 of them; the weighted least squares is then the exact solve. */
function kernelEnumerates(nCoalitions: number, nGroups: number): boolean {
    if (nGroups > 52) return false;
    const all = 2 ** nGroups - 2;
    return (nCoalitions > 0 ? nCoalitions : 2 * nGroups + 2048) >= all;
}

function sessionExactness(method: ExplainMethod, nCoalitions: number, nGroups: number):
    AttributionResult["exactness"] {
    switch (method) {
        case "exact_shap": return "exact";
        case "kernel_shap": return kernelEnumerates(nCoalitions, nGroups) ? "exact" : "sampled";
        case "permutation_shap": return "sampled";
        case "lime": return "surrogate";
        case "occlusion": return "perturbation";
    }
}

// ---- ask/tell session --------------------------------------------------------

/** Explain any model through a native ask/tell session. The host predicts
 *  the requested perturbed rows with predictBatch (sync or async). */
export async function explainModel(input: ExplainModelInput): Promise<AttributionResult> {
    const { X, background, grouping, predictBatch, onProgress, signal } = input;
    const options = input.options ?? {};
    for (const key of Object.keys(options)) {
        if (!OPTION_KEYS.has(key)) throw new Error(`unknown explain option ${key}`);
    }
    const method = options.method ?? "permutation_shap";
    const baseline = options.baseline ?? "background";
    if (!Object.prototype.hasOwnProperty.call(METHOD_CODES, method)) {
        throw new Error(`unknown explain method ${method}`);
    }
    if (!Object.prototype.hasOwnProperty.call(BASELINE_CODES, baseline)) {
        throw new Error(`unknown explain baseline ${baseline}`);
    }
    const nPermutations = options.nPermutations ?? 10;
    const nCoalitions = options.nCoalitions ?? 0;
    const maxBatchRows = options.maxBatchRows ?? DEFAULT_MAX_BATCH_ROWS;
    const seed = options.seed ?? 0;
    if (!Number.isSafeInteger(nPermutations) || nPermutations < 1) {
        throw new Error("nPermutations must be a positive integer");
    }
    if (!Number.isSafeInteger(nCoalitions) || nCoalitions < 0) {
        throw new Error("nCoalitions must be a nonnegative integer");
    }
    if (!Number.isSafeInteger(maxBatchRows) || maxBatchRows < 1) {
        throw new Error("maxBatchRows must be a positive integer");
    }
    if (!Number.isSafeInteger(seed) || seed < 0) {
        throw new Error("seed must be a nonnegative integer below 2^53");
    }
    for (const key of ["limeKernelWidth", "limeRidge"] as const) {
        if (options[key] !== undefined && !Number.isFinite(options[key])) {
            throw new Error(`${key} must be finite`);
        }
    }
    if (typeof predictBatch !== "function") throw new TypeError("predictBatch must be a function");
    checkMatrix("X", X);
    const nFeatures = X.cols;
    if (background !== undefined) {
        checkMatrix("background", background, true);
        if (background.cols !== nFeatures) throw new Error("background must have X.cols columns");
    } else if (baseline !== "interpolate") {
        throw new Error(`the ${baseline} baseline requires a background matrix`);
    }
    const nGroupsIn = checkGrouping(grouping, nFeatures);
    if (signal?.aborted) throw abortError();

    let evaluations = 0;
    let nOutputs = input.nOutputs;
    if (nOutputs === undefined) {
        const probe = await predictBatch(X.data.slice(0, nFeatures), 1, nFeatures);
        if (signal?.aborted) throw abortError();
        if (!(probe instanceof Float64Array) && !Array.isArray(probe)) {
            throw new TypeError("predictBatch must return a Float64Array or number[]");
        }
        nOutputs = probe.length;
        evaluations += 1;
    }
    if (!Number.isSafeInteger(nOutputs) || nOutputs < 1) {
        throw new Error("nOutputs must be a positive integer");
    }

    const m = getModule();
    const arena = new Arena();
    let ctx: Context | null = null;
    let xView: MatrixHandle | null = null;
    let bgView: MatrixHandle | null = null;
    let session = 0;
    try {
        ctx = Context.create();
        xView = makeMatrixView(X.data, X.rows, X.cols);
        bgView = background !== undefined
            ? makeMatrixView(background.data, background.rows, background.cols)
            : makeMatrixView(new Float64Array(0), 0, nFeatures);
        const opt = arena.alloc(OPTIONS_SIZE);
        checkStatus(m.ccall("n4m_explain_options_init_v1", "number", ["number"], [opt]) as number);
        {
            const dv = new DataView(m.HEAPU8.buffer);
            dv.setInt32(opt + OPT_METHOD, METHOD_CODES[method], true);
            dv.setInt32(opt + OPT_BASELINE, BASELINE_CODES[baseline], true);
            dv.setBigInt64(opt + OPT_N_OUTPUTS, BigInt(nOutputs), true);
            dv.setBigInt64(opt + OPT_N_PERMUTATIONS, BigInt(nPermutations), true);
            dv.setBigInt64(opt + OPT_N_COALITIONS, BigInt(nCoalitions), true);
            if (options.limeKernelWidth !== undefined) {
                dv.setFloat64(opt + OPT_LIME_WIDTH, options.limeKernelWidth, true);
            }
            if (options.limeRidge !== undefined) dv.setFloat64(opt + OPT_LIME_RIDGE, options.limeRidge, true);
            dv.setBigInt64(opt + OPT_MAX_BATCH_ROWS, BigInt(maxBatchRows), true);
            dv.setBigUint64(opt + OPT_SEED, BigInt(seed), true);
        }
        const groupPtr = grouping === undefined ? 0 : arena.int64(grouping.featureGroup);
        const outSession = arena.alloc(4);
        m.setValue(outSession, 0, "i32");
        checkStatus(m.ccall("n4m_explain_session_create_v1", "number",
            ["number", "number", "number", "number", "i64", "number", "number"],
            [ctx.handle, xView.viewPtr, bgView.viewPtr, groupPtr,
             BigInt(nGroupsIn), opt, outSession]) as number, ctx.handle);
        session = m.getValue(outSession, "i32");

        const info = arena.alloc(5 * 8);
        checkStatus(m.ccall("n4m_explain_session_info_v1", "number",
            Array(6).fill("number"),
            [session, info, info + 8, info + 16, info + 24, info + 32]) as number);
        const nRows = readI64(info), nGroups = readI64(info + 16);
        const total = readI64(info + 32);
        if (readI64(info + 8) !== nFeatures || readI64(info + 24) !== nOutputs) {
            throw new Error("native explanation session reports unexpected dimensions");
        }

        const capacity = Math.max(1, Math.min(maxBatchRows, total));
        const rowsPtr = arena.alloc(capacity * nFeatures * 8);
        const predPtr = arena.alloc(capacity * nOutputs * 8);
        const writtenPtr = arena.alloc(8);
        let done = 0;
        for (;;) {
            if (signal?.aborted) throw abortError();
            checkStatus(m.ccall("n4m_explain_session_next_batch_v1", "number",
                ["number", "number", "number", "i64", "number"],
                [ctx.handle, session, rowsPtr, BigInt(capacity), writtenPtr]) as number, ctx.handle);
            const written = readI64(writtenPtr);
            if (written === 0) break;
            // Fresh copy: the heap may grow (detaching views) while predictBatch awaits.
            const rows = readF64(rowsPtr, written * nFeatures);
            const predicted = toPredictions(await predictBatch(rows, written, nFeatures), written, nOutputs);
            if (signal?.aborted) throw abortError();
            m.HEAPF64.set(predicted, predPtr / 8);
            checkStatus(m.ccall("n4m_explain_session_submit_v1", "number",
                ["number", "number", "number", "i64"],
                [ctx.handle, session, predPtr, BigInt(written)]) as number, ctx.handle);
            done += written;
            evaluations += written;
            onProgress?.(done, total);
        }

        const attrSize = nRows * nGroups * nOutputs, outSize = nRows * nOutputs;
        const attrPtr = arena.alloc(attrSize * 8);
        const expPtr = arena.alloc(outSize * 8);
        const predOutPtr = arena.alloc(outSize * 8);
        const hasStdError = method === "permutation_shap";
        const stdPtr = hasStdError ? arena.alloc(attrSize * 8) : 0;
        checkStatus(m.ccall("n4m_explain_session_result_v1", "number",
            ["number", "number", "number", "i64", "number", "i64", "number", "i64", "number", "i64"],
            [ctx.handle, session, attrPtr, BigInt(attrSize), expPtr, BigInt(outSize),
             predOutPtr, BigInt(outSize), stdPtr, BigInt(hasStdError ? attrSize : 0)]) as number,
            ctx.handle);
        const attributions = readF64(attrPtr, attrSize);
        const expected = readF64(expPtr, outSize);
        const predictions = readF64(predOutPtr, outSize);
        const additive = method !== "lime" && method !== "occlusion";
        return {
            kind: method,
            baseline,
            exactness: sessionExactness(method, nCoalitions, nGroups),
            additive,
            groupAggregation: "group_game",
            nRows, nGroups, nOutputs, nFeatures,
            attributions, expected, predictions,
            stdError: hasStdError ? readF64(stdPtr, attrSize) : null,
            additivityMaxAbsError: additive
                ? additivityError(attributions, expected, predictions, nRows, nGroups, nOutputs)
                : null,
            evaluations,
            settings: {
                method, baseline,
                // The interpolate baseline ignores background rows.
                n_background: baseline === "interpolate" ? 0 : background?.rows ?? 0,
                n_permutations: method === "permutation_shap" ? nPermutations : null,
                n_coalitions: method === "kernel_shap" || method === "lime" ? nCoalitions : null,
                lime_kernel_width: method === "lime" ? (options.limeKernelWidth ?? 0) : null,
                lime_ridge: method === "lime" ? (options.limeRidge ?? 1) : null,
                max_batch_rows: maxBatchRows,
                seed,
                n_outputs_inferred: input.nOutputs === undefined,
                total_evaluations: total,
            },
            engine: engineInfo(),
        };
    } finally {
        if (session !== 0) m.ccall("n4m_explain_session_destroy", null, ["number"], [session]);
        bgView?.free();
        xView?.free();
        arena.free();
        ctx?.destroy();
    }
}

// ---- model-specific exact explainers ----------------------------------------

/** Exact interventional SHAP of an affine model X . coefficients + intercept. */
export function linearShap(input: {
    X: Matrix; coefficients: Matrix; intercept?: Float64Array; reference: Float64Array;
    grouping?: FeatureGrouping;
}): AttributionResult {
    const { X, coefficients, intercept, reference, grouping } = input;
    checkMatrix("X", X);
    checkMatrix("coefficients", coefficients);
    const nFeatures = X.cols, nOutputs = coefficients.cols;
    if (coefficients.rows !== nFeatures) throw new Error("coefficients must be nFeatures x nOutputs");
    if (!(reference instanceof Float64Array) || reference.length !== nFeatures) {
        throw new Error("reference must be a Float64Array of nFeatures values");
    }
    if (intercept !== undefined && (!(intercept instanceof Float64Array) || intercept.length !== nOutputs)) {
        throw new Error("intercept must be a Float64Array of nOutputs values");
    }
    const nGroups = checkGrouping(grouping, nFeatures);
    const m = getModule();
    const arena = new Arena();
    let ctx: Context | null = null;
    let xView: MatrixHandle | null = null;
    let bView: MatrixHandle | null = null;
    try {
        ctx = Context.create();
        xView = makeMatrixView(X.data, X.rows, X.cols);
        bView = makeMatrixView(coefficients.data, coefficients.rows, coefficients.cols);
        const nRows = X.rows, attrSize = nRows * nGroups * nOutputs, outSize = nRows * nOutputs;
        const interceptPtr = intercept === undefined ? 0 : arena.f64(intercept);
        const referencePtr = arena.f64(reference);
        const groupPtr = grouping === undefined ? 0 : arena.int64(grouping.featureGroup);
        const attrPtr = arena.alloc(attrSize * 8), expPtr = arena.alloc(outSize * 8);
        const predPtr = arena.alloc(outSize * 8);
        checkStatus(m.ccall("n4m_explain_linear_v1", "number",
            ["number", "number", "number", "number", "number", "number", "i64",
             "number", "i64", "number", "i64", "number", "i64"],
            [ctx.handle, xView.viewPtr, bView.viewPtr, interceptPtr, referencePtr, groupPtr,
             BigInt(nGroups), attrPtr, BigInt(attrSize),
             expPtr, BigInt(outSize), predPtr, BigInt(outSize)]) as number, ctx.handle);
        const attributions = readF64(attrPtr, attrSize);
        const expected = readF64(expPtr, outSize);
        const predictions = readF64(predPtr, outSize);
        return {
            kind: "linear_shap", baseline: null, exactness: "exact", additive: true,
            // Linear group sums are exact Shapley values of the group game.
            groupAggregation: grouping === undefined ? "none" : "group_game",
            nRows, nGroups, nOutputs, nFeatures, attributions, expected, predictions,
            stdError: null,
            additivityMaxAbsError: additivityError(attributions, expected, predictions,
                nRows, nGroups, nOutputs),
            evaluations: 0,
            settings: { method: "linear_shap", has_intercept: intercept !== undefined },
            engine: engineInfo(),
        };
    } finally {
        bView?.free();
        xView?.free();
        arena.free();
        ctx?.destroy();
    }
}

/** Exact path-dependent TreeSHAP over a generic flat tree ensemble. Grouped
 *  values are sums of per-feature Shapley values, not group-game values. */
export function treeShap(input: { X: Matrix; trees: FlatTreeEnsemble; grouping?: FeatureGrouping }):
    AttributionResult {
    const { X, trees, grouping } = input;
    checkMatrix("X", X);
    const nFeatures = X.cols, nOutputs = trees.nOutputs;
    if (!Number.isSafeInteger(nOutputs) || nOutputs < 1) throw new Error("trees.nOutputs must be a positive integer");
    const nTrees = trees.treeOffsets.length - 1;
    const nNodes = trees.feature.length;
    if (nTrees < 1) throw new Error("trees.treeOffsets must hold nTrees + 1 offsets");
    checkIntegers("trees.treeOffsets", trees.treeOffsets, nTrees + 1);
    for (let t = 0; t < nTrees; t += 1) {
        if (trees.treeOffsets[t]! >= trees.treeOffsets[t + 1]!) {
            throw new Error("trees.treeOffsets must be strictly increasing");
        }
    }
    if (trees.treeOffsets[0] !== 0 || trees.treeOffsets[nTrees] !== nNodes) {
        throw new Error("trees.treeOffsets must start at 0 and end at the node count");
    }
    checkIntegers("trees.feature", trees.feature, nNodes);
    checkIntegers("trees.left", trees.left, nNodes);
    checkIntegers("trees.right", trees.right, nNodes);
    if (!(trees.threshold instanceof Float64Array) || trees.threshold.length !== nNodes ||
        !(trees.cover instanceof Float64Array) || trees.cover.length !== nNodes ||
        !(trees.value instanceof Float64Array) || trees.value.length !== nNodes * nOutputs) {
        throw new Error("trees.threshold/cover must hold nNodes values and trees.value nNodes * nOutputs");
    }
    if (trees.defaultLeft !== undefined &&
        (!(trees.defaultLeft instanceof Uint8Array) || trees.defaultLeft.length !== nNodes)) {
        throw new Error("trees.defaultLeft must be a Uint8Array of nNodes flags");
    }
    if (trees.baseOffset !== undefined &&
        (!(trees.baseOffset instanceof Float64Array) || trees.baseOffset.length !== nOutputs)) {
        throw new Error("trees.baseOffset must be a Float64Array of nOutputs values");
    }
    const decisionType = trees.decisionType ?? 0;
    if (decisionType !== 0 && decisionType !== 1) throw new Error("trees.decisionType must be 0 or 1");
    const treeScale = trees.treeScale ?? 1;
    if (!Number.isFinite(treeScale)) throw new Error("trees.treeScale must be finite");
    const nGroups = checkGrouping(grouping, nFeatures);
    const m = getModule();
    const arena = new Arena();
    let ctx: Context | null = null;
    let xView: MatrixHandle | null = null;
    try {
        ctx = Context.create();
        xView = makeMatrixView(X.data, X.rows, X.cols);
        const nRows = X.rows, attrSize = nRows * nGroups * nOutputs, outSize = nRows * nOutputs;
        const offsetsPtr = arena.int64(trees.treeOffsets);
        const featurePtr = arena.int64(trees.feature);
        const thresholdPtr = arena.f64(trees.threshold);
        const leftPtr = arena.int64(trees.left);
        const rightPtr = arena.int64(trees.right);
        let defaultLeftPtr = 0;
        if (trees.defaultLeft !== undefined) {
            defaultLeftPtr = arena.alloc(nNodes);
            m.HEAPU8.set(trees.defaultLeft, defaultLeftPtr);
        }
        const coverPtr = arena.f64(trees.cover);
        const valuePtr = arena.f64(trees.value);
        const basePtr = trees.baseOffset === undefined ? 0 : arena.f64(trees.baseOffset);
        const groupPtr = grouping === undefined ? 0 : arena.int64(grouping.featureGroup);
        const attrPtr = arena.alloc(attrSize * 8), expPtr = arena.alloc(outSize * 8);
        const predPtr = arena.alloc(outSize * 8);
        checkStatus(m.ccall("n4m_explain_tree_path_dependent_v1", "number",
            ["number", "number", "i64", "number", "number", "number", "number", "number", "number",
             "number", "number", "i64", "number", "number", "number", "number", "i64",
             "number", "i64", "number", "i64", "number", "i64"],
            [ctx.handle, xView.viewPtr, BigInt(nTrees), offsetsPtr, featurePtr, thresholdPtr,
             leftPtr, rightPtr, defaultLeftPtr, coverPtr, valuePtr, BigInt(nOutputs), decisionType,
             basePtr, treeScale, groupPtr, BigInt(nGroups),
             attrPtr, BigInt(attrSize), expPtr, BigInt(outSize), predPtr, BigInt(outSize)]) as number,
            ctx.handle);
        const attributions = readF64(attrPtr, attrSize);
        const expected = readF64(expPtr, outSize);
        const predictions = readF64(predPtr, outSize);
        return {
            kind: "tree_shap", baseline: null, exactness: "exact", additive: true,
            groupAggregation: grouping === undefined ? "none" : "sum_of_feature_values",
            nRows, nGroups, nOutputs, nFeatures, attributions, expected, predictions,
            stdError: null,
            additivityMaxAbsError: additivityError(attributions, expected, predictions,
                nRows, nGroups, nOutputs),
            evaluations: 0,
            settings: { method: "tree_shap", feature_perturbation: "tree_path_dependent",
                n_trees: nTrees, decision_type: decisionType, tree_scale: treeScale },
            engine: engineInfo(),
        };
    } finally {
        xView?.free();
        arena.free();
        ctx?.destroy();
    }
}

// ---- grouping and summary helpers -------------------------------------------

function toInt32(ptr: number, n: number): Int32Array {
    const view = new BigInt64Array(getModule().HEAPU8.buffer, ptr, n);
    return Int32Array.from(view, v => Number(v));
}

/** Contiguous fixed-width partition: feature j -> group floor(j / width). */
export function groupsFixedWidth(nFeatures: number, width: number): FeatureGrouping {
    if (!Number.isSafeInteger(nFeatures) || nFeatures < 1 || !Number.isSafeInteger(width) || width < 1) {
        throw new Error("nFeatures and width must be positive integers");
    }
    const m = getModule();
    const arena = new Arena();
    try {
        const groups = arena.alloc(nFeatures * 8), count = arena.alloc(8);
        checkStatus(m.ccall("n4m_explain_groups_fixed_width_v1", "number",
            ["i64", "i64", "number", "i64", "number"],
            [BigInt(nFeatures), BigInt(width), groups, BigInt(nFeatures), count]) as number);
        return { featureGroup: toInt32(groups, nFeatures), nGroups: readI64(count) };
    } finally {
        arena.free();
    }
}

/** Partition from half-open [start, end) feature-index ranges. Range r becomes
 *  group r; every maximal uncovered run becomes one further group. */
export function groupsFromRanges(nFeatures: number, ranges: ReadonlyArray<readonly [number, number]>):
    FeatureGrouping {
    if (!Number.isSafeInteger(nFeatures) || nFeatures < 1) throw new Error("nFeatures must be a positive integer");
    if (!Array.isArray(ranges) || ranges.some(r => !Array.isArray(r) || r.length !== 2)) {
        throw new Error("ranges must be an array of [start, end) pairs");
    }
    const flat = ranges.flatMap(r => [r[0], r[1]]);
    checkIntegers("ranges", flat, flat.length);
    const m = getModule();
    const arena = new Arena();
    try {
        const rangesPtr = arena.int64(flat);
        const capacity = ranges.length + nFeatures;
        const groups = arena.alloc(nFeatures * 8), rangeOf = arena.alloc(capacity * 8);
        const count = arena.alloc(8);
        checkStatus(m.ccall("n4m_explain_groups_from_ranges_v1", "number",
            ["i64", "number", "i64", "number", "i64", "number", "i64", "number"],
            [BigInt(nFeatures), rangesPtr, BigInt(ranges.length), groups, BigInt(nFeatures),
             rangeOf, BigInt(capacity), count]) as number);
        const nGroups = readI64(count);
        return { featureGroup: toInt32(groups, nFeatures), nGroups, rangeOfGroup: toInt32(rangeOf, nGroups) };
    } finally {
        arena.free();
    }
}

/** Per-group mean |attribution| and mean signed attribution over rows. */
export function summarizeAttributions(result: AttributionResult, outputIndex = 0):
    { meanAbs: Float64Array; meanSigned: Float64Array } {
    const { nRows, nGroups, nOutputs, attributions } = result;
    if (!(attributions instanceof Float64Array) || attributions.length !== nRows * nGroups * nOutputs) {
        throw new Error("result.attributions must hold nRows * nGroups * nOutputs values");
    }
    if (!Number.isSafeInteger(outputIndex) || outputIndex < 0 || outputIndex >= nOutputs) {
        throw new Error("outputIndex out of range");
    }
    const m = getModule();
    const arena = new Arena();
    try {
        const attrPtr = arena.f64(attributions);
        const absPtr = arena.alloc(nGroups * 8), signedPtr = arena.alloc(nGroups * 8);
        checkStatus(m.ccall("n4m_explain_summarize_v1", "number",
            ["number", "i64", "i64", "i64", "i64", "number", "number", "i64"],
            [attrPtr, BigInt(nRows), BigInt(nGroups), BigInt(nOutputs), BigInt(outputIndex),
             absPtr, signedPtr, BigInt(nGroups)]) as number);
        return { meanAbs: readF64(absPtr, nGroups), meanSigned: readF64(signedPtr, nGroups) };
    } finally {
        arena.free();
    }
}
