// SPDX-License-Identifier: CECILL-2.1
// Thin, caller-buffer wrappers for the public Methods Preflight diagnostics.
import { checkStatus, getModule, makeMatrixView } from "./ffi.js";
import { Context } from "./context.js";
import { N4mError, Status, type Matrix } from "./types.js";

export interface PreflightPcaResult {
    scores: Matrix;
    loadings: Matrix;
    eigenvalues: Float64Array;
    /** [trace(C), trace(C*C)] over the full centered covariance spectrum. */
    fullMoments: readonly [number, number];
}

export interface PreflightRepeatabilityResult {
    pooledSd: Float64Array;
    icc: Float64Array;
    nCompleteGroups: number[];
    /** 0=computed, 1=insufficient, 2=unbalanced, 3=zero total variance. */
    bandStatus: Int32Array;
}

export interface PreflightNeighborResult {
    indices: number[];
    distances: Float64Array;
    rows: number;
    k: number;
}

export interface PreflightLocalContrastResult {
    contrast: number;
    fractionRowsWithNeighbors: number;
    observedSameClassFraction: number;
    expectedSameClassFraction: number;
}

export interface HdbscanOptions {
    minClusterSize: number;
    /** Neighbors defining the core distance, the row itself included (scikit-learn). */
    minSamples: number;
    selection?: "eom" | "leaf";
    allowSingleCluster?: boolean;
}

export interface HdbscanTreeNode {
    /** Parent node index, -1 for the root (node 0). */
    parent: number;
    size: number;
    /** Selected cluster label or -1. */
    label: number;
    lambdaBirth: number;
    lambdaDeath: number;
    stability: number;
}

export interface HdbscanResult {
    /** -1 = noise; clusters 0..nClusters-1. */
    labels: number[];
    /** scikit-learn probabilities_: a membership strength, not a probability. */
    membership: Float64Array;
    /** GLOSH outlier scores in [0, 1]. */
    outlierScores: Float64Array;
    nClusters: number;
    /** scikit-learn-contrib/hdbscan cluster_persistence_, one value per label. */
    persistence: Float64Array;
    tree: HdbscanTreeNode[];
}

export interface PreflightFactorAssociationResult {
    /** eta^2 (continuous) or Cramer's V (categorical); NaN when undefined. */
    effectSize: number;
    /** One-way ANOVA F or Pearson chi-square; NaN when undefined. */
    statistic: number;
    df1: number;
    df2: number;
    nLevels: number;
    nTargetValues: number;
    /** log10 upper bound on distinct permutation outcomes. */
    log10DistinctPermutations: number;
    /** Monte Carlo (1+b)/(1+B); NaN without permutations. */
    pValue: number;
    nAtLeastObserved: number;
    nPermutationsExecuted: number;
}

function validateMatrix(X: Matrix): void {
    if (!Number.isSafeInteger(X.rows) || !Number.isSafeInteger(X.cols) ||
        X.rows < 1 || X.cols < 1 || X.data.length !== X.rows * X.cols) {
        throw new RangeError("invalid Preflight matrix dimensions");
    }
}

function validateGroupIds(values: readonly (number | bigint)[]): void {
    for (const value of values) {
        if (typeof value === "number" && !Number.isSafeInteger(value)) {
            throw new RangeError("group IDs supplied as numbers must be safe integers");
        }
        const id = BigInt(value);
        if (id < -(1n << 63n) || id >= (1n << 63n)) {
            throw new RangeError("group ID is outside signed int64 range");
        }
    }
}

class NativeScope {
    readonly ctx = Context.create();
    private readonly releases: Array<() => void> = [];

    alloc(bytes: number): number {
        if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > 0x7fffffff) {
            throw new RangeError("Preflight buffer exceeds WASM allocation limit");
        }
        const m = getModule();
        const ptr = m._malloc(Math.max(1, bytes));
        if (ptr === 0) throw new N4mError(Status.ERR_OUT_OF_MEMORY, "WASM allocation failed");
        this.releases.push(() => m._free(ptr));
        return ptr;
    }

    matrix(data: Float64Array, rows: number, cols: number): ReturnType<typeof makeMatrixView> {
        const v = makeMatrixView(data, rows, cols);
        this.releases.push(v.free);
        return v;
    }

    i64(values: readonly (number | bigint)[]): number {
        const ptr = this.alloc(values.length * 8);
        const dv = new DataView(getModule().HEAPU8.buffer);
        for (let i = 0; i < values.length; i++) {
            dv.setBigInt64(ptr + i * 8, BigInt(values[i]!), true);
        }
        return ptr;
    }

    close(): void {
        for (let i = this.releases.length - 1; i >= 0; --i) this.releases[i]!();
        this.ctx.destroy();
    }
}

function copyF64(ptr: number, n: number): Float64Array {
    return new Float64Array(getModule().HEAPF64.slice(ptr / 8, ptr / 8 + n));
}

/** Centered PCA using the native Methods SVD. Output matrices are row-major. */
export function preflightPca(X: Matrix, k: number): PreflightPcaResult {
    validateMatrix(X);
    if (!Number.isInteger(k) || k < 1 || k > Math.min(X.rows, X.cols)) {
        throw new RangeError("invalid PCA component count");
    }
    const m = getModule();
    const scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const x = scope.matrix(X.data, X.rows, X.cols);
        const scores = scope.matrix(new Float64Array(X.rows * k), X.rows, k);
        const loadings = scope.matrix(new Float64Array(k * X.cols), k, X.cols);
        const eigen = scope.alloc(k * 8), moments = scope.alloc(16);
        const st = m.ccall("n4m_preflight_pca_compute", "number",
            ["number", "number", "number", "number", "number", "number", "number"],
            [ctx.handle, x.viewPtr, k, scores.viewPtr, loadings.viewPtr, eigen, moments]) as number;
        checkStatus(st, ctx.handle);
        return {
            scores: {data: copyF64(scores.dataPtr, X.rows * k), rows: X.rows, cols: k},
            loadings: {data: copyF64(loadings.dataPtr, k * X.cols), rows: k, cols: X.cols},
            eigenvalues: copyF64(eigen, k),
            fullMoments: [m.HEAPF64[moments / 8]!, m.HEAPF64[moments / 8 + 1]!],
        };
    } finally {
        scope.close();
    }
}

/** Per-feature pooled repeatability and balanced ICC(1,1). */
export function preflightRepeatability(X: Matrix,
    groupIds: readonly (number | bigint)[], validMask?: Uint8Array): PreflightRepeatabilityResult {
    validateMatrix(X);
    if (groupIds.length !== X.rows || (validMask && validMask.length !== X.data.length)) {
        throw new RangeError("repeatability group or mask length mismatch");
    }
    validateGroupIds(groupIds);
    const m = getModule(), scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const x = scope.matrix(X.data, X.rows, X.cols);
        const groups = scope.i64(groupIds);
        const mask = validMask ? scope.alloc(validMask.length) : 0;
        const sd = scope.alloc(X.cols * 8), icc = scope.alloc(X.cols * 8);
        const counts = scope.alloc(X.cols * 8), status = scope.alloc(X.cols * 4);
        if (validMask) m.HEAPU8.set(validMask, mask);
        const st = m.ccall("n4m_preflight_repeatability_compute", "number",
            Array(8).fill("number"),
            [ctx.handle, x.viewPtr, groups, mask, sd, icc, counts, status]) as number;
        checkStatus(st, ctx.handle);
        const dv = new DataView(m.HEAPU8.buffer);
        return {
            pooledSd: copyF64(sd, X.cols), icc: copyF64(icc, X.cols),
            nCompleteGroups: Array.from({length: X.cols}, (_, j) =>
                Number(dv.getBigInt64(counts + j * 8, true))),
            bandStatus: new Int32Array(m.HEAP32.slice(status / 4, status / 4 + X.cols)),
        };
    } finally {
        scope.close();
    }
}

/** Finite-value summary: count, min, max, mean, sample SD, Q1, median, Q3. */
export function preflightSummary(values: Float64Array, validMask?: Uint8Array): Float64Array {
    if (validMask && validMask.length !== values.length) throw new RangeError("summary mask length mismatch");
    const m = getModule(), scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const input = scope.alloc(values.length * 8);
        const mask = validMask ? scope.alloc(validMask.length) : 0;
        const out = scope.alloc(64);
        m.HEAPF64.set(values, input / 8);
        if (validMask) m.HEAPU8.set(validMask, mask);
        const st = m.ccall("n4m_preflight_summary_compute", "number",
            ["number", "number", "number", "i64", "number"],
            [ctx.handle, input, mask, BigInt(values.length), out]) as number;
        checkStatus(st, ctx.handle);
        return copyF64(out, 8);
    } finally {
        scope.close();
    }
}

/** Euclidean reference neighbors, optionally excluding equal group IDs. */
export function preflightNearestNeighbors(query: Matrix, reference: Matrix, k: number,
    queryGroups?: readonly (number | bigint)[],
    referenceGroups?: readonly (number | bigint)[]): PreflightNeighborResult {
    validateMatrix(query); validateMatrix(reference);
    if (query.cols !== reference.cols || !Number.isInteger(k) || k < 1 ||
        (queryGroups === undefined) !== (referenceGroups === undefined) ||
        (queryGroups && queryGroups.length !== query.rows) ||
        (referenceGroups && referenceGroups.length !== reference.rows)) {
        throw new RangeError("invalid neighbor shapes, groups or k");
    }
    if (queryGroups) validateGroupIds(queryGroups);
    if (referenceGroups) validateGroupIds(referenceGroups);
    const m = getModule(), scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const q = scope.matrix(query.data, query.rows, query.cols);
        const r = scope.matrix(reference.data, reference.rows, reference.cols);
        const qg = queryGroups ? scope.i64(queryGroups) : 0;
        const rg = referenceGroups ? scope.i64(referenceGroups) : 0;
        const len = query.rows * k;
        const idx = scope.alloc(len * 8), dist = scope.alloc(len * 8);
        const st = m.ccall("n4m_preflight_nearest_neighbors", "number",
            Array(8).fill("number"),
            [ctx.handle, q.viewPtr, r.viewPtr, qg, rg, k, idx, dist]) as number;
        checkStatus(st, ctx.handle);
        const dv = new DataView(m.HEAPU8.buffer);
        return {
            indices: Array.from({length: len}, (_, j) => Number(dv.getBigInt64(idx + j * 8, true))),
            distances: copyF64(dist, len), rows: query.rows, k,
        };
    } finally {
        scope.close();
    }
}

/** Local Y coherence using an existing row-major n x k neighbor-index table. */
export function preflightLocalContrast(targets: Float64Array,
    neighborIndices: readonly (number | bigint)[], k: number,
    categorical: boolean, groupIds?: readonly (number | bigint)[]): PreflightLocalContrastResult {
    if (!Number.isSafeInteger(k) || k < 1 || targets.length < 1 ||
        neighborIndices.length !== targets.length * k ||
        (groupIds && groupIds.length !== targets.length)) {
        throw new RangeError("invalid local contrast dimensions or groups");
    }
    validateGroupIds(neighborIndices);
    if (groupIds) validateGroupIds(groupIds);
    const m = getModule(), scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const y = scope.alloc(targets.length * 8);
        const neighbors = scope.i64(neighborIndices);
        const groups = groupIds ? scope.i64(groupIds) : 0;
        const out = scope.alloc(32);
        m.HEAPF64.set(targets, y / 8);
        const st = m.ccall("n4m_preflight_local_contrast", "number",
            ["number", "number", "number", "number", "i64", "number", "number", "number"],
            [ctx.handle, y, neighbors, groups, BigInt(targets.length), k,
                categorical ? 1 : 0, out]) as number;
        checkStatus(st, ctx.handle);
        const result = copyF64(out, 4);
        return {
            contrast: result[0]!, fractionRowsWithNeighbors: result[1]!,
            observedSameClassFraction: result[2]!, expectedSameClassFraction: result[3]!,
        };
    } finally {
        scope.close();
    }
}

/** Euclidean HDBSCAN* (scikit-learn 1.9.1 semantics, stable MST tie order). */
export function clusterHdbscan(X: Matrix, options: HdbscanOptions): HdbscanResult {
    validateMatrix(X);
    const {minClusterSize, minSamples} = options;
    if (X.rows < 2 || !Number.isSafeInteger(minClusterSize) || minClusterSize < 2 ||
        !Number.isSafeInteger(minSamples) || minSamples < 1 || minSamples > X.rows) {
        throw new RangeError("invalid HDBSCAN dimensions or parameters");
    }
    const m = getModule(), scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const x = scope.matrix(X.data, X.rows, X.cols);
        const perRow = () => scope.alloc(X.rows * 8);
        const labels = perRow(), membership = perRow(), outlier = perRow();
        const persistence = perRow(), parent = perRow(), size = perRow(), label = perRow();
        const birth = perRow(), death = perRow(), stability = perRow();
        const nClusters = scope.alloc(8), nNodes = scope.alloc(8);
        const st = m.ccall("n4m_cluster_hdbscan", "number",
            ["number", "number", "i64", "i64", "number", "number", ...Array(12).fill("number")],
            [ctx.handle, x.viewPtr, BigInt(minClusterSize), BigInt(minSamples),
                options.selection === "leaf" ? 1 : 0, options.allowSingleCluster ? 1 : 0,
                labels, membership, outlier, nClusters, persistence, nNodes, parent, size, label,
                birth, death, stability]) as number;
        checkStatus(st, ctx.handle);
        const dv = new DataView(m.HEAPU8.buffer);
        const i64 = (ptr: number, n: number) =>
            Array.from({length: n}, (_, j) => Number(dv.getBigInt64(ptr + j * 8, true)));
        const k = Number(dv.getBigInt64(nNodes, true));
        const clusters = Number(dv.getBigInt64(nClusters, true));
        const parents = i64(parent, k), sizes = i64(size, k), treeLabels = i64(label, k);
        const births = copyF64(birth, k), deaths = copyF64(death, k);
        const stabilities = copyF64(stability, k);
        return {
            labels: i64(labels, X.rows),
            membership: copyF64(membership, X.rows),
            outlierScores: copyF64(outlier, X.rows),
            nClusters: clusters,
            persistence: copyF64(persistence, clusters),
            tree: Array.from({length: k}, (_, j) => ({
                parent: parents[j]!, size: sizes[j]!, label: treeLabels[j]!,
                lambdaBirth: births[j]!, lambdaDeath: deaths[j]!, stability: stabilities[j]!,
            })),
        };
    } finally {
        scope.close();
    }
}

/** Adjusted Rand index; NaN for n<2 or identical trivial partitions. Noise is an ordinary label. */
export function preflightAdjustedRandIndex(a: readonly (number | bigint)[],
    b: readonly (number | bigint)[]): number {
    if (a.length !== b.length) throw new RangeError("ARI label length mismatch");
    validateGroupIds(a); validateGroupIds(b);
    const m = getModule(), scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const pa = scope.i64(a), pb = scope.i64(b), out = scope.alloc(8);
        const st = m.ccall("n4m_preflight_adjusted_rand_index", "number",
            ["number", "number", "number", "i64", "number"],
            [ctx.handle, pa, pb, BigInt(a.length), out]) as number;
        checkStatus(st, ctx.handle);
        return m.HEAPF64[out / 8]!;
    } finally {
        scope.close();
    }
}

export interface PreflightNoiseProxyResult {
    /** sqrt(mean residual^2)/gain per feature; NaN at edges or without rows. */
    bandNoise: Float64Array;
    bandCounts: number[];
    /** Root mean square over every evaluable residual cell, divided by gain. */
    pooled: number;
    /** Residual gain sqrt(sum((delta - weights)^2)) for unit white noise. */
    gain: number;
}

export interface PreflightPairsResult {
    /** Row-major listed pairs: [query, reference] for each listed pair. */
    pairs: number[];
    distances: Float64Array;
    queryFlags: Uint8Array;
    referenceFlags: Uint8Array;
    totalPairs: number;
    candidatePairs: number;
}

/** Savitzky-Golay smoothing-residual noise proxy; interior features only. */
export function preflightNoiseProxy(X: Matrix, windowLength: number, polyorder: number,
    validMask?: Uint8Array): PreflightNoiseProxyResult {
    validateMatrix(X);
    if (!Number.isInteger(windowLength) || !Number.isInteger(polyorder) ||
        (validMask && validMask.length !== X.data.length)) {
        throw new RangeError("invalid noise proxy window, polyorder or mask");
    }
    const m = getModule(), scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const x = scope.matrix(X.data, X.rows, X.cols);
        const mask = validMask ? scope.alloc(validMask.length) : 0;
        const band = scope.alloc(X.cols * 8), counts = scope.alloc(X.cols * 8);
        const out = scope.alloc(16);
        if (validMask) m.HEAPU8.set(validMask, mask);
        const st = m.ccall("n4m_preflight_noise_proxy_compute", "number",
            Array(8).fill("number"),
            [ctx.handle, x.viewPtr, mask, windowLength, polyorder, band, counts, out]) as number;
        checkStatus(st, ctx.handle);
        const dv = new DataView(m.HEAPU8.buffer);
        return {
            bandNoise: copyF64(band, X.cols),
            bandCounts: Array.from({length: X.cols}, (_, j) =>
                Number(dv.getBigInt64(counts + j * 8, true))),
            pooled: m.HEAPF64[out / 8]!, gain: m.HEAPF64[out / 8 + 1]!,
        };
    } finally {
        scope.close();
    }
}

/**
 * Row pairs with root-mean-square feature difference <= tolerance. Without a
 * reference, query rows are compared with each other (i < h).
 */
export function preflightPairsWithinTolerance(query: Matrix, tolerance: number,
    maxPairs: number, reference?: Matrix,
    queryGroups?: readonly (number | bigint)[],
    referenceGroups?: readonly (number | bigint)[]): PreflightPairsResult {
    validateMatrix(query);
    if (reference) validateMatrix(reference);
    if (!Number.isSafeInteger(maxPairs) || maxPairs < 0 ||
        (reference && reference.cols !== query.cols) ||
        (queryGroups && queryGroups.length !== query.rows) ||
        (reference && (queryGroups === undefined) !== (referenceGroups === undefined)) ||
        (referenceGroups && (!reference || referenceGroups.length !== reference.rows))) {
        throw new RangeError("invalid pair shapes, groups or listing budget");
    }
    if (queryGroups) validateGroupIds(queryGroups);
    if (referenceGroups) validateGroupIds(referenceGroups);
    const m = getModule(), scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const q = scope.matrix(query.data, query.rows, query.cols);
        const r = reference ? scope.matrix(reference.data, reference.rows, reference.cols) : undefined;
        const qg = queryGroups ? scope.i64(queryGroups) : 0;
        const rg = referenceGroups ? scope.i64(referenceGroups) : 0;
        const idx = maxPairs ? scope.alloc(maxPairs * 16) : 0;
        const dist = maxPairs ? scope.alloc(maxPairs * 8) : 0;
        const qf = scope.alloc(query.rows);
        const rf = reference ? scope.alloc(reference.rows) : 0;
        const counts = scope.alloc(16);
        const st = m.ccall("n4m_preflight_pairs_within_tolerance", "number",
            ["number", "number", "number", "number", "number", "number", "i64",
                "number", "number", "number", "number", "number"],
            [ctx.handle, q.viewPtr, r ? r.viewPtr : 0, qg, rg, tolerance, BigInt(maxPairs),
                idx, dist, qf, rf, counts]) as number;
        checkStatus(st, ctx.handle);
        const dv = new DataView(m.HEAPU8.buffer);
        const total = Number(dv.getBigInt64(counts, true));
        const listed = Math.min(total, maxPairs);
        return {
            pairs: Array.from({length: listed * 2}, (_, j) => Number(dv.getBigInt64(idx + j * 8, true))),
            distances: copyF64(dist, listed),
            queryFlags: new Uint8Array(m.HEAPU8.slice(qf, qf + query.rows)),
            referenceFlags: reference ? new Uint8Array(m.HEAPU8.slice(rf, rf + reference.rows))
                : new Uint8Array(0),
            totalPairs: total,
            candidatePairs: Number(dv.getBigInt64(counts + 8, true)),
        };
    } finally {
        scope.close();
    }
}

/** Factor-target association over exchangeable units with seeded permutations.
 * The seed is an unsigned 64-bit integer; pass a bigint above 2^53. */
export function preflightFactorAssociation(factorCodes: readonly (number | bigint)[],
    targets: Float64Array, categorical: boolean, nPermutations: number,
    seed: number | bigint): PreflightFactorAssociationResult {
    if (factorCodes.length !== targets.length || targets.length < 2 ||
        !Number.isInteger(nPermutations) || nPermutations < 0 || nPermutations > 1_000_000) {
        throw new RangeError("invalid association dimensions or permutation count");
    }
    validateGroupIds(factorCodes);
    if (typeof seed === "number" && !Number.isSafeInteger(seed)) {
        throw new RangeError("seeds supplied as numbers must be safe integers");
    }
    const seed64 = BigInt(seed);
    if (seed64 < 0n || seed64 >= (1n << 64n)) throw new RangeError("seed is outside uint64 range");
    const m = getModule(), scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const codes = scope.i64(factorCodes);
        const y = scope.alloc(targets.length * 8);
        const out = scope.alloc(64), counts = scope.alloc(16);
        m.HEAPF64.set(targets, y / 8);
        const st = m.ccall("n4m_preflight_factor_association", "number",
            ["number", "number", "number", "i64", "number", "number", "i64", "number", "number"],
            [ctx.handle, codes, y, BigInt(targets.length), categorical ? 1 : 0,
                nPermutations, seed64, out, counts]) as number;
        checkStatus(st, ctx.handle);
        const r = copyF64(out, 8);
        const dv = new DataView(m.HEAPU8.buffer);
        return {
            effectSize: r[0]!, statistic: r[1]!, df1: r[2]!, df2: r[3]!,
            nLevels: r[4]!, nTargetValues: r[5]!, log10DistinctPermutations: r[6]!,
            pValue: r[7]!,
            nAtLeastObserved: Number(dv.getBigInt64(counts, true)),
            nPermutationsExecuted: Number(dv.getBigInt64(counts + 8, true)),
        };
    } finally {
        scope.close();
    }
}

/** Holm step-down adjusted p-values for one hypothesis family. */
export function preflightHolmAdjust(pValues: Float64Array): Float64Array {
    if (pValues.length < 1) throw new RangeError("Holm family must not be empty");
    const m = getModule(), scope = new NativeScope();
    try {
        const ctx = scope.ctx;
        const input = scope.alloc(pValues.length * 8), out = scope.alloc(pValues.length * 8);
        m.HEAPF64.set(pValues, input / 8);
        const st = m.ccall("n4m_preflight_holm_adjust", "number",
            ["number", "number", "i64", "number"],
            [ctx.handle, input, BigInt(pValues.length), out]) as number;
        checkStatus(st, ctx.handle);
        return copyF64(out, pValues.length);
    } finally {
        scope.close();
    }
}
