// SPDX-License-Identifier: CECILL-2.1
// Closed fit/apply roles over the seven native filter kernels (ABI 2.12).
import { checkStatus, getModule, makeMatrixView } from "./ffi.js";
import type { Matrix } from "./types.js";

export type SampleFilterKind = "YOutlier" | "XOutlier" | "HighLeverage" |
    "SpectralQuality" | "Composite";
export type FeatureFilterKind = "Variance" | "Correlation";

const SAMPLE_KINDS: Record<SampleFilterKind, readonly [number, number, number]> = {
    YOutlier: [0, 1, 3], XOutlier: [1, 5, 2],
    HighLeverage: [2, 4, 2], SpectralQuality: [3, 3, 5],
    Composite: [4, 1, 0],
};
const FEATURE_KINDS: Record<FeatureFilterKind, number> = {
    Variance: 0, Correlation: 1,
};

function validMatrix(X: Matrix, name: string): void {
    if (!Number.isSafeInteger(X.rows) || !Number.isSafeInteger(X.cols) ||
        X.rows < 1 || X.cols < 1 || X.rows * X.cols > 2147483647 ||
        X.data.length !== X.rows * X.cols) {
        throw new Error(`${name} must be a nonempty row-major matrix`);
    }
}

function aligned(X: Matrix, Y: Matrix | null, required: boolean): void {
    validMatrix(X, "X");
    if (required && Y === null) throw new Error("Y must be supplied");
    if (Y !== null) {
        validMatrix(Y, "Y");
        if (Y.cols !== 1 || Y.rows !== X.rows) {
            throw new Error("Y must have one column and align with X rows");
        }
    }
}

function parameters(integers: readonly number[], doubles: readonly number[],
                    ni: number, nd: number, seed: number | bigint): bigint {
    if (integers.length !== ni || doubles.length !== nd ||
        integers.some(x => !Number.isSafeInteger(x)) ||
        doubles.some(x => !Number.isFinite(x))) {
        throw new Error("invalid native filter parameter vector");
    }
    if (typeof seed === "number" && !Number.isSafeInteger(seed)) {
        throw new Error("seed must be an exact integer");
    }
    const s = BigInt(seed);
    if (s < 0n || s > (1n << 64n) - 1n) throw new Error("seed must fit uint64");
    return s;
}

function createSample(kind: SampleFilterKind, integers: readonly number[],
                      doubles: readonly number[], seed: number | bigint): number {
    if (!Object.prototype.hasOwnProperty.call(SAMPLE_KINDS, kind)) {
        throw new Error("unknown sample filter kind");
    }
    const [code, ni, nd] = SAMPLE_KINDS[kind];
    const s = parameters(integers, doubles, ni, nd, seed);
    const m = getModule();
    const ip = ni ? m._malloc(ni * 8) : 0;
    const dp = nd ? m._malloc(nd * 8) : 0;
    const hp = m._malloc(4);
    try {
        const data = new DataView(m.HEAPU8.buffer);
        integers.forEach((v, j) => data.setBigInt64(ip + j * 8, BigInt(v), true));
        doubles.forEach((v, j) => data.setFloat64(dp + j * 8, v, true));
        m.setValue(hp, 0, "i32");
        checkStatus(m.ccall("n4m_sample_filter_create", "number",
            ["number", "number", "number", "number", "number", "i64", "number"],
            [code, ip, ni, dp, nd, s, hp]) as number);
        return m.getValue(hp, "i32");
    } finally {
        for (const ptr of [ip, dp, hp]) if (ptr) m._free(ptr);
    }
}

export interface NativeFilterStats {
    nSamples: number;
    nKept: number;
    nExcluded: number;
    exclusionRate: number;
}

export class NativeSampleFilter {
    private handle: number;
    readonly kind: SampleFilterKind;

    constructor(kind: SampleFilterKind, integers: readonly number[],
                doubles: readonly number[], seed: number | bigint = 0) {
        this.kind = kind;
        this.handle = createSample(kind, integers, doubles, seed);
    }

    addChild(kind: "HighLeverage" | "SpectralQuality", integers: readonly number[],
             doubles: readonly number[], seed: number | bigint = 0): this {
        if (this.kind !== "Composite" ||
            !Object.prototype.hasOwnProperty.call(SAMPLE_KINDS, kind)) {
            throw new Error("only composite filters accept leverage/quality children");
        }
        const [code, ni, nd] = SAMPLE_KINDS[kind];
        const s = parameters(integers, doubles, ni, nd, seed);
        const m = getModule();
        const ip = ni ? m._malloc(ni * 8) : 0;
        const dp = nd ? m._malloc(nd * 8) : 0;
        try {
            const data = new DataView(m.HEAPU8.buffer);
            integers.forEach((v, j) => data.setBigInt64(ip + j * 8, BigInt(v), true));
            doubles.forEach((v, j) => data.setFloat64(dp + j * 8, v, true));
            checkStatus(m.ccall("n4m_sample_filter_add_child", "number",
                ["number", "number", "number", "number", "number", "number", "i64"],
                [this.live(), code, ip, ni, dp, nd, s]) as number);
            return this;
        } finally {
            for (const ptr of [ip, dp]) if (ptr) m._free(ptr);
        }
    }

    private live(): number {
        if (!this.handle) throw new Error("native sample filter has been disposed");
        return this.handle;
    }

    fit(X: Matrix, Y: Matrix | null = null): this {
        aligned(X, Y, this.kind === "YOutlier");
        const m = getModule();
        const xv = makeMatrixView(X.data, X.rows, X.cols);
        const yv = Y === null ? null : makeMatrixView(Y.data, Y.rows, Y.cols);
        try {
            checkStatus(m.ccall("n4m_sample_filter_fit", "number",
                ["number", "number", "number"],
                [this.live(), xv.viewPtr, yv?.viewPtr ?? 0]) as number);
            return this;
        } finally {
            xv.free(); yv?.free();
        }
    }

    apply(X: Matrix, Y: Matrix | null = null): { mask: Uint8Array; stats: NativeFilterStats } {
        aligned(X, Y, this.kind === "YOutlier");
        const m = getModule();
        const xv = makeMatrixView(X.data, X.rows, X.cols);
        const yv = Y === null ? null : makeMatrixView(Y.data, Y.rows, Y.cols);
        const mp = m._malloc(X.rows);
        const sp = m._malloc(32);
        try {
            checkStatus(m.ccall("n4m_sample_filter_apply", "number",
                ["number", "number", "number", "number", "number"],
                [this.live(), xv.viewPtr, yv?.viewPtr ?? 0, mp, sp]) as number);
            const data = new DataView(m.HEAPU8.buffer);
            return {
                mask: Uint8Array.from(m.HEAPU8.subarray(mp, mp + X.rows)),
                stats: {
                    nSamples: Number(data.getBigInt64(sp, true)),
                    nKept: Number(data.getBigInt64(sp + 8, true)),
                    nExcluded: Number(data.getBigInt64(sp + 16, true)),
                    exclusionRate: data.getFloat64(sp + 24, true),
                },
            };
        } finally {
            xv.free(); yv?.free(); m._free(mp); m._free(sp);
        }
    }

    dispose(): void {
        if (this.handle) {
            getModule().ccall("n4m_sample_filter_destroy", null,
                ["number"], [this.handle]);
            this.handle = 0;
        }
    }
}

export class NativeFeatureFilter {
    private handle: number;
    readonly kind: FeatureFilterKind;

    constructor(kind: FeatureFilterKind, threshold = 0, topK = -1) {
        if (!Object.prototype.hasOwnProperty.call(FEATURE_KINDS, kind) ||
            !Number.isFinite(threshold) || !Number.isInteger(topK)) {
            throw new Error("invalid native feature filter configuration");
        }
        const m = getModule();
        const hp = m._malloc(4);
        try {
            checkStatus(m.ccall("n4m_feature_filter_create", "number",
                ["number", "number", "number", "number"],
                [FEATURE_KINDS[kind], threshold, topK, hp]) as number);
            this.handle = m.getValue(hp, "i32");
        } finally { m._free(hp); }
        this.kind = kind;
    }

    private live(): number {
        if (!this.handle) throw new Error("native feature filter has been disposed");
        return this.handle;
    }

    fit(X: Matrix, Y: Matrix | null = null): this {
        aligned(X, Y, this.kind === "Correlation");
        if (this.kind === "Variance" && Y !== null) throw new Error("variance does not use Y");
        const m = getModule();
        const xv = makeMatrixView(X.data, X.rows, X.cols);
        const yv = Y === null ? null : makeMatrixView(Y.data, Y.rows, Y.cols);
        try {
            checkStatus(m.ccall("n4m_feature_filter_fit", "number",
                ["number", "number", "number"],
                [this.live(), xv.viewPtr, yv?.viewPtr ?? 0]) as number);
            return this;
        } finally { xv.free(); yv?.free(); }
    }

    selectedIndices(): Int32Array {
        const m = getModule();
        const cp = m._malloc(8);
        try {
            checkStatus(m.ccall("n4m_feature_filter_selected_indices", "number",
                ["number", "number", "i64", "number"],
                [this.live(), 0, 0n, cp]) as number);
            const n = Number(new DataView(m.HEAPU8.buffer).getBigInt64(cp, true));
            if (!Number.isSafeInteger(n) || n < 0 || n > 2147483647) {
                throw new Error("invalid native selected-index count");
            }
            const ip = m._malloc(Math.max(n, 1) * 8);
            try {
                checkStatus(m.ccall("n4m_feature_filter_selected_indices", "number",
                    ["number", "number", "i64", "number"],
                    [this.live(), ip, BigInt(n), cp]) as number);
                const data = new DataView(m.HEAPU8.buffer);
                return Int32Array.from(Array.from({ length: n }, (_, j) => {
                    const value = Number(data.getBigInt64(ip + 8 * j, true));
                    if (!Number.isSafeInteger(value) || value < 0 || value > 2147483647) {
                        throw new Error("invalid native selected index");
                    }
                    return value;
                }));
            } finally { m._free(ip); }
        } finally { m._free(cp); }
    }

    transform(X: Matrix): Matrix {
        validMatrix(X, "X");
        const cols = this.selectedIndices().length;
        const m = getModule();
        const xv = makeMatrixView(X.data, X.rows, X.cols);
        const result: Matrix = { rows: X.rows, cols, data: new Float64Array(X.rows * cols) };
        const ov = makeMatrixView(result.data, result.rows, result.cols);
        try {
            checkStatus(m.ccall("n4m_feature_filter_transform", "number",
                ["number", "number", "number"],
                [this.live(), xv.viewPtr, ov.viewPtr]) as number);
            result.data.set(m.HEAPF64.subarray(ov.dataPtr / 8,
                ov.dataPtr / 8 + result.data.length));
            return result;
        } finally { xv.free(); ov.free(); }
    }

    dispose(): void {
        if (this.handle) {
            getModule().ccall("n4m_feature_filter_destroy", null,
                ["number"], [this.handle]);
            this.handle = 0;
        }
    }
}
