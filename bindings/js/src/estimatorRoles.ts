// SPDX-License-Identifier: CECILL-2.1
// Generic native estimator roles (ABI 2.13).
//
// Every class in estimatorRolesGenerated.ts extends NativeEstimator and
// implements exactly the role interfaces its native method declares
// (Regressor, Transformer, ...). Parameters, defaults, required inputs,
// fitting and the portable N4ME state are native; this file marshals only.

import { checkStatus, getModule, makeMatrixView } from "./ffi.js";
import type { Matrix } from "./types.js";

/** Regressor role: Data[n, p] + Target[n, q] -> Prediction[n, q]. */
export interface Regressor {
    predict(X: Matrix): Matrix;
}

/** Transformer role: Data[n, p] (+ Target) -> Data[n, k]. */
export interface Transformer {
    transform(X: Matrix): Matrix;
}

/** Native parameter types, as published by the manifest. */
export type ParamType = "int" | "double" | "bool" | "enum" | "int_array" | "double_array";
export type ParamValue = number | boolean | string | number[];

/** Optional fit inputs; the native core refuses those a method does not use. */
export interface FitInputs {
    sampleWeight?: Float64Array | number[];
    groups?: number[];
    featureGroups?: number[];
    blocks?: number[];
    axis?: Float64Array | number[];
    XTarget?: Matrix;
    foldIds?: number[];
    seed?: number;
}

// n4m_fit_inputs_v1_t on wasm32 (pointers 4 bytes, int64 8-aligned); checked
// against offsetof() of the C header when the layout was written.
const FIT_INPUTS_SIZE = 128;
const OFF = {
    X: 4, Y: 8, sampleWeight: 24, nSampleWeight: 32, groups: 40, nGroups: 48,
    featureGroups: 56, nFeatureGroups: 64, blocks: 72, nBlocks: 80, axis: 88,
    nAxis: 96, XTarget: 104, foldIds: 108, nFoldIds: 112, seed: 120,
} as const;

/** Runs `fn` with a fresh native context, destroyed afterwards. */
function withContext<T>(fn: (ctx: number) => T): T {
    const m = getModule();
    const out = m._malloc(4);
    try {
        m.setValue(out, 0, "i32");
        checkStatus(m.ccall("n4m_context_create", "number", ["number"], [out]) as number);
        const ctx = m.getValue(out, "i32");
        try {
            return fn(ctx);
        } finally {
            m.ccall("n4m_context_destroy", null, ["number"], [ctx]);
        }
    } finally {
        m._free(out);
    }
}

function readI64(ptr: number): number {
    const m = getModule();
    return Number(m.getValue(ptr, "i64") as unknown as bigint);
}

type Alloc = { ptr: number; free: () => void };

function allocF64(values: Float64Array | number[]): Alloc {
    const m = getModule();
    const ptr = m._malloc(Math.max(1, values.length) * 8);
    m.HEAPF64.set(values instanceof Float64Array ? values : Float64Array.from(values), ptr / 8);
    return { ptr, free: () => m._free(ptr) };
}

function allocI64(values: number[]): Alloc {
    const m = getModule();
    const ptr = m._malloc(Math.max(1, values.length) * 8);
    values.forEach((v, i) => m.setValue(ptr + 8 * i, BigInt(v) as unknown as number, "i64"));
    return { ptr, free: () => m._free(ptr) };
}

const registry = new Map<string, new () => NativeEstimator>();

/** Base of every generated estimator: parameters, fit and N4ME state. */
export abstract class NativeEstimator {
    /** Catalog method id, for example "models.pls.pls_regression". */
    abstract readonly methodId: string;
    /** Parameter name -> native type. */
    protected abstract readonly paramTypes: Readonly<Record<string, ParamType>>;
    /** Explicit parameter values (unset ones take the native default). */
    params: Record<string, ParamValue | undefined> = {};
    private ptr = 0;

    /** Registers a generated class so fromN4me() can rebuild it. */
    static register(methodId: string, cls: new () => NativeEstimator): void {
        registry.set(methodId, cls);
    }

    get fitted(): boolean {
        return this.ptr !== 0;
    }

    /** Fit on row-major X and optional targets. Returns this. */
    fit(X: Matrix, y?: Matrix | Float64Array, inputs: FitInputs = {}): this {
        const m = getModule();
        const allocs: Alloc[] = [];
        const hold = (a: Alloc) => (allocs.push(a), a.ptr);
        const struct = m._malloc(FIT_INPUTS_SIZE);
        try {
            m.HEAPU8.fill(0, struct, struct + FIT_INPUTS_SIZE);
            m.setValue(struct, FIT_INPUTS_SIZE, "i32");
            const xv = makeMatrixView(X.data, X.rows, X.cols);
            allocs.push({ ptr: xv.viewPtr, free: xv.free });
            m.setValue(struct + OFF.X, xv.viewPtr, "i32");
            if (y !== undefined) {
                const ym: Matrix = y instanceof Float64Array ? { data: y, rows: y.length, cols: 1 } : y;
                const yv = makeMatrixView(ym.data, ym.rows, ym.cols);
                allocs.push({ ptr: yv.viewPtr, free: yv.free });
                m.setValue(struct + OFF.Y, yv.viewPtr, "i32");
            }
            const setArray = (ptrOff: number, lenOff: number, a: Alloc, n: number) => {
                m.setValue(struct + ptrOff, hold(a), "i32");
                m.setValue(struct + lenOff, BigInt(n) as unknown as number, "i64");
            };
            if (inputs.sampleWeight) setArray(OFF.sampleWeight, OFF.nSampleWeight, allocF64(inputs.sampleWeight), inputs.sampleWeight.length);
            if (inputs.groups) setArray(OFF.groups, OFF.nGroups, allocI64(inputs.groups), inputs.groups.length);
            if (inputs.featureGroups) setArray(OFF.featureGroups, OFF.nFeatureGroups, allocI64(inputs.featureGroups), inputs.featureGroups.length);
            if (inputs.blocks) setArray(OFF.blocks, OFF.nBlocks, allocI64(inputs.blocks), inputs.blocks.length);
            if (inputs.axis) setArray(OFF.axis, OFF.nAxis, allocF64(inputs.axis), inputs.axis.length);
            if (inputs.foldIds) setArray(OFF.foldIds, OFF.nFoldIds, allocI64(inputs.foldIds), inputs.foldIds.length);
            if (inputs.XTarget) {
                const tv = makeMatrixView(inputs.XTarget.data, inputs.XTarget.rows, inputs.XTarget.cols);
                allocs.push({ ptr: tv.viewPtr, free: tv.free });
                m.setValue(struct + OFF.XTarget, tv.viewPtr, "i32");
            }
            m.setValue(struct + OFF.seed, BigInt(inputs.seed ?? 0) as unknown as number, "i64");

            const est = withContext((ctx) => {
                const params = this.nativeParams(ctx);
                const out = m._malloc(4);
                try {
                    m.setValue(out, 0, "i32");
                    const idPtr = hold(this.cString(this.methodId));
                    checkStatus(m.ccall("n4m_estimator_create", "number",
                        ["number", "number", "number", "number"], [ctx, idPtr, params, out]) as number, ctx);
                    const handle = m.getValue(out, "i32");
                    const status = m.ccall("n4m_estimator_fit", "number",
                        ["number", "number", "number"], [ctx, handle, struct]) as number;
                    if (status !== 0) {
                        m.ccall("n4m_estimator_destroy", null, ["number"], [handle]);
                        checkStatus(status, ctx);
                    }
                    return handle;
                } finally {
                    m._free(out);
                    m.ccall("n4m_params_destroy", null, ["number"], [params]);
                }
            });
            this.dispose();
            this.ptr = est;
            return this;
        } finally {
            allocs.forEach((a) => a.free());
            m._free(struct);
        }
    }

    /** Portable fitted state (N4ME bytes), readable by every n4m binding. */
    toN4me(): Uint8Array {
        const m = getModule();
        const handle = this.handle();
        return withContext((ctx) => {
            const sizePtr = m._malloc(4);
            try {
                checkStatus(m.ccall("n4m_estimator_export_size", "number",
                    ["number", "number", "number", "number"], [ctx, handle, 1, sizePtr]) as number, ctx);
                const size = m.getValue(sizePtr, "i32");
                const buf = m._malloc(Math.max(1, size));
                try {
                    checkStatus(m.ccall("n4m_estimator_export_to_buffer", "number",
                        ["number", "number", "number", "number", "number", "number"],
                        [ctx, handle, 1, buf, size, sizePtr]) as number, ctx);
                    return m.HEAPU8.slice(buf, buf + m.getValue(sizePtr, "i32"));
                } finally {
                    m._free(buf);
                }
            } finally {
                m._free(sizePtr);
            }
        });
    }

    /** Rebuilds a fitted estimator of the class registered for its method. */
    static fromN4me(payload: Uint8Array): NativeEstimator {
        const m = getModule();
        const data = m._malloc(Math.max(1, payload.byteLength));
        const out = m._malloc(4);
        try {
            m.HEAPU8.set(payload, data);
            m.setValue(out, 0, "i32");
            const handle = withContext((ctx) => {
                checkStatus(m.ccall("n4m_estimator_import_from_buffer", "number",
                    ["number", "number", "number", "number"], [ctx, data, payload.byteLength, out]) as number, ctx);
                return m.getValue(out, "i32");
            });
            const cls = registry.get(NativeEstimator.methodIdOf(handle));
            if (cls === undefined) {
                m.ccall("n4m_estimator_destroy", null, ["number"], [handle]);
                throw new Error("no JS class registered for this N4ME method");
            }
            const est = new cls();
            est.ptr = handle;
            return est;
        } finally {
            m._free(data);
            m._free(out);
        }
    }

    /** Releases the native estimator. */
    dispose(): void {
        if (this.ptr !== 0) {
            getModule().ccall("n4m_estimator_destroy", null, ["number"], [this.ptr]);
            this.ptr = 0;
        }
    }

    protected predictMatrix(X: Matrix): Matrix {
        return this.matrixOp("n4m_estimator_predict", "n4m_estimator_n_outputs", X);
    }

    protected transformMatrix(X: Matrix): Matrix {
        return this.matrixOp("n4m_estimator_transform", "n4m_estimator_transform_cols", X);
    }

    private matrixOp(symbol: string, widthSymbol: string, X: Matrix): Matrix {
        const m = getModule();
        const handle = this.handle();
        const widthPtr = m._malloc(8);
        try {
            checkStatus(m.ccall(widthSymbol, "number", ["number", "number"], [handle, widthPtr]) as number);
            const cols = readI64(widthPtr);
            const xv = makeMatrixView(X.data, X.rows, X.cols);
            const ov = makeMatrixView(new Float64Array(X.rows * cols), X.rows, cols);
            try {
                withContext((ctx) => checkStatus(m.ccall(symbol, "number",
                    ["number", "number", "number", "number"], [ctx, handle, xv.viewPtr, ov.viewPtr]) as number, ctx));
                return { data: m.HEAPF64.slice(ov.dataPtr / 8, ov.dataPtr / 8 + X.rows * cols), rows: X.rows, cols };
            } finally {
                xv.free();
                ov.free();
            }
        } finally {
            m._free(widthPtr);
        }
    }

    private handle(): number {
        if (this.ptr === 0) throw new Error(`${this.methodId} is not fitted`);
        return this.ptr;
    }

    private cString(s: string): Alloc {
        const m = getModule();
        const n = m.lengthBytesUTF8(s) + 1;
        const ptr = m._malloc(n);
        m.stringToUTF8(s, ptr, n);
        return { ptr, free: () => m._free(ptr) };
    }

    private nativeParams(ctx: number): number {
        const m = getModule();
        const indexPtr = m._malloc(4);
        const out = m._malloc(4);
        const allocs: Alloc[] = [];
        const id = this.cString(this.methodId);
        allocs.push(id);
        try {
            checkStatus(m.ccall("n4m_method_find", "number", ["number", "number"], [id.ptr, indexPtr]) as number);
            m.setValue(out, 0, "i32");
            checkStatus(m.ccall("n4m_params_create", "number", ["number", "number", "number"],
                [ctx, m.getValue(indexPtr, "i32"), out]) as number, ctx);
            const params = m.getValue(out, "i32");
            try {
                for (const [name, value] of Object.entries(this.params)) {
                    if (value === undefined) continue;
                    const type = this.paramTypes[name];
                    if (type === undefined) throw new Error(`${this.methodId}: unknown parameter '${name}'`);
                    const key = this.cString(name);
                    allocs.push(key);
                    let status: number;
                    if (type === "int") {
                        status = m.ccall("n4m_params_set_int", "number", ["number", "number", "i64"], [params, key.ptr, BigInt(value as number)]) as number;
                    } else if (type === "double") {
                        status = m.ccall("n4m_params_set_double", "number", ["number", "number", "number"], [params, key.ptr, value]) as number;
                    } else if (type === "bool") {
                        status = m.ccall("n4m_params_set_bool", "number", ["number", "number", "number"], [params, key.ptr, value ? 1 : 0]) as number;
                    } else if (type === "enum") {
                        const choice = this.cString(String(value));
                        allocs.push(choice);
                        status = m.ccall("n4m_params_set_enum", "number", ["number", "number", "number"], [params, key.ptr, choice.ptr]) as number;
                    } else if (type === "int_array") {
                        const arr = allocI64(value as number[]);
                        allocs.push(arr);
                        status = m.ccall("n4m_params_set_int_array", "number", ["number", "number", "number", "i64"], [params, key.ptr, arr.ptr, BigInt((value as number[]).length)]) as number;
                    } else {
                        const arr = allocF64(value as number[]);
                        allocs.push(arr);
                        status = m.ccall("n4m_params_set_double_array", "number", ["number", "number", "number", "i64"], [params, key.ptr, arr.ptr, BigInt((value as number[]).length)]) as number;
                    }
                    if (status !== 0) throw new Error(`${this.methodId}: invalid value for parameter '${name}'`);
                }
                checkStatus(m.ccall("n4m_params_validate", "number", ["number", "number"], [ctx, params]) as number, ctx);
                return params;
            } catch (error) {
                m.ccall("n4m_params_destroy", null, ["number"], [params]);
                throw error;
            }
        } finally {
            allocs.forEach((a) => a.free());
            m._free(indexPtr);
            m._free(out);
        }
    }

    private static methodIdOf(handle: number): string {
        const m = getModule();
        const indexPtr = m._malloc(4);
        const capsPtr = m._malloc(8);
        const info = m._malloc(72);
        try {
            checkStatus(m.ccall("n4m_estimator_info", "number", ["number", "number", "number"], [handle, indexPtr, capsPtr]) as number);
            m.HEAPU8.fill(0, info, info + 72);
            m.setValue(info, 72, "i32");
            checkStatus(m.ccall("n4m_method_info_v1", "number", ["number", "number"], [m.getValue(indexPtr, "i32"), info]) as number);
            return m.UTF8ToString(m.getValue(info + 8, "i32"));
        } finally {
            m._free(indexPtr);
            m._free(capsPtr);
            m._free(info);
        }
    }
}
