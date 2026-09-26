// SPDX-License-Identifier: CECILL-2.1
// Fitted preprocessing and N4MP portability; all numerics remain in libn4m.

import { checkStatus, getModule, makeMatrixView } from "./ffi.js";
import type { Context } from "./context.js";
import type { Matrix } from "./types.js";

/** The 15 pipeline kinds currently implemented by the native core. */
export enum PipelineOperatorKind {
    IDENTITY = 0, CENTER = 1, AUTOSCALE = 2, PARETO_SCALE = 3,
    SNV = 4, MSC = 5, EMSC = 6, DETREND_POLY = 7,
    SAVGOL_SMOOTH = 8, SAVGOL_DERIVATIVE = 9, NORRIS_WILLIAMS = 10,
    ASLS_BASELINE = 11, OSC = 12, EPO = 13, WAVELET_DENOISE = 14,
}

export interface PipelineStep {
    readonly kind: PipelineOperatorKind;
    /** Original positional native parameters; defaults remain an empty vector. */
    readonly params: readonly number[];
}

function validateMatrix(X: Matrix, label: string): void {
    if (!Number.isSafeInteger(X.rows) || !Number.isSafeInteger(X.cols) ||
        X.rows < 1 || X.cols < 1 || X.data.length !== X.rows * X.cols ||
        !X.data.every(Number.isFinite)) {
        throw new Error(`${label} must be a nonempty finite row-major matrix`);
    }
}

function validateSteps(steps: readonly PipelineStep[]): void {
    if (steps.length < 1 || steps.length > 256) {
        throw new Error("native pipeline requires 1–256 ordered steps");
    }
    for (const step of steps) {
        if (!Number.isInteger(step.kind) || step.kind < 0 || step.kind > 14 ||
            !Array.isArray(step.params) || step.params.length > 256 ||
            !step.params.every(Number.isFinite)) {
            throw new Error("invalid or unsupported native pipeline step");
        }
    }
}

function readPlan(ptr: number): { features: number; steps: PipelineStep[] } {
    const m = getModule();
    const info = m._malloc(12); // int64_t feature width, int32_t operator count
    try {
        checkStatus(m.ccall("n4m_pipeline_get_info", "number",
            ["number", "number", "number"], [ptr, info, info + 8]) as number);
        const features = Number(m.getValue(info, "i64"));
        const count = m.getValue(info + 8, "i32");
        if (!Number.isSafeInteger(features) || features < 1 || count < 1 || count > 256) {
            throw new Error("invalid native pipeline metadata");
        }
        const kindPtr = m._malloc(4);
        const countPtr = m._malloc(4);
        try {
            const steps: PipelineStep[] = [];
            for (let index = 0; index < count; ++index) {
                checkStatus(m.ccall("n4m_pipeline_get_operator", "number",
                    ["number", "number", "number", "number", "number", "number"],
                    [ptr, index, kindPtr, 0, 0, countPtr]) as number);
                const kind = m.getValue(kindPtr, "i32");
                const nParams = m.getValue(countPtr, "i32");
                if (kind < 0 || kind > 14 || nParams < 0 || nParams > 256) {
                    throw new Error("invalid native pipeline operator metadata");
                }
                const paramsPtr = m._malloc(Math.max(1, nParams) * 8);
                try {
                    if (nParams > 0) {
                        checkStatus(m.ccall("n4m_pipeline_get_operator", "number",
                            ["number", "number", "number", "number", "number", "number"],
                            [ptr, index, kindPtr, paramsPtr, nParams, countPtr]) as number);
                    }
                    steps.push({ kind, params: Array.from(
                        m.HEAPF64.subarray(paramsPtr / 8, paramsPtr / 8 + nParams)) });
                } finally {
                    m._free(paramsPtr);
                }
            }
            return { features, steps };
        } finally {
            m._free(kindPtr);
            m._free(countPtr);
        }
    } finally {
        m._free(info);
    }
}

function samePlan(a: readonly PipelineStep[], b: readonly PipelineStep[]): boolean {
    return a.length === b.length && a.every((step, index) => {
        const other = b[index];
        return other !== undefined && step.kind === other.kind &&
            step.params.length === other.params.length &&
            step.params.every((value, i) => Object.is(value, other.params[i]));
    });
}

/** Owning JS façade for an ordered, fitted C ABI preprocessing pipeline. */
export class NativePreprocessingPipeline {
    private _ptr: number;
    private readonly _ctx: Context;
    readonly nFeatures: number;
    readonly steps: readonly PipelineStep[];

    private constructor(ctx: Context, ptr: number, features: number, steps: PipelineStep[]) {
        this._ctx = ctx;
        this._ptr = ptr;
        this.nFeatures = features;
        this.steps = steps.map(step => Object.freeze({
            kind: step.kind, params: Object.freeze([...step.params]),
        }));
        Object.freeze(this.steps);
    }

    /** Fit an ordered recipe. OSC and EPO require Y; others may omit it. */
    static fit(ctx: Context, steps: readonly PipelineStep[], X: Matrix,
               Y?: Matrix): NativePreprocessingPipeline {
        validateSteps(steps);
        validateMatrix(X, "X");
        if (Y !== undefined) {
            validateMatrix(Y, "Y");
            if (Y.rows !== X.rows) throw new Error("Y rows must match X rows");
        }
        const m = getModule();
        const outPtr = m._malloc(4);
        let ptr = 0;
        try {
            m.setValue(outPtr, 0, "i32");
            checkStatus(m.ccall("n4m_pipeline_create", "number", ["number"],
                [outPtr]) as number);
            ptr = m.getValue(outPtr, "i32");
            for (const step of steps) {
                const params = m._malloc(Math.max(1, step.params.length) * 8);
                try {
                    m.HEAPF64.set(step.params, params / 8);
                    checkStatus(m.ccall("n4m_pipeline_add_operator", "number",
                        ["number", "number", "number", "number"],
                        [ptr, step.kind, step.params.length ? params : 0,
                         step.params.length]) as number);
                } finally {
                    m._free(params);
                }
            }
            const xView = makeMatrixView(X.data, X.rows, X.cols);
            let yView: ReturnType<typeof makeMatrixView> | undefined;
            try {
                yView = Y === undefined ? undefined :
                    makeMatrixView(Y.data, Y.rows, Y.cols);
                checkStatus(m.ccall("n4m_pipeline_fit", "number",
                    ["number", "number", "number", "number"],
                    [ctx.handle, ptr, xView.viewPtr, yView?.viewPtr ?? 0]) as number,
                    ctx.handle);
            } finally {
                yView?.free();
                xView.free();
            }
            const plan = readPlan(ptr);
            if (!samePlan(plan.steps, steps)) {
                throw new Error("native fitted plan differs from requested recipe");
            }
            const result = new NativePreprocessingPipeline(ctx, ptr, plan.features, plan.steps);
            ptr = 0;
            return result;
        } finally {
            if (ptr !== 0) m.ccall("n4m_pipeline_destroy", null, ["number"], [ptr]);
            m._free(outPtr);
        }
    }

    /** Import fitted N4MP bytes; optionally attest against an external recipe. */
    static fromBytes(ctx: Context, bytes: Uint8Array,
                     expectedSteps?: readonly PipelineStep[]): NativePreprocessingPipeline {
        if (!(bytes instanceof Uint8Array) || bytes.length === 0 ||
            bytes.length > 64 * 1024 * 1024) {
            throw new Error("N4MP payload must be a Uint8Array of at most 64 MiB");
        }
        if (expectedSteps !== undefined) validateSteps(expectedSteps);
        const m = getModule();
        const data = m._malloc(bytes.length);
        const outPtr = m._malloc(4);
        let ptr = 0;
        try {
            m.HEAPU8.set(bytes, data);
            m.setValue(outPtr, 0, "i32");
            checkStatus(m.ccall("n4m_pipeline_import_from_buffer", "number",
                ["number", "number", "number", "number"],
                [ctx.handle, data, bytes.length, outPtr]) as number, ctx.handle);
            ptr = m.getValue(outPtr, "i32");
            const plan = readPlan(ptr);
            if (expectedSteps !== undefined && !samePlan(plan.steps, expectedSteps)) {
                throw new Error("N4MP ordered plan does not match expected recipe");
            }
            const result = new NativePreprocessingPipeline(ctx, ptr, plan.features, plan.steps);
            ptr = 0;
            return result;
        } finally {
            if (ptr !== 0) m.ccall("n4m_pipeline_destroy", null, ["number"], [ptr]);
            m._free(data);
            m._free(outPtr);
        }
    }

    /** Transform new rows using only native fitted state. */
    transform(X: Matrix): Matrix {
        if (this._ptr === 0) throw new Error("native pipeline has been destroyed");
        validateMatrix(X, "X");
        if (X.cols !== this.nFeatures) throw new Error("X feature width must match fit");
        const m = getModule();
        const input = makeMatrixView(X.data, X.rows, X.cols);
        try {
            const output = makeMatrixView(new Float64Array(X.data.length), X.rows, X.cols);
            try {
                checkStatus(m.ccall("n4m_pipeline_transform", "number",
                    ["number", "number", "number", "number"],
                    [this._ctx.handle, this._ptr, input.viewPtr, output.viewPtr]) as number,
                    this._ctx.handle);
                return { data: Float64Array.from(m.HEAPF64.subarray(
                    output.dataPtr / 8, output.dataPtr / 8 + X.data.length)),
                    rows: X.rows, cols: X.cols };
            } finally {
                output.free();
            }
        } finally {
            input.free();
        }
    }

    /** Export the native fitted state and ordered recipe in N4MP format. */
    toBytes(): Uint8Array {
        if (this._ptr === 0) throw new Error("native pipeline has been destroyed");
        const m = getModule();
        const sizePtr = m._malloc(4); // WASM32 size_t
        const writtenPtr = m._malloc(4);
        try {
            checkStatus(m.ccall("n4m_pipeline_export_size", "number",
                ["number", "number"], [this._ptr, sizePtr]) as number);
            const size = m.getValue(sizePtr, "i32");
            const data = m._malloc(Math.max(1, size));
            try {
                checkStatus(m.ccall("n4m_pipeline_export_to_buffer", "number",
                    ["number", "number", "number", "number"],
                    [this._ptr, data, size, writtenPtr]) as number);
                return Uint8Array.from(m.HEAPU8.subarray(
                    data, data + m.getValue(writtenPtr, "i32")));
            } finally {
                m._free(data);
            }
        } finally {
            m._free(sizePtr);
            m._free(writtenPtr);
        }
    }

    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall("n4m_pipeline_destroy", null, ["number"], [this._ptr]);
        this._ptr = 0;
    }
}
