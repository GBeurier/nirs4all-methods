// SPDX-License-Identifier: CECILL-2.1
// Native predict-only model promoted from an affine MethodResult.

import { checkStatus, getModule, makeMatrixView, readArrayView } from "./ffi.js";
import type { Context } from "./context.js";
import type { MethodResult } from "./methodResult.js";
import { ModelArray, type Matrix } from "./types.js";

/** Component resolution of a PLS regression or PCR model (ABI 2.20). */
export interface ComponentResolution {
    /** Absolute X-Y cross-covariance resolution, in centred (scaled) X times
     *  Y units, shared by every component. */
    resolution: number;
    /** Per component: true when its cross-covariance is not above the
     *  resolution, whether mathematically undefined (exact value zero) or
     *  merely not resolved numerically (below the bound). PLS publishes such
     *  a component, and every later one, as a null component (zero latent
     *  columns, no contribution to the coefficients); PCR keeps a resolved
     *  X direction (n4m.h, N4M_MODEL_UNRESOLVED_COMPONENTS). */
    unresolved: boolean[];
}

export class NativeModel {
    private _ptr: number;
    private readonly _ctx: Context;

    private constructor(ctx: Context, ptr: number) {
        this._ctx = ctx;
        this._ptr = ptr;
    }

    /** Copy an affine MethodResult into a standalone native model. */
    static fromMethodResult(ctx: Context, result: MethodResult): NativeModel {
        const m = getModule();
        const out = m._malloc(4);
        try {
            m.setValue(out, 0, "i32");
            const status = m.ccall(
                "n4m_model_from_method_result", "number",
                ["number", "number", "number"],
                [ctx.handle, result.handle, out]) as number;
            checkStatus(status, ctx.handle);
            return new NativeModel(ctx, m.getValue(out, "i32"));
        } finally {
            m._free(out);
        }
    }

    /** Import a complete native N4MM model payload. */
    static fromN4mm(ctx: Context, payload: Uint8Array): NativeModel {
        const m = getModule();
        const data = m._malloc(Math.max(1, payload.byteLength));
        const out = m._malloc(4);
        try {
            m.HEAPU8.set(payload, data);
            m.setValue(out, 0, "i32");
            const status = m.ccall(
                "n4m_model_import_from_buffer", "number",
                ["number", "number", "number", "number"],
                [ctx.handle, data, payload.byteLength, out]) as number;
            checkStatus(status, ctx.handle);
            return new NativeModel(ctx, m.getValue(out, "i32"));
        } finally {
            m._free(data);
            m._free(out);
        }
    }

    /** Predict through libn4m; no coefficients are evaluated in JS. */
    predict(X: Matrix): Matrix {
        const m = getModule();
        const targetsPtr = m._malloc(4);
        try {
            checkStatus(m.ccall(
                "n4m_model_get_n_targets", "number",
                ["number", "number"], [this._ptr, targetsPtr]) as number);
            const targets = m.getValue(targetsPtr, "i32");
            const input = makeMatrixView(X.data, X.rows, X.cols);
            try {
                const output = makeMatrixView(
                    new Float64Array(X.rows * targets), X.rows, targets);
                try {
                    checkStatus(m.ccall(
                        "n4m_model_predict", "number",
                        ["number", "number", "number", "number"],
                        [this._ctx.handle, this._ptr, input.viewPtr, output.viewPtr]
                    ) as number, this._ctx.handle);
                    return {
                        data: new Float64Array(m.HEAPF64.subarray(
                            output.dataPtr / 8, output.dataPtr / 8 + X.rows * targets)),
                        rows: X.rows,
                        cols: targets,
                    };
                } finally {
                    output.free();
                }
            } finally {
                input.free();
            }
        } finally {
            m._free(targetsPtr);
        }
    }

    /** Copy one fitted-model array (n4m_model_get_array). */
    getArray(kind: ModelArray): Matrix {
        const m = getModule();
        const out = m._malloc(4);
        try {
            m.setValue(out, 0, "i32");
            checkStatus(m.ccall(
                "n4m_model_get_array", "number",
                ["number", "number", "number", "number"],
                [this._ctx.handle, this._ptr, kind, out]) as number, this._ctx.handle);
            const arr = m.getValue(out, "i32");
            try {
                return readArrayView(arr);
            } finally {
                m.ccall("n4m_array_free", null, ["number"], [arr]);
            }
        } finally {
            m._free(out);
        }
    }

    /** Component resolution and unresolved flags; throws N4mError
     *  (unsupported) for models that do not record them. */
    componentResolution(): ComponentResolution {
        const resolution = this.getArray(ModelArray.COMPONENT_RESOLUTION).data[0]!;
        const flags = this.getArray(ModelArray.UNRESOLVED_COMPONENTS).data;
        return { resolution, unresolved: Array.from(flags, (v) => v !== 0) };
    }

    /** Export the complete native model as N4MM bytes. */
    toN4mm(): Uint8Array {
        const m = getModule();
        const sizePtr = m._malloc(4); // size_t on WASM32
        const writtenPtr = m._malloc(4);
        try {
            checkStatus(m.ccall(
                "n4m_model_export_size", "number",
                ["number", "number"], [this._ptr, sizePtr]) as number);
            const size = m.getValue(sizePtr, "i32");
            const data = m._malloc(Math.max(1, size));
            try {
                checkStatus(m.ccall(
                    "n4m_model_export_to_buffer", "number",
                    ["number", "number", "number", "number"],
                    [this._ptr, data, size, writtenPtr]) as number);
                return new Uint8Array(m.HEAPU8.subarray(
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
        getModule().ccall("n4m_model_destroy", null, ["number"], [this._ptr]);
        this._ptr = 0;
    }
}
