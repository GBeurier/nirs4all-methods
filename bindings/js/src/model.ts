// SPDX-License-Identifier: CeCILL-2.1

import { Context } from "./context.js";
import { Config } from "./config.js";
import { checkStatus, getModule, makeMatrixView } from "./ffi.js";
import { Matrix } from "./types.js";

/** RAII wrapper around p4a_model_t. */
export class Model {
    private _ptr: number;

    private constructor(ptr: number) {
        this._ptr = ptr;
    }

    /** Fit a PLS model on `X` (n × p) predicting `Y` (n × q). */
    static fit(ctx: Context, cfg: Config, X: Matrix, Y: Matrix): Model {
        const m = getModule();
        const x = makeMatrixView(X.data, X.rows, X.cols);
        const y = makeMatrixView(Y.data, Y.rows, Y.cols);
        const out = m._malloc(4);
        try {
            const status = m.ccall(
                "p4a_model_fit", "number",
                ["number", "number", "number", "number", "number"],
                [ctx.handle, cfg.handle, x.viewPtr, y.viewPtr, out]) as number;
            checkStatus(status, ctx.handle);
            return new Model(m.getValue(out, "i32"));
        } finally {
            m._free(out);
            x.free();
            y.free();
        }
    }

    get handle(): number {
        return this._ptr;
    }

    /** Predict targets for `X` (n × p). Returns an (n × n_targets) matrix. */
    predict(ctx: Context, X: Matrix): Matrix {
        const m = getModule();
        const n_targets_buf = m._malloc(4);
        const status_nt = m.ccall(
            "p4a_model_get_n_targets", "number",
            ["number", "number"], [this._ptr, n_targets_buf]) as number;
        checkStatus(status_nt);
        const n_targets = m.getValue(n_targets_buf, "i32");
        m._free(n_targets_buf);

        const xView = makeMatrixView(X.data, X.rows, X.cols);
        const outData = new Float64Array(X.rows * n_targets);
        const outView = makeMatrixView(outData, X.rows, n_targets);
        try {
            const status = m.ccall(
                "p4a_model_predict", "number",
                ["number", "number", "number", "number"],
                [ctx.handle, this._ptr, xView.viewPtr,
                 outView.viewPtr]) as number;
            checkStatus(status, ctx.handle);
            // Copy back from WASM heap.
            const copied = new Float64Array(outData.length);
            copied.set(
                m.HEAPF64.subarray(outView.dataPtr / 8,
                                    outView.dataPtr / 8 + outData.length));
            return { data: copied, rows: X.rows, cols: n_targets };
        } finally {
            xView.free();
            outView.free();
        }
    }

    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall(
            "p4a_model_destroy", null, ["number"], [this._ptr]);
        this._ptr = 0;
    }
}
