// SPDX-License-Identifier: CECILL-2.1
//
// AOM/POP support handles and result wrappers.

import {
    asI64,
    checkStatus,
    getModule,
    makeMatrixView,
    readI64,
} from "./ffi.js";
import {
    GatingMode,
    Matrix,
    OperatorKind,
    Pls4allError,
    Status,
} from "./types.js";

export type HandleLike = number | { handle: number };

export function handleOf(value: HandleLike): number {
    return typeof value === "number" ? value : value.handle;
}

export function nullableHandleOf(value: HandleLike | null | undefined): number {
    return value == null ? 0 : handleOf(value);
}

function readOutPtr(outPtr: number, name: string): number {
    const ptr = Number(getModule().getValue(outPtr, "i32"));
    if (ptr === 0) {
        throw new Pls4allError(
            Status.ERR_INTERNAL,
            `${name} returned a NULL handle`,
        );
    }
    return ptr;
}

function allocF64(values: ArrayLike<number>): { ptr: number; free: () => void } {
    const m = getModule();
    const data = values instanceof Float64Array ?
        values : Float64Array.from(values);
    const ptr = m._malloc(data.length * 8);
    if (data.length > 0) m.HEAPF64.set(data, ptr / 8);
    return { ptr, free: () => m._free(ptr) };
}

function allocI64(values: ArrayLike<number | bigint>): {
    ptr: number;
    length: number;
    free: () => void;
} {
    const m = getModule();
    const ptr = m._malloc(values.length * 8);
    const view = new BigInt64Array(m.HEAPU8.buffer, ptr, values.length);
    for (let i = 0; i < values.length; i += 1) {
        const value = values[i];
        if (value === undefined) {
            throw new Error(`missing int64 value at index ${i}`);
        }
        view[i] = asI64(value);
    }
    return { ptr, length: values.length, free: () => m._free(ptr) };
}

function readInt32(getter: string, handle: number): number {
    const m = getModule();
    const out = m._malloc(4);
    try {
        const status = m.ccall(getter, "number",
            ["number", "number"], [handle, out]) as number;
        checkStatus(status);
        return Number(m.getValue(out, "i32"));
    } finally {
        m._free(out);
    }
}

function readDouble(getter: string, handle: number): number {
    const m = getModule();
    const out = m._malloc(8);
    try {
        const status = m.ccall(getter, "number",
            ["number", "number"], [handle, out]) as number;
        checkStatus(status);
        return Number(m.getValue(out, "double"));
    } finally {
        m._free(out);
    }
}

function readInt32Vector(getter: string, handle: number): Int32Array {
    const m = getModule();
    const dataPtrPtr = m._malloc(4);
    const sizePtr = m._malloc(4);
    try {
        const status = m.ccall(getter, "number",
            ["number", "number", "number"],
            [handle, dataPtrPtr, sizePtr]) as number;
        checkStatus(status);
        const dataPtr = Number(m.getValue(dataPtrPtr, "i32"));
        const size = Number(m.getValue(sizePtr, "i32"));
        if (size === 0) return new Int32Array(0);
        return m.HEAP32.subarray(dataPtr / 4, dataPtr / 4 + size).slice();
    } finally {
        m._free(dataPtrPtr);
        m._free(sizePtr);
    }
}

function readFloat64Vector(getter: string, handle: number): Float64Array {
    const m = getModule();
    const dataPtrPtr = m._malloc(4);
    const sizePtr = m._malloc(4);
    try {
        const status = m.ccall(getter, "number",
            ["number", "number", "number"],
            [handle, dataPtrPtr, sizePtr]) as number;
        checkStatus(status);
        const dataPtr = Number(m.getValue(dataPtrPtr, "i32"));
        const size = Number(m.getValue(sizePtr, "i32"));
        if (size === 0) return new Float64Array(0);
        return m.HEAPF64.subarray(dataPtr / 8, dataPtr / 8 + size).slice();
    } finally {
        m._free(dataPtrPtr);
        m._free(sizePtr);
    }
}

function readFloat64MatrixI32(getter: string, handle: number): Matrix {
    const m = getModule();
    const dataPtrPtr = m._malloc(4);
    const rowsPtr = m._malloc(4);
    const colsPtr = m._malloc(4);
    try {
        const status = m.ccall(getter, "number",
            ["number", "number", "number", "number"],
            [handle, dataPtrPtr, rowsPtr, colsPtr]) as number;
        checkStatus(status);
        const dataPtr = Number(m.getValue(dataPtrPtr, "i32"));
        const rows = Number(m.getValue(rowsPtr, "i32"));
        const cols = Number(m.getValue(colsPtr, "i32"));
        const n = rows * cols;
        const data = n === 0 ? new Float64Array(0) :
            m.HEAPF64.subarray(dataPtr / 8, dataPtr / 8 + n).slice();
        return { data, rows, cols };
    } finally {
        m._free(dataPtrPtr);
        m._free(rowsPtr);
        m._free(colsPtr);
    }
}

function readFloat64MatrixI64(getter: string, handle: number): Matrix {
    const m = getModule();
    const dataPtrPtr = m._malloc(4);
    const rowsPtr = m._malloc(8);
    const colsPtr = m._malloc(8);
    try {
        const status = m.ccall(getter, "number",
            ["number", "number", "number", "number"],
            [handle, dataPtrPtr, rowsPtr, colsPtr]) as number;
        checkStatus(status);
        const dataPtr = Number(m.getValue(dataPtrPtr, "i32"));
        const rows = readI64(rowsPtr);
        const cols = readI64(colsPtr);
        const n = rows * cols;
        const data = n === 0 ? new Float64Array(0) :
            m.HEAPF64.subarray(dataPtr / 8, dataPtr / 8 + n).slice();
        return { data, rows, cols };
    } finally {
        m._free(dataPtrPtr);
        m._free(rowsPtr);
        m._free(colsPtr);
    }
}

export class OperatorBank {
    private _ptr: number;

    constructor() {
        const m = getModule();
        const out = m._malloc(4);
        try {
            const status = m.ccall("p4a_operator_bank_create", "number",
                ["number"], [out]) as number;
            checkStatus(status);
            this._ptr = readOutPtr(out, "p4a_operator_bank_create");
        } finally {
            m._free(out);
        }
    }

    get handle(): number {
        return this._ptr;
    }

    add(kind: OperatorKind | number, params: ArrayLike<number> = []): void {
        const m = getModule();
        const buffer = allocF64(params);
        try {
            const status = m.ccall("p4a_operator_bank_add", "number",
                ["number", "number", "number", "number"],
                [this._ptr, kind, buffer.ptr, params.length]) as number;
            checkStatus(status);
        } finally {
            buffer.free();
        }
    }

    size(): number {
        const m = getModule();
        const out = m._malloc(4);
        try {
            const status = m.ccall("p4a_operator_bank_size", "number",
                ["number", "number"], [this._ptr, out]) as number;
            checkStatus(status);
            return Number(m.getValue(out, "i32"));
        } finally {
            m._free(out);
        }
    }

    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall("p4a_operator_bank_destroy", null,
            ["number"], [this._ptr]);
        this._ptr = 0;
    }
}

export class GatingStrategy {
    private _ptr: number;

    constructor(mode: GatingMode | number) {
        const m = getModule();
        const out = m._malloc(4);
        try {
            const status = m.ccall("p4a_gating_strategy_create", "number",
                ["number", "number"], [out, mode]) as number;
            checkStatus(status);
            this._ptr = readOutPtr(out, "p4a_gating_strategy_create");
        } finally {
            m._free(out);
        }
    }

    get handle(): number {
        return this._ptr;
    }

    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall("p4a_gating_strategy_destroy", null,
            ["number"], [this._ptr]);
        this._ptr = 0;
    }
}

export class ValidationPlan {
    private _ptr: number;

    constructor(nSamples?: number) {
        const m = getModule();
        const out = m._malloc(4);
        try {
            const status = m.ccall("p4a_validation_plan_create", "number",
                ["number"], [out]) as number;
            checkStatus(status);
            this._ptr = readOutPtr(out, "p4a_validation_plan_create");
        } finally {
            m._free(out);
        }
        if (nSamples !== undefined) this.setNSamples(nSamples);
    }

    get handle(): number {
        return this._ptr;
    }

    setNSamples(nSamples: number): void {
        const status = getModule().ccall(
            "p4a_validation_plan_set_n_samples", "number",
            ["number", "number"], [this._ptr, asI64(nSamples)]) as number;
        checkStatus(status);
    }

    getNSamples(): number {
        const m = getModule();
        const out = m._malloc(8);
        try {
            const status = m.ccall("p4a_validation_plan_get_n_samples",
                "number", ["number", "number"], [this._ptr, out]) as number;
            checkStatus(status);
            return readI64(out);
        } finally {
            m._free(out);
        }
    }

    getNFolds(): number {
        const m = getModule();
        const out = m._malloc(4);
        try {
            const status = m.ccall("p4a_validation_plan_get_n_folds",
                "number", ["number", "number"], [this._ptr, out]) as number;
            checkStatus(status);
            return Number(m.getValue(out, "i32"));
        } finally {
            m._free(out);
        }
    }

    addFold(train: ArrayLike<number | bigint>,
            test: ArrayLike<number | bigint>): void {
        const trainBuf = allocI64(train);
        const testBuf = allocI64(test);
        try {
            const status = getModule().ccall(
                "p4a_validation_plan_add_fold", "number",
                ["number", "number", "number", "number", "number"],
                [
                    this._ptr,
                    trainBuf.ptr,
                    asI64(trainBuf.length),
                    testBuf.ptr,
                    asI64(testBuf.length),
                ],
            ) as number;
            checkStatus(status);
        } finally {
            trainBuf.free();
            testBuf.free();
        }
    }

    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall("p4a_validation_plan_destroy", null,
            ["number"], [this._ptr]);
        this._ptr = 0;
    }
}

export class AomGlobalResult {
    private _ptr: number;

    constructor(ptr: number) {
        this._ptr = ptr;
    }

    get handle(): number {
        return this._ptr;
    }

    get nOperators(): number {
        return readInt32("p4a_aom_global_result_get_n_operators", this._ptr);
    }

    get maxComponents(): number {
        return readInt32("p4a_aom_global_result_get_max_components", this._ptr);
    }

    get selectedOperatorIndex(): number {
        return readInt32(
            "p4a_aom_global_result_get_selected_operator_index", this._ptr);
    }

    get selectedNComponents(): number {
        return readInt32(
            "p4a_aom_global_result_get_selected_n_components", this._ptr);
    }

    get bestScore(): number {
        return readDouble("p4a_aom_global_result_get_best_score", this._ptr);
    }

    operatorKinds(): Int32Array {
        return readInt32Vector(
            "p4a_aom_global_result_get_operator_kinds", this._ptr);
    }

    operatorScores(): Float64Array {
        return readFloat64Vector(
            "p4a_aom_global_result_get_operator_scores", this._ptr);
    }

    rmseCurves(): Matrix {
        return readFloat64MatrixI32(
            "p4a_aom_global_result_get_rmse_curves", this._ptr);
    }

    predictions(): Matrix {
        return readFloat64MatrixI64(
            "p4a_aom_global_result_get_predictions", this._ptr);
    }

    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall("p4a_aom_global_result_destroy", null,
            ["number"], [this._ptr]);
        this._ptr = 0;
    }
}

export class AomPerComponentResult {
    private _ptr: number;

    constructor(ptr: number) {
        this._ptr = ptr;
    }

    get handle(): number {
        return this._ptr;
    }

    get nOperators(): number {
        return readInt32(
            "p4a_aom_per_component_result_get_n_operators", this._ptr);
    }

    get maxComponents(): number {
        return readInt32(
            "p4a_aom_per_component_result_get_max_components", this._ptr);
    }

    get selectedNComponents(): number {
        return readInt32(
            "p4a_aom_per_component_result_get_selected_n_components",
            this._ptr,
        );
    }

    get bestScore(): number {
        return readDouble(
            "p4a_aom_per_component_result_get_best_score", this._ptr);
    }

    operatorKinds(): Int32Array {
        return readInt32Vector(
            "p4a_aom_per_component_result_get_operator_kinds", this._ptr);
    }

    selectedOperatorIndices(): Int32Array {
        return readInt32Vector(
            "p4a_aom_per_component_result_get_selected_operator_indices",
            this._ptr,
        );
    }

    componentScores(): Matrix {
        return readFloat64MatrixI32(
            "p4a_aom_per_component_result_get_component_scores", this._ptr);
    }

    prefixScores(): Float64Array {
        return readFloat64Vector(
            "p4a_aom_per_component_result_get_prefix_scores", this._ptr);
    }

    predictions(): Matrix {
        return readFloat64MatrixI64(
            "p4a_aom_per_component_result_get_predictions", this._ptr);
    }

    destroy(): void {
        if (this._ptr === 0) return;
        getModule().ccall("p4a_aom_per_component_result_destroy", null,
            ["number"], [this._ptr]);
        this._ptr = 0;
    }
}

export function aom_global_select(
    ctx: HandleLike,
    cfg: HandleLike,
    bank: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    maxComponents: number,
): AomGlobalResult {
    const m = getModule();
    const xView = makeMatrixView(X.data, X.rows, X.cols);
    const yView = makeMatrixView(Y.data, Y.rows, Y.cols);
    const out = m._malloc(4);
    const ctxPtr = handleOf(ctx);
    try {
        const status = m.ccall("p4a_aom_global_select", "number",
            ["number", "number", "number", "number", "number", "number",
             "number", "number"],
            [
                ctxPtr,
                handleOf(cfg),
                handleOf(bank),
                xView.viewPtr,
                yView.viewPtr,
                handleOf(plan),
                maxComponents,
                out,
            ]) as number;
        checkStatus(status, ctxPtr);
        return new AomGlobalResult(readOutPtr(out, "p4a_aom_global_select"));
    } finally {
        m._free(out);
        yView.free();
        xView.free();
    }
}

export function aom_per_component_select(
    ctx: HandleLike,
    cfg: HandleLike,
    bank: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    maxComponents: number,
): AomPerComponentResult {
    const m = getModule();
    const xView = makeMatrixView(X.data, X.rows, X.cols);
    const yView = makeMatrixView(Y.data, Y.rows, Y.cols);
    const out = m._malloc(4);
    const ctxPtr = handleOf(ctx);
    try {
        const status = m.ccall("p4a_aom_per_component_select", "number",
            ["number", "number", "number", "number", "number", "number",
             "number", "number"],
            [
                ctxPtr,
                handleOf(cfg),
                handleOf(bank),
                xView.viewPtr,
                yView.viewPtr,
                handleOf(plan),
                maxComponents,
                out,
            ]) as number;
        checkStatus(status, ctxPtr);
        return new AomPerComponentResult(
            readOutPtr(out, "p4a_aom_per_component_select"),
        );
    } finally {
        m._free(out);
        yView.free();
        xView.free();
    }
}
