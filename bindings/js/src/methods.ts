// SPDX-License-Identifier: CECILL-2.1
//
// Complete JS bindings for the p4a_method_result_t method surface.

import {
    asI64,
    asU64,
    checkStatus,
    getModule,
    hasExportedFunction,
    makeMatrixView,
    MATRIX_VIEW_SIZE,
} from "./ffi.js";
import { handleOf, HandleLike, nullableHandleOf } from "./aom.js";
import { MethodResult } from "./methodResult.js";
import { Dtype, Matrix, Pls4allError, Status } from "./types.js";

type HeapBuffer = {
    ptr: number;
    length: number;
    free: () => void;
};

function resultFromCall(ctx: HandleLike | null, name: string,
                        args: unknown[]): MethodResult {
    const m = getModule();
    if (!hasExportedFunction(name)) {
        throw new Pls4allError(
            Status.ERR_NOT_IMPLEMENTED,
            `${name} is not exported by the loaded WASM module`,
        );
    }
    const out = m._malloc(4);
    const ctxPtr = nullableHandleOf(ctx);
    try {
        const status = m.ccall(
            name,
            "number",
            new Array(args.length + 1).fill("number"),
            [...args, out],
        ) as number;
        checkStatus(status, ctxPtr);
        const ptr = Number(m.getValue(out, "i32"));
        if (ptr === 0) {
            throw new Pls4allError(
                Status.ERR_INTERNAL,
                `${name} returned a NULL handle`,
            );
        }
        return new MethodResult(ptr);
    } finally {
        m._free(out);
    }
}

function allocF64(values: ArrayLike<number>): HeapBuffer {
    const m = getModule();
    const data = values instanceof Float64Array ?
        values : Float64Array.from(values);
    const ptr = m._malloc(data.length * 8);
    if (data.length > 0) m.HEAPF64.set(data, ptr / 8);
    return { ptr, length: data.length, free: () => m._free(ptr) };
}

function allocI32(values: ArrayLike<number>): HeapBuffer {
    const m = getModule();
    const ptr = m._malloc(values.length * 4);
    const view = new Int32Array(m.HEAPU8.buffer, ptr, values.length);
    for (let i = 0; i < values.length; i += 1) {
        const value = values[i];
        if (value === undefined) {
            throw new Error(`missing int32 value at index ${i}`);
        }
        view[i] = value | 0;
    }
    return { ptr, length: values.length, free: () => m._free(ptr) };
}

function allocI64(values: ArrayLike<number | bigint>): HeapBuffer {
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

function withMatrix<T>(matrix: Matrix, fn: (viewPtr: number) => T): T {
    const view = makeMatrixView(matrix.data, matrix.rows, matrix.cols);
    try {
        return fn(view.viewPtr);
    } finally {
        view.free();
    }
}

function withTwoMatrices<T>(X: Matrix, Y: Matrix,
                            fn: (xPtr: number, yPtr: number) => T): T {
    return withMatrix(X, (xPtr) =>
        withMatrix(Y, (yPtr) => fn(xPtr, yPtr)));
}

function makeMatrixViewArray(blocks: Matrix[]): {
    viewPtr: number;
    free: () => void;
} {
    const m = getModule();
    const viewPtr = m._malloc(MATRIX_VIEW_SIZE * blocks.length);
    const dataPtrs: number[] = [];
    try {
        for (let i = 0; i < blocks.length; i += 1) {
            const block = blocks[i];
            if (block === undefined) {
                throw new Error(`missing matrix block at index ${i}`);
            }
            const expected = block.rows * block.cols;
            if (block.data.length !== expected) {
                throw new Error(
                    `block ${i} data length ${block.data.length} != ` +
                    `rows*cols (${expected})`,
                );
            }
            const dataPtr = m._malloc(block.data.length * 8);
            dataPtrs.push(dataPtr);
            if (block.data.length > 0) {
                m.HEAPF64.set(block.data, dataPtr / 8);
            }
            const status = m.ccall(
                "p4a_matrix_view_init_rowmajor",
                "number",
                ["number", "number", "number", "number", "number"],
                [
                    viewPtr + i * MATRIX_VIEW_SIZE,
                    dataPtr,
                    asI64(block.rows),
                    asI64(block.cols),
                    Dtype.F64,
                ],
            ) as number;
            checkStatus(status);
        }
        return {
            viewPtr,
            free: () => {
                m._free(viewPtr);
                for (const ptr of dataPtrs) m._free(ptr);
            },
        };
    } catch (error) {
        m._free(viewPtr);
        for (const ptr of dataPtrs) m._free(ptr);
        throw error;
    }
}

export function sparse_simpls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    sparsityLambda: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_sparse_simpls_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, sparsityLambda],
    ));
}

export function di_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    XSource: Matrix,
    YSource: Matrix,
    XTarget: Matrix,
    diLambda: number,
): MethodResult {
    return withMatrix(XSource, (xsPtr) =>
        withMatrix(YSource, (ysPtr) =>
            withMatrix(XTarget, (xtPtr) => resultFromCall(
                ctx, "p4a_di_pls_fit",
                [handleOf(ctx), handleOf(cfg), xsPtr, ysPtr, xtPtr, diLambda],
            ))));
}

export function recursive_pls_run(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    windowSize: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_recursive_pls_run",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, windowSize],
    ));
}

export function cppls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    gamma: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_cppls_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, gamma],
    ));
}

export function weighted_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    sampleWeights: ArrayLike<number>,
): MethodResult {
    const weights = allocF64(sampleWeights);
    try {
        return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
            ctx, "p4a_weighted_pls_fit",
            [
                handleOf(ctx),
                handleOf(cfg),
                xPtr,
                yPtr,
                weights.ptr,
                asI64(weights.length),
            ],
        ));
    } finally {
        weights.free();
    }
}

export function robust_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    huberK: number,
    maxIrlsIter: number = 5,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_robust_pls_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, huberK, maxIrlsIter],
    ));
}

export function ridge_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    ridgeLambda: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_ridge_pls_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, ridgeLambda],
    ));
}

export function continuum_regression_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    tau: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_continuum_regression_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, tau],
    ));
}

export function n_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    XFlat: Matrix,
    modeJ: number,
    modeK: number,
    Y: Matrix,
): MethodResult {
    return withMatrix(XFlat, (xPtr) => withMatrix(Y, (yPtr) => resultFromCall(
        ctx, "p4a_n_pls_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, modeJ, modeK, yPtr],
    )));
}

export function kernel_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    kernelType: number,
    gamma: number = 0.0,
    coef0: number = 1.0,
    degree: number = 3,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_kernel_pls_fit",
        [
            handleOf(ctx),
            handleOf(cfg),
            kernelType,
            gamma,
            coef0,
            degree,
            xPtr,
            yPtr,
        ],
    ));
}

export function o2pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    nPredictive: number,
    nXOrthogonal: number = 0,
    nYOrthogonal: number = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_o2pls_fit",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            nPredictive,
            nXOrthogonal,
            nYOrthogonal,
        ],
    ));
}

export function approximate_press_compute(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    maxComponents: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_approximate_press_compute",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, maxComponents],
    ));
}

export function sparse_pls_da_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    yLabels: ArrayLike<number>,
): MethodResult {
    const labels = allocI32(yLabels);
    try {
        return withMatrix(X, (xPtr) => resultFromCall(
            ctx, "p4a_sparse_pls_da_fit",
            [
                handleOf(ctx),
                handleOf(cfg),
                xPtr,
                labels.ptr,
                asI64(labels.length),
            ],
        ));
    } finally {
        labels.free();
    }
}

export function group_sparse_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    groupAssignment: ArrayLike<number>,
    groupLambda: number,
): MethodResult {
    const groups = allocI32(groupAssignment);
    try {
        return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
            ctx, "p4a_group_sparse_pls_fit",
            [
                handleOf(ctx),
                handleOf(cfg),
                xPtr,
                yPtr,
                groups.ptr,
                asI64(groups.length),
                groupLambda,
            ],
        ));
    } finally {
        groups.free();
    }
}

export function fused_sparse_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    l1Lambda: number,
    fusionLambda: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_fused_sparse_pls_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, l1Lambda, fusionLambda],
    ));
}

export function pls_monitoring_run(
    ctx: HandleLike,
    model: HandleLike,
    XReference: Matrix,
    XMonitor: Matrix,
    alpha: number = 0.05,
): MethodResult {
    return withTwoMatrices(XReference, XMonitor, (xrPtr, xmPtr) =>
        resultFromCall(
            ctx, "p4a_pls_monitoring_run",
            [handleOf(ctx), handleOf(model), xrPtr, xmPtr, alpha],
        ));
}

export function one_se_rule_compute(
    ctx: HandleLike,
    foldRmseMatrix: Matrix,
    maxComponents: number = foldRmseMatrix.rows,
    nFolds: number = foldRmseMatrix.cols,
): MethodResult {
    const rmse = allocF64(foldRmseMatrix.data);
    try {
        return resultFromCall(ctx, "p4a_one_se_rule_compute", [
            handleOf(ctx),
            rmse.ptr,
            maxComponents,
            nFolds,
        ]);
    } finally {
        rmse.free();
    }
}

export function so_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    XBlocks: Matrix[],
    Y: Matrix,
    nComponentsPerBlock: ArrayLike<number>,
): MethodResult {
    const blocks = makeMatrixViewArray(XBlocks);
    const comps = allocI32(nComponentsPerBlock);
    try {
        return withMatrix(Y, (yPtr) => resultFromCall(
            ctx, "p4a_so_pls_fit",
            [
                handleOf(ctx),
                handleOf(cfg),
                blocks.viewPtr,
                XBlocks.length,
                yPtr,
                comps.ptr,
                asI64(comps.length),
            ],
        ));
    } finally {
        comps.free();
        blocks.free();
    }
}

export function on_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    XBlocks: Matrix[],
    nJoint: number,
    nUniquePerBlock: ArrayLike<number>,
): MethodResult {
    const blocks = makeMatrixViewArray(XBlocks);
    const uniques = allocI32(nUniquePerBlock);
    try {
        return resultFromCall(ctx, "p4a_on_pls_fit", [
            handleOf(ctx),
            handleOf(cfg),
            blocks.viewPtr,
            XBlocks.length,
            nJoint,
            uniques.ptr,
            asI64(uniques.length),
        ]);
    } finally {
        uniques.free();
        blocks.free();
    }
}

export function rosa_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    XBlocks: Matrix[],
    Y: Matrix,
    nComponents: number,
): MethodResult {
    const blocks = makeMatrixViewArray(XBlocks);
    try {
        return withMatrix(Y, (yPtr) => resultFromCall(
            ctx, "p4a_rosa_fit",
            [
                handleOf(ctx),
                handleOf(cfg),
                blocks.viewPtr,
                XBlocks.length,
                yPtr,
                nComponents,
            ],
        ));
    } finally {
        blocks.free();
    }
}

export function bagging_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    nEstimators: number,
    seed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_bagging_pls_fit",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            nEstimators,
            asU64(seed),
        ],
    ));
}

export function gpr_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    lengthScale: number,
    noiseLevel: number,
    seed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_gpr_pls_fit",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            lengthScale,
            noiseLevel,
            asU64(seed),
        ],
    ));
}

export function boosting_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    nEstimators: number,
    learningRate: number = 0.1,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_boosting_pls_fit",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            nEstimators,
            learningRate,
        ],
    ));
}

export function random_subspace_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    nEstimators: number,
    featuresPerSubspace: number,
    seed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_random_subspace_pls_fit",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            nEstimators,
            featuresPerSubspace,
            asU64(seed),
        ],
    ));
}

export function pls_glm_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    poisson: boolean = false,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_pls_glm_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, poisson ? 1 : 0],
    ));
}

export function pls_qda_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    yLabels: ArrayLike<number>,
): MethodResult {
    const labels = allocI32(yLabels);
    try {
        return withMatrix(X, (xPtr) => resultFromCall(
            ctx, "p4a_pls_qda_fit",
            [
                handleOf(ctx),
                handleOf(cfg),
                xPtr,
                labels.ptr,
                asI64(labels.length),
            ],
        ));
    } finally {
        labels.free();
    }
}

export function pls_cox_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    survivalTimes: ArrayLike<number>,
    eventIndicators: ArrayLike<number>,
): MethodResult {
    const times = allocF64(survivalTimes);
    const events = allocI32(eventIndicators);
    try {
        return withMatrix(X, (xPtr) => resultFromCall(
            ctx, "p4a_pls_cox_fit",
            [
                handleOf(ctx),
                handleOf(cfg),
                xPtr,
                times.ptr,
                asI64(times.length),
                events.ptr,
                asI64(events.length),
            ],
        ));
    } finally {
        events.free();
        times.free();
    }
}

export function pds_fit(
    ctx: HandleLike,
    XSource: Matrix,
    XTarget: Matrix,
    windowHalfWidth: number,
): MethodResult {
    return withTwoMatrices(XSource, XTarget, (xsPtr, xtPtr) => resultFromCall(
        ctx, "p4a_pds_fit",
        [handleOf(ctx), xsPtr, xtPtr, windowHalfWidth],
    ));
}

export function ds_fit(
    ctx: HandleLike,
    XSource: Matrix,
    XTarget: Matrix,
): MethodResult {
    return withTwoMatrices(XSource, XTarget, (xsPtr, xtPtr) => resultFromCall(
        ctx, "p4a_ds_fit",
        [handleOf(ctx), xsPtr, xtPtr],
    ));
}

export function mir_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_mir_pls_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr],
    ));
}

export function missing_aware_nipals_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_missing_aware_nipals_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr],
    ));
}

export function pls_diagnostics_compute(
    ctx: HandleLike,
    model: HandleLike,
    X: Matrix,
    XReference: Matrix | null = null,
): MethodResult {
    return withMatrix(X, (xPtr) => {
        if (XReference === null) {
            return resultFromCall(ctx, "p4a_pls_diagnostics_compute", [
                handleOf(ctx),
                handleOf(model),
                xPtr,
                0,
            ]);
        }
        return withMatrix(XReference, (xrPtr) => resultFromCall(
            ctx, "p4a_pls_diagnostics_compute",
            [handleOf(ctx), handleOf(model), xPtr, xrPtr],
        ));
    });
}

export function mb_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    blockSizes: ArrayLike<number | bigint>,
): MethodResult {
    const sizes = allocI64(blockSizes);
    try {
        return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
            ctx, "p4a_mb_pls_fit",
            [
                handleOf(ctx),
                handleOf(cfg),
                xPtr,
                yPtr,
                sizes.ptr,
                asI64(sizes.length),
            ],
        ));
    } finally {
        sizes.free();
    }
}

export function lw_pls_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    nNeighbors: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_lw_pls_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, nNeighbors],
    ));
}

export function pls_lda_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    yLabels: ArrayLike<number>,
    nClasses: number,
): MethodResult {
    const labels = allocI32(yLabels);
    try {
        return withMatrix(X, (xPtr) => resultFromCall(
            ctx, "p4a_pls_lda_fit",
            [
                handleOf(ctx),
                handleOf(cfg),
                xPtr,
                labels.ptr,
                asI64(labels.length),
                nClasses,
            ],
        ));
    } finally {
        labels.free();
    }
}

export function pls_logistic_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    yLabels: ArrayLike<number>,
    nClasses: number,
): MethodResult {
    const labels = allocI32(yLabels);
    try {
        return withMatrix(X, (xPtr) => resultFromCall(
            ctx, "p4a_pls_logistic_fit",
            [
                handleOf(ctx),
                handleOf(cfg),
                xPtr,
                labels.ptr,
                asI64(labels.length),
                nClasses,
            ],
        ));
    } finally {
        labels.free();
    }
}

export function aom_preprocess_fit(
    ctx: HandleLike,
    bank: HandleLike,
    gate: HandleLike,
    X: Matrix,
    Y: Matrix | null = null,
): MethodResult {
    return withMatrix(X, (xPtr) => {
        if (Y === null) {
            return resultFromCall(ctx, "p4a_aom_preprocess_fit", [
                handleOf(ctx),
                handleOf(bank),
                handleOf(gate),
                xPtr,
                0,
            ]);
        }
        return withMatrix(Y, (yPtr) => resultFromCall(
            ctx, "p4a_aom_preprocess_fit",
            [handleOf(ctx), handleOf(bank), handleOf(gate), xPtr, yPtr],
        ));
    });
}

export function variable_select_rank(
    ctx: HandleLike,
    model: HandleLike,
    X: Matrix,
    method: number,
    topK: number,
): MethodResult {
    return withMatrix(X, (xPtr) => resultFromCall(
        ctx, "p4a_variable_select_rank",
        [handleOf(ctx), handleOf(model), xPtr, method, topK],
    ));
}

export function interval_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    intervalWidth: number,
    step: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_interval_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            intervalWidth,
            step,
        ],
    ));
}

export function stability_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    topK: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_stability_select",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, handleOf(plan), topK],
    ));
}

export function uve_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    noiseFeatures: number,
    noiseSeed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_uve_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            noiseFeatures,
            asU64(noiseSeed),
        ],
    ));
}

export function spa_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    topK: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_spa_select",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, topK],
    ));
}

export function cars_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nIterations: number,
    minFeatures: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_cars_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nIterations,
            minFeatures,
        ],
    ));
}

export function random_frog_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nIterations: number,
    initialSize: number,
    minSize: number,
    maxSize: number,
    topK: number,
    seed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_random_frog_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nIterations,
            initialSize,
            minSize,
            maxSize,
            topK,
            asU64(seed),
        ],
    ));
}

export function scars_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nIterations: number,
    minFeatures: number,
    sampleFraction: number,
    seed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_scars_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nIterations,
            minFeatures,
            sampleFraction,
            asU64(seed),
        ],
    ));
}

export function ga_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nGenerations: number,
    populationSize: number,
    minFeatures: number,
    maxFeatures: number,
    mutationRate: number,
    seed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_ga_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nGenerations,
            populationSize,
            minFeatures,
            maxFeatures,
            mutationRate,
            asU64(seed),
        ],
    ));
}

export function pso_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nSwarm: number,
    nIterations: number,
    w: number = 0.729,
    c1: number = 1.494,
    c2: number = 1.494,
    vMax: number = 4.0,
    seed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_pso_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nSwarm,
            nIterations,
            w,
            c1,
            c2,
            vMax,
            asU64(seed),
        ],
    ));
}

export function vissa_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nIterations: number = 20,
    nSubmodels: number = 100,
    ratioKept: number = 0.1,
    threshold: number = 0.5,
    floorProbability: number = 0.01,
    seed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_vissa_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nIterations,
            nSubmodels,
            ratioKept,
            threshold,
            floorProbability,
            asU64(seed),
        ],
    ));
}

export function shaving_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nSteps: number,
    minFeatures: number,
    shaveFraction: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_shaving_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nSteps,
            minFeatures,
            shaveFraction,
        ],
    ));
}

export function bve_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nSteps: number,
    minFeatures: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_bve_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nSteps,
            minFeatures,
        ],
    ));
}

export function t2_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    alphaThresholds: ArrayLike<number>,
    minSelected: number,
): MethodResult {
    const alphas = allocF64(alphaThresholds);
    try {
        return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
            ctx, "p4a_t2_select",
            [
                handleOf(ctx),
                handleOf(cfg),
                xPtr,
                yPtr,
                handleOf(plan),
                alphas.ptr,
                asI64(alphas.length),
                minSelected,
            ],
        ));
    } finally {
        alphas.free();
    }
}

export function wvc_select(
    ctx: HandleLike,
    X: Matrix,
    Y: Matrix,
    nComponents: number,
    topK: number,
    normalize: boolean = true,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_wvc_select",
        [handleOf(ctx), xPtr, yPtr, nComponents, topK, normalize ? 1 : 0],
    ));
}

export function wvc_threshold_select(
    ctx: HandleLike,
    X: Matrix,
    Y: Matrix,
    nComponents: number,
    normalize: boolean = true,
    scoreThreshold: number = 0.0,
    thresholdFactor: number = 1.0,
    minSelected: number = 1,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_wvc_threshold_select",
        [
            handleOf(ctx),
            xPtr,
            yPtr,
            nComponents,
            normalize ? 1 : 0,
            scoreThreshold,
            thresholdFactor,
            minSelected,
        ],
    ));
}

export function emcuve_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    noiseFeatures: number,
    noiseSeed: number | bigint,
    nEnsembles: number,
    voteThreshold: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_emcuve_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            noiseFeatures,
            asU64(noiseSeed),
            nEnsembles,
            voteThreshold,
        ],
    ));
}

export function randomization_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    nPermutations: number,
    randomizationSeed: number | bigint = 0,
    alpha: number = 0.05,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_randomization_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            nPermutations,
            asU64(randomizationSeed),
            alpha,
        ],
    ));
}

export function bipls_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    intervalWidth: number,
    minIntervals: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_bipls_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            intervalWidth,
            minIntervals,
        ],
    ));
}

export function sipls_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    intervalWidth: number,
    combinationSize: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_sipls_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            intervalWidth,
            combinationSize,
        ],
    ));
}

export function rep_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nSteps: number,
    minFeatures: number,
    removeCount: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_rep_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nSteps,
            minFeatures,
            removeCount,
        ],
    ));
}

export function ipw_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nIterations: number,
    topK: number,
    damping: number,
    weightFloor: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_ipw_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nIterations,
            topK,
            damping,
            weightFloor,
        ],
    ));
}

export function st_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    thresholds: ArrayLike<number>,
    minSelected: number,
): MethodResult {
    const thr = allocF64(thresholds);
    try {
        return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
            ctx, "p4a_st_select",
            [
                handleOf(ctx),
                handleOf(cfg),
                xPtr,
                yPtr,
                handleOf(plan),
                thr.ptr,
                asI64(thr.length),
                minSelected,
            ],
        ));
    } finally {
        thr.free();
    }
}

export function ecr_fit(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    alpha: number,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_ecr_fit",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, alpha],
    ));
}

export function iriv_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    maxRounds: number,
    seed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_iriv_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            maxRounds,
            asU64(seed),
        ],
    ));
}

export function irf_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    plan: HandleLike,
    nIterations: number,
    windowSize: number,
    initialIntervals: number,
    topK: number,
    seed: number | bigint = 0,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_irf_select",
        [
            handleOf(ctx),
            handleOf(cfg),
            xPtr,
            yPtr,
            handleOf(plan),
            nIterations,
            windowSize,
            initialIntervals,
            topK,
            asU64(seed),
        ],
    ));
}

export function vip_spa_select(
    ctx: HandleLike,
    cfg: HandleLike,
    X: Matrix,
    Y: Matrix,
    topK: number,
    vipThreshold: number = 0.3,
): MethodResult {
    return withTwoMatrices(X, Y, (xPtr, yPtr) => resultFromCall(
        ctx, "p4a_vip_spa_select",
        [handleOf(ctx), handleOf(cfg), xPtr, yPtr, vipThreshold, topK],
    ));
}
