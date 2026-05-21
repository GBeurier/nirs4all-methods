// SPDX-License-Identifier: CECILL-2.1

/** Float row-major or column-major matrix view passed to pls4all. */
export interface Matrix {
    /** Row-major data buffer. Length must be `rows * cols`. */
    data: Float64Array;
    rows: number;
    cols: number;
}

/** Mirror of the C enum p4a_status_t. */
export enum Status {
    OK = 0,
    ERR_INVALID_ARGUMENT = 1,
    ERR_NULL_POINTER = 2,
    ERR_SHAPE_MISMATCH = 3,
    ERR_DTYPE_MISMATCH = 4,
    ERR_STRIDE_INVALID = 5,
    ERR_NOT_FITTED = 6,
    ERR_NUMERICAL_FAILURE = 7,
    ERR_CONVERGENCE_FAILED = 8,
    ERR_OUT_OF_MEMORY = 9,
    ERR_UNSUPPORTED = 10,
    ERR_NOT_IMPLEMENTED = 11,
    ERR_ABI_MISMATCH = 12,
    ERR_IO = 13,
    ERR_CORRUPT_BUFFER = 14,
    ERR_VERSION_INCOMPATIBLE = 15,
    ERR_BACKEND_UNAVAILABLE = 16,
    ERR_CANCELLED = 17,
    ERR_INTERNAL = 255,
}

/** Mirror of p4a_dtype_t. */
export enum Dtype {
    UNKNOWN = 0,
    F64 = 1,
    F32 = 2,
    I32 = 3,
    I64 = 4,
}

/** Mirror of p4a_algorithm_t. Values must match cpp/include/pls4all/p4a.h. */
export enum Algorithm {
    PLS_REGRESSION = 0,
    PLS_CANONICAL = 1,
    PLS_SVD = 2,
    PLS_DA = 3,
    OPLS = 4,
    OPLS_DA = 5,
    SPARSE_PLS = 6,
    MB_PLS = 7,
    LW_PLS = 8,
    AOM_PLS = 9,
    PCR = 10,
}

/** Mirror of p4a_solver_t. */
export enum Solver {
    NIPALS = 0,
    SIMPLS = 1,
    ORTHOGONAL_SCORES = 2,
    KERNEL_ALGORITHM = 3,
    WIDE_KERNEL = 4,
    SVD = 5,
    POWER = 6,
    RANDOMIZED_SVD = 7,
}

/** Mirror of p4a_deflation_t. */
export enum Deflation {
    REGRESSION = 0,
    CANONICAL = 1,
    X_ONLY = 2,
    XY = 3,
    ORTHOGONAL = 4,
}

/** Mirror of p4a_operator_kind_t. */
export enum OperatorKind {
    IDENTITY = 0,
    CENTER = 1,
    AUTOSCALE = 2,
    PARETO_SCALE = 3,
    SNV = 4,
    MSC = 5,
    EMSC = 6,
    DETREND_POLY = 7,
    SAVGOL_SMOOTH = 8,
    SAVGOL_DERIVATIVE = 9,
    NORRIS_WILLIAMS = 10,
    ASLS_BASELINE = 11,
    OSC = 12,
    EPO = 13,
    WAVELET_DENOISE = 14,
    FINITE_DIFFERENCE = 15,
    WHITTAKER = 16,
    FCK = 17,
}

/** Mirror of p4a_gating_mode_t. */
export enum GatingMode {
    HARD = 0,
    SOFT = 1,
    SPARSE = 2,
    PER_COMPONENT = 3,
    PER_BLOCK = 4,
    PER_TARGET = 5,
}

export class Pls4allError extends Error {
    readonly status: number;
    constructor(status: number, message: string) {
        super(`pls4all error ${status}: ${message}`);
        this.status = status;
    }
}
