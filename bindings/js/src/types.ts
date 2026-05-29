// SPDX-License-Identifier: CECILL-2.1

/** Float row-major or column-major matrix view passed to pls4all. */
export interface Matrix {
    /** Row-major data buffer. Length must be `rows * cols`. */
    data: Float64Array;
    rows: number;
    cols: number;
}

/** Mirror of the C enum n4m_status_t. */
export enum Status {
    OK = 0,
    ERR_INVALID_ARGUMENT = 1,
    ERR_NULL_POINTER = 2,
    ERR_OUT_OF_MEMORY = 3,
    ERR_DTYPE_MISMATCH = 4,
    ERR_SHAPE_MISMATCH = 5,
    ERR_DIM_MISMATCH = 6,
    ERR_NUMERICAL_FAILURE = 7,
    ERR_NOT_FITTED = 8,
    ERR_INTERNAL = 9,
    ERR_ABI_MISMATCH = 10,
    ERR_VERSION_INCOMPATIBLE = 11,
    ERR_BACKEND_UNAVAILABLE = 12,
    ERR_IO = 13,
    ERR_PERMISSION = 14,
    ERR_NOT_IMPLEMENTED = 15,
    ERR_TIMEOUT = 16,
    ERR_CANCELED = 17,
}

/** Mirror of n4m_dtype_t (cpp/include/n4m/n4m.h §2). */
export enum Dtype {
    UNKNOWN = 0,
    F64 = 1,
    F32 = 2,
    I32 = 3,
    I64 = 4,
}

/** Mirror of n4m_algorithm_t. Values must match cpp/include/pls4all/p4a.h. */
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

/** Mirror of n4m_solver_t. */
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

/** Mirror of n4m_deflation_t. */
export enum Deflation {
    REGRESSION = 0,
    CANONICAL = 1,
    X_ONLY = 2,
    XY = 3,
    ORTHOGONAL = 4,
}

export class Pls4allError extends Error {
    readonly status: number;
    constructor(status: number, message: string) {
        super(`pls4all error ${status}: ${message}`);
        this.status = status;
    }
}
