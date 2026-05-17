"""C++ direct baseline: calls p4a_pls_fit_simple via ctypes for the
purest possible timing (no Python pls4all wrapper overhead)."""
from _common import parse_args, time_runs, emit

import ctypes
import os
import numpy as np


def main():
    X, y, nc, runs = parse_args()
    lib_dir = os.environ.get("PLS4ALL_LIB_DIR",
                              "/home/delete/nirs4all/pls4all/build/dev-release/cpp/src")
    so = ctypes.CDLL(os.path.join(lib_dir, "libp4a.so"))
    so.p4a_pls_fit_simple.restype = ctypes.c_int
    so.p4a_pls_fit_simple.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]

    n, p = X.shape
    q = 1
    # Row-major contiguous buffers (already so for X; reshape y).
    Xrm = np.ascontiguousarray(X, dtype=np.float64)
    Yrm = np.ascontiguousarray(y.reshape(n, q), dtype=np.float64)
    coefs = np.zeros((p, q), dtype=np.float64)
    xm = np.zeros((p,), dtype=np.float64)
    ym = np.zeros((q,), dtype=np.float64)
    preds = np.zeros((n, q), dtype=np.float64)

    def do_one():
        # Reset output buffers each call (the C side overwrites them).
        rc = so.p4a_pls_fit_simple(
            Xrm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Yrm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n, p, q, nc,
            coefs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            xm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ym.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"p4a_pls_fit_simple returned status {rc}")

    rec = time_runs(do_one, runs)
    emit(rec)


if __name__ == "__main__":
    main()
