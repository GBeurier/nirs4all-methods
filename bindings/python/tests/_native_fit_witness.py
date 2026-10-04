# SPDX-License-Identifier: CECILL-2.1
"""Mandatory test-only native FIT probe; never imported by product bindings."""

from __future__ import annotations

import ctypes
import json
import os
from contextlib import contextmanager
from pathlib import Path

from n4m._ffi import lib


class NativeFitProbe:
    def __init__(self, path: Path, *, serialized: bool, capacity: int):
        if not path.is_file():
            raise RuntimeError(f"Native FIT witness library is unavailable: {path}")
        self.path = path.resolve()
        self.native = ctypes.CDLL(str(self.path))
        reset = self.native.n4m_test_native_fit_reset
        reset.argtypes = [ctypes.c_int, ctypes.c_uint64]
        reset.restype = ctypes.c_int
        snapshot = self.native.n4m_test_native_fit_snapshot
        snapshot.argtypes = []
        snapshot.restype = ctypes.c_char_p
        address = self.native.n4m_test_native_fit_real_address
        address.argtypes = [ctypes.c_int]
        address.restype = ctypes.c_uint64
        self.functions = {}
        for kind, name in enumerate(("role", "multimodal"), 1):
            original = getattr(lib, "n4m_" + name + "_pipeline_fit")
            # Refuse a shim linked to another engine instance/runtime artifact.
            if address(kind) != ctypes.cast(original, ctypes.c_void_p).value:
                raise RuntimeError("Native FIT witness is not linked to the loaded Methods engine")
            function = getattr(self.native, "n4m_test_probe_" + name + "_fit")
            function.argtypes = original.argtypes
            function.restype = original.restype
            self.functions[name] = function
        if type(serialized) is not bool or type(capacity) is not int or not 1 <= capacity <= 65536:
            raise RuntimeError("Native FIT witness requires boolean serialization and capacity 1..65536")
        if reset(int(serialized), capacity) != 0:
            raise RuntimeError("Native FIT witness reset refused; all workers must be joined and inactive")

    def snapshot(self) -> dict:
        """Read only after all admitted workers join; refuse truncation/active calls."""
        raw = self.native.n4m_test_native_fit_snapshot()
        if raw is None:
            raise RuntimeError("Native FIT witness snapshot requires all workers joined/inactive")
        result = json.loads(raw)
        if result["active"] or result["inflight"] or result["overflow"]:
            raise RuntimeError("Native FIT witness incomplete: active calls or bounded-store overflow")
        if len(result["records"]) != result["total_calls"]:
            raise RuntimeError("Native FIT witness record count is incomplete")
        return result


@contextmanager
def witness(monkeypatch, *, serialized: bool = False, capacity: int = 65536):
    """Route real public FIT calls through native probes with automatic restoration.

    Use around real SDK/DAG HPO windows. Keep capability admission on the
    production engine; this test library only probes FIT. No sleep, fake model
    callback, environment mutation, kernel or Python timing is used. Positive
    evidence describes overlapping in-flight native calls, not CPU utilization.
    """
    configured = os.environ.get("N4M_NATIVE_FIT_WITNESS_LIBRARY")
    if not configured:
        raise RuntimeError("Mandatory native FIT witness requires N4M_NATIVE_FIT_WITNESS_LIBRARY")
    probe = NativeFitProbe(Path(configured), serialized=serialized, capacity=capacity)
    with monkeypatch.context() as scoped:
        for name, function in probe.functions.items():
            scoped.setattr(lib, "n4m_" + name + "_pipeline_fit", function)
        yield probe
