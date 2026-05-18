# SPDX-License-Identifier: CECILL-2.1
"""PCG64 RNG handle wrapper.

Used by stochastic augmenters that take an ``rng`` parameter at create time.
The augmenter does not own the RNG; the caller must keep the wrapper alive
for the augmenter's lifetime.
"""
from __future__ import annotations

import ctypes

from ._errors import check
from ._ffi import lib


class PCG64:
    """Owned :c:type:`c4a_rng_pcg64_state_t` handle.

    Seeding from a single ``seed`` reproduces ``np.random.default_rng(seed)``
    bit-for-bit; see ``c4a.h §7`` for the parity guarantees.
    """

    def __init__(self, seed: int) -> None:
        handle = ctypes.c_void_p()
        status = lib.c4a_rng_pcg64_create(ctypes.c_uint64(seed), ctypes.byref(handle))
        check(status, f"c4a_rng_pcg64_create(seed={seed})")
        self._handle = handle

    def set_seed(self, seed: int) -> None:
        check(lib.c4a_rng_pcg64_set_seed(self._handle, ctypes.c_uint64(seed)),
              "c4a_rng_pcg64_set_seed")

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None and self._handle.value:
            lib.c4a_rng_pcg64_destroy(self._handle)
            self._handle = ctypes.c_void_p(0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._handle


__all__ = ["PCG64"]
