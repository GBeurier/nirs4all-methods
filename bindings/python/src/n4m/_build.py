# SPDX-License-Identifier: CECILL-2.1
"""Compile-profile metadata read through the existing public C ABI."""

from __future__ import annotations

import ctypes
import re
from dataclasses import dataclass

from ._ffi import lib

_STAMP_PREFIX = "n4m-build-capabilities-"
_STAMP_V1 = re.compile(r"n4m-build-capabilities-v1;blas=([01]);openmp=([01]);cuda=([01])")


@dataclass(frozen=True)
class BuildCapabilities:
    """Flags compiled into the loaded numerical core, not device availability.

    ``sequential_cpu`` identifies the reference build without accelerated
    backends. It does not promise that Context.num_threads caps BLAS/OpenMP or
    CUDA: callers must refuse those profiles when they require one numerical
    thread per independently owned pipeline.
    """

    schema_version: int
    blas: bool
    openmp: bool
    cuda: bool

    @property
    def sequential_cpu(self) -> bool:
        """Whether all optional numerical acceleration was compiled out."""
        return not (self.blas or self.openmp or self.cuda)


def build_info() -> str:
    """Return native descriptive build information and its capability stamp."""
    raw = ctypes.c_char_p(lib.n4m_get_build_info()).value
    if raw is None:
        raise RuntimeError("libn4m returned null build information")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("libn4m build information is not UTF-8") from error


def build_capabilities() -> BuildCapabilities:
    """Read a complete v1 compile stamp, refusing missing or ambiguous evidence.

    Older runtimes remain loadable; only this capability query refuses an
    unstamped build. Accelerated profiles are reported accurately even when
    no accelerator is available at runtime. No environment or context changes
    are made by this reader.
    """
    lines = [line for line in build_info().split("\n") if line.startswith(_STAMP_PREFIX)]
    match = _STAMP_V1.fullmatch(lines[0]) if len(lines) == 1 else None
    if match is None:
        raise RuntimeError("libn4m requires one complete supported build capability stamp v1")
    blas, openmp, cuda = (flag == "1" for flag in match.groups())
    return BuildCapabilities(schema_version=1, blas=blas, openmp=openmp, cuda=cuda)
