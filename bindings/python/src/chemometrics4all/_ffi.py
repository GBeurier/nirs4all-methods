# SPDX-License-Identifier: CECILL-2.1
"""Single point of access to libc4a via :mod:`ctypes`.

The module discovers the shared library on import and binds ``argtypes`` /
``restype`` for every exported symbol declared in ``c4a.h``. The auto-
generated declarations live in :mod:`._ffi_decls` and are produced by
``scripts/generate_ffi_decls.py``.

Discovery order:

  1. ``CHEMOMETRICS4ALL_LIB_PATH`` environment variable (full path to the
     shared library or its versioned symlink).
  2. ``<package_dir>/lib/`` — bundled with the wheel.
  3. Development fallback: ``<repo>/build/dev-debug/cpp/src/`` searched
     relative to the package directory.

The lookup is OS-aware: ``libc4a.so.1.9.0`` on Linux, ``libc4a.1.9.0.dylib``
on macOS, ``c4a.dll`` on Windows.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path

from . import _ffi_decls

_PKG_DIR = Path(__file__).resolve().parent

# Match the C4A_ABI_VERSION_* macros in cpp/include/chemometrics4all/c4a_version.h.
ABI_VERSION_MAJOR = 1
ABI_VERSION_MINOR = 9
ABI_VERSION_PATCH = 0
ABI_VERSION_STRING = f"{ABI_VERSION_MAJOR}.{ABI_VERSION_MINOR}.{ABI_VERSION_PATCH}"


def _candidate_library_names() -> list[str]:
    system = platform.system()
    if system == "Linux":
        return [
            f"libc4a.so.{ABI_VERSION_STRING}",
            f"libc4a.so.{ABI_VERSION_MAJOR}",
            "libc4a.so",
        ]
    if system == "Darwin":
        return [
            f"libc4a.{ABI_VERSION_STRING}.dylib",
            f"libc4a.{ABI_VERSION_MAJOR}.dylib",
            "libc4a.dylib",
        ]
    if system == "Windows":
        return ["c4a.dll", "libc4a.dll"]
    return ["libc4a.so", "libc4a.dylib", "c4a.dll"]


def _candidate_search_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("CHEMOMETRICS4ALL_LIB_PATH")
    if env:
        # The env var may point to either a directory or a full library path.
        envp = Path(env)
        if envp.is_file():
            paths.append(envp.parent)
            paths.append(envp)
        else:
            paths.append(envp)
    paths.append(_PKG_DIR / "lib")
    # Development fallbacks: walk upward looking for a chemometrics4all checkout.
    here = _PKG_DIR
    for _ in range(8):
        here = here.parent
        if not here or here == here.parent:
            break
        dev = here / "build" / "dev-debug" / "cpp" / "src"
        if dev.exists():
            paths.append(dev)
        ci = here / "build" / "ci-linux-gcc12-release" / "cpp" / "src"
        if ci.exists():
            paths.append(ci)
    return paths


def _resolve_library_path() -> Path:
    env = os.environ.get("CHEMOMETRICS4ALL_LIB_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    names = _candidate_library_names()
    for directory in _candidate_search_paths():
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    searched = ", ".join(str(p) for p in _candidate_search_paths())
    raise FileNotFoundError(
        f"libc4a not found. Searched: {searched}. Set CHEMOMETRICS4ALL_LIB_PATH "
        f"to the absolute path of libc4a.so.{ABI_VERSION_STRING} (or the "
        "matching .dylib / .dll). Looked for: " + ", ".join(names)
    )


def _load() -> ctypes.CDLL:
    path = _resolve_library_path()
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:  # pragma: no cover - host-specific
        raise RuntimeError(f"Failed to load libc4a from {path}: {exc}") from exc
    _ffi_decls.apply_argtypes(lib)
    # Verify ABI compatibility: the loaded library must expose a MAJOR ==
    # ours and a MINOR >= ours. ``c4a_check_abi_compatibility`` returns
    # C4A_OK on success.
    status = lib.c4a_check_abi_compatibility(ABI_VERSION_MAJOR, ABI_VERSION_MINOR)
    if status != 0:
        raise RuntimeError(
            f"ABI mismatch: library at {path} rejected header "
            f"v{ABI_VERSION_STRING} with status {status}."
        )
    return lib


lib: ctypes.CDLL = _load()
"""The loaded libc4a handle. Argtypes/restype are already bound."""


def library_path() -> str:
    """Path of the loaded library (for diagnostics)."""
    # NB: ctypes doesn't expose the path on every platform, so re-resolve.
    return str(_resolve_library_path())


__all__ = [
    "ABI_VERSION_MAJOR",
    "ABI_VERSION_MINOR",
    "ABI_VERSION_PATCH",
    "ABI_VERSION_STRING",
    "lib",
    "library_path",
]
