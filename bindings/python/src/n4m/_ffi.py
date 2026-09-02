# SPDX-License-Identifier: CECILL-2.1
"""Single point of access to libn4m via :mod:`ctypes`.

The module discovers the shared library on import and binds ``argtypes`` /
``restype`` for every exported symbol declared in ``n4m.h``. The auto-
generated declarations live in :mod:`._ffi_decls` and are produced by
``scripts/generate_ffi_decls.py``.

Discovery order (highest priority first):

  1. ``N4M_LIB_PATH`` environment variable (full path to the shared
     library, or a directory to search).
  2. ``<site-packages>/<dist>.libs/`` — the sibling directory that
     ``auditwheel`` (Linux), ``delocate`` (macOS) and ``delvewheel``
     (Windows) graft the repaired, content-addressed library into. The
     Python *package* is ``n4m`` but the *distribution* is
     ``nirs4all-methods``, so the repaired dir is ``nirs4all_methods.libs``
     — never ``n4m.libs``. We therefore glob **all** ``*.libs`` siblings
     rather than assuming the package name.
  3. ``<package_dir>/lib/`` — wheel layout when libn4m is bundled directly
     as package data (the cibuildwheel fallback when no repair tool runs).
  4. Development fallback: ``<repo>/build/{dev-release,dev-debug,...}/cpp/src``.
  5. System search path (``ctypes.util.find_library`` / DLL search path).

The library is matched by glob (``libn4m*.so*`` / ``libn4m*.dylib`` /
``n4m*.dll`` / ``libn4m*.dll``) so a versioned SONAME (e.g.
``libn4m.so.1.10.0``) or an auditwheel-hashed name (``libn4m-7c4d2a1f.so.1``)
is picked up without hard-coding a version string.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

from . import _ffi_decls

_PKG_DIR = Path(__file__).resolve().parent

# Match the N4M_ABI_VERSION_* macros in cpp/include/n4m/n4m_version.h.
# Bumped additively with each minor ABI change; kept in sync with the header
# (see scripts/bump_version.sh, which fails --check on drift).
ABI_VERSION_MAJOR = 2
ABI_VERSION_MINOR = 5
ABI_VERSION_PATCH = 0
ABI_VERSION_STRING = f"{ABI_VERSION_MAJOR}.{ABI_VERSION_MINOR}.{ABI_VERSION_PATCH}"

# auditwheel renames the bundled library to ``libn4m-<8hexhash>.so.2``, so we
# accept ``libn4m*.so*`` in addition to the plain SONAME forms. On Windows the
# MSVC DLL is ``n4m.dll`` (CMake OUTPUT_NAME "n4m", no lib prefix); MinGW
# produces ``libn4m.dll``. Match both.
_LIB_PATTERNS = ("libn4m*.so*", "libn4m*.dylib", "n4m*.dll", "libn4m*.dll")

_loaded_path: Path | None = None


def _glob_libs(directory: Path) -> list[Path]:
    """Return all libn4m candidates under ``directory``, sorted for determinism."""
    found: list[Path] = []
    if not directory.is_dir():
        return found
    for pattern in _LIB_PATTERNS:
        found.extend(sorted(directory.glob(pattern)))
    return found


def _sibling_libs_dirs() -> list[Path]:
    """All ``*.libs`` sibling dirs of the package (auditwheel/delocate/delvewheel).

    The repair tools name the dir after the *distribution* (``nirs4all-methods``
    -> ``nirs4all_methods.libs``), not the package (``n4m``), so we glob every
    ``*.libs`` sibling rather than assuming a name.
    """
    parent = _PKG_DIR.parent
    dirs = [d for d in parent.glob("*.libs") if d.is_dir()]
    # Prefer THIS distribution's own repaired dir ("nirs4all-methods" ->
    # "nirs4all_methods.libs") over any other package's "*.libs" sibling that
    # might also contain an ABI-compatible libn4m and shadow ours.
    dirs.sort(key=lambda d: (d.name != "nirs4all_methods.libs", d.name))
    return dirs


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []

    # (1) Explicit override always wins. May be a file or a directory.
    env = os.environ.get("N4M_LIB_PATH")
    if env:
        envp = Path(env)
        if envp.is_file():
            paths.append(envp)
        else:
            paths.extend(_glob_libs(envp))

    # (2) auditwheel/delocate/delvewheel sibling ``<dist>.libs/``.
    for libs_dir in _sibling_libs_dirs():
        paths.extend(_glob_libs(libs_dir))

    # (3) Direct wheel layout: package_data ships libn4m under n4m/lib/.
    paths.extend(_glob_libs(_PKG_DIR / "lib"))

    # (4) Developer convenience: repo-root build directories.
    here = _PKG_DIR
    for _ in range(8):
        here = here.parent
        if not here or here == here.parent:
            break
        for preset in (
            "dev-release",
            "dev-debug",
            "blas-omp",
            "ci-linux-gcc12-release",
        ):
            paths.extend(_glob_libs(here / "build" / preset / "cpp" / "src"))

    # (5) System search path.
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        sys_path = ctypes.util.find_library("n4m")
        if sys_path:
            paths.append(Path(sys_path))
    elif sys.platform == "win32":
        for name in ("n4m.dll", "libn4m.dll"):
            paths.append(Path(name))
    return paths


def _load() -> ctypes.CDLL:
    global _loaded_path
    errors: list[tuple[Path, OSError]] = []
    candidates = _candidate_paths()
    for path in candidates:
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:  # pragma: no cover - host-specific
            errors.append((path, exc))
            continue
        # A globbed candidate may be some other shared object that loaded but
        # is not libn4m (missing symbols -> AttributeError). Treat that like a
        # load failure and continue rather than aborting import.
        try:
            _ffi_decls.apply_argtypes(lib)
            # Verify ABI compatibility: the loaded library must expose MAJOR ==
            # ours and MINOR >= ours. ``n4m_check_abi_compatibility`` returns
            # N4M_OK (0) on success. A too-old library is skipped so a stale dev
            # build cannot shadow a compatible one later in the list.
            status = lib.n4m_check_abi_compatibility(
                ABI_VERSION_MAJOR, ABI_VERSION_MINOR
            )
        except (AttributeError, OSError) as exc:
            errors.append((path, OSError(f"not a usable libn4m ({exc})")))
            continue
        if status != 0:
            errors.append(
                (
                    path,
                    OSError(
                        f"ABI mismatch: library rejected header v{ABI_VERSION_STRING} (status {status})"
                    ),
                )
            )
            continue
        _loaded_path = path
        return lib

    searched = (
        "\n".join(f"  - {p}: {e}" for p, e in errors) or "  (no candidates found)"
    )
    raise FileNotFoundError(
        f"libn4m (ABI {ABI_VERSION_MAJOR}.x, >= {ABI_VERSION_STRING}) could not be loaded.\n"
        f"Tried:\n{searched}\n"
        "Set N4M_LIB_PATH to the absolute path of libn4m.so / libn4m.dylib / n4m.dll."
    )


lib: ctypes.CDLL = _load()
"""The loaded libn4m handle. Argtypes/restype are already bound."""


def library_path() -> str:
    """Path of the loaded library (for diagnostics)."""
    return str(_loaded_path) if _loaded_path is not None else ""


__all__ = [
    "ABI_VERSION_MAJOR",
    "ABI_VERSION_MINOR",
    "ABI_VERSION_PATCH",
    "ABI_VERSION_STRING",
    "lib",
    "library_path",
]
