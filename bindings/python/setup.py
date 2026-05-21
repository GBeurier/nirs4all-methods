# SPDX-License-Identifier: CECILL-2.1
#
# setup.py shim — declares the package as **non-pure** so cibuildwheel
# and setuptools treat it as platform-specific.
#
# This binding is **ctypes-only**: there is no CPython extension to
# compile. By default, setuptools detects this absence and produces a
# pure-Python wheel tagged ``py3-none-any``. That tag is wrong for us
# because:
#
# 1. The wheel embeds a native shared library (libn4m) shipped under
#    ``pls4all/lib/``, so the artefact is platform-specific.
# 2. cibuildwheel skips its ``CIBW_REPAIR_WHEEL_COMMAND_*`` step on
#    pure wheels — auditwheel / delocate / delvewheel never runs and
#    the manylinux policy is not enforced.
#
# Overriding ``Distribution.has_ext_modules`` flips both behaviours:
# setuptools generates a platform-tagged wheel
# (``cp3X-cp3X-manylinux_2_31_x86_64`` initially), and cibuildwheel
# runs the repair pass. scripts/retag_wheels.py then rewrites the
# filename + ``WHEEL`` metadata + ``RECORD`` to the canonical
# ``py3-none-${platform}`` form so the wheel applies to every
# CPython supported by ``requires-python``.

from setuptools import Distribution, setup
from setuptools.dist import Distribution as _Distribution


def _has_native_payload(self: _Distribution) -> bool:
    return True


# Force the platform-tagged wheel: setuptools.bdist_wheel checks both
# ``Distribution.has_ext_modules()`` and ``Distribution.is_pure()``.
Distribution.has_ext_modules = _has_native_payload  # type: ignore[method-assign]
Distribution.is_pure = lambda self: False           # type: ignore[method-assign]

# All other metadata (name, version, license, dependencies, package_data, …)
# comes from pyproject.toml. Calling setup() with no kwargs lets the
# declarative metadata drive the build.
setup()
