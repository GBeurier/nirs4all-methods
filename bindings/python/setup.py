# SPDX-License-Identifier: CECILL-2.1
"""Setup hook for platform-tagged ctypes wheels.

The binding has no CPython extension module, but wheels include the native
``libc4a`` payload under ``chemometrics4all/lib``. Marking the distribution as
non-pure makes setuptools and cibuildwheel produce platform wheels and run the
repair step; ``scripts/retag_wheels.py`` then rewrites the interpreter/ABI tag
to ``py3-none-${platform}``.
"""
from setuptools import Distribution, setup
from setuptools.dist import Distribution as _Distribution


def _has_native_payload(self: _Distribution) -> bool:
    return True


Distribution.has_ext_modules = _has_native_payload  # type: ignore[method-assign]
Distribution.is_pure = lambda self: False  # type: ignore[method-assign]

setup()
