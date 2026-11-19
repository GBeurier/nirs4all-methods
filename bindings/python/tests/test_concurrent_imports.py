# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Cold concurrent imports must not eagerly acquire sibling module locks."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "modules", [("scatter", "smoothing"), ("smoothing", "scatter")]
)
def test_cold_concurrent_transform_imports(modules):
    """Reproduce the module-lock ownership of two first submodule imports."""
    code = textwrap.dedent(
        """
        import importlib
        from importlib import _bootstrap
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        assert "n4m" not in sys.modules
        barrier = Barrier(2)

        def load(name):
            # CPython normally acquires the requested child's lock before its
            # parents. Hold both child locks before either parent import starts
            # so this reproduces the cold-import race without timing sleeps.
            with _bootstrap._ModuleLockManager(name):
                barrier.wait(timeout=10)
                return importlib.import_module(name)

        names = ["n4m.transform." + name for name in sys.argv[1:]]
        with ThreadPoolExecutor(max_workers=2) as executor:
            loaded = list(executor.map(load, names))
        assert [module.__name__ for module in loaded] == names
        from n4m.transform.scatter import SNV
        from n4m.transform.smoothing import SavitzkyGolay
        assert SNV is not None and SavitzkyGolay is not None
        """
    )
    process = subprocess.run(
        [sys.executable, "-c", code, *modules],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_transform_public_submodule_attributes():
    """Attribute access, from-imports and discovery retain the public surface."""
    import n4m

    namespace = {}
    exec("from n4m.transform import *", namespace)  # noqa: S102 - literal public-API probe
    for name in n4m.transform.__all__:
        assert name in dir(n4m.transform)
        assert namespace[name] is getattr(n4m.transform, name)
    with pytest.raises(AttributeError, match="no attribute 'missing_role'"):
        _ = n4m.transform.missing_role
