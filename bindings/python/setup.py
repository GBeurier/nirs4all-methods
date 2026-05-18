# SPDX-License-Identifier: CECILL-2.1
"""Setup hook for cibuildwheel.

The actual project metadata lives in :file:`pyproject.toml` (PEP 621). This
stub exists so that ``setuptools`` is invoked through the legacy entry point
when ``pip install -e .`` is run on Python distributions whose tooling has
not yet migrated to PEP 660.
"""
from setuptools import setup

setup()
