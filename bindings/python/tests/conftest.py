# SPDX-License-Identifier: CECILL-2.1
"""pytest fixtures for the chemometrics4all binding test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]  # bindings/python/tests -> bindings -> repo
_PARITY_DIR = _REPO_ROOT / "parity"


# Make the parity comparator importable as ``parity.binding_parity``.
sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session")
def parity_dir() -> Path:
    return _PARITY_DIR


@pytest.fixture(scope="session")
def fixtures_dir(parity_dir: Path) -> Path:
    d = parity_dir / "fixtures"
    if not d.exists():
        pytest.skip(f"parity fixtures directory not found at {d}")
    return d
