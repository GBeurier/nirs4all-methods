#!/usr/bin/env python3
"""Run binding contracts against the selected actual shared-library artifact."""

import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
preset = os.environ.get("N4M_ANALYSIS_PRESET", "analysis-all")
libdir = root / "build" / preset / "cpp" / "src"
paths = [
    libdir / "libn4m.so",
    libdir / "libn4m.dylib",
    libdir / "Release" / "n4m.dll",
    libdir / "n4m.dll",
]
library = next((p for p in paths if p.is_file()), None)
if library is None:
    raise SystemExit(f"No compiled analysis library under {libdir}")
env = {
    **os.environ,
    "N4M_LIB_PATH": str(library),
    "PYTHONPATH": str(root / "bindings/python/src"),
}
subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "bindings/python/tests/test_component_analysis.py",
        "-q",
    ],
    cwd=root,
    env=env,
    check=True,
)
