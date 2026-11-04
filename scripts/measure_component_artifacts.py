#!/usr/bin/env python3
"""Measure actual same-source optimized engine artifacts and built wheels.

Run all four analysis-* presets first. Uses a temporary generated package copy;
never stages a library into tracked sources. Linux native measurement uses strip
--strip-unneeded, exactly as wheel preparation. No static archive size estimates.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
subprocess.run(
    [
        sys.executable,
        str(root / "bindings/python/scripts/make_python_package.py"),
        "--name",
        "nirs4all-methods",
    ],
    check=True,
    cwd=root,
)
version_header = (root / "cpp/include/n4m/n4m_version.h").read_text()
abi = ".".join(
    re.search(r"#define N4M_ABI_VERSION_" + part + r"\s+(\d+)", version_header)[1]
    for part in ["MAJOR", "MINOR", "PATCH"]
)
results = {"platform": sys.platform, "abi": abi, "native": {}, "wasm": {}}
with tempfile.TemporaryDirectory(prefix="n4m-analysis-size-") as scratch:
    scratch = Path(scratch)
    package = scratch / "package"
    shutil.copytree(
        root / "bindings/python_nirs4all_methods",
        package,
        ignore=shutil.ignore_patterns("lib", "build", "*.egg-info", "dist"),
    )
    libdir = package / "src/n4m/lib"
    libdir.mkdir(parents=True)
    for label in ["baseline", "mixed", "fanova", "all"]:
        source = root / f"build/analysis-{label}/cpp/src/libn4m.so.{abi}"
        target = libdir / "libn4m.so"
        shutil.copyfile(source, target)
        unstripped = target.stat().st_size
        subprocess.run(["strip", "--strip-unneeded", str(target)], check=True)
        data = target.read_bytes()
        out = scratch / label
        out.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(out),
                str(package),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        wheel = next(out.glob("*.whl"))
        results["native"][label] = {
            "release_library_bytes": unstripped,
            "stripped_library_bytes": len(data),
            "library_gzip_bytes": len(gzip.compress(data, compresslevel=9, mtime=0)),
            "wheel_bytes": wheel.stat().st_size,
        }
        # Prove this actual wheel payload loads and executes without a Rust toolchain.
        if label == "all":
            venv = scratch / "installed"
            subprocess.run(
                [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
                check=True,
            )
            py = venv / "bin/python"
            subprocess.run(
                [str(py), "-m", "pip", "install", str(wheel)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    str(py),
                    "-c",
                    """from n4m import component_analysis as c
import numpy as np
assert c.normalize_loss([2],[4]).tolist()==[.5]
g=np.repeat(np.arange(5,dtype=np.int64),4)
x=np.tile([-1.5,-.5,.5,1.5],5)
y=3+.75*x+np.repeat([-2.,-1.,0.,1.,2.],4)+np.tile([.1,-.1,-.1,.1],5)
r=c.mixed_effects(np.column_stack([np.ones(20),x]),y,g)
assert r['state']=='ready',r
assert np.max(np.abs(np.array(r['beta'])-[3,.75]))<1e-7
X=np.tile([[0,0],[0,1],[1,0],[1,1]],(16,1))
f=c.fanova(X,X[:,0]+2*X[:,1],[2,2],[[0,1],[0,1]],n_trees=12,min_samples_leaf=1,max_depth=4)
assert f['state']=='ready',f
assert abs(f['feature_importance'][0]['fraction']-.2)<.025
print('installed wheel: both static engines fitted successfully')""",
                ],
                check=True,
                cwd=scratch,
            )
    for label, preset in [
        ("baseline", "emscripten"),
        ("mixed", "emscripten-mixed"),
        ("fanova", "emscripten-fanova"),
        ("all", "emscripten-analysis"),
    ]:
        source = root / f"build/{preset}/bindings/js/n4m.wasm"
        if source.exists():
            data = source.read_bytes()
            results["wasm"][label] = {
                "wasm_bytes": len(data),
                "wasm_gzip_bytes": len(gzip.compress(data, compresslevel=9, mtime=0)),
            }
for group in results.values():
    if isinstance(group, dict) and "baseline" in group:
        base = group["baseline"]
        for label, values in group.items():
            if label != "baseline":
                values["increment_bytes"] = {
                    key: value - base[key] for key, value in values.copy().items()
                }
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
