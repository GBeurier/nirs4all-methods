#!/usr/bin/env python3
"""Ecosystem E2E gate for nirs4all-methods cross-binding parity.

This script is intentionally a thin orchestrator around the repository's
existing parity runners. It does not implement numerical logic itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_R_ENV = Path("/home/delete/miniconda3/envs/pls4all_r")
REQUIRED_BACKENDS = ("cpp", "python_tier1", "r_tier1", "ref_python_scikit_learn")
STRICT_TOLERANCES = {
    "binding_parity_max_diff": 1e-12,
    "reference_parity_rmse_rel": 1e-12,
    "wasm_rmse_rel": 1e-12,
}


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path = REPO,
         timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def _tail(text: str, lines: int = 20) -> list[str]:
    return text.strip().splitlines()[-lines:]


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _ensure_dev_release_build(timeout: int) -> dict[str, Any]:
    lib = REPO / "build/dev-release/cpp/src/libn4m.so"
    cli = REPO / "build/dev-release/cpp/cli/n4m_cli"

    env = os.environ.copy()
    env.setdefault("FC", "/nonexistent")
    if not (REPO / "build/dev-release/CMakeCache.txt").exists():
        configure = _run(["cmake", "--preset", "dev-release"], env=env, timeout=timeout)
        if configure.returncode != 0:
            raise RuntimeError(
                "cmake --preset dev-release failed:\n"
                + "\n".join(_tail(configure.stdout + "\n" + configure.stderr))
            )
    build = _run(
        [
            "cmake",
            "--build",
            "--preset",
            "dev-release",
            "--target",
            "n4m_c",
            "n4m_cli",
            "--parallel",
        ],
        env=env,
        timeout=timeout,
    )
    if build.returncode != 0:
        raise RuntimeError(
            "cmake build dev-release n4m_c/n4m_cli failed:\n"
            + "\n".join(_tail(build.stdout + "\n" + build.stderr))
    )
    if not lib.exists():
        raise RuntimeError(f"dev-release libn4m missing after build: {lib}")
    return {"build_invoked": True, "lib": str(lib), "cli": str(cli)}


def _prepend_env_paths(env: dict[str, str], name: str, paths: list[Path]) -> None:
    existing = env.get(name, "")
    parts = [str(path) for path in paths if path.exists()]
    if existing:
        parts.append(existing)
    if parts:
        env[name] = os.pathsep.join(parts)


def _prepend_r_library_env(env: dict[str, str], r_lib: Path) -> None:
    _prepend_env_paths(env, "R_LIBS", [r_lib])
    if env.get("R_LIBS_USER"):
        _prepend_env_paths(env, "R_LIBS_USER", [r_lib])


def _base_env(r_lib: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        part
        for part in (
            str(REPO),
            str(REPO / "bindings/python/src"),
            env.get("PYTHONPATH", ""),
        )
        if part
    )
    if "N4M_R_ENV" not in env and DEFAULT_R_ENV.exists():
        env["N4M_R_ENV"] = str(DEFAULT_R_ENV)
    r_env = Path(env["N4M_R_ENV"]) if env.get("N4M_R_ENV") else None
    if r_env is not None and r_env.exists():
        _prepend_env_paths(env, "PATH", [r_env / "bin"])
        _prepend_env_paths(
            env,
            "LD_LIBRARY_PATH",
            [
                REPO / "build/dev-release/cpp/src",
                r_env / "lib",
                r_env / "lib/octave/10.3.0",
            ],
        )
    if r_lib is not None:
        r_lib.mkdir(parents=True, exist_ok=True)
        env["N4M_R_GATE_LIB"] = str(r_lib)
        _prepend_r_library_env(env, r_lib)
    return env


def _resolve_rscript(env: dict[str, str]) -> Path:
    override = env.get("N4M_RSCRIPT")
    if override:
        path = Path(override)
        if path.exists():
            return path
        raise RuntimeError(f"N4M_RSCRIPT points to a missing executable: {path}")
    found = shutil.which("Rscript", path=env.get("PATH"))
    if found:
        return Path(found)
    r_env = Path(env["N4M_R_ENV"]) if env.get("N4M_R_ENV") else None
    if r_env is not None:
        candidate = r_env / "bin/Rscript"
        if candidate.exists():
            return candidate
    raise RuntimeError("Rscript not found; set N4M_R_ENV or N4M_RSCRIPT")


def _r_binary_from_rscript(rscript: Path) -> Path:
    r_binary = rscript.with_name("R")
    if r_binary.exists():
        return r_binary
    found = shutil.which("R")
    if found:
        return Path(found)
    raise RuntimeError(f"R binary not found next to {rscript}")


def _prepare_r_gate_library(artifacts_dir: Path, timeout: int) -> dict[str, Any]:
    r_lib = artifacts_dir / "_r-lib"
    if r_lib.exists():
        shutil.rmtree(r_lib)
    r_lib.mkdir(parents=True, exist_ok=True)

    env = _base_env(r_lib)
    env["N4M_R_LINK_PREBUILT"] = "1"
    env["PLS4ALL_LIB_DIR"] = str(REPO / "build/dev-release/cpp/src")
    env["PLS4ALL_GENERATED_DIR"] = str(REPO / "build/dev-release/generated")
    env["PLS4ALL_INCLUDE_DIR"] = str(REPO / "cpp/include")

    makevars = artifacts_dir / "r-gate-Makevars"
    makevars.write_text(
        "CC=gcc\n"
        "CXX=g++\n"
        "CXX11=g++\n"
        "CXX14=g++\n"
        "CXX17=g++\n"
        "CXX17STD=-std=gnu++17\n",
        encoding="utf-8",
    )
    env["R_MAKEVARS_USER"] = str(makevars)

    rscript = _resolve_rscript(env)
    r_binary = _r_binary_from_rscript(rscript)
    package_dir = REPO / "bindings/r/pls4all"
    install = _run(
        [str(r_binary), "CMD", "INSTALL", "--preclean", "--library", str(r_lib), str(package_dir)],
        env=env,
        timeout=timeout,
    )
    install_transcript = install.stdout + ("\n" + install.stderr if install.stderr else "")
    if install.returncode != 0:
        raise RuntimeError("R CMD INSTALL bindings/r/pls4all failed:\n" + "\n".join(_tail(install_transcript)))

    probe_expr = (
        ".libPaths(c(Sys.getenv('N4M_R_GATE_LIB'), .libPaths())); "
        "suppressPackageStartupMessages(library(pls4all)); "
        "abi <- paste(n4m_abi_version(), collapse='.'); "
        "cat('N4A_R_GATE', as.character(packageVersion('pls4all')), abi, "
        "find.package('pls4all'), sep='\\t'); cat('\\n'); "
        "if (strsplit(abi, '.', fixed=TRUE)[[1]][1] != '2') "
        "stop(sprintf('pls4all ABI %s is not ABI major 2', abi))"
    )
    probe = _run([str(rscript), "-e", probe_expr], env=env, timeout=60)
    probe_transcript = probe.stdout + ("\n" + probe.stderr if probe.stderr else "")
    if probe.returncode != 0:
        raise RuntimeError("R pls4all ABI preflight failed:\n" + "\n".join(_tail(probe_transcript)))
    marker = next((line for line in probe.stdout.splitlines() if line.startswith("N4A_R_GATE\t")), "")
    if not marker:
        raise RuntimeError("R pls4all ABI preflight did not emit N4A_R_GATE marker")
    _, version, abi, package_path = marker.split("\t", 3)
    return {
        "package": "pls4all",
        "version": version,
        "abi": abi,
        "rscript": str(rscript),
        "library": str(r_lib),
        "package_path": package_path,
        "makevars": str(makevars),
        "link_mode": "prebuilt-dev-release",
    }


def _run_orchestrator(artifacts_dir: Path, timeout: int, *, r_lib: Path | None = None) -> tuple[Path, str]:
    csv_path = artifacts_dir / "methods-cross-binding-matrix.csv"
    cmd = [
        sys.executable,
        "benchmarks/cross_binding/orchestrator.py",
        "--algorithms",
        "pls",
        "--sizes",
        "20x8",
        "--threads",
        "1",
        "--libn4m-build",
        "dev-release",
        "--n-runs",
        "2",
        "--n-components",
        "3",
        "--reference-backends",
        "registry",
        "--only",
        *REQUIRED_BACKENDS,
        "--timeout",
        str(timeout),
        "--out-csv",
        str(csv_path),
        "--force",
        "--workers",
        "1",
    ]
    proc = _run(cmd, env=_base_env(r_lib), timeout=timeout + 60)
    transcript = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError("cross-binding orchestrator failed:\n" + "\n".join(_tail(transcript)))
    if not csv_path.exists():
        raise RuntimeError(f"orchestrator did not write {csv_path}")
    return csv_path, transcript


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _bool_cell(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    return value.strip().lower() == "true"


def _prediction_digest(path: Path) -> dict[str, Any]:
    arr = np.load(path)
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "preview": [float(x) for x in arr.ravel()[:5]],
    }


def _rmse_rel(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=np.float64).ravel()
    expected_arr = np.asarray(expected, dtype=np.float64).ravel()
    if actual_arr.shape != expected_arr.shape:
        raise RuntimeError(f"shape mismatch: {actual_arr.shape} != {expected_arr.shape}")
    diff = actual_arr - expected_arr
    denom = max(1e-12, float(np.sqrt(np.mean(expected_arr * expected_arr))))
    return float(np.sqrt(np.mean(diff * diff)) / denom)


def _load_orchestrator_dataset(row: dict[str, str]) -> tuple[Path, np.ndarray, np.ndarray]:
    n = int(row["n"])
    p = int(row["p"])
    seed = int(row.get("prediction_seed") or row.get("seed_base") or 0)
    csv_path = REPO / "benchmarks/cross_binding/data" / f"data_{n}x{p}_seed{seed}.csv"
    if not csv_path.exists():
        raise RuntimeError(f"orchestrator dataset CSV missing for WASM ledger fixture: {csv_path}")
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    X = np.ascontiguousarray(arr[:, :-1], dtype=np.float64)
    y = np.ascontiguousarray(arr[:, -1].reshape(-1, 1), dtype=np.float64)
    if X.shape != (n, p):
        raise RuntimeError(f"unexpected orchestrator dataset shape {X.shape}; expected {(n, p)}")
    return csv_path, X, y


def _native_pls_fixture_arrays(X: np.ndarray, Y: np.ndarray, n_components: int) -> dict[str, np.ndarray | str]:
    python_src = REPO / "bindings/python/src"
    if str(python_src) not in sys.path:
        sys.path.insert(0, str(python_src))
    import pls4all

    with pls4all.Context() as ctx, pls4all.Config() as cfg:
        cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
        cfg.solver = pls4all.Solver.SIMPLS
        cfg.deflation = pls4all.Deflation.REGRESSION
        cfg.n_components = int(n_components)
        cfg.center_x = True
        cfg.scale_x = False
        cfg.center_y = True
        cfg.scale_y = False
        model = pls4all.Model.fit(ctx, cfg, X, Y)
        try:
            return {
                "version": pls4all.version(),
                "coefficients": np.asarray(
                    model.get_array(ctx, pls4all.ModelArrayKind.COEFFICIENTS),
                    dtype=np.float64,
                ),
                "x_mean": np.asarray(
                    model.get_array(ctx, pls4all.ModelArrayKind.X_MEAN),
                    dtype=np.float64,
                ),
                "y_mean": np.asarray(
                    model.get_array(ctx, pls4all.ModelArrayKind.Y_MEAN),
                    dtype=np.float64,
                ),
                "predictions": np.asarray(model.predict(ctx, X), dtype=np.float64),
            }
        finally:
            model.close()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def _build_wasm_orchestrator_fixture(
    rows: list[dict[str, str]],
    prediction_rows: list[dict[str, Any]],
    artifacts_dir: Path,
    *,
    n_components: int,
) -> Path:
    by_backend = {row["backend"]: row for row in rows}
    cpp_row = by_backend["cpp"]
    csv_path, X, Y = _load_orchestrator_dataset(cpp_row)
    cpp_predictions = np.asarray(np.load(Path(cpp_row["predictions_path"])), dtype=np.float64).reshape(-1, 1)
    native = _native_pls_fixture_arrays(X, Y, n_components)
    native_predictions = np.asarray(native["predictions"], dtype=np.float64).reshape(-1, 1)
    cpp_native_rmse_rel = _rmse_rel(cpp_predictions, native_predictions)
    if cpp_native_rmse_rel > STRICT_TOLERANCES["binding_parity_max_diff"]:
        raise RuntimeError(
            "native Python fixture predictions do not match orchestrator cpp ledger: "
            f"{cpp_native_rmse_rel}"
        )

    prediction_digests = {
        row["backend"]: {
            "path": row["path"],
            "sha256": row["sha256"],
            "shape": row["shape"],
        }
        for row in prediction_rows
        if row["backend"] in {"cpp", "python_tier1", "r_tier1"}
    }
    fixture = {
        "schema": "n4a.methods.wasm_orchestrator_fixture.v1",
        "status": "pass",
        "algorithm": "pls",
        "source": "benchmarks/cross_binding/orchestrator.py",
        "n": int(cpp_row["n"]),
        "p": int(cpp_row["p"]),
        "q": 1,
        "n_components": int(n_components),
        "tolerances": dict(STRICT_TOLERANCES),
        "seed_base": int(cpp_row.get("seed_base") or 0),
        "prediction_seed": int(cpp_row.get("prediction_seed") or cpp_row.get("seed_base") or 0),
        "dataset_csv": str(csv_path),
        "dataset_csv_sha256": _sha256_file(csv_path),
        "X": [float(value) for value in X.ravel()],
        "Y": [float(value) for value in Y.ravel()],
        "X_sha256": _array_sha256(X),
        "Y_sha256": _array_sha256(Y),
        "reference_backend": "cpp",
        "reference_predictions_path": str(cpp_row["predictions_path"]),
        "reference_predictions_sha256": _array_sha256(cpp_predictions),
        "prediction_digests": prediction_digests,
        "native_python_version": native["version"],
        "cpp_native_predictions_rmse_rel": cpp_native_rmse_rel,
        "coefficients": [float(value) for value in np.asarray(native["coefficients"], dtype=np.float64).ravel()],
        "x_mean": [float(value) for value in np.asarray(native["x_mean"], dtype=np.float64).ravel()],
        "y_mean": [float(value) for value in np.asarray(native["y_mean"], dtype=np.float64).ravel()],
        "predictions": [float(value) for value in cpp_predictions.ravel()],
    }
    fixture_path = artifacts_dir / "wasm-orchestrator-fixture.json"
    _json_write(fixture_path, fixture)
    return fixture_path


def _validate_orchestrator_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_backend = {row["backend"]: row for row in rows}
    missing = [name for name in REQUIRED_BACKENDS if name not in by_backend]
    if missing:
        raise RuntimeError(f"orchestrator output missing backends: {missing}")

    parity_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for backend in REQUIRED_BACKENDS:
        row = by_backend[backend]
        ok = _bool_cell(row.get("ok"))
        binding_ok = _bool_cell(row.get("binding_parity_ok"))
        reference_ok = _bool_cell(row.get("reference_parity_ok"))
        is_external = row.get("kind") == "external"
        if ok is not True:
            failures.append(f"{backend}: ok={row.get('ok')} reason={row.get('reason')}")
        if not is_external and binding_ok is not True:
            failures.append(
                f"{backend}: binding_parity_ok={row.get('binding_parity_ok')} "
                f"note={row.get('binding_parity_note')}"
            )
        if reference_ok is not True:
            failures.append(
                f"{backend}: reference_parity_ok={row.get('reference_parity_ok')} "
                f"note={row.get('reference_parity_note')}"
            )

        parity_rows.append(
            {
                "backend": backend,
                "language": row.get("language"),
                "kind": row.get("kind"),
                "ok": ok,
                "binding_parity_ok": binding_ok,
                "binding_parity_max_diff": _float_or_none(row.get("binding_parity_max_diff")),
                "reference_parity_ok": reference_ok,
                "reference_parity_rmse_rel": _float_or_none(row.get("reference_parity_rmse_rel")),
                "reference_parity_note": row.get("reference_parity_note", ""),
            }
        )

        pred = row.get("predictions_path")
        if pred and Path(pred).exists():
            prediction_rows.append(
                {
                    "backend": backend,
                    "language": row.get("language"),
                    "algorithm": row.get("algorithm"),
                    **_prediction_digest(Path(pred)),
                }
            )

    if failures:
        raise RuntimeError("cross-binding parity failures:\n" + "\n".join(failures))
    return parity_rows, prediction_rows


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except ValueError:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _summarize_parity_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_external = [row for row in rows if row.get("kind") != "external"]
    binding_diffs = _finite_values(non_external, "binding_parity_max_diff")
    reference_diffs = _finite_values(rows, "reference_parity_rmse_rel")
    backends = {str(row["backend"]) for row in rows}
    return {
        "all_required_backends_present": set(REQUIRED_BACKENDS).issubset(backends),
        "backend_count": len(rows),
        "binding_backend_count": len(non_external),
        "binding_parity_all_ok": all(row.get("binding_parity_ok") is True for row in non_external),
        "binding_parity_max_diff": max(binding_diffs, default=0.0),
        "reference_parity_all_ok": all(row.get("reference_parity_ok") is True for row in rows),
        "reference_parity_rmse_rel_max": max(reference_diffs, default=0.0),
    }


def _summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_backend = {row["backend"]: row for row in rows}
    row_counts = [
        int(row["shape"][0])
        for row in rows
        if isinstance(row.get("shape"), list) and row["shape"]
    ]
    core_backends = ("cpp", "python_tier1", "r_tier1")
    core_hashes = [
        str(by_backend[backend]["sha256"])
        for backend in core_backends
        if backend in by_backend and by_backend[backend].get("sha256")
    ]
    return {
        "backend_count": len(rows),
        "prediction_rows_min": min(row_counts, default=0),
        "prediction_rows_max": max(row_counts, default=0),
        "shared_cpp_python_r_sha256": len(core_hashes) == len(core_backends) and len(set(core_hashes)) == 1,
        "cpp_python_r_sha256": core_hashes[0] if core_hashes and len(set(core_hashes)) == 1 else None,
    }


def _find_node() -> Path | None:
    env_node = os.environ.get("NODE")
    if env_node and Path(env_node).exists():
        return Path(env_node)
    found = shutil.which("node")
    if found and not found.startswith("/mnt/c/"):
        return Path(found)
    nvm_root = Path.home() / ".nvm/versions/node"
    if nvm_root.exists():
        candidates = sorted(nvm_root.glob("v*/bin/node"), reverse=True)
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_git_tracked(path: Path) -> bool:
    rel = path.relative_to(REPO)
    proc = _run(["git", "ls-files", "--error-unmatch", str(rel)], timeout=30)
    return proc.returncode == 0


def _wasm_artifact_metadata(wasm_js: Path, wasm_bin: Path) -> dict[str, Any]:
    emcc = shutil.which("emcc")
    return {
        "head_rebuild_verified": False,
        "rebuild_reason": (
            "emcc/EMSDK unavailable in this environment; smoke runs the local "
            "build/emscripten artifact and records its digest instead"
            if not emcc or not os.environ.get("EMSDK")
            else "WASM rebuild not requested by this smoke gate"
        ),
        "emcc": emcc,
        "emsdk": os.environ.get("EMSDK", ""),
        "artifacts": {
            "n4m.js": {
                "path": str(wasm_js),
                "git_tracked": _is_git_tracked(wasm_js),
                "sha256": _sha256_file(wasm_js),
            },
            "n4m.wasm": {
                "path": str(wasm_bin),
                "git_tracked": _is_git_tracked(wasm_bin),
                "sha256": _sha256_file(wasm_bin),
            },
        },
    }


def _run_wasm_smoke(timeout: int, *, fixture_path: Path | None = None) -> dict[str, Any]:
    node = _find_node()
    if node is None:
        raise RuntimeError("Linux node executable not found for WASM parity smoke")
    script = REPO / "bindings/js/test/run_smoke.mjs"
    wasm_js = REPO / "build/emscripten/bindings/js/n4m.js"
    wasm_bin = REPO / "build/emscripten/bindings/js/n4m.wasm"
    if not wasm_js.exists() or not wasm_bin.exists():
        raise RuntimeError(
            "WASM artifacts missing; run cmake --preset emscripten && "
            "cmake --build --preset emscripten --target n4m_wasm"
        )
    env = _base_env()
    if fixture_path is not None:
        env["N4M_WASM_PARITY_FIXTURE"] = str(fixture_path)
    proc = _run(
        [str(node), "--experimental-vm-modules", str(script)],
        env=env,
        timeout=timeout,
    )
    transcript = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError("WASM smoke failed:\n" + "\n".join(_tail(transcript)))
    metrics: dict[str, float] = {}
    for line in transcript.splitlines():
        match = re.search(r"^\s*(coefficients|x_mean|y_mean|predictions)\s+rmse_rel:\s+([0-9.eE+-]+)", line)
        if match:
            metrics[f"{match.group(1)}_rmse_rel"] = float(match.group(2))
    if "predictions_rmse_rel" not in metrics:
        raise RuntimeError("WASM smoke did not emit the expected predictions rmse_rel metric")
    metrics_max = max(metrics.values(), default=0.0)
    if metrics_max > STRICT_TOLERANCES["wasm_rmse_rel"]:
        raise RuntimeError(
            "WASM parity exceeded strict tolerance: "
            f"{metrics_max} > {STRICT_TOLERANCES['wasm_rmse_rel']}"
        )
    return {
        "backend": "js_wasm",
        "language": "WASM",
        "kind": "pls4all_binding",
        "ok": True,
        "node": str(node),
        "wasm_js": str(wasm_js),
        "wasm": str(wasm_bin),
        "fixture": str(fixture_path or (REPO / "bindings/js/test/parity_fixture.json")),
        "metrics": metrics,
        "metrics_max_rmse_rel": metrics_max,
        "tolerance": STRICT_TOLERANCES["wasm_rmse_rel"],
        "artifact_metadata": _wasm_artifact_metadata(wasm_js, wasm_bin),
    }


def _rust_archive_status() -> dict[str, Any]:
    manifest = REPO / "bindings/_archive/rust/pls4all/Cargo.toml"
    lib = REPO / "build/dev-release/cpp/src/libn4m.so"
    has_legacy_symbol = False
    if lib.exists():
        proc = _run(["nm", "-D", "--defined-only", str(lib)], timeout=30)
        has_legacy_symbol = "n4m_pls_fit_simple" in proc.stdout
    return {
        "backend": "rust_archive",
        "language": "Rust",
        "kind": "archived_binding",
        "release_target": False,
        "gate": "not_applicable",
        "manifest": str(manifest),
        "reason": (
            "Rust lives under bindings/_archive and is not a current release "
            "target in CLAUDE.md. The archived wrapper still targets the "
            "removed n4m_pls_fit_simple helper."
        ),
        "legacy_symbol_present": has_legacy_symbol,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts_dir = args.artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    build = {"build_invoked": False}
    if not args.skip_build:
        build = _ensure_dev_release_build(args.timeout)

    r_gate = _prepare_r_gate_library(artifacts_dir, args.timeout)
    csv_path, orchestrator_transcript = _run_orchestrator(
        artifacts_dir,
        args.timeout,
        r_lib=Path(r_gate["library"]),
    )
    rows = _read_rows(csv_path)
    parity_rows, prediction_rows = _validate_orchestrator_rows(rows)
    wasm_fixture = _build_wasm_orchestrator_fixture(
        rows,
        prediction_rows,
        artifacts_dir,
        n_components=3,
    )
    wasm = _run_wasm_smoke(args.timeout, fixture_path=wasm_fixture)
    rust_archive = _rust_archive_status()
    binding_summary = _summarize_parity_rows(parity_rows)
    prediction_summary = _summarize_prediction_rows(prediction_rows)

    binding_payload = {
        "schema": "n4a.methods.cross_binding_parity.v1",
        "status": "pass",
        "build": build,
        "tolerances": STRICT_TOLERANCES,
        "r_gate": r_gate,
        "orchestrator_csv": str(csv_path),
        "required_backends": list(REQUIRED_BACKENDS),
        "parity_rows": parity_rows,
        "binding_summary": binding_summary,
        "wasm": wasm,
        "rust_archive": rust_archive,
    }
    predictions_payload = {
        "schema": "n4a.methods.predictions_by_language.v1",
        "status": "pass",
        "tolerances": STRICT_TOLERANCES,
        "predictions": prediction_rows,
        "prediction_summary": prediction_summary,
        "wasm": {
            "backend": wasm["backend"],
            "language": wasm["language"],
            "fixture": wasm["fixture"],
            "metrics": wasm["metrics"],
            "metrics_max_rmse_rel": wasm["metrics_max_rmse_rel"],
            "tolerance": wasm["tolerance"],
        },
        "rust_archive": rust_archive,
    }
    _json_write(artifacts_dir / "binding-parity.json", binding_payload)
    _json_write(artifacts_dir / "predictions-by-language.json", predictions_payload)
    (artifacts_dir / "orchestrator.log").write_text(orchestrator_transcript)

    print(
        json.dumps(
            {
                "status": "pass",
                "binding_parity": str(artifacts_dir / "binding-parity.json"),
                "predictions_by_language": str(artifacts_dir / "predictions-by-language.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
