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
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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


def _base_env() -> dict[str, str]:
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
    return env


def _run_orchestrator(artifacts_dir: Path, timeout: int) -> tuple[Path, str]:
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
    proc = _run(cmd, env=_base_env(), timeout=timeout + 60)
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
        return float(value)
    except ValueError:
        return None


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


def _run_wasm_smoke(timeout: int) -> dict[str, Any]:
    node = _find_node()
    if node is None:
        raise RuntimeError("Linux node executable not found for WASM parity smoke")
    script = REPO / "bindings/js/test/run_smoke.mjs"
    wasm_js = REPO / "build/emscripten/bindings/js/n4m.js"
    wasm_bin = REPO / "build/emscripten/bindings/js/n4m.wasm"
    if not wasm_js.exists() or not wasm_bin.exists():
        raise RuntimeError(
            "WASM artifacts missing; run cmake --preset emscripten && "
            "cmake --build --preset emscripten --target pls4all_wasm"
        )
    proc = _run(
        [str(node), "--experimental-vm-modules", str(script)],
        env=_base_env(),
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
    if len(metrics) != 4:
        raise RuntimeError("WASM smoke did not emit all expected rmse_rel metrics")
    return {
        "backend": "js_wasm",
        "language": "WASM",
        "kind": "pls4all_binding",
        "ok": True,
        "node": str(node),
        "wasm_js": str(wasm_js),
        "wasm": str(wasm_bin),
        "metrics": metrics,
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

    csv_path, orchestrator_transcript = _run_orchestrator(artifacts_dir, args.timeout)
    rows = _read_rows(csv_path)
    parity_rows, prediction_rows = _validate_orchestrator_rows(rows)
    wasm = _run_wasm_smoke(args.timeout)
    rust_archive = _rust_archive_status()

    binding_payload = {
        "schema": "n4a.methods.cross_binding_parity.v1",
        "status": "pass",
        "build": build,
        "orchestrator_csv": str(csv_path),
        "required_backends": list(REQUIRED_BACKENDS),
        "parity_rows": parity_rows,
        "wasm": wasm,
        "rust_archive": rust_archive,
    }
    predictions_payload = {
        "schema": "n4a.methods.predictions_by_language.v1",
        "status": "pass",
        "predictions": prediction_rows,
        "wasm": {
            "backend": wasm["backend"],
            "language": wasm["language"],
            "metrics": wasm["metrics"],
            "fixture": str(REPO / "bindings/js/test/parity_fixture.json"),
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
