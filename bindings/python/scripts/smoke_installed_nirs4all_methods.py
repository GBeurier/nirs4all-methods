#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Build and load-test the installed ``nirs4all-methods`` Python wheel.

This is a release/loadability smoke, not a numerical parity suite. It proves the
full Python distribution installs as ``nirs4all-methods``, imports as ``n4m``,
loads the bundled ``libn4m`` from the installed wheel, and exposes the exact
SNV/PLS imports consumed by the opt-in ``nirs4all.operators.methods`` route.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GENERATOR = REPO / "bindings" / "python" / "scripts" / "make_python_package.py"
BUILD_PRESETS = (
    "dev-release",
    "dev-debug",
    "blas-omp",
    "blas-on",
    "ci-linux-gcc12-release",
)
LIB_PATTERNS = ("libn4m*.so*", "libn4m*.dylib", "n4m*.dll", "libn4m*.dll")


class SmokeError(RuntimeError):
    """Expected smoke failure with a user-facing diagnostic."""


def _run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        rendered = " ".join(cmd)
        raise SmokeError(
            f"command failed ({proc.returncode}): {rendered}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _abi_expected() -> tuple[int, int]:
    header = REPO / "cpp" / "include" / "n4m" / "n4m_version.h"
    values: dict[str, int] = {}
    for line in header.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if (
            len(parts) == 3
            and parts[0] == "#define"
            and parts[1].startswith("N4M_ABI_VERSION_")
        ):
            values[parts[1].removeprefix("N4M_ABI_VERSION_")] = int(parts[2])
    try:
        return values["MAJOR"], values["MINOR"]
    except KeyError as exc:
        raise SmokeError(f"could not read ABI version from {header}") from exc


def _glob_libs(directory: Path) -> list[Path]:
    found: list[Path] = []
    if not directory.is_dir():
        return found
    for pattern in LIB_PATTERNS:
        found.extend(path for path in directory.glob(pattern) if path.is_file())
    return sorted(set(found), key=lambda p: (p.name.count(".so.") == 0, p.name))


def _is_abi_compatible(path: Path, major: int, minor: int) -> tuple[bool, str]:
    try:
        lib = ctypes.CDLL(str(path))
        check = lib.n4m_check_abi_compatibility
        check.argtypes = [ctypes.c_int32, ctypes.c_int32]
        check.restype = ctypes.c_int
        status = int(check(major, minor))
    except Exception as exc:  # pragma: no cover - host/linker-specific detail
        return False, str(exc)
    if status != 0:
        return (
            False,
            f"ABI check rejected header {major}.{minor}.x with status {status}",
        )
    return True, "ok"


def discover_lib() -> Path:
    major, minor = _abi_expected()
    candidates: list[Path] = []
    for preset in BUILD_PRESETS:
        candidates.extend(_glob_libs(REPO / "build" / preset / "cpp" / "src"))
    candidates.extend(
        path
        for path in sorted((REPO / "build").glob("*/cpp/src/libn4m*"))
        if path.is_file()
    )

    diagnostics: list[str] = []
    for path in candidates:
        ok, reason = _is_abi_compatible(path.resolve(), major, minor)
        if ok:
            return path.resolve()
        diagnostics.append(f"  - {path}: {reason}")

    searched = (
        "\n".join(diagnostics) if diagnostics else "  (no libn4m candidates found)"
    )
    raise SmokeError(
        f"no ABI-compatible libn4m build found for header ABI {major}.{minor}.x.\n"
        f"Run `make build PRESET=dev-release` or pass `--lib /path/to/libn4m`.\n"
        f"Searched:\n{searched}"
    )


def _load_generator():
    spec = importlib.util.spec_from_file_location("n4m_package_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise SmokeError(f"could not load package generator from {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generate_package(tmp: Path) -> Path:
    generator = _load_generator()
    generator.REPO = tmp
    generator.SRC_PKG = REPO / "bindings" / "python"
    return generator.generate("nirs4all-methods")


def _stage_lib(package_dir: Path, lib_path: Path) -> Path:
    lib_dir = package_dir / "src" / "n4m" / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    staged = lib_dir / lib_path.name
    shutil.copy2(lib_path, staged)
    return staged


def _build_wheel(
    python: str, package_dir: Path, wheel_dir: Path, *, no_build_isolation: bool
) -> Path:
    wheel_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-m", "build", "--wheel", "--outdir", str(wheel_dir)]
    if no_build_isolation:
        cmd.append("--no-isolation")
    cmd.append(str(package_dir))
    proc = _run(cmd)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SmokeError(
            f"expected exactly one wheel in {wheel_dir}, found {wheels}\n"
            f"build stdout:\n{proc.stdout}\n"
            f"build stderr:\n{proc.stderr}"
        )
    wheel_name = wheels[0].name
    if not wheel_name.startswith("nirs4all_methods-"):
        raise SmokeError(
            f"built unexpected wheel {wheel_name}; expected nirs4all_methods-*.whl\n"
            f"build stdout:\n{proc.stdout}\n"
            f"build stderr:\n{proc.stderr}"
        )
    return wheels[0]


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _child_program() -> str:
    return textwrap.dedent(
        r"""
        import json
        import os
        from pathlib import Path

        import numpy as np

        import n4m
        from n4m.estimators.regression.latent import PLS
        from n4m.transform.scatter import SNV

        module_path = Path(n4m.__file__).resolve()
        prefix = Path(os.environ["N4M_EXPECTED_PREFIX"]).resolve()
        try:
            module_path.relative_to(prefix)
        except ValueError as exc:
            raise AssertionError(f"n4m imported outside smoke venv: {module_path}") from exc

        lib_path = Path(n4m.library_path()).resolve()
        expected_lib_dir = module_path.parent / "lib"
        try:
            lib_path.relative_to(expected_lib_dir)
        except ValueError as exc:
            raise AssertionError(f"libn4m did not load from installed package lib dir: {lib_path}") from exc

        abi = tuple(int(v) for v in n4m.abi_version())
        assert abi[0] == 2, abi

        ctx = n4m.Context()
        try:
            ctx.num_threads = 1
            assert ctx.num_threads == 1
        finally:
            ctx.close()

        X = np.array(
            [
                [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
                [2.0, 3.0, 5.0, 7.0, 11.0, 13.0],
                [3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
                [5.0, 9.0, 2.0, 6.0, 5.0, 3.0],
                [8.0, 5.0, 9.0, 7.0, 9.0, 3.0],
                [2.0, 7.0, 1.0, 8.0, 2.0, 8.0],
                [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        )
        y = X[:, 0] - 0.25 * X[:, 2] + 0.1 * X[:, 5]

        Xs = SNV().fit_transform(X)
        np.testing.assert_allclose(Xs.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(Xs.std(axis=1, ddof=0), 1.0, atol=1e-12)

        model = PLS(pls_components=[2], cv=2, scale_x=True).fit(Xs, y)
        pred = np.asarray(model.predict(Xs), dtype=np.float64).reshape(-1)
        assert pred.shape == (X.shape[0],)
        assert np.all(np.isfinite(pred))
        diagnostics = model.get_diagnostics()
        assert diagnostics["method"] == "pls"

        print(json.dumps({
            "status": "INSTALLED_N4M_OK",
            "module": str(module_path),
            "library": str(lib_path),
            "abi": abi,
            "prediction_checksum": float(np.sum(pred)),
        }, sort_keys=True))
        """
    )


def _install_and_run(wheel: Path, tmp: Path, python: str) -> dict[str, object]:
    venv_dir = tmp / "venv"
    _run([python, "-m", "venv", "--system-site-packages", str(venv_dir)])
    vpy = _venv_python(venv_dir)
    _run(
        [str(vpy), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)]
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {"PYTHONPATH", "PYTHONNOUSERSITE", "N4M_LIB_PATH", "PLS4ALL_LIB_PATH"}
    }
    env["N4M_EXPECTED_PREFIX"] = str(venv_dir)
    proc = _run([str(vpy), "-c", _child_program()], cwd=tmp, env=env)
    last_line = proc.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lib",
        type=Path,
        help="Path to a built libn4m shared library. Defaults to an ABI-compatible build/* discovery.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for wheel build and smoke venv creation.",
    )
    parser.add_argument(
        "--no-build-isolation",
        action="store_true",
        help="Pass --no-isolation to python -m build. Useful only when local setuptools satisfies pyproject.toml.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the generated package, wheelhouse, and venv for debugging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="n4m_install_smoke_"))
    try:
        lib_path = args.lib.resolve() if args.lib else discover_lib()
        if not lib_path.exists():
            raise SmokeError(f"libn4m path does not exist: {lib_path}")
        major, minor = _abi_expected()
        ok, reason = _is_abi_compatible(lib_path, major, minor)
        if not ok:
            raise SmokeError(
                f"{lib_path} is not ABI-compatible with header {major}.{minor}.x: {reason}"
            )

        package_dir = _generate_package(tmp)
        staged = _stage_lib(package_dir, lib_path)
        wheel = _build_wheel(
            args.python,
            package_dir,
            tmp / "wheelhouse",
            no_build_isolation=args.no_build_isolation,
        )
        result = _install_and_run(wheel, tmp, args.python)

        print(
            json.dumps(
                {
                    "status": "OK",
                    "input_lib": str(lib_path),
                    "staged_lib": str(staged),
                    "wheel": str(wheel),
                    "installed": result,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except SmokeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.keep_temp:
            print(f"kept temp dir: {tmp}", file=sys.stderr)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
