#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Build and load-test the installed ``nirs4all-methods`` Python wheel.

This is a release/loadability smoke, not a numerical parity suite. It proves the
full Python distribution installs as ``nirs4all-methods``, imports as ``n4m``,
loads the bundled ``libn4m`` from the installed wheel, and exposes the exact
SNV/PLS imports consumed by the opt-in ``nirs4all.operators.methods`` route,
plus the public owning native-finetune path from an isolated installed venv.
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
import tarfile
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


def _build_distribution(
    python: str,
    package_dir: Path,
    dist_dir: Path,
    *,
    artifact: str,
    no_build_isolation: bool,
) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-m", "build", f"--{artifact}", "--outdir", str(dist_dir)]
    if no_build_isolation:
        cmd.append("--no-isolation")
    cmd.append(str(package_dir))
    proc = _run(cmd)
    pattern = "*.whl" if artifact == "wheel" else "*.tar.gz"
    artifacts = sorted(dist_dir.glob(pattern))
    if len(artifacts) != 1:
        raise SmokeError(
            f"expected exactly one {artifact} in {dist_dir}, found {artifacts}\n"
            f"build stdout:\n{proc.stdout}\n"
            f"build stderr:\n{proc.stderr}"
        )
    artifact_name = artifacts[0].name
    if not artifact_name.startswith("nirs4all_methods-"):
        raise SmokeError(
            f"built unexpected {artifact} {artifact_name}; expected nirs4all_methods-*"
            f"{'.whl' if artifact == 'wheel' else '.tar.gz'}\n"
            f"build stdout:\n{proc.stdout}\n"
            f"build stderr:\n{proc.stderr}"
        )
    return artifacts[0]


def _inspect_sdist(sdist: Path, staged_lib_name: str) -> dict[str, object]:
    with tarfile.open(sdist, "r:gz") as archive:
        members = sorted(member.name for member in archive.getmembers() if member.name)

    if not members:
        raise SmokeError(f"sdist {sdist} is empty")

    top_level = members[0].split("/", 1)[0]
    expected = (
        f"{top_level}/LICENSE",
        f"{top_level}/pyproject.toml",
        f"{top_level}/setup.py",
        f"{top_level}/src/n4m/__init__.py",
        f"{top_level}/src/n4m/_ffi.py",
        f"{top_level}/src/n4m/lib/{staged_lib_name}",
        f"{top_level}/tests/test_import.py",
    )
    missing = [member for member in expected if member not in members]
    if missing:
        raise SmokeError(
            f"sdist {sdist.name} is missing expected members:\n"
            + "\n".join(f"  - {member}" for member in missing)
        )

    bundled_libs = [
        member
        for member in members
        if member.startswith(f"{top_level}/src/n4m/lib/") and not member.endswith("/")
    ]
    return {
        "member_count": len(members),
        "top_level": top_level,
        "bundled_libs": bundled_libs,
    }


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
        from n4m.model_selection import (
            Algorithm,
            Direction,
            Metric,
            Optimizer,
            Sampler,
            SearchSpace,
            ValidationPlan,
            finetune_estimator,
        )
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
        assert issubclass(n4m.PartialBatchError, n4m.N4MError)

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

        fold_ids = np.arange(X.shape[0], dtype=np.int32) % 2
        with ValidationPlan.from_fold_ids(fold_ids) as plan:
            with SearchSpace() as finetune_space:
                finetune_space.add_int("n_components", 1, 2)
                finetune = finetune_estimator(
                    Algorithm.PCR,
                    Xs,
                    y,
                    plan,
                    finetune_space,
                    n_trials=4,
                    sampler=Sampler.RANDOM,
                    metric=Metric.RMSE,
                    seed=19,
                )
        assert finetune.estimator is Algorithm.PCR
        assert finetune.requested_trials == 4
        assert finetune.timed_out is False
        assert len(finetune.trials) == 4
        assert finetune.best_params["n_components"] in (1, 2)
        assert np.isfinite(finetune.best_score)

        search_space = SearchSpace().add_int("n_components", 1, 3)
        optimizer = Optimizer(
            search_space,
            sampler=Sampler.RANDOM,
            direction=Direction.MINIMIZE,
            seed=17,
        )
        initial_trials = optimizer.ask_batch(4)
        assert [trial.id for trial in initial_trials] == [0, 1, 2, 3]
        for trial in initial_trials:
            n_components = trial.get_int("n_components")
            optimizer.tell(trial.id, float((n_components - 2) ** 2))
        checkpoint = optimizer.save()
        assert checkpoint.startswith(b"N4MOPT\r\n")
        restored_optimizer = Optimizer.load(checkpoint)
        original_next = optimizer.ask()
        restored_next = restored_optimizer.ask()
        assert original_next.id == restored_next.id
        assert original_next.get_int("n_components") == restored_next.get_int("n_components")
        resumed_score = float((original_next.get_int("n_components") - 2) ** 2)
        optimizer.tell(original_next.id, resumed_score)
        restored_optimizer.tell(restored_next.id, resumed_score)
        restored_optimizer.close()
        best = optimizer.best()
        assert best is not None
        best_trial, best_score = best
        best_n_components = best_trial.get_int("n_components")
        assert 1 <= best_n_components <= 3
        assert np.isfinite(best_score)

        print(json.dumps({
            "status": "INSTALLED_N4M_OK",
            "module": str(module_path),
            "library": str(lib_path),
            "abi": abi,
            "prediction_checksum": float(np.sum(pred)),
            "optimizer_best_n_components": best_n_components,
            "optimizer_best_score": best_score,
            "finetune_estimator": finetune.estimator.name,
            "finetune_best_n_components": finetune.best_params["n_components"],
            "finetune_best_score": finetune.best_score,
        }, sort_keys=True))
        """
    )


def _install_and_run(artifact: Path, tmp: Path, python: str) -> dict[str, object]:
    venv_dir = tmp / "venv"
    _run([python, "-m", "venv", "--system-site-packages", str(venv_dir)])
    vpy = _venv_python(venv_dir)
    _run(
        [
            str(vpy),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(artifact),
        ]
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
        "--artifact",
        choices=("wheel", "sdist"),
        default="wheel",
        help="Release artifact to build and smoke-install. `sdist` also inspects the tarball contents.",
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
        artifact = _build_distribution(
            args.python,
            package_dir,
            tmp / "dist",
            artifact=args.artifact,
            no_build_isolation=args.no_build_isolation,
        )
        inspection = (
            _inspect_sdist(artifact, staged.name) if args.artifact == "sdist" else None
        )
        result = _install_and_run(artifact, tmp, args.python)

        print(
            json.dumps(
                {
                    "status": "OK",
                    "artifact_kind": args.artifact,
                    "artifact_inspection": inspection,
                    "input_lib": str(lib_path),
                    "staged_lib": str(staged),
                    "artifact": str(artifact),
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
