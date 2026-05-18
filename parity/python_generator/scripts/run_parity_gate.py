"""Unified parity gate for chemometrics4all.

Mirrors `pls4all/parity/python_generator/scripts/...` and the
`parity-gate.yml` CI workflow contract.

Three stages:

  Stage 1 — Fixture determinism: regenerate every `parity/fixtures/*_v1.json`
            from the per-phase generators and verify byte-for-byte equality
            with the committed copy. Catches algorithmic drift in the
            frozen NumPy references.

  Stage 2 — Frozen-ref cross-validation: where a frozen `c4a_parity_*_ref/`
            module ships with a `validate_*.py` helper, run it against the
            actually-installed upstream package (numpy, scipy, sklearn,
            pywt, pybaselines, nirs4all). Catches drift of the frozen ref
            from the canonical reference. Informational — failures here do
            NOT fail the gate (the frozen ref is the contract).

  Stage 3 — C++ parity tests: invoke the built `chemometrics4all_tests`
            binary; every fixture under `parity/fixtures/` is loaded and
            asserted to within its declared tolerance.

Usage:
    python parity/python_generator/scripts/run_parity_gate.py \
        --build-dir build/dev-debug \
        [--skip-fixture-regen] [--skip-cpp-tests] [--skip-cross-validation]

Exit codes:
    0  — all gates pass
    1  — Stage 1 drift detected (fixture regeneration changed bytes)
    2  — Stage 3 failure (C++ parity tests broke)
    3  — Stage 1 generator crashed
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "parity" / "fixtures"
GENERATOR_DIR = HERE


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_fixtures() -> dict[str, str]:
    """SHA256 of every committed fixture for drift comparison."""
    return {
        p.name: sha256_file(p)
        for p in sorted(FIXTURE_DIR.glob("*.json"))
    }


def list_generators() -> list[Path]:
    """All per-phase generator scripts in dependency-ordered execution order."""
    return sorted(GENERATOR_DIR.glob("generate_*.py"))


def stage1_fixture_determinism(verbose: bool = True) -> tuple[int, list[str]]:
    """Regenerate every fixture and check SHA256 against the committed copy."""
    failures: list[str] = []
    crashes: list[str] = []

    before = snapshot_fixtures()
    generators = list_generators()

    if verbose:
        print(f"\n=== Stage 1: fixture determinism ({len(generators)} generators) ===\n")

    for gen in generators:
        rel = gen.relative_to(REPO_ROOT)
        if verbose:
            print(f"  Running {rel.name} ...", end=" ", flush=True)
        try:
            subprocess.run(
                [sys.executable, str(gen)],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=120,
            )
            if verbose:
                print("ok")
        except subprocess.CalledProcessError as e:
            crashes.append(str(rel))
            if verbose:
                tail = (e.stderr or "").splitlines()[-5:]
                print(f"CRASHED\n     {chr(10).join(tail)}")
        except subprocess.TimeoutExpired:
            crashes.append(str(rel))
            if verbose:
                print("TIMEOUT")

    after = snapshot_fixtures()

    # Compare hashes
    drifted = []
    for fname, sha_before in before.items():
        sha_after = after.get(fname)
        if sha_after is None:
            drifted.append(f"{fname}: vanished after regen")
        elif sha_before != sha_after:
            drifted.append(f"{fname}: SHA changed ({sha_before[:8]} -> {sha_after[:8]})")

    new_files = set(after) - set(before)
    if new_files:
        for f in sorted(new_files):
            drifted.append(f"{f}: new file created (uncommitted)")

    if verbose:
        if crashes:
            print(f"\n  Generators crashed: {len(crashes)}")
            for c in crashes:
                print(f"    - {c}")
        if drifted:
            print(f"\n  Fixtures drifted: {len(drifted)}")
            for d in drifted[:20]:
                print(f"    - {d}")
        if not crashes and not drifted:
            print(f"\n  [PASS] All {len(before)} fixtures bit-identical after regen.")

    exit_code = 0
    if crashes:
        exit_code = 3
        failures.extend([f"generator crash: {c}" for c in crashes])
    if drifted:
        exit_code = 1 if exit_code == 0 else exit_code
        failures.extend([f"fixture drift: {d}" for d in drifted])

    return exit_code, failures


def stage2_cross_validation(verbose: bool = True) -> tuple[int, list[str]]:
    """Run validate_*_ref.py scripts; failures are informational."""
    if verbose:
        print(f"\n=== Stage 2: frozen-ref cross-validation (informational) ===\n")

    validators = sorted(GENERATOR_DIR.glob("validate_*.py"))
    if not validators:
        if verbose:
            print("  No validators present.")
        return 0, []

    issues: list[str] = []
    for v in validators:
        rel = v.relative_to(REPO_ROOT)
        if verbose:
            print(f"  {rel.name} ...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, str(v)],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=180,
            )
            if result.returncode == 0:
                if verbose:
                    print("ok")
            else:
                issues.append(f"{rel}: rc={result.returncode}")
                if verbose:
                    print(f"WARN (rc={result.returncode})")
                    for line in (result.stdout + "\n" + result.stderr).splitlines()[-8:]:
                        print(f"     {line}")
        except subprocess.TimeoutExpired:
            issues.append(f"{rel}: timeout")
            if verbose:
                print("TIMEOUT")
        except Exception as e:  # noqa: BLE001
            issues.append(f"{rel}: {e}")
            if verbose:
                print(f"ERROR ({e})")

    if verbose:
        if issues:
            print(f"\n  Validators reported {len(issues)} issues (informational).")
        else:
            print(f"\n  [PASS] All cross-validations agree with upstream.")

    return 0, issues  # Always returns 0 — informational


def stage3_cpp_tests(build_dir: Path, verbose: bool = True) -> tuple[int, list[str]]:
    """Run the C++ test binary."""
    if verbose:
        print(f"\n=== Stage 3: C++ parity tests ({build_dir}) ===\n")

    test_bin = build_dir / "cpp" / "tests" / "chemometrics4all_tests"
    if not test_bin.exists():
        return 2, [f"test binary missing: {test_bin}"]

    try:
        result = subprocess.run(
            [str(test_bin)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return 2, [f"{test_bin} timed out after 300 s"]

    last_lines = result.stdout.splitlines()[-3:]
    for line in last_lines:
        if verbose:
            print(f"  {line}")

    if result.returncode != 0:
        return 2, [f"chemometrics4all_tests exited {result.returncode}"]
    return 0, []


def main() -> int:
    parser = argparse.ArgumentParser(description="chemometrics4all parity gate runner.")
    parser.add_argument("--build-dir", default="build/dev-debug",
                        help="CMake build directory (default: build/dev-debug)")
    parser.add_argument("--skip-fixture-regen", action="store_true",
                        help="Skip Stage 1 (fixture determinism)")
    parser.add_argument("--skip-cross-validation", action="store_true",
                        help="Skip Stage 2 (frozen-ref cross-validation)")
    parser.add_argument("--skip-cpp-tests", action="store_true",
                        help="Skip Stage 3 (C++ parity tests)")
    args = parser.parse_args()

    exit_code = 0
    all_failures: list[str] = []

    if not args.skip_fixture_regen:
        rc, fails = stage1_fixture_determinism()
        exit_code = exit_code or rc
        all_failures.extend(fails)

    if not args.skip_cross_validation:
        _, infos = stage2_cross_validation()
        # Don't influence exit code; just record
        all_failures.extend(f"[INFO] {i}" for i in infos)

    if not args.skip_cpp_tests:
        rc, fails = stage3_cpp_tests(REPO_ROOT / args.build_dir)
        exit_code = exit_code or rc
        all_failures.extend(fails)

    print("\n=== Parity gate summary ===")
    if exit_code == 0:
        print("  [PASS]")
    else:
        print(f"  [FAIL] exit_code={exit_code}")
        for f in all_failures:
            if not f.startswith("[INFO]"):
                print(f"    - {f}")
    print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
