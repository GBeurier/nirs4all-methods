"""The ctypes retagger must preserve native extension tags and wheelhouse contents."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "retag_wheels.py"
PLATFORM = "manylinux_2_35_x86_64"


def _wheel(path: Path, dist: str, version: str, tag: str) -> bytes:
    dist_info = f"{dist}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n",
        )
        archive.writestr(
            f"{dist_info}/RECORD",
            f"{dist_info}/WHEEL,,\n{dist_info}/RECORD,,\n",
        )
    return path.read_bytes()


def _run(wheelhouse: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(wheelhouse), "--inplace"],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "dag_ml-0.3.30-cp311-abi3-manylinux_2_34_x86_64.whl",
        "nirs4all_methods-1.2.2-cp311-abi3-manylinux_2_35_x86_64.whl",
    ],
)
def test_mixed_wheelhouse_fails_without_retagging_anything(
    tmp_path: Path, unsafe_name: str
) -> None:
    legitimate = tmp_path / f"nirs4all_methods-1.2.1-cp311-cp311-{PLATFORM}.whl"
    original = _wheel(
        legitimate, "nirs4all_methods", "1.2.1", f"cp311-cp311-{PLATFORM}"
    )
    unsafe = tmp_path / unsafe_name
    unsafe.write_bytes(b"native-extension-sentinel")

    result = _run(tmp_path)

    assert result.returncode == 1
    assert (
        "retag supports only ctypes wheels" in result.stderr
        or "non-ctypes ABI tag" in result.stderr
    )
    assert legitimate.read_bytes() == original
    assert unsafe.read_bytes() == b"native-extension-sentinel"
    assert len(list(tmp_path.glob("*.whl"))) == 2


@pytest.mark.parametrize(
    "malformed", ["bad_zip", "missing_record", "missing_wheel_tag"]
)
def test_malformed_later_wheel_does_not_retag_earlier_wheel(
    tmp_path: Path, malformed: str
) -> None:
    first = tmp_path / f"nirs4all_methods-1.2.1-cp311-cp311-{PLATFORM}.whl"
    original = _wheel(first, "nirs4all_methods", "1.2.1", f"cp311-cp311-{PLATFORM}")
    second = tmp_path / f"pls4all-1.2.1-cp311-cp311-{PLATFORM}.whl"
    info = "pls4all-1.2.1.dist-info"
    if malformed == "bad_zip":
        second.write_bytes(b"corrupt-wheel")
    else:
        with zipfile.ZipFile(second, "w") as archive:
            wheel_text = "Wheel-Version: 1.0\n"
            if malformed == "missing_record":
                wheel_text += f"Tag: cp311-cp311-{PLATFORM}\n"
            archive.writestr(f"{info}/WHEEL", wheel_text)
            if malformed != "missing_record":
                archive.writestr(f"{info}/RECORD", f"{info}/WHEEL,,\n")
    second_original = second.read_bytes()

    result = _run(tmp_path)

    assert result.returncode == 1
    assert (
        "invalid wheel archive" in result.stderr
        or "missing WHEEL or RECORD" in result.stderr
    )
    assert first.read_bytes() == original
    assert second.read_bytes() == second_original
    assert len(list(tmp_path.glob("*.whl"))) == 2


def test_wheel_with_build_tag_is_retagged(tmp_path: Path) -> None:
    original = tmp_path / f"pls4all-1.2.1-1-cp312-cp312-{PLATFORM}.whl"
    _wheel(original, "pls4all", "1.2.1", f"cp312-cp312-{PLATFORM}")

    result = _run(tmp_path)

    retagged = tmp_path / f"pls4all-1.2.1-1-py3-none-{PLATFORM}.whl"
    assert result.returncode == 0, result.stderr
    assert not original.exists()
    assert retagged.exists()
    with zipfile.ZipFile(retagged) as archive:
        wheel = archive.read("pls4all-1.2.1.dist-info/WHEEL").decode()
        record = archive.read("pls4all-1.2.1.dist-info/RECORD").decode()
    assert f"Tag: py3-none-{PLATFORM}" in wheel
    assert "pls4all-1.2.1.dist-info/WHEEL,sha256=" in record
