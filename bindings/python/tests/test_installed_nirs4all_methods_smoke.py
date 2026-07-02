from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load_smoke_module():
    script = REPO / "bindings/python/scripts/smoke_installed_nirs4all_methods.py"
    spec = importlib.util.spec_from_file_location("n4m_install_smoke_under_test", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discover_lib_returns_first_abi_compatible_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke_module()
    first = tmp_path / "build" / "dev-release" / "cpp" / "src" / "libn4m.so"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("first", encoding="utf-8")
    second = tmp_path / "build" / "dev-debug" / "cpp" / "src" / "libn4m.so.2"
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_text("second", encoding="utf-8")

    monkeypatch.setattr(smoke, "REPO", tmp_path)
    monkeypatch.setattr(smoke, "BUILD_PRESETS", ("dev-release", "dev-debug"))
    monkeypatch.setattr(smoke, "_abi_expected", lambda: (2, 0))

    seen: list[tuple[Path, int, int]] = []

    def fake_is_abi_compatible(path: Path, major: int, minor: int) -> tuple[bool, str]:
        seen.append((path, major, minor))
        if path == first.resolve():
            return False, "too old"
        return True, "ok"

    monkeypatch.setattr(smoke, "_is_abi_compatible", fake_is_abi_compatible)

    assert smoke.discover_lib() == second.resolve()
    assert seen == [
        (first.resolve(), 2, 0),
        (second.resolve(), 2, 0),
    ]


def test_discover_lib_reports_why_candidates_were_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke_module()
    candidate = tmp_path / "build" / "dev-release" / "cpp" / "src" / "libn4m.so"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("candidate", encoding="utf-8")

    monkeypatch.setattr(smoke, "REPO", tmp_path)
    monkeypatch.setattr(smoke, "BUILD_PRESETS", ("dev-release",))
    monkeypatch.setattr(smoke, "_abi_expected", lambda: (2, 1))
    monkeypatch.setattr(
        smoke,
        "_is_abi_compatible",
        lambda path, major, minor: (False, "ABI check rejected header 2.1.x with status 7"),
    )

    with pytest.raises(smoke.SmokeError) as excinfo:
        smoke.discover_lib()

    message = str(excinfo.value)
    assert "no ABI-compatible libn4m build found for header ABI 2.1.x." in message
    assert str(candidate) in message
    assert "status 7" in message


def test_install_and_run_scrubs_host_python_and_lib_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke_module()
    wheel = tmp_path / "wheelhouse" / "nirs4all_methods-1.0.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    wheel.write_text("wheel", encoding="utf-8")

    monkeypatch.setenv("PYTHONPATH", "/host/pythonpath")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("N4M_LIB_PATH", "/host/libn4m.so")
    monkeypatch.setenv("PLS4ALL_LIB_PATH", "/host/pls4all/libn4m.so")
    monkeypatch.setattr(smoke, "_venv_python", lambda venv_dir: venv_dir / "bin" / "python")

    calls: list[tuple[list[str], Path | None, dict[str, str] | None]] = []

    def fake_run(
        cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        calls.append((cmd, cwd, env))
        if env is None:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        payload = {
            "status": "INSTALLED_N4M_OK",
            "module": str(tmp_path / "venv" / "lib" / "python3.11" / "site-packages" / "n4m" / "__init__.py"),
            "library": str(tmp_path / "venv" / "lib" / "python3.11" / "site-packages" / "n4m" / "lib" / "libn4m.so"),
            "abi": [2, 0, 0],
            "prediction_checksum": 1.25,
        }
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(smoke, "_run", fake_run)

    result = smoke._install_and_run(wheel, tmp_path, "python3")

    assert result["status"] == "INSTALLED_N4M_OK"
    assert [call[0][:3] for call in calls[:2]] == [
        ["python3", "-m", "venv"],
        [str(tmp_path / "venv" / "bin" / "python"), "-m", "pip"],
    ]
    child_cmd, child_cwd, child_env = calls[2]
    assert child_cmd[:2] == [str(tmp_path / "venv" / "bin" / "python"), "-c"]
    assert child_cwd == tmp_path
    assert child_env is not None
    assert child_env["N4M_EXPECTED_PREFIX"] == str(tmp_path / "venv")
    for forbidden in (
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "N4M_LIB_PATH",
        "PLS4ALL_LIB_PATH",
    ):
        assert forbidden not in child_env
