# SPDX-License-Identifier: CECILL-2.1
"""Release package surface guards for the dual Python distributions."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _header_abi() -> tuple[int, int, int]:
    """Canonical ABI triple from ``cpp/include/n4m/n4m_version.h``."""
    text = (REPO / "cpp/include/n4m/n4m_version.h").read_text(encoding="utf-8")

    def macro(name: str) -> int:
        match = re.search(rf"#define\s+{name}\s+(\d+)", text)
        assert match is not None, f"{name} not found in n4m_version.h"
        return int(match.group(1))

    return (
        macro("N4M_ABI_VERSION_MAJOR"),
        macro("N4M_ABI_VERSION_MINOR"),
        macro("N4M_ABI_VERSION_PATCH"),
    )


def _load_package_generator():
    script = REPO / "bindings/python/scripts/make_python_package.py"
    spec = importlib.util.spec_from_file_location(
        "make_python_package_under_test", script
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_packages_explain_distribution_import_split(tmp_path, monkeypatch):
    generator = _load_package_generator()
    monkeypatch.setattr(generator, "REPO", tmp_path)
    monkeypatch.setattr(generator, "SRC_PKG", REPO / "bindings/python")

    full = generator.generate("nirs4all-methods")
    slim = generator.generate("pls4all")

    full_meta = (full / "pyproject.toml").read_text(encoding="utf-8")
    slim_meta = (slim / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "nirs4all-methods"' in full_meta
    assert 'Repository = "https://github.com/GBeurier/nirs4all-methods"' in full_meta
    assert 'name = "pls4all"' in slim_meta
    assert 'Repository = "https://github.com/GBeurier/nirs4all-methods"' in slim_meta

    full_readme = (full / "README.md").read_text(encoding="utf-8")
    full_readme_oneline = " ".join(full_readme.split())
    assert "distribution name is `nirs4all-methods`; the import package is `n4m`" in (
        full_readme_oneline
    )
    assert "from n4m.estimators.regression.latent import PLS" in full_readme
    assert "there is no flat `n4m.sklearn` package" in full_readme_oneline
    assert (full / "src/n4m/__init__.py").is_file()
    assert not (full / "src/pls4all").exists()

    slim_readme = (slim / "README.md").read_text(encoding="utf-8")
    slim_readme_oneline = " ".join(slim_readme.split())
    assert "slim, PLS-focused subset distribution" in slim_readme
    assert "from pls4all.sklearn import PLSRegression" in slim_readme
    assert "Install `nirs4all-methods` when you need the full" in slim_readme_oneline
    assert (slim / "src/pls4all/__init__.py").is_file()
    assert (slim / "src/pls4all/migration.py").is_file()
    assert "from n4m" not in (slim / "src/pls4all/migration.py").read_text(encoding="utf-8")
    assert not (slim / "src/n4m").exists()


def test_generated_readmes_document_current_abi(tmp_path, monkeypatch):
    """The generated PyPI READMEs are the packages' PyPI long_description, and
    they bake the ABI version literally (``+abi.2.0.0`` / ``(2, 0, 0)``) in the
    make_python_package.py templates — only the project ``{version}`` prefix is
    interpolated. That hardcoded ABI silently rots on an ABI bump (a checked-in
    stale artifact was found at ``abi.1.22.0``). Pin the generated ABI claim to
    ``cpp/include/n4m/n4m_version.h`` so bumping the ABI forces the templates to
    move with it. The static checked-in binding READMEs are guarded separately in
    test_binding_readme_abi_claims.py.
    """
    generator = _load_package_generator()
    monkeypatch.setattr(generator, "REPO", tmp_path)
    monkeypatch.setattr(generator, "SRC_PKG", REPO / "bindings/python")

    major, minor, patch = _header_abi()
    expected = f"{major}.{minor}.{patch}"

    for dist in ("nirs4all-methods", "pls4all"):
        readme = (generator.generate(dist) / "README.md").read_text(encoding="utf-8")

        abi_strings = sorted({s for s in re.findall(r"abi\.(\d+\.\d+\.\d+)", readme)})
        assert abi_strings, f"{dist} generated README documents no `+abi.<x.y.z>` string"
        stale = [s for s in abi_strings if s != expected]
        assert not stale, (
            f"{dist} generated README documents stale ABI {stale}; header is "
            f"{expected} — update the readme templates in make_python_package.py"
        )

        match = re.search(
            r"abi_version\(\)[^\n]*?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", readme
        )
        assert match is not None, (
            f"{dist} generated README documents no abi_version() example triple"
        )
        triple = tuple(int(x) for x in match.groups())
        assert triple == (major, minor, patch), (
            f"{dist} generated README documents ABI triple {triple}; header is "
            f"{(major, minor, patch)} — update make_python_package.py"
        )
