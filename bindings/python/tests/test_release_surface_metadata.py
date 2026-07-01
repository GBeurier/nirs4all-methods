# SPDX-License-Identifier: CECILL-2.1
"""Release package surface guards for the dual Python distributions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


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
    assert not (slim / "src/n4m").exists()
