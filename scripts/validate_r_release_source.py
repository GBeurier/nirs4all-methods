#!/usr/bin/env python3
"""Allow the first R tarball's packaging repair without changing a release tag.

The numerical core, R transport, tests, and fixtures must still match the
immutable stable release. No registry artifact is replaced by this workflow.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

ALLOWED = frozenset({
    "bindings/r/n4m/DESCRIPTION",
    ".github/workflows/release-r.yml",
    "scripts/validate_r_release_source.py",
    "scripts/tests/test_validate_r_release_source.py",
})


def validate(repo: Path, tag: str, expected: str) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag) is None:
        raise ValueError("An existing stable release tag is required")
    if re.fullmatch(r"[0-9a-f]{40}", expected) is None or git("rev-parse", "HEAD") != expected:
        raise ValueError("Exact reviewed source SHA is required")
    base = git("rev-parse", f"refs/tags/{tag}^{{commit}}")
    subprocess.run(["git", "merge-base", "--is-ancestor", base, expected], cwd=repo, check=True)
    changed = set(git("diff", "--name-only", base, expected).splitlines())
    if changed - ALLOWED:
        raise ValueError(f"Recovery changes files outside R packaging: {sorted(changed - ALLOWED)}")
    desc = (repo / "bindings/r/n4m/DESCRIPTION").read_text()
    if re.search(rf"(?m)^Version: {re.escape(tag[1:])}$", desc) is None:
        raise ValueError("R package version differs from the immutable release tag")
    return {"tag": tag, "native_source": base, "packaging_source": expected}


if __name__ == "__main__":
    print(validate(Path(__file__).resolve().parents[1], os.environ.get("RELEASE_TAG", ""), os.environ.get("EXPECTED_SOURCE_SHA", "")))
