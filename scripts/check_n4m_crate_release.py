#!/usr/bin/env python3
"""Validate the n4m crate release identity and workflow safety contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "bindings/rust/n4m/Cargo.toml"
LOCKFILE = ROOT / "Cargo.lock"
WORKFLOW = ROOT / ".github/workflows/release-n4m-crate.yml"


def crate_identity() -> tuple[str, str]:
    manifest = MANIFEST.read_text(encoding="utf-8")
    package = re.search(r"(?ms)^\[package\]\s*(.*?)(?=^\[|\Z)", manifest)
    if package is None:
        raise SystemExit("bindings/rust/n4m/Cargo.toml has no [package] table")
    fields = dict(
        re.findall(r'^([a-z-]+)\s*=\s*"([^"]+)"\s*$', package.group(1), re.MULTILINE)
    )
    name = fields.get("name")
    version = fields.get("version")
    if name != "n4m" or version is None:
        raise SystemExit("bindings/rust/n4m/Cargo.toml must declare package n4m with a version")
    if re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version) is None:
        raise SystemExit(f"n4m crate version is not strict semver X.Y.Z: {version!r}")

    lock = LOCKFILE.read_text(encoding="utf-8")
    entries = re.findall(r"(?ms)^\[\[package\]\]\s*(.*?)(?=^\[\[package\]\]|\Z)", lock)
    matches = 0
    for entry in entries:
        lock_fields = dict(
            re.findall(r'^([a-z-]+)\s*=\s*"([^"]+)"\s*$', entry, re.MULTILINE)
        )
        matches += lock_fields.get("name") == name and lock_fields.get("version") == version
    if matches != 1:
        raise SystemExit(f"Cargo.lock must contain exactly one {name} {version} package entry")
    return name, version


def validate_tag(tag: str, name: str, version: str) -> None:
    expected = f"{name}-v{version}"
    if tag != expected:
        raise SystemExit(f"release tag must be exactly {expected!r}, got {tag!r}")


def validate_workflow_contract() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    required = (
        'tags: ["n4m-v*.*.*"]',
        "workflow_dispatch: {}",
        "cargo package --locked -p n4m",
        "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/n4m-v')",
        "name: crates-io",
        'if [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then',
        "cargo publish --locked -p n4m",
    )
    missing = [fragment for fragment in required if fragment not in workflow]
    if missing:
        raise SystemExit("n4m release workflow contract is missing: " + ", ".join(missing))
    if workflow.count("cargo publish") != 1:
        raise SystemExit("n4m release workflow must contain exactly one cargo publish command")
    if "workflow_dispatch:\n" in workflow and "workflow_dispatch: {}" not in workflow:
        raise SystemExit("workflow_dispatch must remain input-free and dry-run only")


def write_outputs(name: str, version: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"crate={name}\nversion={version}\ntag={name}-v{version}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="component tag to compare with Cargo.toml")
    parser.add_argument("--workflow-contract", action="store_true")
    args = parser.parse_args()

    name, version = crate_identity()
    if args.tag:
        validate_tag(args.tag, name, version)
    if args.workflow_contract:
        validate_workflow_contract()
    write_outputs(name, version)
    print(f"validated {name} {version} ({name}-v{version})")


if __name__ == "__main__":
    main()
