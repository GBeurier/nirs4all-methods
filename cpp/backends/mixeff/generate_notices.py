"""Regenerate the locked native/WASM dependency inventory and complete notices.

Run after cargo fetch --locked. Metadata paths are local build inputs only;
absolute paths are not written into distribution artifacts.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
manifest = ROOT / "Cargo.toml"
all_packages: dict[str, dict] = {}
target_packages = {}
for target in ("x86_64-unknown-linux-gnu", "wasm32-unknown-emscripten"):
    metadata = json.loads(
        subprocess.check_output(
            [
                "cargo",
                "metadata",
                "--locked",
                "--manifest-path",
                str(manifest),
                "--filter-platform",
                target,
                "--format-version",
                "1",
            ],
            text=True,
        )
    )
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    root_id = metadata["resolve"]["root"]
    seen, pending = {root_id}, [root_id]
    while pending:
        for dep in nodes[pending.pop()]["deps"]:
            if (
                any(kind["kind"] in (None, "build") for kind in dep["dep_kinds"])
                and dep["pkg"] not in seen
            ):
                seen.add(dep["pkg"])
                pending.append(dep["pkg"])
    packages = [
        package
        for package in metadata["packages"]
        if package["id"] in seen and package["id"] != root_id
    ]
    target_packages[target] = sorted(
        f"{package['name']}@{package['version']}" for package in packages
    )
    all_packages.update({package["id"]: package for package in packages})

inventory, notices = [], []
for package in sorted(
    all_packages.values(), key=lambda item: (item["name"], item["version"])
):
    folder = Path(package["manifest_path"]).parent
    files = {
        path
        for path in folder.iterdir()
        if path.is_file()
        and re.match(
            r"^(license|licence|copying|copyright|unlicense)([-._]|$)",
            path.name,
            re.IGNORECASE,
        )
    }
    for name in ("LICENSES", "licenses"):
        if (folder / name).is_dir():
            files.update(path for path in (folder / name).rglob("*") if path.is_file())
    if package["license_file"]:
        files.add(folder / package["license_file"])
    # Some tiny crates publish only a repository-wide license and headers.
    # Refuse an incomplete notice inventory instead of guessing a license text.
    if not files:
        raise RuntimeError(
            f"No license file found for {package['name']}@{package['version']}"
        )
    names = []
    for path in sorted(files):
        relative = path.relative_to(folder).as_posix()
        names.append(relative)
        notices.append(
            f"===== {package['name']}@{package['version']}: {relative} =====\n\n{path.read_text(encoding='utf-8')}"
        )
    inventory.append(
        {
            "name": package["name"],
            "version": package["version"],
            "declared_license": package["license"],
            "license_files": names,
            "repository": package["repository"],
        }
    )

sysroot = Path(
    subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
)
compiler = subprocess.check_output(["rustc", "--version"], text=True).strip()
# Staticlib includes the Rust standard runtime. Preserve its distributed notice
# bundle as well as individual registry-crate notices, including third parties.
compiler_version = compiler.split()[1]
for name in ("LICENSE-MIT", "LICENSE-APACHE", "COPYRIGHT"):
    path = sysroot / "share/doc/rust" / name
    if not path.exists():
        path = ROOT / "licenses" / f"rust-{compiler_version}" / name
    notices.append(
        f"===== Rust standard runtime ({compiler}): {name} =====\n\n{path.read_text(encoding='utf-8')}"
    )
library_notice = sysroot / "share/doc/rust/COPYRIGHT-library.html"
# Remove trailing layout-only whitespace; all copyright and license words remain.
(ROOT / "RUST_STANDARD_LIBRARY_NOTICES.html").write_text(
    "\n".join(
        line.rstrip()
        for line in library_notice.read_text(encoding="utf-8").splitlines()
    )
    + "\n",
    encoding="utf-8",
)
(ROOT / "DEPENDENCIES.json").write_text(
    json.dumps(
        {
            "schema_version": "n4m.mixeff-dependencies/1",
            "compiler": compiler,
            "targets": target_packages,
            "packages": inventory,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
(ROOT / "THIRD_PARTY_NOTICES.txt").write_text("\n\n".join(notices), encoding="utf-8")
print(f"Recorded {len(inventory)} registry crates plus Rust standard-runtime notices")
