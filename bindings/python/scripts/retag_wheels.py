# SPDX-License-Identifier: CECILL-2.1
"""Rewrite cibuildwheel/auditwheel-tagged wheels to ``py3-none-${platform}``.

The pls4all Python binding is ctypes-only — no CPython extension is
compiled, no symbols are linked against a specific CPython ABI. Yet
``cibuildwheel`` produces wheels tagged ``cp3X-cp3X-${platform}``
because the :file:`setup.py` shim forces ``has_ext_modules() == True``
(so the repair step bundles libp4a). We *want* the platform tag, but
not the CPython-ABI prefix.

This script post-processes a directory of wheels (typically
``wheelhouse/``) and rewrites each ``pls4all-${VER}-cp3X-cp3X-${plat}.whl``
to ``pls4all-${VER}-py3-none-${plat}.whl`` in three lockstep changes:

1. the wheel filename;
2. the ``Tag:`` lines inside ``${distinfo}/WHEEL``;
3. the corresponding entries in ``${distinfo}/RECORD`` (including the
   sha256 + size for ``WHEEL`` itself after step 2).

Renaming just the filename produces a wheel that pip rejects because the
``WHEEL`` metadata and the filename disagree. Rewriting ``WHEEL`` without
updating ``RECORD`` also fails ``pip install --check`` because the
recorded hash no longer matches.

Usage::

    python scripts/retag_wheels.py wheelhouse/

The script is idempotent — already-retagged wheels are left untouched.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Format reference: PEP 427 (wheel binary format) + PEP 425 (compatibility tags).
# Tag triple: <interpreter>-<abi>-<platform>, where interpreter is "cp3X"
# (CPython) / "pp3X" (PyPy) / "py3" (any-Python), abi is "cp3X" / "abi3" /
# "none", platform is e.g. "manylinux_2_31_x86_64".
_WHEEL_NAME_RE = re.compile(
    r"^(?P<dist>[A-Za-z0-9_]+(?:-[A-Za-z0-9_.]+)*)-(?P<ver>[^-]+)"
    r"-(?P<build>[^-]+-)?"
    r"(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^.]+)"
    r"\.whl$"
)


def _record_hash(payload: bytes) -> str:
    """Return the URL-safe base64-encoded sha256 used by PEP 376 RECORDs."""
    digest = hashlib.sha256(payload).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _retag_wheel(src: Path, dest_dir: Path) -> Path:
    """Rewrite a single wheel; return the destination path."""
    m = _WHEEL_NAME_RE.match(src.name)
    if m is None:
        raise ValueError(f"unrecognised wheel filename: {src.name}")

    python_tag = m.group("python")
    abi_tag = m.group("abi")
    platform_tag = m.group("platform")

    if python_tag == "py3" and abi_tag == "none":
        # Already py3-none-${platform}; nothing to do beyond a copy.
        dest = dest_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return dest

    new_python_tag = "py3"
    new_abi_tag = "none"
    new_name = (
        f"{m.group('dist')}-{m.group('ver')}"
        f"{('-' + m.group('build').rstrip('-')) if m.group('build') else ''}"
        f"-{new_python_tag}-{new_abi_tag}-{platform_tag}.whl"
    )
    dest = dest_dir / new_name

    # Operate on a temp file so we never half-write the destination wheel.
    fd, tmp_path = tempfile.mkstemp(suffix=".whl", dir=dest_dir)
    tmp = Path(tmp_path)
    new_wheel_payload: bytes | None = None
    record_member: str | None = None
    wheel_member: str | None = None

    try:
        # Find ${distinfo}/WHEEL + ${distinfo}/RECORD members from the source.
        with zipfile.ZipFile(src, "r") as zin:
            members = zin.namelist()
            for name in members:
                if name.endswith(".dist-info/WHEEL"):
                    wheel_member = name
                elif name.endswith(".dist-info/RECORD"):
                    record_member = name
            if wheel_member is None or record_member is None:
                raise RuntimeError(
                    f"wheel {src.name} is missing WHEEL or RECORD; "
                    "cannot retag a malformed wheel"
                )

            # Build the new WHEEL metadata in memory.
            original_wheel = zin.read(wheel_member).decode("utf-8")
            new_wheel_payload = _rewrite_wheel_metadata(
                original_wheel,
                python_tag,
                abi_tag,
                platform_tag,
                new_python_tag,
                new_abi_tag,
            ).encode("utf-8")

            # Compute the new RECORD entry for WHEEL (must be done before
            # writing the new zip).
            wheel_hash = _record_hash(new_wheel_payload)
            wheel_size = len(new_wheel_payload)

            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
                for name in members:
                    if name == wheel_member:
                        zout.writestr(name, new_wheel_payload)
                    elif name == record_member:
                        new_record = _rewrite_record(
                            zin.read(record_member).decode("utf-8"),
                            wheel_member,
                            wheel_hash,
                            wheel_size,
                        )
                        zout.writestr(name, new_record.encode("utf-8"))
                    else:
                        zout.writestr(name, zin.read(name))
        tmp.replace(dest)
    finally:
        # mkstemp opened a fd we never use — close it to avoid leaks under
        # pytest-temp scoping. The file itself is renamed above if we got
        # that far; otherwise it lingers under dest_dir as a fragment and
        # the next run will overwrite it.
        try:
            import os as _os
            _os.close(fd)
        except OSError:
            pass
        if tmp.exists() and tmp.name != dest.name:
            tmp.unlink(missing_ok=True)

    return dest


def _rewrite_wheel_metadata(
    text: str,
    old_python: str,
    old_abi: str,
    old_platform: str,
    new_python: str,
    new_abi: str,
) -> str:
    """Rewrite the ``Tag:`` line(s) inside a ``WHEEL`` metadata blob."""
    old_tag = f"{old_python}-{old_abi}-{old_platform}"
    new_tag = f"{new_python}-{new_abi}-{old_platform}"
    out_lines: list[str] = []
    saw_tag = False
    for line in text.splitlines(keepends=True):
        if line.startswith("Tag:"):
            saw_tag = True
            current = line[len("Tag:"):].strip()
            # Only rewrite the tag we know about, leave any siblings.
            replacement = new_tag if current == old_tag else current
            out_lines.append(f"Tag: {replacement}\n")
        else:
            out_lines.append(line)
    if not saw_tag:
        raise RuntimeError("WHEEL metadata has no Tag: line")
    return "".join(out_lines)


def _rewrite_record(
    text: str,
    wheel_member: str,
    new_hash: str,
    new_size: int,
) -> str:
    """Update the ``WHEEL`` row in a ``RECORD`` CSV (per PEP 376)."""
    in_stream = io.StringIO(text)
    out_stream = io.StringIO()
    writer = csv.writer(out_stream, lineterminator="\n")
    reader = csv.reader(in_stream)
    updated = False
    for row in reader:
        if not row:
            continue
        path = row[0]
        if path == wheel_member:
            writer.writerow([path, new_hash, str(new_size)])
            updated = True
        elif path.endswith(".dist-info/RECORD"):
            # RECORD's own row records its own hash/size as empty strings
            # per PEP 376 (sha256, length intentionally blank). Preserve.
            writer.writerow(row)
        else:
            writer.writerow(row)
    if not updated:
        raise RuntimeError(
            f"RECORD has no entry for {wheel_member}; "
            "the wheel is malformed."
        )
    return out_stream.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "wheelhouse",
        type=Path,
        help="directory containing the wheels to retag (e.g. wheelhouse/)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="rewrite into the same directory and remove the original",
    )
    args = parser.parse_args(argv)

    if not args.wheelhouse.is_dir():
        print(f"error: {args.wheelhouse} is not a directory", file=sys.stderr)
        return 2

    wheels = sorted(args.wheelhouse.glob("*.whl"))
    if not wheels:
        print(f"warning: no wheels found in {args.wheelhouse}", file=sys.stderr)
        return 0

    out_dir = args.wheelhouse if args.inplace else (args.wheelhouse / "retagged")
    out_dir.mkdir(exist_ok=True)

    for src in wheels:
        try:
            dest = _retag_wheel(src, out_dir)
        except Exception as exc:  # pragma: no cover — fail loudly for diagnostics.
            print(f"error: failed to retag {src.name}: {exc}", file=sys.stderr)
            return 1
        if args.inplace and dest != src:
            src.unlink()
        rel_src = src.relative_to(args.wheelhouse)
        rel_dest = dest.relative_to(args.wheelhouse)
        print(f"  {rel_src} -> {rel_dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
