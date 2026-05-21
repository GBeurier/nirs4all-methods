# SPDX-License-Identifier: CECILL-2.1
"""Rewrite cibuildwheel/auditwheel-tagged wheels to ``py3-none-${platform}``.

The pls4all Python binding is ctypes-only — no CPython extension is
compiled, no symbols are linked against a specific CPython ABI. Yet
``cibuildwheel`` produces wheels tagged ``cp3X-cp3X-${platform}``
because the :file:`setup.py` shim forces ``has_ext_modules() == True``
(so the repair step bundles libn4m). We *want* the platform tag, but
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
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Format reference: PEP 427 (wheel binary format) + PEP 425 (compatibility tags).
# Tag triple: <interpreter>-<abi>-<platform>, where interpreter is "cp3X"
# (CPython) / "pp3X" (PyPy) / "py3" (any-Python), abi is "cp3X" / "abi3" /
# "none", platform is e.g. "manylinux_2_31_x86_64" — or, when a wheel is
# compatible with several platform tags, dot-joined like
# "manylinux_2_17_x86_64.manylinux2014_x86_64".
#
# We deliberately allow dots inside the platform segment so that compressed
# tags are not rejected.
_WHEEL_NAME_RE = re.compile(
    r"^(?P<dist>[A-Za-z0-9_]+(?:-[A-Za-z0-9_.]+)*)-(?P<ver>[^-]+)"
    r"-(?P<build>[^-]+-)?"
    r"(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[A-Za-z0-9_.]+)"
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
    # Close the fd immediately — on Windows, opening the path again as a
    # ZipFile fails if the fd is still held open by this process.
    fd, tmp_path = tempfile.mkstemp(suffix=".whl", dir=dest_dir)
    os.close(fd)
    tmp = Path(tmp_path)

    try:
        with zipfile.ZipFile(src, "r") as zin:
            infos = zin.infolist()
            wheel_info = next(
                (i for i in infos if i.filename.endswith(".dist-info/WHEEL")),
                None,
            )
            record_info = next(
                (i for i in infos if i.filename.endswith(".dist-info/RECORD")),
                None,
            )
            if wheel_info is None or record_info is None:
                raise RuntimeError(
                    f"wheel {src.name} is missing WHEEL or RECORD; "
                    "cannot retag a malformed wheel"
                )

            # Build the new WHEEL metadata in memory.
            original_wheel = zin.read(wheel_info).decode("utf-8")
            new_wheel_payload = _rewrite_wheel_metadata(
                original_wheel,
                python_tag,
                abi_tag,
                platform_tag,
                new_python_tag,
                new_abi_tag,
            ).encode("utf-8")
            wheel_hash = _record_hash(new_wheel_payload)
            wheel_size = len(new_wheel_payload)

            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
                # Preserve each member's ZipInfo metadata (mtime, external_attr,
                # compression, file flags) so the rewritten wheel matches the
                # source byte-for-byte except for the two updated members.
                for info in infos:
                    if info.filename == wheel_info.filename:
                        new_info = _clone_zipinfo(info)
                        zout.writestr(new_info, new_wheel_payload)
                    elif info.filename == record_info.filename:
                        new_record = _rewrite_record(
                            zin.read(record_info).decode("utf-8"),
                            wheel_info.filename,
                            wheel_hash,
                            wheel_size,
                        ).encode("utf-8")
                        zout.writestr(_clone_zipinfo(info), new_record)
                    else:
                        # Stream the original member bytes back through a
                        # ZipInfo clone to keep metadata intact.
                        zout.writestr(_clone_zipinfo(info), zin.read(info))
        # Final rename. .replace is atomic on POSIX + Windows when source +
        # destination are on the same filesystem (they are — both inside
        # dest_dir).
        tmp.replace(dest)
        tmp = None  # mark as moved so the finally branch leaves it alone.
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink(missing_ok=True)

    return dest


def _clone_zipinfo(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    """Return a fresh ZipInfo carrying the same metadata as ``info``.

    Reusing the original ZipInfo across ZipFile boundaries can mutate it; a
    clone is safer.
    """
    clone = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
    clone.compress_type = info.compress_type
    clone.external_attr = info.external_attr
    clone.create_system = info.create_system
    clone.internal_attr = info.internal_attr
    clone.flag_bits = info.flag_bits
    clone.extract_version = info.extract_version
    clone.create_version = info.create_version
    return clone


def _rewrite_wheel_metadata(
    text: str,
    old_python: str,
    old_abi: str,
    old_platform: str,  # noqa: ARG001 — kept for API symmetry
    new_python: str,
    new_abi: str,
) -> str:
    """Rewrite every matching ``Tag:`` line inside a ``WHEEL`` metadata blob.

    auditwheel / delocate often expand a single compressed filename tag like
    ``manylinux_2_17_x86_64.manylinux2014_x86_64`` into multiple separate
    ``Tag:`` lines inside ``WHEEL`` — one per dot-joined component. We must
    therefore rewrite **every** line whose interpreter/abi prefix matches the
    wheel name, not just the one that exactly equals the filename's
    dot-joined platform form.
    """
    old_prefix = f"{old_python}-{old_abi}-"
    new_prefix = f"{new_python}-{new_abi}-"
    out_lines: list[str] = []
    saw_tag = False
    rewritten = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("Tag:"):
            saw_tag = True
            current = line[len("Tag:"):].strip()
            if current.startswith(old_prefix):
                # Strip the interpreter/abi prefix and re-prefix; preserves
                # the platform segment exactly (including dotted forms).
                platform_part = current[len(old_prefix):]
                out_lines.append(f"Tag: {new_prefix}{platform_part}\n")
                rewritten += 1
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)
    if not saw_tag:
        raise RuntimeError("WHEEL metadata has no Tag: line")
    if rewritten == 0:
        raise RuntimeError(
            f"WHEEL metadata has no Tag: line matching prefix '{old_prefix}*'; "
            "filename and metadata disagree."
        )
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
