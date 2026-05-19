# SPDX-License-Identifier: CECILL-2.1
"""Rewrite cibuildwheel/auditwheel-tagged wheels to ``py3-none-${platform}``.

The chemometrics4all Python binding is ctypes-only: no CPython extension is
compiled, but the wheel embeds the native ``libc4a`` runtime. Setuptools first
emits a platform wheel with a CPython ABI tag so repair tools run; this script
then rewrites the filename, WHEEL metadata and RECORD hash to the canonical
``py3-none-${platform}`` form.
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

_WHEEL_NAME_RE = re.compile(
    r"^(?P<dist>[A-Za-z0-9_]+(?:-[A-Za-z0-9_.]+)*)-(?P<ver>[^-]+)"
    r"-(?P<build>[^-]+-)?"
    r"(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[A-Za-z0-9_.]+)"
    r"\.whl$"
)


def _record_hash(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _clone_zipinfo(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
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
    new_python: str,
    new_abi: str,
) -> str:
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
            f"WHEEL metadata has no Tag: line matching prefix '{old_prefix}*'"
        )
    return "".join(out_lines)


def _rewrite_record(
    text: str,
    wheel_member: str,
    new_hash: str,
    new_size: int,
) -> str:
    in_stream = io.StringIO(text)
    out_stream = io.StringIO()
    writer = csv.writer(out_stream, lineterminator="\n")
    reader = csv.reader(in_stream)
    updated = False
    for row in reader:
        if not row:
            continue
        if row[0] == wheel_member:
            writer.writerow([row[0], new_hash, str(new_size)])
            updated = True
        else:
            writer.writerow(row)
    if not updated:
        raise RuntimeError(f"RECORD has no entry for {wheel_member}")
    return out_stream.getvalue()


def _retag_wheel(src: Path, dest_dir: Path) -> Path:
    match = _WHEEL_NAME_RE.match(src.name)
    if match is None:
        raise ValueError(f"unrecognised wheel filename: {src.name}")

    python_tag = match.group("python")
    abi_tag = match.group("abi")
    platform_tag = match.group("platform")
    if python_tag == "py3" and abi_tag == "none":
        dest = dest_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return dest

    build = match.group("build")
    build_part = f"-{build.rstrip('-')}" if build else ""
    dest = dest_dir / (
        f"{match.group('dist')}-{match.group('ver')}{build_part}"
        f"-py3-none-{platform_tag}.whl"
    )

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
                raise RuntimeError(f"wheel {src.name} is missing WHEEL or RECORD")

            wheel_payload = _rewrite_wheel_metadata(
                zin.read(wheel_info).decode("utf-8"),
                python_tag,
                abi_tag,
                "py3",
                "none",
            ).encode("utf-8")
            wheel_hash = _record_hash(wheel_payload)
            wheel_size = len(wheel_payload)

            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
                for info in infos:
                    if info.filename == wheel_info.filename:
                        zout.writestr(_clone_zipinfo(info), wheel_payload)
                    elif info.filename == record_info.filename:
                        record_payload = _rewrite_record(
                            zin.read(record_info).decode("utf-8"),
                            wheel_info.filename,
                            wheel_hash,
                            wheel_size,
                        ).encode("utf-8")
                        zout.writestr(_clone_zipinfo(info), record_payload)
                    else:
                        zout.writestr(_clone_zipinfo(info), zin.read(info))
        tmp.replace(dest)
        tmp = None
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink(missing_ok=True)

    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "wheelhouse",
        type=Path,
        help="directory containing wheels to retag",
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

    out_dir = args.wheelhouse if args.inplace else args.wheelhouse / "retagged"
    out_dir.mkdir(exist_ok=True)

    for src in wheels:
        try:
            dest = _retag_wheel(src, out_dir)
        except Exception as exc:
            print(f"error: failed to retag {src.name}: {exc}", file=sys.stderr)
            return 1
        if args.inplace and dest != src:
            src.unlink()
        print(f"  {src.name} -> {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
