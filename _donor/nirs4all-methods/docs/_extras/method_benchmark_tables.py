"""Per-method benchmark table refresh for Sphinx builds.

Generated method pages contain a committed ``### Benchmarks`` section. This
hook rebuilds just that section at ``sphinx-build`` time from the same payload
that powers the interactive dashboard, so new benchmark CSV rows update the
documentation without re-running ``build_methods.py`` by hand.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "benchmarks" / "cross_binding" / "results"
STATIC_JSON = ROOT / "docs" / "_static" / "bench-data.json"
SNAPSHOT_PATH = ROOT / "docs" / "_extras" / "method-benchmarks.json"

_SECTION_RE = re.compile(
    r"^### Benchmarks\b.*?(?=\n---\s*\n\s*_See also_:|\Z)",
    re.MULTILINE | re.DOTALL,
)
_LOG = logging.getLogger(__name__)


@dataclass
class _MethodRef:
    method_id: str


def _import_build_methods():
    import build_methods  # type: ignore[import-not-found]

    return build_methods


def _import_build_landing():
    import build_landing  # type: ignore[import-not-found]

    return build_landing


def has_live_csv(results_dir: Path = RESULTS_DIR) -> bool:
    build_landing = _import_build_landing()
    if hasattr(build_landing, "_result_csv_paths"):
        return bool(build_landing._result_csv_paths(results_dir))
    return (results_dir / "full_matrix.csv").exists()


def build_payload_from_csv(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    build_landing = _import_build_landing()
    payload = build_landing.build_payload(results_dir)["payload"]
    return payload


def save_static_payload(payload: dict[str, Any], path: Path = STATIC_JSON) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        _LOG.warning("benchmark payload snapshot write failed (%s)", exc)
        return False
    return True


def build_blocks_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    build_methods = _import_build_methods()
    algos = sorted({
        str(row.get("algo"))
        for row in payload.get("rows", [])
        if row.get("algo")
    })
    blocks: dict[str, str] = {}
    for algo in algos:
        blocks[algo] = (
            build_methods.render_benchmarks(_MethodRef(algo), payload).rstrip() + "\n"
        )
    return blocks


def save_snapshot(blocks: dict[str, str], path: Path = SNAPSHOT_PATH) -> bool:
    payload = {
        "schema": 1,
        "source": "build_methods.render_benchmarks",
        "blocks": blocks,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        _LOG.warning(
            "method-benchmark snapshot write failed (%s); continuing with in-memory blocks.",
            exc,
        )
        return False
    return True


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    blocks = data.get("blocks", {})
    return {key: value for key, value in blocks.items() if isinstance(value, str)}


def load_static_payload(path: Path = STATIC_JSON) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_or_build_blocks(
    bench_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, str], str]:
    """Return ``(blocks, source_tag)`` for the current docs build.

    Live CSV/dashboard data wins over the committed snapshot. If a partial
    benchmark refresh only covers a subset of methods, the old snapshot fills
    the untouched methods so pages do not lose their last known benchmark rows.
    """
    snapshot = load_snapshot()
    source = "payload"
    payload = bench_payload

    if payload is None and has_live_csv():
        payload = build_payload_from_csv()
        save_static_payload(payload)
        source = "csv"
    elif payload is None:
        payload = load_static_payload()
        source = "static-json"

    if payload and payload.get("rows"):
        fresh = build_blocks_from_payload(payload)
        merged = {**snapshot, **fresh}
        if fresh:
            save_snapshot(merged)
        tag = f"{source}+snapshot" if snapshot and len(fresh) < len(merged) else source
        return merged, tag

    if snapshot:
        return snapshot, "snapshot"
    return {}, "none"


def replace_benchmark_section(markdown: str, new_block: str) -> str:
    if "### Benchmarks" not in markdown:
        return markdown
    return _SECTION_RE.sub(new_block.rstrip() + "\n", markdown, count=1)


def install_sphinx_hook(app, blocks: dict[str, str]) -> None:
    if not blocks:
        return

    def _on_source_read(_app, docname, source):
        if not docname.startswith("methods/"):
            return
        method = docname.split("/", 1)[1]
        block = blocks.get(method)
        if block is None:
            return
        source[0] = replace_benchmark_section(source[0], block)

    app.connect("source-read", _on_source_read)


if __name__ == "__main__":  # pragma: no cover
    payload = build_payload_from_csv()
    save_static_payload(payload)
    blocks = build_blocks_from_payload(payload)
    save_snapshot(blocks)
    print(f"wrote {len(blocks)} method-benchmark blocks to {SNAPSHOT_PATH.relative_to(ROOT)}")
