"""Build the JSON payload for the benchmark-status landing page."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import subprocess
from pathlib import Path


PLACEHOLDER_COLUMN = "chemometrics4all.reference-registry"


def _validation_payload() -> dict:
    try:
        from validation_registry import load_snapshot

        return load_snapshot().dashboard_payload()
    except Exception as exc:  # pragma: no cover - defensive docs fallback
        return {
            "schema_version": 0,
            "available": False,
            "errors": [str(exc)],
            "methods": {},
            "comparators": {},
            "datasets": {},
            "references": {},
            "suites": [],
        }


def host_info() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu_model": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count() or "?",
        "cpu_count_physical": "?",
        "ram_gb": "?",
        "kernel": platform.release(),
        "cpu_mhz": None,
        "cpu_cache": None,
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            text = cpuinfo.read_text(encoding="utf-8")
            model = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
            if model:
                info["cpu_model"] = model.group(1).strip()
            pairs = set(re.findall(r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", text, re.S))
            if pairs:
                info["cpu_count_physical"] = len(pairs)
            mhz = re.search(r"^cpu MHz\s*:\s*([\d.]+)$", text, re.M)
            if mhz:
                info["cpu_mhz"] = f"{float(mhz.group(1)):.0f}"
            cache = re.search(r"^cache size\s*:\s*(.+)$", text, re.M)
            if cache:
                info["cpu_cache"] = cache.group(1).strip()
        except OSError:
            pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            total = re.search(r"^MemTotal:\s*(\d+)\s*kB", meminfo.read_text(encoding="utf-8"), re.M)
            if total:
                info["ram_gb"] = f"{int(total.group(1)) / 1024 / 1024:.1f}"
        except OSError:
            pass
    try:
        info["kernel"] = subprocess.check_output(["uname", "-r"], timeout=2).decode().strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def placeholder_payload(generated_at: str | None = None) -> dict:
    generated_at = generated_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payload = {
        "generated_at": generated_at,
        "host": host_info(),
        "columns": [
            {
                "id": PLACEHOLDER_COLUMN,
                "label": PLACEHOLDER_COLUMN,
                "short": "registry",
                "group": "python",
                "tier": "benchmark scaffold",
                "lang": "Python",
                "what": "Placeholder column: the chemometrics4all benchmark registry exists, but no timing matrix has been generated yet.",
                "build": "",
                "kind": "chemometrics4all",
            }
        ],
        "presets": {
            "all": {"label": "Scaffold", "cols": [PLACEHOLDER_COLUMN]},
            "chemometrics4all": {"label": "chemometrics4all only", "cols": [PLACEHOLDER_COLUMN]},
        },
        "rows": [
            {
                "algo": "registry_scaffold",
                "n": 0,
                "p": 0,
                "threads": 0,
                "cells": {
                    PLACEHOLDER_COLUMN: {
                        "parity": "not_run",
                        "ok": False,
                        "reason": "Placeholder: real chemometrics4all benchmark results have not been generated yet.",
                    }
                },
            }
        ],
        "versions": {
            PLACEHOLDER_COLUMN: "benchmark_registry=benchmarks/benchmark_registry.json; truth_sources=benchmarks/truth_sources.lock.json"
        },
        "reference_by_algo": {},
        "algo_to_group": {"registry_scaffold": "scaffold"},
        "algo_groups": [
            {"key": "scaffold", "label": "Scaffold", "algos": ["registry_scaffold"]}
        ],
        "languages": ["Python"],
        "stats": {
            "algos": 0,
            "backends": 1,
            "groups": 1,
            "sizes": 0,
            "threads": [0],
            "rows": 1,
            "cells": 1,
            "ok": 0,
        },
        "validation": _validation_payload(),
    }
    return {
        "payload": payload,
        "json": json.dumps(payload, separators=(",", ":")),
        "generated_at": payload["generated_at"],
    }


def _as_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "ok", "pass"}


def _tri_bool(value: str | bool | None) -> bool | None:
    text = str(value or "").strip().lower()
    if text == "":
        return None
    if isinstance(value, bool):
        return value
    return text in {"1", "true", "yes", "ok", "pass"}


def _float_or_none(value: str | float | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _float_list(value: str | None) -> list[float]:
    out: list[float] = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def _cell_parity(raw: dict, ok: bool) -> str:
    parity = str(raw.get("parity") or "").strip().lower()
    aliases = {
        "ok": "exact",
        "covered": "exact",
        "reference_ok": "exact",
        "reference_snapshot": "exact",
        "binding_ok": "exact",
        "external_ok": "exact",
        "context": "context",
        "pass": "exact",
        "binding_mismatch": "divergent",
        "reference_diverged": "divergent",
        "external_diverged": "divergent",
        "reference_snapshot_drift": "drift",
        "diverged": "divergent",
        "divergent": "divergent",
        "timing_only": "error",
        "reference_snapshot_missing": "error",
        "reference_not_compared": "error",
        "warn": "drift",
        "drift": "drift",
        "fail": "error",
        "error": "error",
        "external_not_compared": "error",
        "candidate": "candidate",
        "candidate_reference": "candidate",
        "reference_candidate": "candidate",
        "reference_needed": "reference_needed",
        "missing_reference": "reference_needed",
        "timeout": "not_run",
        "not_run": "not_run",
        "not_available": "not_available",
        "na": "not_available",
    }
    if parity in aliases:
        return aliases[parity]
    if ok:
        return "exact"
    return "error"


def _column_version(raw: dict) -> str:
    backend = raw.get("backend", "").strip()
    if backend == "ref.nirs4all":
        return raw.get("lib_build") or "nirs4all local checkout"
    if backend == "ref.sklearn":
        return raw.get("lib_build") or "scikit-learn"
    if backend == "ref.pybaselines":
        return raw.get("lib_build") or "pybaselines"
    if backend == "ref.pywavelets":
        return raw.get("lib_build") or "PyWavelets"
    if backend == "ref.numpy":
        return raw.get("lib_build") or "numpy"
    if backend == "ref.scipy":
        return raw.get("lib_build") or "scipy"
    if backend == "ref.pycaltransfer":
        return raw.get("lib_build") or "pycaltransfer"
    if backend == "ref.dtw_python":
        return raw.get("lib_build") or "dtw-python"
    if backend == "ref.statsmodels":
        return raw.get("lib_build") or "statsmodels"
    if backend == "ref.cowarp":
        return raw.get("lib_build") or "cowarp"
    if backend == "ref.icoshift":
        return raw.get("lib_build") or "Icoshift"
    if backend == "ref.reference_needed":
        return raw.get("lib_build") or "reference needed"
    return raw.get("lib_build") or raw.get("reference_library") or "not reported"


def _reference_display_name(raw: dict) -> str:
    backend = raw.get("backend", "").strip()
    library = raw.get("reference_library", "").strip()
    by_backend = {
        "ref.nirs4all": "nirs4all",
        "ref.sklearn": "sklearn",
        "ref.pybaselines": "pybaselines",
        "ref.pywavelets": "PyWavelets",
        "ref.numpy": "NumPy",
        "ref.scipy": "scipy",
        "ref.pycaltransfer": "pycaltransfer",
        "ref.r.prospectr": "prospectr",
        "ref.dtw_python": "dtw-python",
        "ref.statsmodels": "statsmodels",
        "ref.cowarp": "cowarp",
        "ref.icoshift": "Icoshift",
        "ref.reference_needed": "reference needed",
    }
    if backend in by_backend:
        return by_backend[backend]
    if library:
        return re.split(r"[.(\s]", library, maxsplit=1)[0]
    return backend.removeprefix("ref.") if backend.startswith("ref.") else backend


def _column_id(row: dict) -> str:
    backend = row.get("backend", "unknown").strip() or "unknown"
    build = row.get("lib_build", "").strip()
    if backend == "cpp" and build:
        return f"chemometrics4all.cpp.{build}"
    if backend.startswith("ref."):
        return backend
    return f"chemometrics4all.{backend}"


def _column_meta(cid: str, row: dict) -> dict:
    lang = row.get("language") or row.get("backend", "unknown").title()
    backend = row.get("backend", "unknown")
    library = row.get("reference_library", "").strip()
    if backend.startswith("ref."):
        lang_key = (lang or "").strip().lower()
        group = {
            "python": "ext-py",
            "r": "ext-r",
            "matlab": "ext-ml",
            "matlab/octave": "ext-ml",
            "julia": "ext-jl",
            "rust": "ext-rs",
            "go": "ext-go",
            "javascript": "ext-js",
            "java": "ext-java",
            "c++": "ext-cpp",
        }.get(lang_key, "external")
        short = {
            "ref.nirs4all": "nirs4all",
            "ref.sklearn": "sklearn",
            "ref.scipy": "scipy",
            "ref.pybaselines": "pybase",
            "ref.pywavelets": "pywt",
            "ref.numpy": "numpy",
            "ref.pycaltransfer": "pycal",
            "ref.r.prospectr": "prospectr",
            "ref.dtw_python": "dtw",
            "ref.statsmodels": "stats",
            "ref.cowarp": "cowarp",
            "ref.icoshift": "icoshift",
            "ref.reference_needed": "needed",
        }.get(backend, library or backend.replace("ref.", ""))
        return {
            "id": cid,
            "label": cid,
            "short": short,
            "group": group,
            "tier": row.get("tier", ""),
            "lang": lang,
            "what": row.get("kind", ""),
            "build": row.get("lib_build", ""),
            "kind": row.get("kind", "external_reference"),
            "reference": library,
        }
    group = {
        "cpp": "cpp",
        "python": "python",
        "sklearn": "python",
        "r": "r",
        "matlab": "matlab",
    }.get(backend, backend)
    short = {
        "cpp": "C4A.cpp",
        "python": "C4A.python",
        "sklearn": "C4A.sklearn",
        "r": "C4A.R",
        "matlab": "C4A.MATLAB",
    }.get(backend, backend)
    return {
        "id": cid,
        "label": cid,
        "short": short,
        "group": group,
        "tier": row.get("tier", ""),
        "lang": lang,
        "what": row.get("kind", ""),
        "build": row.get("lib_build", ""),
        "kind": row.get("kind", "chemometrics4all"),
        "reference": library,
    }


_LANG_ORDER = {
    "C++": 0,
    "C": 0,
    "Python": 1,
    "R": 2,
    "MATLAB/Octave": 3,
    "MATLAB": 3,
    "Octave": 3,
    "JavaScript": 4,
    "Julia": 5,
    "Rust": 6,
    "Go": 7,
    "Java": 8,
    "External": 9,
}


CANDIDATE_REFERENCE_ROWS = {}


def _column_sort_key(col: dict) -> tuple:
    lang = col.get("lang", "")
    lang_rank = _LANG_ORDER.get(lang, 90)
    kind_rank = 0 if col.get("kind") == "chemometrics4all" else 1
    group = str(col.get("group", ""))
    label = str(col.get("short") or col.get("label") or col.get("id") or "")
    return (lang_rank, lang, kind_rank, group, label)


def _load_registry_groups(repo_root: Path) -> tuple[dict[str, str], list[dict]]:
    path = repo_root / "benchmarks" / "benchmark_registry.json"
    if not path.exists():
        return {}, []
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, []
    algo_to_group = {}
    grouped: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    for method in registry.get("methods", []):
        algo = method.get("method_id")
        group = method.get("family", "other")
        if not algo:
            continue
        algo_to_group[algo] = group
        grouped.setdefault(group, []).append(algo)
        labels[group] = group.replace("_", " ").title()
    groups = [
        {"key": key, "label": labels.get(key, key), "algos": sorted(values)}
        for key, values in sorted(grouped.items())
    ]
    return algo_to_group, groups


def _result_csv_paths(results_dir: Path) -> list[Path]:
    """Return CSV sources that define the dashboard state.

    ``full_matrix.csv`` is the canonical snapshot. Additional generated
    benchmark shards can refresh subsets of it without requiring a manual
    merge step; temporary probe/check CSVs are intentionally ignored.
    """
    if not results_dir.exists():
        return []
    names = ["full_matrix.csv"]
    paths = [results_dir / name for name in names if (results_dir / name).exists()]
    paths.extend(sorted(results_dir.glob("dashboard_refresh_*.csv")))
    return paths


def build_payload(results_dir: Path) -> dict:
    """Read chemometrics4all cross-binding CSVs and build dashboard payload."""
    csv_paths = _result_csv_paths(results_dir)
    if not csv_paths:
        return placeholder_payload()

    rows_in: list[dict] = []
    latest_source = max(path.stat().st_mtime for path in csv_paths)
    for source_index, path in enumerate(csv_paths):
        source_mtime = path.stat().st_mtime
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source_index"] = source_index
                row["_source_mtime"] = source_mtime
                rows_in.append(row)
    if not rows_in:
        return placeholder_payload()

    generated_at = dt.datetime.fromtimestamp(latest_source, dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    seen: dict[tuple, dict] = {}
    for raw in rows_in:
        try:
            n = int(raw.get("n", "0"))
            p = int(raw.get("p", "0"))
            threads = int(raw.get("threads", "0"))
        except ValueError:
            continue
        algo = raw.get("algorithm", "").strip()
        backend = raw.get("backend", "").strip()
        if not algo or not backend:
            continue
        key = (algo, backend, raw.get("lib_build", "").strip(), n, p, threads)
        rank = (
            float(raw.get("_source_mtime") or 0.0),
            int(raw.get("_source_index") or 0),
        )
        old = seen.get(key)
        old_rank = (
            float(old.get("_source_mtime") or 0.0),
            int(old.get("_source_index") or 0),
        ) if old is not None else None
        if old is None or rank >= old_rank:
            seen[key] = raw
    rows_in = list(seen.values())

    columns: dict[str, dict] = {}
    versions: dict[str, str] = {}
    by_row: dict[tuple[str, int, int, int], dict] = {}
    families: dict[str, str] = {}
    references_by_algo: dict[str, dict[str, object]] = {}
    ok_cells = 0

    for raw in rows_in:
        try:
            n = int(raw.get("n", "0"))
            p = int(raw.get("p", "0"))
            threads = int(raw.get("threads", "0"))
            ms = float(raw.get("median_ms", "nan"))
        except ValueError:
            continue
        algo = raw.get("algorithm", "").strip()
        if not algo:
            continue
        if raw.get("family"):
            families[algo] = raw["family"].strip()
        cid = _column_id(raw)
        columns.setdefault(cid, _column_meta(cid, raw))
        versions[cid] = _column_version(raw)

        ok = _as_bool(raw.get("ok"))
        if ok:
            ok_cells += 1
        backend = raw.get("backend", "").strip()
        kind = raw.get("kind", "").strip()
        reference_library = raw.get("reference_library", "").strip()
        if backend.startswith("ref.") or kind == "external_reference":
            ref_label = _reference_display_name(raw)
            raw_ref_label = reference_library or backend
            ref_info = references_by_algo.setdefault(
                algo,
                {"primary": "", "libraries": [], "raw_primary": "", "raw_libraries": []},
            )
            refs = ref_info["libraries"]
            if isinstance(refs, list) and ref_label and ref_label not in refs:
                refs.append(ref_label)
            raw_refs = ref_info["raw_libraries"]
            if isinstance(raw_refs, list) and raw_ref_label and raw_ref_label not in raw_refs:
                raw_refs.append(raw_ref_label)
            if raw.get("reference_role", "").strip() == "canonical" and ref_label and not ref_info.get("primary"):
                ref_info["primary"] = ref_label
                ref_info["raw_primary"] = raw_ref_label
        gate = "reference" if (backend.startswith("ref.") or kind == "external_reference") else (
            "binding" if backend in {"python", "sklearn", "r", "matlab"} else "native"
        )
        key = (algo, n, p, threads)
        row = by_row.setdefault(key, {
            "algo": algo,
            "n": n,
            "p": p,
            "threads": threads,
            "cells": {},
        })
        row["cells"][cid] = {
            "ok": ok,
            "ms": ms if ok else None,
            "fmt": f"{ms:.3f} ms" if ok else "fail",
            "warmup_runs": int(raw.get("warmup_runs") or 0),
            "timed_runs": int(raw.get("timed_runs") or 0),
            "batch_loops": int(raw.get("batch_loops") or 1),
            "samples_ms": _float_list(raw.get("samples_ms")),
            "parity": _cell_parity(raw, ok),
            "raw_parity": raw.get("parity") or ("ok" if ok else "fail"),
            "gate": gate,
            "reason": raw.get("reason", ""),
            "reference": reference_library,
            "reference_role": raw.get("reference_role", "").strip(),
            "binding_parity_ok": _tri_bool(raw.get("binding_parity_ok")),
            "reference_parity_ok": _tri_bool(raw.get("reference_parity_ok")),
            "reference_max_abs_diff": _float_or_none(raw.get("reference_max_abs_diff")),
            "reference_rms_diff": _float_or_none(raw.get("reference_rms_diff")),
            "reference_rel_l2_diff": _float_or_none(raw.get("reference_rel_l2_diff")),
            "binding_max_abs_diff": _float_or_none(raw.get("binding_max_abs_diff")),
            "binding_rms_diff": _float_or_none(raw.get("binding_rms_diff")),
            "binding_rel_l2_diff": _float_or_none(raw.get("binding_rel_l2_diff")),
        }

    for algo, refs in CANDIDATE_REFERENCE_ROWS.items():
        matching_rows = [row for row in by_row.values() if row.get("algo") == algo]
        if not matching_rows:
            continue
        for ref in refs:
            raw = {
                "backend": ref["backend"],
                "language": ref["language"],
                "tier": "candidate_reference",
                "kind": "external_reference",
                "reference_library": ref["library"],
                "lib_build": ref["build"],
            }
            cid = _column_id(raw)
            columns.setdefault(cid, _column_meta(cid, raw))
            versions.setdefault(cid, _column_version(raw))
            ref_label = _reference_display_name(raw)
            ref_info = references_by_algo.setdefault(
                algo,
                {"primary": "", "libraries": [], "raw_primary": "", "raw_libraries": []},
            )
            refs_list = ref_info["libraries"]
            if isinstance(refs_list, list) and ref_label and ref_label not in refs_list:
                refs_list.append(ref_label)
            raw_refs = ref_info["raw_libraries"]
            if isinstance(raw_refs, list) and ref["library"] not in raw_refs:
                raw_refs.append(ref["library"])
            if not ref_info.get("primary"):
                ref_info["primary"] = ref_label
                ref_info["raw_primary"] = ref["library"]
            for row in matching_rows:
                row["cells"].setdefault(cid, {
                    "ok": False,
                    "ms": None,
                    "fmt": "—",
                    "warmup_runs": 0,
                    "timed_runs": 0,
                    "batch_loops": 1,
                    "samples_ms": [],
                    "parity": ref["parity"],
                    "raw_parity": ref["parity"],
                    "gate": "candidate_reference",
                    "reason": ref["reason"],
                    "reference": ref["library"],
                    "reference_role": "candidate" if ref["parity"] == "candidate" else "reference_needed",
                    "binding_parity_ok": None,
                    "reference_parity_ok": None,
                    "reference_max_abs_diff": None,
                    "reference_rms_diff": None,
                    "reference_rel_l2_diff": None,
                    "binding_max_abs_diff": None,
                    "binding_rms_diff": None,
                    "binding_rel_l2_diff": None,
                })

    repo_root = Path(__file__).resolve().parents[2]
    algo_to_group, _registry_groups = _load_registry_groups(repo_root)
    result_algos = sorted({row["algo"] for row in by_row.values()})
    grouped: dict[str, list[str]] = {}
    for algo in result_algos:
        group = algo_to_group.get(algo) or families.get(algo) or "other"
        algo_to_group[algo] = group
        grouped.setdefault(group, []).append(algo)
    algo_groups = [
        {"key": key, "label": key.replace("_", " ").title(), "algos": sorted(values)}
        for key, values in sorted(grouped.items())
    ]

    rows = [by_row[key] for key in sorted(by_row)]
    col_list = sorted(columns.values(), key=_column_sort_key)
    threads = sorted({row["threads"] for row in rows})
    languages = sorted({col["lang"] for col in col_list}, key=lambda lang: (_LANG_ORDER.get(lang, 90), lang))
    reference_by_algo = {}
    for algo, ref_info in sorted(references_by_algo.items()):
        refs = ref_info.get("libraries", [])
        if not isinstance(refs, list):
            refs = []
        primary = str(ref_info.get("primary") or (refs[0] if refs else ""))
        raw_refs = ref_info.get("raw_libraries", [])
        if not isinstance(raw_refs, list):
            raw_refs = []
        raw_primary = str(ref_info.get("raw_primary") or (raw_refs[0] if raw_refs else ""))
        reference_by_algo[algo] = {
            "label": primary or (" / ".join(refs[:3]) + (f" +{len(refs) - 3}" if len(refs) > 3 else "")),
            "primary": primary,
            "libraries": refs,
            "raw_primary": raw_primary,
            "raw_libraries": raw_refs,
        }
    payload = {
        "generated_at": generated_at,
        "host": host_info(),
        "columns": col_list,
        "presets": {
            "all": {"label": "All", "cols": [c["id"] for c in col_list]},
            "chemometrics4all": {
                "label": "chemometrics4all",
                "cols": [c["id"] for c in col_list if c["kind"] == "chemometrics4all"],
            },
            "externals": {
                "label": "externals",
                "cols": [c["id"] for c in col_list if c["kind"] == "external_reference"],
            },
        },
        "rows": rows,
        "versions": versions,
        "reference_by_algo": reference_by_algo,
        "algo_to_group": algo_to_group,
        "algo_groups": algo_groups,
        "languages": languages,
        "stats": {
            "algos": len({row["algo"] for row in rows}),
            "backends": len(col_list),
            "groups": len(algo_groups),
            "sizes": len({(row["n"], row["p"]) for row in rows}),
            "threads": threads,
            "rows": len(rows),
            "cells": sum(len(row["cells"]) for row in rows),
            "ok": ok_cells,
        },
        "validation": _validation_payload(),
    }
    return {
        "payload": payload,
        "json": json.dumps(payload, separators=(",", ":")),
        "generated_at": payload["generated_at"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="benchmarks/cross_binding/results",
        help="reserved for the future generated result directory",
    )
    parser.add_argument(
        "--out",
        default="docs/_static/bench-data.json",
        help="output path for the JSON payload",
    )
    args = parser.parse_args()
    payload = build_payload(Path(args.results))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload["payload"], indent=2), encoding="utf-8")
    stats = payload["payload"]["stats"]
    print(
        f"wrote {out}: "
        f"{stats['cells']} cells / {stats['backends']} columns / {stats['algos']} algos"
    )


if __name__ == "__main__":
    main()
