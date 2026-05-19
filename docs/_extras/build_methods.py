#!/usr/bin/env python3
"""Generate one chemometrics4all documentation page per public method.

The layout deliberately mirrors the generated pls4all method pages:
title, description, parameter table, bibliography, mathematical principle,
implementation notes, binding tabs, external reference card, and benchmark
tables.

Inputs are intentionally local and deterministic:
  * benchmarks/cross_binding/orchestrator.py for benchmark MethodSpec entries;
  * docs/algorithms/*.md for existing scientific explanations;
  * cpp/include/chemometrics4all/c4a.h for C ABI signatures;
  * bindings/python/src/chemometrics4all/sklearn/*.py for Python signatures;
  * bindings/r/chemometrics4all/R/*.R for R signatures;
  * docs/_static/bench-data.json for dashboard timing/parity rows.
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / "benchmarks" / "cross_binding" / "orchestrator.py"
ALGORITHMS_DIR = ROOT / "docs" / "algorithms"
METHODS_DIR = ROOT / "docs" / "methods"
HEADER = ROOT / "cpp" / "include" / "chemometrics4all" / "c4a.h"
PY_SKLEARN_DIR = ROOT / "bindings" / "python" / "src" / "chemometrics4all" / "sklearn"
R_DIR = ROOT / "bindings" / "r" / "chemometrics4all" / "R"
BENCH_JSON = ROOT / "docs" / "_static" / "bench-data.json"


GROUP_LABELS = {
    "preprocessing": "Preprocessing",
    "baseline": "Baseline",
    "wavelet": "Wavelet",
    "feature_extraction": "Feature extraction",
    "resampling": "Resampling",
    "splitter": "Splitter",
    "filter": "Filter",
    "diagnostic": "Diagnostic",
    "augmentation": "Augmentation",
    "metric": "Metric",
    "utility": "Utility",
    "other": "Other",
}

GROUP_ORDER = [
    "preprocessing",
    "baseline",
    "wavelet",
    "feature_extraction",
    "resampling",
    "splitter",
    "filter",
    "diagnostic",
    "augmentation",
    "metric",
    "utility",
    "other",
]

SOURCE_ALIASES = {
    "area_norm": "area_normalization",
    "frac_to_pct": "fraction_to_percent",
    "pct_to_frac": "percent_to_fraction",
    "x_outlier_mahalanobis": "x_outlier_filter",
    "y_outlier_iqr": "y_outlier_filter",
    "kennard_stone": "splitters",
    "spxy": "splitters",
    "kbins_stratified": "splitters",
}

PY_CLASS_OVERRIDES = {
    "osc": "OSC",
    "epo": "EPO",
    "fck_static": "FCKStaticTransformer",
    "crop": "CropTransformer",
    "resample": "ResampleTransformer",
    "resampler": "Resampler",
    "range_discretizer": "RangeDiscretizer",
    "kbins_discretizer": "IntegerKBinsDiscretizer",
    "kennard_stone": "KennardStoneSplitter",
    "spxy": "SPXYSplitter",
    "kbins_stratified": "KBinsStratifiedSplitter",
    "y_outlier_iqr": "YOutlierFilter",
    "x_outlier_mahalanobis": "XOutlierFilter",
}

C_PREFIX_OVERRIDES = {
    "osc": "c4a_pp_osc",
    "epo": "c4a_pp_epo",
    "flexible_pca": "c4a_pp_flex_pca",
    "flexible_svd": "c4a_pp_flex_svd",
    "fck_static": "c4a_pp_fck_static",
    "crop": "c4a_pp_crop",
    "resample": "c4a_pp_resample",
    "resampler": "c4a_pp_resampler",
    "range_discretizer": "c4a_pp_range_disc",
    "kbins_discretizer": "c4a_pp_kbins_disc",
    "kennard_stone": "c4a_split_kennard_stone",
    "spxy": "c4a_split_spxy",
    "kbins_stratified": "c4a_split_kbins_stratified",
    "y_outlier_iqr": "c4a_filter_y_outlier",
    "x_outlier_mahalanobis": "c4a_filter_x_outlier",
    "hotelling_t2": "c4a_util_hotelling_t2",
    "q_residuals": "c4a_util_q_residuals",
    "transfer_metrics": "c4a_transfer_metrics_compute",
    "nirs_metrics": "c4a_metric_",
    "signal_type_detector": "c4a_signal_detect",
    "rng_pcg64": "c4a_rng_pcg64",
}

REFERENCE_BACKENDS = {
    "nirs_ref": ("ref.nirs4all", "Python"),
    "sklearn_ref": ("ref.sklearn", "Python"),
    "scipy_ref": ("ref.scipy", "Python"),
    "pywavelets_ref": ("ref.pywavelets", "Python"),
    "frozen_ref": ("ref.frozen", "Python"),
    "r_base": ("ref.r.base", "R"),
    "r_stats": ("ref.r.stats", "R"),
}

PARAM_NOTES = {
    "X": "Input matrix with shape n_samples x n_features.",
    "y": "Response vector or matrix used by supervised transforms and splitters.",
    "d": "External disturbance vector used by EPO.",
    "lam": "Smoothness penalty; larger values produce a smoother baseline.",
    "p": "Asymmetry parameter in penalized least-squares baseline correction.",
    "max_iter": "Maximum number of solver or reweighting iterations.",
    "tol": "Convergence tolerance.",
    "polyorder": "Polynomial order.",
    "window_length": "Odd Savitzky-Golay window length.",
    "deriv": "Derivative order.",
    "delta": "Sample spacing along the wavelength axis.",
    "mode": "Boundary handling mode.",
    "sigma": "Gaussian standard deviation or kernel scale, depending on method.",
    "n_components": "Number of latent components or projected columns.",
    "scale": "Whether to standardize internal variables before fitting.",
    "family": "Wavelet family.",
    "level": "Wavelet decomposition level.",
    "threshold_mode": "Wavelet thresholding rule.",
    "noise_estimator": "Noise estimator used for denoising threshold.",
    "kernel_size": "Convolution kernel width.",
    "filter_orders": "Fractional derivative orders in the kernel bank.",
    "filter_scales": "Scale factors in the kernel bank.",
    "test_size": "Fraction of samples assigned to the test set.",
    "seed": "Random seed for deterministic splitting or filtering.",
    "n_bins": "Number of bins.",
    "strategy": "Binning strategy.",
    "method": "Algorithm variant selected by the wrapper.",
    "threshold": "Outlier or quality threshold.",
    "contamination": "Expected outlier fraction.",
}


@dataclass
class Reference:
    backend: str
    library: str
    language: str
    compare: bool = True
    note: str = ""


@dataclass
class Method:
    method_id: str
    label: str
    family: str
    r_expr: str = ""
    py_class: str = ""
    c_prefix: str = ""
    refs: list[Reference] = field(default_factory=list)
    source_slug: str = ""
    source_title: str = ""
    source_text: str = ""
    benchmarked: bool = False


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _literal_string(node: ast.AST | None) -> str:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else ""


def _literal_bool(node: ast.AST | None, default: bool) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return default


def _first_c_symbol(node: ast.AST | None) -> str:
    if node is None:
        return ""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if child.value.startswith("c4a_"):
                return child.value
    return ""


def _first_c4a_class(node: ast.AST | None) -> str:
    if node is None:
        return ""
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            if child.value.id == "c4a":
                return child.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _parse_reference(node: ast.AST) -> Reference | None:
    if not isinstance(node, ast.Call):
        return None
    name = _call_name(node.func)
    compare = _literal_bool(_keyword(node, "compare"), True)
    note = _literal_string(_keyword(node, "contract_note"))
    if name == "ReferenceSpec" and len(node.args) >= 2:
        backend = _literal_string(node.args[0])
        library = _literal_string(node.args[1])
        language = _literal_string(node.args[4]) if len(node.args) > 4 else "Python"
        return Reference(backend, library, language or "Python", compare, note)
    if name in REFERENCE_BACKENDS:
        backend, language = REFERENCE_BACKENDS[name]
        if name in {"r_base", "r_stats"}:
            library = "R base" if name == "r_base" else "R stats"
        else:
            library = _literal_string(node.args[0]) if node.args else backend
        return Reference(backend, library, language, compare, note)
    return None


def parse_methods_from_orchestrator() -> list[Method]:
    tree = ast.parse(ORCHESTRATOR.read_text(encoding="utf-8"))
    out: list[Method] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "MethodSpec":
            continue
        if len(node.args) < 6:
            continue
        method_id = _literal_string(node.args[0])
        label = _literal_string(node.args[1])
        family = _literal_string(node.args[2]) or "other"
        if not method_id or not label:
            continue
        py_class = _first_c4a_class(node.args[3]) or PY_CLASS_OVERRIDES.get(method_id, "")
        c_prefix = _first_c_symbol(node.args[4]) or C_PREFIX_OVERRIDES.get(method_id, "")
        r_expr = _literal_string(node.args[5])
        refs: list[Reference] = []
        if len(node.args) >= 7:
            ref_node = node.args[6]
            items = ref_node.elts if isinstance(ref_node, ast.Tuple) else [ref_node]
            for item in items:
                ref = _parse_reference(item)
                if ref:
                    refs.append(ref)
        out.append(Method(method_id, label, family, r_expr, py_class, c_prefix, refs, benchmarked=True))
    return sorted(out, key=lambda m: m.method_id)


def _clean_title(raw: str) -> str:
    title = raw.strip().lstrip("#").strip()
    title = title.replace("`", "")
    return title


def _section(text: str, headings: tuple[str, ...]) -> str:
    lines = text.splitlines()
    start = None
    level = None
    heading_lc = {h.lower() for h in headings}
    for i, line in enumerate(lines):
        m = re.match(r"^(#{2,4})\s+(.+?)\s*$", line)
        if not m:
            continue
        if m.group(2).strip().lower() in heading_lc:
            start = i + 1
            level = len(m.group(1))
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start, len(lines)):
        m = re.match(r"^(#{2,4})\s+", lines[j])
        if m and len(m.group(1)) <= int(level):
            end = j
            break
    return "\n".join(lines[start:end]).strip()


def _intro(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    out = []
    for line in lines:
        if line.startswith("## "):
            break
        out.append(line)
    return "\n".join(out).strip()


def load_algorithm_pages() -> dict[str, dict[str, str]]:
    pages: dict[str, dict[str, str]] = {}
    for path in sorted(ALGORITHMS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        first = next((line for line in text.splitlines() if line.startswith("# ")), f"# {path.stem}")
        pages[path.stem] = {
            "title": _clean_title(first),
            "text": text,
            "intro": _intro(text),
            "parameters": _section(text, ("parameters",)),
            "algorithm": _section(text, ("algorithm", "mathematical principle", "principle")),
            "contract": _section(text, ("numerical contract", "determinism", "lifecycle")),
            "reference": _section(text, ("reference", "references", "bibliography")),
        }
    return pages


def infer_family(slug: str, page: dict[str, str]) -> str:
    title = page.get("title", "").lower()
    if slug.startswith("aug_") or "augmenter" in title:
        return "augmentation"
    if "wavelet" in slug or "haar" in slug:
        return "wavelet"
    if "filter" in slug or "outlier" in slug or "quality" in slug or "leverage" in slug:
        return "filter"
    if "metric" in slug:
        return "metric"
    if slug in {"hotelling_t2", "q_residuals", "signal_type_detector"}:
        return "diagnostic"
    if slug in {"rng_pcg64"}:
        return "utility"
    if "baseline" in title or slug in {"asls", "airpls", "arpls", "modpoly", "imodpoly", "snip", "rolling_ball", "detrend"}:
        return "baseline"
    if slug in {"osc", "epo", "flexible_pca", "flexible_svd", "fck_static"}:
        return "feature_extraction"
    if "splitter" in title or "splitters" in slug:
        return "splitter"
    return "preprocessing"


def enrich_with_sources(methods: list[Method], pages: dict[str, dict[str, str]]) -> list[Method]:
    by_id = {m.method_id: m for m in methods}
    used_sources: set[str] = set()
    for m in methods:
        slug = SOURCE_ALIASES.get(m.method_id, m.method_id)
        page = pages.get(slug)
        if page:
            m.source_slug = slug
            m.source_title = page["title"]
            m.source_text = page["text"]
            used_sources.add(slug)
    for slug, page in pages.items():
        if slug in by_id or slug in used_sources:
            continue
        method_id = slug
        label = page["title"].split(" — ", 1)[0].split(" (", 1)[0]
        extra = Method(
            method_id=method_id,
            label=label,
            family=infer_family(slug, page),
            py_class=PY_CLASS_OVERRIDES.get(method_id, ""),
            c_prefix=C_PREFIX_OVERRIDES.get(method_id, ""),
            source_slug=slug,
            source_title=page["title"],
            source_text=page["text"],
            benchmarked=False,
        )
        methods.append(extra)
    return sorted(methods, key=lambda m: (GROUP_ORDER.index(m.family) if m.family in GROUP_ORDER else 999, m.method_id))


def _annotation(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _default_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        if isinstance(node, ast.Constant):
            return repr(node.value)
        return ""


def load_python_classes() -> dict[str, dict[str, Any]]:
    classes: dict[str, dict[str, Any]] = {}
    for path in sorted(PY_SKLEARN_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            params = []
            if init:
                args = init.args.args[1:]
                defaults = [None] * (len(args) - len(init.args.defaults)) + list(init.args.defaults)
                for arg, default in zip(args, defaults, strict=False):
                    params.append({
                        "name": arg.arg,
                        "type": _annotation(arg.annotation),
                        "default": _default_text(default),
                        "note": PARAM_NOTES.get(arg.arg, ""),
                    })
                for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults, strict=False):
                    params.append({
                        "name": arg.arg,
                        "type": _annotation(arg.annotation),
                        "default": _default_text(default),
                        "note": PARAM_NOTES.get(arg.arg, ""),
                    })
            c_prefix = ""
            for item in node.body:
                if isinstance(item, ast.Assign):
                    if any(isinstance(t, ast.Name) and t.id == "_C_PREFIX" for t in item.targets):
                        c_prefix = _literal_string(item.value)
            classes[node.name] = {
                "doc": ast.get_docstring(node) or "",
                "params": params,
                "c_prefix": c_prefix,
                "module": path.stem,
            }
    return classes


def load_r_signatures() -> dict[str, str]:
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(R_DIR.glob("*.R")))
    sigs: dict[str, str] = {}
    pattern = re.compile(r"^([A-Za-z0-9_.]+)\s*<-\s*function\s*\((.*?)\)\s*\{", re.M | re.S)
    for m in pattern.finditer(text):
        name = m.group(1)
        args = " ".join(m.group(2).split())
        sigs[name] = f"{name}({args})"
    return sigs


def load_c_prototypes() -> dict[str, list[str]]:
    if not HEADER.exists():
        return {}
    prototypes: dict[str, list[str]] = defaultdict(list)
    lines = HEADER.read_text(encoding="utf-8").splitlines()
    buf: list[str] = []
    active = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("C4A_API"):
            active = True
            buf = [stripped]
            if ";" in stripped:
                active = False
        elif active:
            buf.append(stripped)
            if ";" in stripped:
                active = False
        else:
            continue
        if not active and buf:
            proto = " ".join(buf)
            proto = re.sub(r"\s+", " ", proto)
            m = re.search(r"\b(c4a_[A-Za-z0-9_]+)\s*\(", proto)
            if m:
                prototypes[m.group(1)].append(proto.replace("C4A_API ", ""))
            buf = []
    return prototypes


def prototypes_for(prefix: str, prototypes: dict[str, list[str]]) -> list[str]:
    if not prefix:
        return []
    if prefix.endswith("_"):
        names = [n for n in prototypes if n.startswith(prefix)]
    else:
        names = [n for n in prototypes if n.startswith(prefix + "_") or n == prefix]
    return [p for n in sorted(names) for p in prototypes[n]]


def load_bench() -> dict[str, Any]:
    if not BENCH_JSON.exists():
        return {"columns": [], "rows": [], "reference_by_algo": {}}
    return json.loads(BENCH_JSON.read_text(encoding="utf-8"))


def esc_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def code(value: str) -> str:
    return f"`{value}`" if value else "—"


def method_description(m: Method, page: dict[str, str] | None, py_classes: dict[str, dict[str, Any]]) -> str:
    parts = []
    if page and page.get("intro"):
        parts.append(page["intro"])
    if m.py_class and m.py_class in py_classes and py_classes[m.py_class].get("doc"):
        doc = py_classes[m.py_class]["doc"].strip()
        doc_lines = doc.splitlines()
        first_line = doc_lines[0] if doc_lines else doc
        parts.append(
            f"From the `chemometrics4all.{m.py_class}` Python wrapper docstring:\n\n"
            f"> {first_line}"
        )
        if len(doc_lines) > 1:
            parts.append(
                "<details><summary>Full wrapper docstring</summary>\n\n"
                "```text\n"
                f"{doc}\n"
                "```\n\n"
                "</details>"
            )
    if not parts:
        parts.append(
            f"`{m.method_id}` is a chemometrics4all {m.family.replace('_', ' ')} method "
            "exposed through the C ABI and the generated language bindings."
        )
    return "\n\n".join(parts)


def render_parameters(m: Method, py_classes: dict[str, dict[str, Any]], page: dict[str, str] | None) -> str:
    params = py_classes.get(m.py_class, {}).get("params", []) if m.py_class else []
    if params:
        out = ["### Parameters\n", "| Name | Type | Default | Notes |", "|------|------|---------|-------|"]
        for p in params:
            out.append(
                f"| `{esc_md(p['name'])}` | `{esc_md(p.get('type') or 'object')}` | "
                f"{code(esc_md(p.get('default') or 'required'))} | {esc_md(p.get('note') or '')} |"
            )
        return "\n".join(out) + "\n"
    if page and page.get("parameters"):
        return "### Parameters\n\n" + page["parameters"] + "\n"
    return "### Parameters\n\nNo public constructor parameters are required for the documented default call.\n"


def render_bibliography(m: Method, page: dict[str, str] | None) -> str:
    if page and page.get("reference"):
        return page["reference"]
    if m.refs:
        refs = [f"- `{ref.backend}` — {ref.library} ({ref.language})." for ref in m.refs]
        return "\n".join(refs)
    return (
        "No separate external paper is registered in the local documentation. "
        "The page documents the in-tree chemometrics4all implementation contract."
    )


def render_principle(m: Method, page: dict[str, str] | None) -> str:
    if page and page.get("algorithm"):
        return page["algorithm"]
    family = m.family.replace("_", " ")
    return (
        f"`{m.method_id}` follows the standard {family} operator contract: "
        "an input matrix $\\mathbf{X}\\in\\mathbb{R}^{n\\times p}$ is transformed, "
        "scored, split, or filtered by the method-specific kernel while preserving "
        "the parity contract recorded by the benchmark snapshots. The precise "
        "numerical convention is the one exposed by the C ABI signatures and the "
        "registered external references below."
    )


def render_implementation_text(m: Method, page: dict[str, str] | None, prototypes: list[str]) -> str:
    parts = []
    if page and page.get("contract"):
        parts.append(page["contract"])
    if prototypes:
        parts.append("C ABI entry points used by the language bindings:\n\n```c\n" + "\n".join(prototypes[:12]) + "\n```")
    elif m.c_prefix:
        parts.append(f"C ABI prefix: `{m.c_prefix}`.")
    if m.refs:
        parts.append("Reference backends are registered in the benchmark matrix and stored as reproducible snapshots when they define the canonical contract.")
    return "\n\n".join(parts) if parts else "Implementation details are defined by the public C ABI and the generated binding wrappers."


def render_implementation_table(m: Method, py_classes: dict[str, dict[str, Any]], r_sigs: dict[str, str], c_protos: list[str]) -> str:
    rows = []
    c_entry = m.c_prefix or (c_protos[0].split("(", 1)[0].split()[-1] if c_protos else "")
    rows.append(("C ABI", code(c_entry), "C/C++", "Stable libc4a entry point family."))
    if m.py_class:
        py = f"chemometrics4all.{m.py_class}"
        rows.append(("Python", code(py), "Python", "sklearn-style wrapper backed by ctypes."))
    r_name = ""
    if m.r_expr:
        r_name = re.match(r"\{?([A-Za-z0-9_]+)\(", m.r_expr.strip())
        r_entry = r_name.group(1) if r_name else m.r_expr
        rows.append(("R", code(r_sigs.get(r_entry, r_entry)), "R", "Public package wrapper around the C ABI."))
    for ref in m.refs:
        role = "canonical/comparator" if ref.compare else "context only"
        rows.append((ref.backend, code(ref.library), ref.language, role + (f"; {ref.note}" if ref.note else "")))
    out = ["### Implementations\n", "| Layer | Entry point | Language | Contract |", "|-------|-------------|----------|----------|"]
    for layer, entry, lang, note in rows:
        out.append(f"| {esc_md(layer)} | {entry} | {esc_md(lang)} | {esc_md(note)} |")
    return "\n".join(out) + "\n"


def _py_default_args(params: list[dict[str, str]]) -> str:
    parts = []
    for p in params:
        default = p.get("default") or ""
        if default and default != "required":
            parts.append(f"{p['name']}={default}")
    return ", ".join(parts[:6])


def render_usage(m: Method, py_classes: dict[str, dict[str, Any]], c_protos: list[str]) -> str:
    parts = [
        "### Usage\n",
        "Every chemometrics4all binding dispatches into the same C kernel. "
        "The registry references are listed in the parity card below.\n",
        "::::{tab-set}\n:class: chemometrics4all-bindings\n\n",
    ]
    c_body = "\n".join(c_protos[:10]) if c_protos else f"/* C ABI prefix */\n{m.c_prefix or 'c4a_*'}"
    parts.append(":::{tab-item} C ABI · libc4a\n:sync: c\n:class-label: lang-c\n\n```c\n" + c_body + "\n```\n\n:::\n")
    if m.py_class:
        params = py_classes.get(m.py_class, {}).get("params", [])
        args = _py_default_args(params)
        call = f"{m.py_class}({args})" if args else f"{m.py_class}()"
        if m.family == "splitter":
            body = f"from chemometrics4all import {m.py_class}\n\nsplitter = {call}\ntrain_idx, test_idx = splitter.split(X)"
        elif m.family == "filter":
            body = f"from chemometrics4all import {m.py_class}\n\nflt = {call}\nmask, stats = flt.fit_apply(X)"
        elif m.method_id in {"osc", "epo"}:
            aux = "y" if m.method_id == "osc" else "d"
            body = f"from chemometrics4all import {m.py_class}\n\nop = {call}\nXt = op.fit_transform(X, {aux})"
        else:
            body = f"from chemometrics4all import {m.py_class}\n\nop = {call}\nXt = op.fit_transform(X)"
        parts.append(":::{tab-item} Python · chemometrics4all\n:sync: python\n:class-label: lang-python\n\n```python\n" + body + "\n```\n\n:::\n")
    if m.r_expr:
        body = "library(chemometrics4all)\nres <- " + m.r_expr
        parts.append(":::{tab-item} R · chemometrics4all\n:sync: r\n:class-label: lang-r\n\n```r\n" + body + "\n```\n\n:::\n")
    parts.append("::::\n")
    return "\n".join(parts)


def reference_cells(m: Method, bench: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bench.get("rows", []):
        if row.get("algo") != m.method_id:
            continue
        for backend, cell in row.get("cells", {}).items():
            if cell:
                cells[backend].append(cell)
    return cells


def render_reference_card(m: Method, bench: dict[str, Any]) -> str:
    versions = bench.get("versions", {})
    cells_by_backend = reference_cells(m, bench)
    rows = []
    if not m.refs:
        rows.append(
            "- ℹ No external parity reference row is registered for this public helper; "
            "the page is generated from the in-tree API and algorithm documentation."
        )
    for ref in m.refs:
        cells = cells_by_backend.get(ref.backend, [])
        role = next((cell.get("reference_role") for cell in cells if cell.get("reference_role")), "")
        role = role or ("canonical" if ref.compare else "context")
        version = versions.get(ref.backend, "")
        version_text = f" · {version}" if version and version != ref.library else ""
        icon = "📐" if ref.compare else "ℹ"
        note = f" — {ref.note}" if ref.note else ""
        rows.append(
            f"- {icon} **`{ref.backend}`** ({ref.language} · {role}) — "
            f"`{ref.library}`{version_text}{note}"
        )
    return "\n**Registry parity references** 📐\n\n:::{card}\n:class-card: external-refs\n\n" + "\n".join(rows) + "\n:::\n"


def parity_icon(value: str) -> str:
    return {
        "exact": "✓",
        "context": "≈",
        "drift": "≈",
        "divergent": "✗",
        "error": "⚠",
        "not_run": "—",
        "not_available": "⊘",
    }.get(value or "", "—")


def registry_tolerance_label(m: Method) -> str:
    if m.method_id == "iasls":
        return "`rtol=1e-5`, `atol=5e-6`"
    if m.method_id in {"flexible_pca", "flexible_svd"}:
        return "sign-invariant `rtol=1e-5`, `atol=1e-8`"
    return "`rtol=1e-5`, `atol=1e-8`"


def benchmark_band(cid: str, meta: dict[str, Any]) -> tuple[int, str, str]:
    group = str(meta.get("group") or "")
    kind = str(meta.get("kind") or "")
    lang = str(meta.get("lang") or "")
    if kind == "chemometrics4all":
        if group == "cpp":
            return (0, "cpp", "C++ native · libc4a")
        if group == "python":
            return (1, "python", "Python · chemometrics4all")
        if group == "r":
            return (2, "r", "R · chemometrics4all")
        return (3, "ext", "chemometrics4all")
    if lang == "Python":
        return (4, "python", "Python · external")
    if lang == "R":
        return (5, "r", "R · external")
    if lang == "MATLAB":
        return (6, "matlab", "MATLAB · external")
    return (7, "ext", "Other")


def benchmark_backend_label(cid: str, meta: dict[str, Any]) -> str:
    if cid.startswith("ref."):
        return cid
    return str(meta.get("short") or cid)


def benchmark_parity_label(cell: dict[str, Any] | None) -> tuple[str, str]:
    if not cell:
        return ("parity-not_run", "—")
    parity = str(cell.get("parity") or "not_run")
    gate = str(cell.get("gate") or "")
    if cell.get("ok"):
        if gate == "reference":
            return (f"parity-{parity}", "✓ ref")
        if gate == "binding":
            return (f"parity-{parity}", "✓ bind")
        if gate == "native" and parity == "exact":
            return (f"parity-{parity}", "✓ exact")
    return (f"parity-{parity}", f"{parity_icon(parity)} {parity}")


def benchmark_truth_marker(cid: str, meta: dict[str, Any], cells: list[dict[str, Any]]) -> tuple[str, str]:
    ref_cell = next((cell for cell in cells if cell.get("gate") == "reference"), None)
    if not ref_cell:
        return ("", "")
    role = str(ref_cell.get("reference_role") or "canonical")
    versions = meta.get("version_label") or meta.get("version") or ""
    reference = str(ref_cell.get("reference") or meta.get("reference") or cid)
    title = f"Registry parity reference ({meta.get('lang') or 'external'}): {reference}"
    if versions:
        title += f" · {versions}"
    if role:
        title += f" — {role}"
    klass = "truth-source-strict" if role == "canonical" else "truth-source-relaxed"
    return (klass, f'<span class="truth-mark" title="{html.escape(title, quote=True)}">📐</span>')


def render_benchmarks(m: Method, bench: dict[str, Any]) -> str:
    rows = [row for row in bench.get("rows", []) if row.get("algo") == m.method_id]
    if not rows:
        return (
            "### Benchmarks\n\n"
            "No cross-binding timing row is currently registered for this method. "
            "The implementation table above is still generated from the public API surface.\n"
        )
    versions = bench.get("versions", {})
    cols = {col["id"]: {**col, "version_label": versions.get(col["id"], "")} for col in bench.get("columns", [])}
    by_thread: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_thread[int(row.get("threads", 0))].append(row)
    parts = [
        "### Benchmarks\n",
        "Median wall-clock per cell from [`docs/_static/bench-data.json`](../benchmarks/overview.md). "
        "Verdict legend: ✓ exact · ≈ context/drift · ✗ divergent · ⊘ not available · — not run · ⚠ error.\n",
        "::::{tab-set}\n:class: parity-tabs\n\n",
    ]
    for thread in sorted(by_thread):
        thread_rows = sorted(by_thread[thread], key=lambda r: (r.get("n", 0), r.get("p", 0)))
        sizes = [(int(r["n"]), int(r["p"])) for r in thread_rows]
        backends = sorted(
            {cid for r in thread_rows for cid in r.get("cells", {})},
            key=lambda cid: (benchmark_band(cid, cols.get(cid, {}))[0], benchmark_backend_label(cid, cols.get(cid, {}))),
        )
        best_by_size = []
        for row in thread_rows:
            vals = [
                float(cell["ms"])
                for cell in row.get("cells", {}).values()
                if cell and cell.get("ok") and cell.get("ms") is not None
            ]
            best_by_size.append(min(vals) if vals else None)
        parts.append(f":::" + f"{{tab-item}} {thread} thread{'s' if thread != 1 else ''}\n:sync: threads-{thread}\n\n")
        parts.append('<div class="parity-table-wrap">\n<table class="docutils parity-grouped">\n')
        parts.append("<thead><tr><th>Backend</th><th>Parity</th>")
        for n, p in sizes:
            parts.append(f"<th>{n}×{p}</th>")
        parts.append("</tr></thead>\n")
        current_band: tuple[str, str] | None = None
        col_span = 2 + len(sizes)
        for cid in backends:
            meta = cols.get(cid, {})
            _, band_tag, band_label = benchmark_band(cid, meta)
            if current_band != (band_tag, band_label):
                if current_band is not None:
                    parts.append("</tbody>\n")
                parts.append(
                    f'<tbody class="lang-band lang-{html.escape(band_tag)}">'
                    f'<tr class="lang-band-row" data-lang="{html.escape(band_tag)}">'
                    f'<th colspan="{col_span}" scope="rowgroup">'
                    f'<span class="lang-band-dot"></span>{html.escape(band_label)}</th></tr>\n'
                )
                current_band = (band_tag, band_label)
            label = benchmark_backend_label(cid, meta)
            cells = [r.get("cells", {}).get(cid) for r in thread_rows]
            present = [c for c in cells if c]
            parity_cls, parity_text = benchmark_parity_label(present[0] if present else None)
            truth_cls, truth_mark = benchmark_truth_marker(cid, meta, present)
            row_cls = f"bk-row {truth_cls}".strip()
            parts.append(
                f'<tr class="{html.escape(row_cls)}"><td class="bk-name">{truth_mark}'
                f'<code>{html.escape(label)}</code></td>'
                f'<td class="parity {html.escape(parity_cls)}">{html.escape(parity_text)}</td>'
            )
            for idx, c in enumerate(cells):
                if c and c.get("ok") and c.get("ms") is not None:
                    ms = float(c["ms"])
                    best = best_by_size[idx]
                    best_cls = " ms-best" if best is not None and abs(ms - best) <= max(1e-12, abs(best) * 1e-9) else ""
                    fmt = str(c.get("fmt") or f"{ms:.3f} ms")
                    medal = "🏆 " if best_cls else ""
                    parts.append(f'<td class="ms{best_cls}">{medal}{html.escape(fmt)}</td>')
                else:
                    parts.append('<td class="ms">—</td>')
            parts.append("</tr>\n")
        if current_band is not None:
            parts.append("</tbody>\n")
        parts.append("</table>\n</div>\n\n:::\n")
    parts.append("::::\n")
    return "".join(parts)


def render_page(
    m: Method,
    pages: dict[str, dict[str, str]],
    py_classes: dict[str, dict[str, Any]],
    r_sigs: dict[str, str],
    c_prototypes: dict[str, list[str]],
    bench: dict[str, Any],
) -> str:
    page = pages.get(m.source_slug or SOURCE_ALIASES.get(m.method_id, m.method_id))
    if not m.c_prefix and m.py_class in py_classes:
        m.c_prefix = py_classes[m.py_class].get("c_prefix", "")
    c_protos = prototypes_for(m.c_prefix, c_prototypes)
    title = m.label or (page["title"] if page else m.method_id)
    group_label = GROUP_LABELS.get(m.family, m.family.replace("_", " ").title())
    source = f" · _Source_: [`docs/algorithms/{m.source_slug}.md`](../algorithms/{m.source_slug}.md)" if m.source_slug else ""
    parts = [
        f"# `{m.method_id}` — {title}\n\n",
        f"_Group_: **{group_label}** · _Registry tolerance_: {registry_tolerance_label(m)}{source}\n\n",
        "## Description\n\n",
        method_description(m, page, py_classes),
        "\n\n",
        render_parameters(m, py_classes, page),
        "\n## Explanations\n\n",
        "### Bibliographic source\n\n",
        render_bibliography(m, page),
        "\n\n### Mathematical principle\n\n",
        render_principle(m, page),
        "\n\n### Implementation\n\n",
        render_implementation_text(m, page, c_protos),
        "\n\n",
        render_implementation_table(m, py_classes, r_sigs, c_protos),
        "\n",
        render_usage(m, py_classes, c_protos),
        "\n",
        render_reference_card(m, bench),
        "\n",
        render_benchmarks(m, bench),
        "\n---\n\n",
        "_See also_: [benchmark overview](../benchmarks/overview.md) · "
        "[methods index](index.md) · [interactive dashboard](../landing/dashboard.md)\n",
    ]
    return "".join(parts)


def render_index(methods: list[Method]) -> str:
    by_group: dict[str, list[Method]] = defaultdict(list)
    for m in methods:
        by_group[m.family].append(m)
    out = [
        "# Methods catalogue\n",
        "Every public chemometrics4all method documented with its parameters, "
        "bibliographic source, mathematical principle, binding signatures, "
        "external references, and benchmark rows when available.\n",
        f"_Total methods_: **{len(methods)}**. Grouped by family below.\n",
        "```{toctree}\n:hidden:\n:glob:\n:maxdepth: 1\n\n*\n```\n",
    ]
    for group in GROUP_ORDER:
        entries = sorted(by_group.get(group, []), key=lambda m: m.method_id)
        if not entries:
            continue
        out.append(f"## {GROUP_LABELS.get(group, group.title())}\n")
        out.append("| Method | Name | Benchmark | References |")
        out.append("|--------|------|-----------|------------|")
        for m in entries:
            refs = ", ".join(sorted({r.backend.replace("ref.", "") for r in m.refs})) if m.refs else "—"
            out.append(
                f"| [`{m.method_id}`]({m.method_id}.md) | {esc_md(m.label)} | "
                f"{'yes' if m.benchmarked else 'no'} | {esc_md(refs)} |"
            )
        out.append("")
    out.append("---\n")
    out.append("See the [benchmark overview](../benchmarks/overview.md) and the [interactive dashboard](../landing/dashboard.md).")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(METHODS_DIR))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    missing = [p for p in (ORCHESTRATOR, ALGORITHMS_DIR, HEADER, PY_SKLEARN_DIR, R_DIR) if not p.exists()]
    if missing and args.strict:
        raise SystemExit("missing required inputs: " + ", ".join(str(p) for p in missing))

    methods = parse_methods_from_orchestrator() if ORCHESTRATOR.exists() else []
    pages = load_algorithm_pages() if ALGORITHMS_DIR.exists() else {}
    methods = enrich_with_sources(methods, pages)
    py_classes = load_python_classes() if PY_SKLEARN_DIR.exists() else {}
    r_sigs = load_r_signatures() if R_DIR.exists() else {}
    c_protos = load_c_prototypes()
    bench = load_bench()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.md"):
        old.unlink()
    for m in methods:
        (out_dir / f"{m.method_id}.md").write_text(
            render_page(m, pages, py_classes, r_sigs, c_protos, bench),
            encoding="utf-8",
        )
    (out_dir / "index.md").write_text(render_index(methods), encoding="utf-8")
    print(
        f"wrote {len(methods)} method pages + index in {out_dir} "
        f"(benchmarked={sum(1 for m in methods if m.benchmarked)}, "
        f"python classes={len(py_classes)}, R signatures={len(r_sigs)})"
    )


if __name__ == "__main__":
    main()
