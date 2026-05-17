"""Sphinx config for pls4all docs.

Root document is `about.md` (the project info page). The actual landing
page at `/index.html` is a fully custom interactive benchmark dashboard
generated from `_templates/landing.html` via `html_additional_pages`;
the JSON payload is built at sphinx-build time by `_extras/build_landing.py`
and injected through `html_context`.

Build locally:
    pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html

CI: see ``.github/workflows/docs.yml`` — builds + deploys to
gh-pages on every push to main.
"""

from __future__ import annotations

import datetime as _dt
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).parent / "_extras"))

# ---- project --------------------------------------------------------

project = "pls4all"
author = "G. Beurier and contributors"
copyright = f"{_dt.datetime.now().year}, {author}"

# Read version from cpp version header when available, else fallback.
release = "0.94.0"
version = "0.94"

# ---- extensions -----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
]

# myst-parser: enable the optional features we actually use.
myst_enable_extensions = [
    "colon_fence",          # ::: blocks
    "deflist",
    "fieldlist",
    "linkify",              # auto-detect URLs
    "substitution",
    "tasklist",
    "attrs_inline",
]
myst_heading_anchors = 3
# Some long titles in benchmarks; allow.
suppress_warnings = ["myst.header"]

# Treat .md as source files (alongside .rst for compatibility).
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# Master doc — `about.md` (project info). The `index` slot is taken
# over by the custom landing dashboard (see html_additional_pages below).
root_doc = "about"
master_doc = "about"

# Skip the per-phase implementation logs (huge + transient).
exclude_patterns = [
    "_build",
    "reviews/**",
    "parity_audit_2026_05/**",
    "_bench/**",
    "**/.ipynb_checkpoints",
]

# autosectionlabel: prefix labels with the document so duplicate
# section names across pages don't collide.
autosectionlabel_prefix_document = True

# ---- html theme -----------------------------------------------------

html_theme = "alabaster"  # ships with sphinx, no extra dep
html_title = "pls4all"
html_short_title = "pls4all"
html_show_sourcelink = True

html_theme_options = {
    "description": "Portable PLS / NIRS engine in C++17 with a stable C ABI "
                    "and thin first-class bindings for Python, R, MATLAB, "
                    "JavaScript, Android, Go, Rust, Julia, Ruby, .NET, Lua, Nim.",
    "github_user": "GBeurier",
    "github_repo": "pls4all",
    "github_button": True,
    "fixed_sidebar": True,
    "sidebar_width": "260px",
    "page_width": "1200px",
    "show_relbars": True,
}

html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "searchbox.html",
    ]
}

# Static assets dir.
html_static_path = ["_static"]

# Narrow technical font + tighter tables for the benchmark pages.
html_css_files = ["custom.css"]

# Templates dir — Sphinx auto-discovers `_templates/`, but list it
# explicitly so the custom landing.html template is picked up.
templates_path = ["_templates"]

# Override `/index.html` with the custom interactive dashboard.
# Sphinx still renders about.md (master_doc) at `/about.html`.
html_additional_pages = {"index": "landing.html"}


def setup(app):
    """Inject the benchmark JSON payload into html_context.

    Strategy:
      1. If `benchmarks/cross_binding/results/niveau_*.csv` are present
         (local dev environment after a benchmark run), regenerate the
         payload fresh and also write it to `docs/_static/bench-data.json`
         for downstream consumers.
      2. Otherwise (CI, fresh clone, etc.), fall back to the committed
         `docs/_static/bench-data.json`.

    Either way, the landing.html template receives `bench_data_json` +
    `generated_at` via Jinja2 substitution.
    """
    import json as _json
    from build_landing import build_payload

    here = _Path(__file__).parent
    results_dir = here.parent / "benchmarks" / "cross_binding" / "results"
    static_json = here / "_static" / "bench-data.json"

    csvs_present = results_dir.exists() and any(results_dir.glob("niveau_*.csv"))

    if csvs_present:
        payload = build_payload(results_dir)
        # Persist the JSON so CI / fresh clones can render without the raw CSVs.
        static_json.parent.mkdir(parents=True, exist_ok=True)
        static_json.write_text(_json.dumps(payload["payload"], indent=2))
        bench_json = payload["json"]
        generated_at = payload["generated_at"]
    elif static_json.exists():
        raw = _json.loads(static_json.read_text())
        bench_json = _json.dumps(raw, separators=(",", ":"))
        generated_at = raw.get("generated_at", "unknown")
    else:
        # Empty fallback so the template still renders gracefully.
        bench_json = _json.dumps({
            "generated_at": "n/a",
            "host": {}, "columns": [], "presets": {},
            "rows": [], "versions": {},
            "stats": {"algos": 0, "backends": 0, "sizes": 0,
                       "threads": [], "rows": 0, "cells": 0, "ok": 0},
        })
        generated_at = "n/a"

    app.config.html_context = dict(app.config.html_context or {})
    app.config.html_context["bench_data_json"] = bench_json
    app.config.html_context["generated_at"] = generated_at
