"""Sphinx config for chemometrics4all docs.

Root document is `about.md` (the project info page). The actual landing
page at `/index.html` is a custom benchmark-status page generated from
`_templates/landing.html` via `html_additional_pages`; the JSON payload is
built at sphinx-build time by `_extras/build_landing.py` and injected through
`html_context`.

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

project = "chemometrics4all"
author = "G. Beurier and contributors"
copyright = f"{_dt.datetime.now().year}, {author}"

# Read version from cpp version header when available, else fallback.
def _parse_version() -> tuple[str, str]:
    header = _Path(__file__).parent.parent / "cpp" / "include" / "chemometrics4all" / "c4a_version.h"
    if not header.is_file():
        return "0.1.0", "0.1"
    text = header.read_text(encoding="utf-8")
    import re
    major = re.search(r"#\s*define\s+C4A_PROJECT_VERSION_MAJOR\s+(\d+)", text)
    minor = re.search(r"#\s*define\s+C4A_PROJECT_VERSION_MINOR\s+(\d+)", text)
    patch = re.search(r"#\s*define\s+C4A_PROJECT_VERSION_PATCH\s+(\d+)", text)
    if not (major and minor and patch):
        return "0.1.0", "0.1"
    return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}", f"{major.group(1)}.{minor.group(1)}"

release, version = _parse_version()

# ---- extensions -----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
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
    "dollarmath",           # $...$ inline and $$...$$ block math
    "amsmath",              # \begin{align}...\end{align} environments
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
html_title = "chemometrics4all"
html_short_title = "chemometrics4all"
html_show_sourcelink = True

html_theme_options = {
    "description": "Portable chemometrics / NIRS operator engine in C++17 "
                    "with a stable C ABI and thin first-class bindings for Python, R, MATLAB, "
                    "JavaScript, Android, Go, Rust, Julia, Ruby, .NET, Lua, Nim.",
    "github_user": "GBeurier",
    "github_repo": "chemometrics4all",
    "github_button": False,
    "fixed_sidebar": True,
    "sidebar_width": "280px",
    "page_width": "1380px",
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

# Alabaster auto-injects _static/custom.css via its layout, so
# listing it again in html_css_files would duplicate the <link>.
html_js_files = ["tab-combo.js", "sidebar-scroll.js"]

# Templates dir — Sphinx auto-discovers `_templates/`, but list it
# explicitly so the custom landing.html template is picked up.
templates_path = ["_templates"]

# Override `/index.html` with the custom benchmark-status/dashboard page.
# Sphinx still renders about.md (master_doc) at `/about.html`.
html_additional_pages = {"index": "landing.html"}


def setup(app):
    """Inject dashboard data and refresh per-method benchmark tables.

    Live benchmark CSVs win over the committed JSON snapshot. The same payload
    drives the landing dashboard and the ``### Benchmarks`` sections in
    ``docs/methods/*.md`` via a Sphinx ``source-read`` hook.
    """
    import json as _json
    from build_landing import build_payload, _result_csv_paths, _validation_payload
    from method_benchmark_tables import load_or_build_blocks, install_sphinx_hook

    here = _Path(__file__).parent
    results_dir = here.parent / "benchmarks" / "cross_binding" / "results"
    static_json = here / "_static" / "bench-data.json"

    csvs_present = bool(_result_csv_paths(results_dir))

    if csvs_present:
        payload = build_payload(results_dir)
        static_json.parent.mkdir(parents=True, exist_ok=True)
        static_json.write_text(
            _json.dumps(payload["payload"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        bench_json = payload["json"]
        generated_at = payload["generated_at"]
        bench_payload = payload["payload"]
    elif static_json.exists():
        raw = _json.loads(static_json.read_text())
        raw.setdefault("validation", _validation_payload())
        bench_json = _json.dumps(raw, separators=(",", ":"))
        generated_at = raw.get("generated_at", "unknown")
        bench_payload = raw
    else:
        payload = build_payload(results_dir)
        bench_json = payload["json"]
        generated_at = payload["generated_at"]
        bench_payload = payload["payload"]

    app.config.html_context = dict(app.config.html_context or {})
    app.config.html_context["bench_data_json"] = bench_json
    app.config.html_context["generated_at"] = generated_at

    from sphinx.util import logging as _sphinx_logging

    _docs_log = _sphinx_logging.getLogger("chemometrics4all-docs")
    method_blocks, method_blocks_source = load_or_build_blocks(bench_payload)
    if method_blocks_source == "none":
        _docs_log.warning(
            "method benchmark blocks: no CSV, JSON, or snapshot found; "
            "method pages will keep their committed ### Benchmarks sections."
        )
    else:
        _docs_log.info(
            "method benchmark blocks: %d from %s",
            len(method_blocks),
            method_blocks_source,
        )
    install_sphinx_hook(app, method_blocks)
