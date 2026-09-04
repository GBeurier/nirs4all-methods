"""Sphinx config for nirs4all-methods docs.

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
import html as _html
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).parent / "_extras"))

# ---- project --------------------------------------------------------

project = "nirs4all-methods"
author = "G. Beurier and contributors"
copyright = f"{_dt.datetime.now().year}, {author}"

# Read version from cpp version header when available, else fallback.
release = "1.0.16"
version = "1.0"

# ---- extensions -----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxext.opengraph",
]

# ---- Open Graph (social cards) --------------------------------------

ogp_site_url = "https://methods.nirs4all.org/"
ogp_image = "https://methods.nirs4all.org/_static/brand/og.png"
html_baseurl = "https://methods.nirs4all.org/"
ogp_custom_meta_tags = [
    '<meta property="og:image:width" content="1200" />',
    '<meta property="og:image:height" content="630" />',
    '<meta name="twitter:card" content="summary_large_image" />',
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
    # Internal maintainer docs relocated out of the repo root (not part of the
    # rendered site — they were never under docs/ before).
    "archives/**",
    "dev/RELEASE_READINESS.md",
    "dev/PRODUCTION_AUDIT.md",
    "dev/DISTRIBUTION.md",
]

# autosectionlabel: prefix labels with the document so duplicate
# section names across pages don't collide.
autosectionlabel_prefix_document = True

# ---- html theme -----------------------------------------------------

html_theme = "alabaster"  # ships with sphinx, no extra dep
html_title = "nirs4all-methods"
html_short_title = "nirs4all-methods"
html_show_sourcelink = True
html_logo = "_static/brand/stacked.png"
html_favicon = "_static/brand/favicon.ico"

html_theme_options = {
    "description": "Portable PLS / NIRS engine in C++17 with a stable C ABI "
                    "and thin first-class bindings for Python, R, MATLAB, "
                    "JavaScript, Android, Go, Rust, Julia, Ruby, .NET, Lua, Nim.",
    "github_user": "GBeurier",
    "github_repo": "nirs4all-methods",
    # Alabaster's github_button is a third-party iframe (ghbtns.com) that
    # interrupts the sidebar visually. Disabled — a quiet repo link in
    # the footer (or wordmark hover) is sufficient.
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
# The last entry is the GoatCounter beacon (privacy-friendly, no cookies) on the
# themed pages; the custom landing.html carries its own copy. All methods pages
# report the single ecosystem path "/methods" so the cockpit shows one row.
html_js_files = [
    "tab-combo.js",
    "sidebar-scroll.js",
    (
        "https://gc.zgo.at/count.js",
        {
            "data-goatcounter": "https://nirs4all.goatcounter.com/count",
            "data-goatcounter-settings": '{"path": "/methods"}',
            "async": "async",
        },
    ),
]

# Templates dir — Sphinx auto-discovers `_templates/`, but list it
# explicitly so the custom landing.html template is picked up.
templates_path = ["_templates"]

# Override `/index.html` with the custom card-explorer dashboard, and add
# `/matrix.html` for the full methods × implementation parity/benchmark grid.
# Both read the same injected `window.BENCH_DATA` payload (see setup()).
# Sphinx still renders about.md (master_doc) at `/about.html`.
html_additional_pages = {"index": "landing.html", "matrix": "matrix.html"}


def _canonical_url_for_pagename(pagename: str) -> str:
    base = html_baseurl.rstrip("/")
    return f"{base}/" if pagename == "index" else f"{base}/{pagename}.html"


def _seo_page_context(app, pagename, templatename, context, doctree):
    context["n4a_canonical_url"] = _canonical_url_for_pagename(pagename)
    context["n4a_sitemap_url"] = html_baseurl.rstrip("/") + "/sitemap.xml"


def _write_seo_files(app, exception):
    if exception:
        return
    outdir = _Path(app.outdir)
    base = html_baseurl.rstrip("/")
    urls: list[str] = []
    for html_file in sorted(outdir.rglob("*.html")):
        rel = html_file.relative_to(outdir).as_posix()
        urls.append(f"{base}/" if rel == "index.html" else f"{base}/{rel}")
    body = "".join(f"  <url><loc>{_html.escape(url)}</loc></url>\n" for url in urls)
    (outdir / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{body}</urlset>\n",
        encoding="utf-8",
    )
    (outdir / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\n\nSitemap: {base}/sitemap.xml\n",
        encoding="utf-8",
    )
    (outdir / "CNAME").write_text("methods.nirs4all.org\n", encoding="utf-8")
    (outdir / ".nojekyll").write_text("", encoding="utf-8")


def setup(app):
    """Inject benchmark data into the build.

    Two refresh paths run here, both follow the same priority chain:
    live CSV → committed snapshot → empty fallback.

      1. **Landing dashboard** payload — `docs/_static/bench-data.json`,
         consumed by `_templates/landing.html` via Jinja2.
      2. **Per-method benchmark tables** in `docs/methods/*.md` —
         re-rendered on the fly via a `source-read` hook so the
         published pages always reflect the latest results.
    """
    import json as _json
    from build_landing import build_payload
    from method_benchmark_tables import load_or_build_blocks, install_sphinx_hook

    here = _Path(__file__).parent
    results_dir = here.parent / "benchmarks" / "cross_binding" / "results"
    static_json = here / "_static" / "bench-data.json"

    csvs_present = results_dir.exists() and (results_dir / "full_matrix.csv").exists()

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

    # Per-method benchmark tables — refresh from the live CSV (or fall
    # back to the committed snapshot) and swap each method page's
    # `### Benchmarks` section in `source-read`. See
    # `_extras/method_benchmark_tables.py` for the data-flow.
    from sphinx.util import logging as _sphinx_logging
    _docs_log = _sphinx_logging.getLogger("nirs4all-methods-docs")
    method_blocks, method_blocks_source = load_or_build_blocks()
    if method_blocks_source == "none":
        _docs_log.warning(
            "method benchmark blocks: no CSV and no snapshot found; "
            "method pages will keep their committed ### Benchmarks "
            "sections (which may be stale).")
    else:
        _docs_log.info("method benchmark blocks: %d from %s",
                        len(method_blocks), method_blocks_source)
    install_sphinx_hook(app, method_blocks)
    app.connect("html-page-context", _seo_page_context)
    app.connect("build-finished", _write_seo_files)
