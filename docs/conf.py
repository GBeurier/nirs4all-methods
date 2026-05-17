"""Sphinx config for pls4all docs.

Renders the existing flat-markdown docs tree via myst-parser. Root
document is `index.md`; sphinx auto-discovers nested .md via
``include_patterns`` and the per-page toctree entries.

Build locally:
    pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html

CI: see ``.github/workflows/docs.yml`` — builds + deploys to
gh-pages on every push to main.
"""

from __future__ import annotations

import datetime as _dt

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

# Master doc.
root_doc = "index"
master_doc = "index"

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
