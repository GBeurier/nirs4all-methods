"""Build Sphinx navigation from the maintained documentation hierarchy."""

from __future__ import annotations

import io
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.environment.adapters.toctree import TocTree

DOCS = Path(__file__).resolve().parents[1]
ASSESSMENTS = {
    "hdbscan_assessment": "HDBSCAN for Preflight: reuse assessment and reference gate",
    "cluster_matching_assessment": (
        "Cluster matching and Gaussian null kernels: reuse assessment and reference gate"
    ),
}


def test_method_sidebar_excludes_assessment_notes(tmp_path: Path) -> None:
    """Residual Markdown must not enter the method tree or detail sidebar."""
    source = tmp_path / "source"
    source.mkdir()
    # Keep the real hierarchy and generated method membership. Short bodies
    # avoid coupling this navigation regression to numerical docs and links.
    for path in DOCS.rglob("*.md"):
        if "_build" in path.parts:
            continue
        target = source / path.relative_to(DOCS)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8")
        if path.relative_to(DOCS).as_posix() not in {
            "about.md",
            "methods/index.md",
            "dev/index.md",
        }:
            text = next(
                (line for line in text.splitlines() if line.startswith("# ")),
                "# Document",
            )
        target.write_text(text, encoding="utf-8")

    # These may be left by an old checkout or a local documentation build.
    # Neither title belongs in canonical method navigation.
    for name, title in ASSESSMENTS.items():
        (source / "methods" / f"{name}.md").write_text(f"# {title}\n", encoding="utf-8")
    (source / "conf.py").write_text(
        "extensions = ['myst_parser', 'sphinx_design']\n"
        "root_doc = 'about'\nhtml_theme = 'alabaster'\n"
        "myst_enable_extensions = ['colon_fence']\n",
        encoding="utf-8",
    )
    app = Sphinx(
        str(source),
        str(source),
        str(tmp_path / "html"),
        str(tmp_path / "doctrees"),
        "html",
        status=io.StringIO(),
        warning=io.StringIO(),
    )
    app.build()
    tree = TocTree(app.env)
    method_tree = tree.get_toctree_for(
        "methods/index", app.builder, collapse=True, includehidden=True
    )
    assert method_tree is not None
    assert all(title not in method_tree.astext() for title in ASSESSMENTS.values())

    sidebar = tree.get_toctree_for(
        "methods/pls", app.builder, collapse=True, includehidden=True
    )
    assert sidebar is not None
    labels = sidebar.astext()
    for title in ASSESSMENTS.values():
        assert title not in labels
    assert "Development" in labels
    assert "pls — PLS regression (SIMPLS)" in labels
    assert "hdbscan — n4m.cluster.hdbscan" in labels
    assert (tmp_path / "html" / "dev" / "hdbscan.html").is_file()
    development_tree = tree.get_toctree_for(
        "dev/index", app.builder, collapse=True, includehidden=True
    )
    assert development_tree is not None
    assert ASSESSMENTS["hdbscan_assessment"] in development_tree.astext()
    assert ASSESSMENTS["cluster_matching_assessment"] in development_tree.astext()
