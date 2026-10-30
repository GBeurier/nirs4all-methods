"""Numerical display policy and browser interactions for both parity views.

Run with pytest; browser checks additionally require Playwright and Chromium.
These checks use the committed snapshot and never run numerical benchmarks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
STATIC = REPO / "docs" / "_static"


def test_shared_display_policy() -> None:
    """Only a finite delta for the applicable gate can establish a green match."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the display policy check")
    script = r"""
const assert = require('assert');
const fs = require('fs');
global.window = {};
require(process.argv[1]);
const policy = window.N4MParity;
const column = {id: 'external', kind: 'external'};
function display(overrides) {
  const cell = Object.assign({ok: true, parity: 'cross_check',
    reference_parity: 'cross_check', binding_parity: 'not_run',
    divergence_basis: 'cross_check', divergence_metric: 'rmse',
    divergence: 1e-16}, overrides);
  return policy.prepare({columns: [column], rows: [{algo: 'test', cells: {external: cell}}],
    method_scores: {test: {reference: {cross_check: 1}, binding: {}}}});
}
function status(overrides) {
  return display(overrides).rows[0].cells.external.reference_parity;
}
assert.equal(status({}), 'exact');
assert.equal(status({divergence: 0}), 'exact');
assert.equal(status({divergence: -1e-16}), 'exact');
assert.equal(status({divergence: 0.999999e-8}), 'exact');
assert.equal(status({divergence: 1e-8}), 'cross_check');
assert.equal(status({divergence: 0.1}), 'cross_check');
for (const divergence of [null, undefined, NaN, Infinity, '1e-16']) {
  assert.equal(status({divergence}), 'cross_check');
}
assert.equal(status({divergence_metric: 'jaccard', divergence: 0}), 'cross_check');
assert.equal(status({ok: false}), 'cross_check');
assert.equal(status({parity: 'error'}), 'cross_check');
assert.equal(status({reference_parity: 'error'}), 'error');
assert.equal(status({reference_parity: 'not_run'}), 'not_run');
assert.equal(status({reference_parity: 'not_available'}), 'not_available');
assert.equal(status({divergence_basis: 'binding'}), 'cross_check');
assert.equal(status({divergence_basis: 'self'}), 'cross_check');
assert.equal(status({reference_parity: 'divergent'}), 'exact');
assert.equal(status({reference_parity: 'drift'}), 'exact');
assert.deepEqual(display({}).method_scores.test.reference, {exact: 1});
const raw = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const before = JSON.stringify(raw);
const prepared = policy.prepare(raw);
assert.equal(JSON.stringify(raw), before, 'raw scientific verdicts must stay unchanged');
const row = prepared.rows.find(r => r.algo === 'group_sparse_pls');
assert.equal(row.cells.sgPLS.reference_parity, 'exact');
assert.equal(row.cells.sgPLS.divergence, raw.rows.find(r => r.algo === row.algo).cells.sgPLS.divergence);
assert(prepared.method_scores.group_sparse_pls.reference.exact >
  raw.method_scores.group_sparse_pls.reference.exact);
"""
    subprocess.run(
        [
            node,
            "-e",
            script,
            str(STATIC / "parity-display.js"),
            str(STATIC / "bench-data.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def browser_page(tmp_path_factory: pytest.TempPathFactory):
    playwright = pytest.importorskip("playwright.sync_api")
    jinja = pytest.importorskip("jinja2")
    preview = tmp_path_factory.mktemp("parity-preview")
    (preview / "_static").symlink_to(STATIC, target_is_directory=True)
    env = jinja.Environment(loader=jinja.FileSystemLoader(REPO / "docs" / "_templates"))
    for template, target in (
        ("landing.html", "index.html"),
        ("matrix.html", "matrix.html"),
    ):
        html = env.get_template(template).render(
            bench_data_json=(STATIC / "bench-data.json").read_text(),
            pathto=lambda path, *args: path if args else path + ".html",
        )
        (preview / target).write_text(html)
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(
            executable_path=shutil.which("google-chrome"), args=["--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.route("https://**", lambda route: route.abort())
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        yield page, preview, playwright.expect
        browser.close()
        assert not errors, errors


def test_matrix_compact_focus_and_full_grid(browser_page) -> None:
    page, preview, expect = browser_page
    page.goto((preview / "matrix.html").as_uri())
    expect(page.locator("table.mx")).to_have_class("mx mx-compact")
    assert page.evaluate(
        'document.querySelector("table.mx").scrollWidth <= '
        'document.querySelector("#matrix").clientWidth'
    )
    assert page.locator("#mxHead tr.colrow th").count() < 12
    page.locator("#mxSearch").fill("group_sparse_pls")
    expect(page.locator("#mxBody [data-row-key]")).to_have_count(3)
    # Clicking a cell (not just the method name) focuses the entire row.
    page.locator("#mxBody tr").first.locator("td").nth(3).click()
    expect(page.locator("#mxBody [data-row-key]")).to_have_count(1)
    expect(page.locator("#mxFocus")).to_be_visible()
    headers = page.locator("#mxHead tr.colrow [data-col]").evaluate_all(
        "elements => elements.map(e => e.dataset.col)"
    )
    # External libraries precede the C++ builds and language bindings.
    assert headers.index("sgPLS") < headers.index("pls4all.cpp.blas+omp")
    assert "pls4all.matlab" not in headers
    assert "pls4all.cpp.blas+omp" in headers  # A zero delta stays visible.
    expect(page.locator('#mxHead [data-col="sgPLS"]')).to_be_visible()
    page.locator('[data-viz="divergence"]').click()
    cells = page.locator("#mxBody tr td")
    sgpls_index = headers.index("sgPLS") + 1  # The reference header has no data-col.
    expect(cells.nth(sgpls_index).locator(".q-exact")).to_be_visible()
    assert page.locator("#mxBody .q-cross_check").count() > 0
    assert page.locator("#mxBody .q-exact").filter(has_text="δ 0").count() > 0
    page.keyboard.press("Escape")
    expect(page.locator("#mxFocus")).to_be_hidden()
    expect(page.locator("#mxBody [data-row-key]")).to_have_count(3)
    page.locator('[data-layout="all"]').click()
    raw = json.loads((STATIC / "bench-data.json").read_text())
    assert page.locator("#mxHead tr.colrow th").count() == len(raw["columns"]) + 3
    # The native row button supports keyboard activation in the full grid too.
    page.locator("[data-focus-row]").first.focus()
    page.keyboard.press("Enter")
    expect(page.locator("#mxFocus")).to_be_visible()
    page.locator("#mxClearFocus").click()
    expect(page.locator("#mxBody [data-row-key]")).to_have_count(3)


def test_explorer_uses_the_same_match_policy(browser_page) -> None:
    page, preview, expect = browser_page
    page.goto((preview / "index.html").as_uri() + "#m/group_sparse_pls")
    sgpls = page.locator("#ptableMount tr").filter(
        has=page.locator("td.col-impl", has_text="sgPLS")
    )
    expect(sgpls.locator(".badge.exact")).to_have_text("✓match")
    expect(sgpls.locator(".numeric-match")).to_be_visible()
    expect(sgpls.locator(".badge.cross")).to_have_count(0)
    page.goto((preview / "index.html").as_uri())
    expect(page.locator("#kpis")).to_contain_text("MATCHING CHECKS", ignore_case=True)
    expect(page.locator("#kpis")).to_contain_text("INFORMATIONAL", ignore_case=True)


def test_matrix_contains_horizontal_scroll_on_small_screens(browser_page) -> None:
    page, preview, _ = browser_page
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto((preview / "matrix.html").as_uri())
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.evaluate(
        'document.querySelector("#matrix").scrollWidth > '
        'document.querySelector("#matrix").clientWidth'
    )
    page.set_viewport_size({"width": 1280, "height": 900})
