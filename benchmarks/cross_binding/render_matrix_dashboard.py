#!/usr/bin/env python3
# ruff: noqa: E501
"""Render a standalone interactive parity matrix from an orchestrator CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path
from typing import Any

import render_docs

STATUS_LABELS = {
    "pass": "OK",
    "reference_failed": "REF",
    "binding_failed": "BIND",
    "run_failed": "ERR",
    "paper_only": "PAPER",
    "not_applicable": "NA",
}


def as_bool(value: str | None) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def as_float(value: str | None) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def classify(row: dict[str, str]) -> str:
    if as_bool(row.get("ok")) is False:
        return "run_failed"
    if as_bool(row.get("binding_parity_ok")) is False:
        return "binding_failed"
    if as_bool(row.get("reference_parity_ok")) is False:
        return "reference_failed"
    if row.get("reference_kind") == "paper_only":
        return "paper_only"
    if as_bool(row.get("reference_parity_ok")) is True:
        return "pass"
    if as_bool(row.get("binding_parity_ok")) is True:
        return "pass"
    return "not_applicable"


def display_backend(row: dict[str, str]) -> str:
    backend = row.get("backend", "")
    build = row.get("libp4a_build") or "blas-omp"
    return render_docs.disp(backend, build)


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    versions_raw = row.get("versions_json") or "{}"
    try:
        parsed = json.loads(versions_raw)
        if isinstance(parsed, dict):
            versions = parsed
    except json.JSONDecodeError:
        versions = {"raw": versions_raw}

    status = classify(row)
    return {
        "algorithm": row.get("algorithm", ""),
        "backend": row.get("backend", ""),
        "backend_label": display_backend(row),
        "language": row.get("language", ""),
        "tier": row.get("tier", ""),
        "kind": row.get("kind", ""),
        "n": row.get("n", ""),
        "p": row.get("p", ""),
        "threads": row.get("threads", ""),
        "build": row.get("libp4a_build", ""),
        "status": status,
        "status_label": STATUS_LABELS[status],
        "ok": as_bool(row.get("ok")),
        "reported_ms": as_float(row.get("reported_ms")),
        "reference_parity_ok": as_bool(row.get("reference_parity_ok")),
        "reference_parity_ref": row.get("reference_parity_ref", ""),
        "reference_parity_rmse_abs": as_float(row.get("reference_parity_rmse_abs")),
        "reference_parity_rmse_rel": as_float(row.get("reference_parity_rmse_rel")),
        "reference_parity_tolerance": as_float(row.get("reference_parity_tolerance")),
        "reference_parity_note": row.get("reference_parity_note", ""),
        "binding_parity_ok": as_bool(row.get("binding_parity_ok")),
        "binding_parity_ref": row.get("binding_parity_ref", ""),
        "binding_parity_max_diff": as_float(row.get("binding_parity_max_diff")),
        "binding_parity_note": row.get("binding_parity_note", ""),
        "reference_kind": row.get("reference_kind", ""),
        "is_canonical_reference": as_bool(row.get("is_canonical_reference")),
        "reason": row.get("reason", ""),
        "prediction_path": row.get("predictions_path", ""),
        "versions": versions,
    }


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [normalize_row(row) for row in csv.DictReader(handle)]


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def render(csv_path: Path, out: Path) -> None:
    rows = load_rows(csv_path)
    payload = {
        "source": str(csv_path),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "rows": rows,
        "methods": ordered_unique([row["algorithm"] for row in rows]),
        "backends": ordered_unique([row["backend_label"] for row in rows]),
        "statuses": STATUS_LABELS,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(payload), encoding="utf-8")


def page(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    title = "n4m parity matrix"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg: #f6f7f9;
  --surface: #ffffff;
  --ink: #20242a;
  --muted: #5f6875;
  --line: #d8dde5;
  --ok-bg: #e9f7ef;
  --ok-fg: #11643b;
  --ref-bg: #fff2cc;
  --ref-fg: #805800;
  --bind-bg: #fce8e6;
  --bind-fg: #a52714;
  --err-bg: #f9d8d5;
  --err-fg: #8c1d18;
  --paper-bg: #e7eef8;
  --paper-fg: #294b77;
  --na-bg: #edf0f4;
  --na-fg: #59616b;
  --focus: #006b7a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
header {{
  padding: 18px 22px 14px;
  background: var(--surface);
  border-bottom: 1px solid var(--line);
}}
h1 {{
  margin: 0 0 6px;
  font-size: 24px;
  font-weight: 680;
}}
.meta {{
  color: var(--muted);
  display: flex;
  flex-wrap: wrap;
  gap: 10px 18px;
}}
main {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 14px;
  padding: 14px;
}}
.toolbar, .summary, .panel {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
}}
.toolbar {{
  padding: 12px;
  display: grid;
  grid-template-columns: minmax(220px, 1fr) auto;
  gap: 12px;
  align-items: start;
}}
input[type="search"] {{
  width: 100%;
  height: 38px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0 10px;
  font: inherit;
}}
.filters {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  justify-content: flex-end;
}}
.filters label {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}}
.summary {{
  margin: 14px 0;
  padding: 10px;
  display: grid;
  grid-template-columns: repeat(6, minmax(90px, 1fr));
  gap: 8px;
}}
.metric {{
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  min-height: 58px;
}}
.metric strong {{
  display: block;
  font-size: 20px;
}}
.metric span {{
  color: var(--muted);
  font-size: 12px;
}}
.matrix-wrap {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: auto;
  max-height: calc(100vh - 220px);
}}
table {{
  border-collapse: separate;
  border-spacing: 0;
  min-width: 100%;
}}
th, td {{
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  padding: 6px 8px;
  vertical-align: middle;
}}
th {{
  position: sticky;
  top: 0;
  z-index: 2;
  background: #f9fafb;
  color: #38404a;
  font-size: 12px;
  text-align: left;
}}
th.method, td.method {{
  position: sticky;
  left: 0;
  z-index: 3;
  background: #fbfcfd;
  min-width: 220px;
  max-width: 260px;
}}
td.method {{
  font-weight: 620;
}}
.cell {{
  width: 112px;
  min-width: 112px;
  height: 44px;
  border: 0;
  border-radius: 6px;
  padding: 5px 6px;
  text-align: left;
  cursor: pointer;
  font: inherit;
  color: inherit;
}}
.cell:hover, .cell:focus {{
  outline: 2px solid var(--focus);
  outline-offset: 1px;
}}
.empty {{
  color: #9aa2ad;
  text-align: center;
}}
.status {{
  display: inline-block;
  min-width: 38px;
  border-radius: 999px;
  padding: 1px 7px;
  font-size: 11px;
  font-weight: 700;
  text-align: center;
}}
.pass {{ background: var(--ok-bg); color: var(--ok-fg); }}
.reference_failed {{ background: var(--ref-bg); color: var(--ref-fg); }}
.binding_failed {{ background: var(--bind-bg); color: var(--bind-fg); }}
.run_failed {{ background: var(--err-bg); color: var(--err-fg); }}
.paper_only {{ background: var(--paper-bg); color: var(--paper-fg); }}
.not_applicable {{ background: var(--na-bg); color: var(--na-fg); }}
.time {{
  display: block;
  margin-top: 3px;
  color: #2f353d;
  font-size: 12px;
}}
aside {{
  min-width: 0;
}}
.panel {{
  padding: 14px;
  position: sticky;
  top: 14px;
  max-height: calc(100vh - 28px);
  overflow: auto;
}}
.panel h2 {{
  margin: 0 0 8px;
  font-size: 18px;
}}
.details-grid {{
  display: grid;
  grid-template-columns: 125px minmax(0, 1fr);
  gap: 6px 10px;
}}
.details-grid dt {{
  color: var(--muted);
}}
.details-grid dd {{
  margin: 0;
  overflow-wrap: anywhere;
}}
pre {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #f4f6f8;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  max-height: 180px;
  overflow: auto;
}}
button.link {{
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  padding: 7px 10px;
  cursor: pointer;
}}
button.link:hover {{
  border-color: var(--focus);
}}
@media (max-width: 980px) {{
  main {{ grid-template-columns: 1fr; }}
  .panel {{ position: static; max-height: none; }}
  .summary {{ grid-template-columns: repeat(2, minmax(110px, 1fr)); }}
}}
</style>
</head>
<body>
<header>
  <h1>n4m parity matrix</h1>
  <div class="meta">
    <span id="source"></span>
    <span id="generated"></span>
    <span>Click any populated cell for run details.</span>
  </div>
</header>
<main>
  <section>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Filter methods or backends">
      <div class="filters" id="filters"></div>
    </div>
    <div class="summary" id="summary"></div>
    <div class="matrix-wrap">
      <table id="matrix" aria-label="Parity matrix"></table>
    </div>
  </section>
  <aside>
    <div class="panel" id="details">
      <h2>Cell details</h2>
      <p class="meta">Select a cell in the matrix.</p>
    </div>
  </aside>
</main>
<script>
const DATA = {data};
const STATUS_ORDER = ["pass", "reference_failed", "binding_failed", "run_failed", "paper_only", "not_applicable"];
const state = {{
  query: "",
  statuses: new Set(STATUS_ORDER),
  focusFailures: false,
  selected: null
}};
const rows = DATA.rows;
const methods = DATA.methods;
const backends = DATA.backends;

function byMethod(method) {{
  return rows.filter(r => r.algorithm === method);
}}

function rowFor(method, backend) {{
  return rows.find(r => r.algorithm === method && r.backend_label === backend);
}}

function fmtMs(value) {{
  return value === null || Number.isNaN(value) ? "--" : `${{value.toFixed(value >= 100 ? 1 : 2)}} ms`;
}}

function statusText(status) {{
  return DATA.statuses[status] || status;
}}

function isFailure(row) {{
  return row && ["reference_failed", "binding_failed", "run_failed"].includes(row.status);
}}

function matchesQuery(method) {{
  const q = state.query.trim().toLowerCase();
  if (!q) return true;
  if (method.toLowerCase().includes(q)) return true;
  return byMethod(method).some(r =>
    r.backend_label.toLowerCase().includes(q) ||
    r.language.toLowerCase().includes(q) ||
    r.tier.toLowerCase().includes(q) ||
    r.kind.toLowerCase().includes(q)
  );
}}

function visibleRowsFor(method) {{
  let xs = byMethod(method).filter(r => state.statuses.has(r.status));
  if (state.focusFailures) xs = xs.filter(isFailure);
  return xs;
}}

function visibleMethods() {{
  return methods.filter(m => matchesQuery(m) && visibleRowsFor(m).length > 0);
}}

function visibleBackends(methodsInView) {{
  let list = backends.filter(b => methodsInView.some(m => {{
    const r = rowFor(m, b);
    if (!r || !state.statuses.has(r.status)) return false;
    return !state.focusFailures || isFailure(r);
  }}));
  if (!list.length) list = backends;
  return list;
}}

function renderFilters() {{
  const el = document.getElementById("filters");
  el.innerHTML = "";
  for (const status of STATUS_ORDER) {{
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.statuses.has(status);
    input.addEventListener("change", () => {{
      input.checked ? state.statuses.add(status) : state.statuses.delete(status);
      render();
    }});
    const badge = document.createElement("span");
    badge.className = `status ${{status}}`;
    badge.textContent = statusText(status);
    label.append(input, badge);
    el.append(label);
  }}
  const focus = document.createElement("button");
  focus.className = "link";
  focus.textContent = state.focusFailures ? "Show all columns" : "Focus failures";
  focus.addEventListener("click", () => {{
    state.focusFailures = !state.focusFailures;
    render();
  }});
  el.append(focus);
}}

function renderSummary(rowsInView) {{
  const counts = Object.fromEntries(STATUS_ORDER.map(s => [s, 0]));
  for (const row of rowsInView) counts[row.status] = (counts[row.status] || 0) + 1;
  const items = [
    ["Rows", rowsInView.length],
    ["Methods", new Set(rowsInView.map(r => r.algorithm)).size],
    ["Backends", new Set(rowsInView.map(r => r.backend_label)).size],
    ["OK", counts.pass || 0],
    ["Reference fails", counts.reference_failed || 0],
    ["Run errors", counts.run_failed || 0],
  ];
  const el = document.getElementById("summary");
  el.innerHTML = "";
  for (const [label, value] of items) {{
    const div = document.createElement("div");
    div.className = "metric";
    div.innerHTML = `<strong>${{value}}</strong><span>${{label}}</span>`;
    el.append(div);
  }}
}}

function renderMatrix() {{
  const methodList = visibleMethods();
  const backendList = visibleBackends(methodList);
  const rowsInView = rows.filter(r => methodList.includes(r.algorithm) && backendList.includes(r.backend_label));
  renderSummary(rowsInView);
  const table = document.getElementById("matrix");
  table.innerHTML = "";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  const methodHead = document.createElement("th");
  methodHead.className = "method";
  methodHead.textContent = "Method";
  headRow.append(methodHead);
  for (const backend of backendList) {{
    const th = document.createElement("th");
    th.textContent = backend;
    th.title = backend;
    headRow.append(th);
  }}
  thead.append(headRow);
  table.append(thead);

  const tbody = document.createElement("tbody");
  for (const method of methodList) {{
    const tr = document.createElement("tr");
    const methodCell = document.createElement("td");
    methodCell.className = "method";
    methodCell.textContent = method;
    tr.append(methodCell);
    for (const backend of backendList) {{
      const td = document.createElement("td");
      const row = rowFor(method, backend);
      if (!row || !state.statuses.has(row.status) || (state.focusFailures && !isFailure(row))) {{
        td.className = "empty";
        td.textContent = "--";
      }} else {{
        const button = document.createElement("button");
        button.className = `cell ${{row.status}}`;
        button.innerHTML = `<span class="status ${{row.status}}">${{statusText(row.status)}}</span><span class="time">${{fmtMs(row.reported_ms)}}</span>`;
        button.title = `${{method}} / ${{backend}}`;
        button.addEventListener("click", () => {{
          state.selected = row;
          renderDetails(row);
        }});
        td.append(button);
      }}
      tr.append(td);
    }}
    tbody.append(tr);
  }}
  table.append(tbody);
}}

function renderDetails(row) {{
  const details = document.getElementById("details");
  const versions = JSON.stringify(row.versions || {{}}, null, 2);
  details.innerHTML = `
    <h2>${{row.algorithm}}</h2>
    <p><span class="status ${{row.status}}">${{statusText(row.status)}}</span></p>
    <dl class="details-grid">
      <dt>Backend</dt><dd>${{row.backend_label}}</dd>
      <dt>Shape</dt><dd>${{row.n}} x ${{row.p}}, threads=${{row.threads}}, build=${{row.build}}</dd>
      <dt>Timing</dt><dd>${{fmtMs(row.reported_ms)}}</dd>
      <dt>Reference</dt><dd>${{row.reference_parity_ref || "--"}}</dd>
      <dt>RMSE rel</dt><dd>${{row.reference_parity_rmse_rel ?? "--"}}</dd>
      <dt>Tolerance</dt><dd>${{row.reference_parity_tolerance ?? "--"}}</dd>
      <dt>Binding ref</dt><dd>${{row.binding_parity_ref || "--"}}</dd>
      <dt>Max diff</dt><dd>${{row.binding_parity_max_diff ?? "--"}}</dd>
      <dt>Reason</dt><dd>${{row.reason || row.reference_parity_note || row.binding_parity_note || "--"}}</dd>
      <dt>Prediction</dt><dd>${{row.prediction_path || "--"}}</dd>
    </dl>
    <h2>Versions</h2>
    <pre>${{versions}}</pre>`;
}}

function render() {{
  renderFilters();
  renderMatrix();
  if (state.selected) renderDetails(state.selected);
}}

document.getElementById("search").addEventListener("input", event => {{
  state.query = event.target.value;
  render();
}});
document.getElementById("source").textContent = `Source: ${{DATA.source}}`;
document.getElementById("generated").textContent = `Generated: ${{DATA.generated_at}}`;
render();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="orchestrator CSV")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("build/cross_binding_dashboard/index.html"),
        help="standalone HTML output",
    )
    args = parser.parse_args()
    render(args.csv, args.out)
    print(f"dashboard -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
