/* ==========================================================================
   nirs4all-methods · FULL PARITY MATRIX (reskinned)
   Methods × implementation-columns grid in the datasets/formats ecosystem
   style. Vanilla JS, zero deps, zero network. Reads window.BENCH_DATA.
   Preserves the real signal from the legacy landing matrix:
     - binding-vs-reference parity verdicts (gate-aware, never faked)
     - divergence: ΔRMSE for numeric rows, Jaccard for selector rows
     - NR (not run) vs n/a (not available) kept distinct, never coerced
     - build-insensitive "≡ native" sentinel
     - synthetic per-row canonical-reference column (library name + verdict)
     - per-method score cards
   Doc links are RELATIVE (methods/<stem>.html) — served from the site root.
   ========================================================================== */
'use strict';

const DATA = window.BENCH_DATA;
const mount = document.getElementById('matrix');
if (!DATA || !Array.isArray(DATA.columns)) {
  mount.innerHTML = '<div class="mx-empty"><h3>benchmark data failed to load</h3>' +
    '<p>window.BENCH_DATA is undefined. The Sphinx build injects it inline before this script.</p></div>';
  throw new Error('BENCH_DATA missing');
}

const REF_COL = '__reference';

/* ---------- verdict vocabulary: glyph + css class, never colour alone ---------- */
const VERDICT = {
  exact:        { cls: 'exact',  gly: '✓' },
  cross_check:  { cls: 'cross',  gly: '⇄' },
  divergent:    { cls: 'diverg', gly: '✗' },
  drift:        { cls: 'cross',  gly: '≈' },
  error:        { cls: 'error',  gly: '!' },
  deferred:     { cls: 'nr',     gly: '…' },
  not_run:      { cls: 'nr',     gly: 'NR' },
  not_available:{ cls: 'na',     gly: '–' },
};
function verdict(v) { return VERDICT[v] || { cls: 'na', gly: '–' }; }

/* ---------- language slugs (match the legacy LANG_SLUG + css .lt-* tokens) ---------- */
const LANG_SLUG = {
  'C++': 'cpp', 'Python': 'python', 'R': 'r', 'MATLAB/Octave': 'matlab',
  'Benchmark': 'ext', 'Julia': 'ext', 'Rust': 'ext', 'Go': 'ext',
  'JavaScript': 'ext', 'Java': 'ext', 'external': 'ext',
};
function langSlug(lang) { return LANG_SLUG[lang] || 'ext'; }

const COLS = DATA.columns;
const COL_BY_ID = Object.fromEntries(COLS.map(c => [c.id, c]));
const NATIVE_COL = COLS.find(c => c.id === 'pls4all.cpp.native') ? 'pls4all.cpp.native'
                 : (COLS.find(c => c.id === 'pls4all.cpp.blas+omp') ? 'pls4all.cpp.blas+omp' : null);

const GROUP_LABEL = Object.fromEntries((DATA.algo_groups || []).map(g => [g.key, g.label]));
const GROUP_ORDER = (DATA.algo_groups || []).map(g => g.key);
const ACCENTS = ['#0d9488', '#0891b2', '#4f46e5', '#d97706', '#10b981', '#0f766e', '#06b6d4', '#4338ca'];
const GROUP_ACCENT = {};
GROUP_ORDER.forEach((k, i) => { GROUP_ACCENT[k] = ACCENTS[i % ACCENTS.length]; });

/* ---------- gate semantics (verbatim from the legacy renderer) ---------- */
function usesReferenceGate(meta) {
  if (!meta) return false;
  const kind = (meta.kind || '').toLowerCase();
  const id = meta.id || '';
  return kind === 'external' || kind === 'reference' || id.startsWith('pls4all.cpp.');
}
function effectiveParity(cell, meta) {
  if (!cell) return 'not_available';
  if (usesReferenceGate(meta)) return cell.reference_parity || cell.parity || 'not_available';
  return cell.binding_parity || cell.parity || 'not_available';
}

/* ---------- formatting ---------- */
function fmtMs(ms) {
  if (ms == null || !isFinite(ms)) return '—';
  if (ms >= 1000) return (ms / 1000).toFixed(1) + ' s';
  if (ms >= 10) return ms.toFixed(1) + ' ms';
  return ms.toFixed(2) + ' ms';
}
function fmtSpeedup(r) {
  if (!isFinite(r)) return '–';
  if (r >= 100) return r.toFixed(0) + '×';
  if (r >= 10) return r.toFixed(1) + '×';
  return r.toFixed(2) + '×';
}
function fmtPct(n, d) { return d ? Math.round((n / d) * 100) : 0; }
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function escAttr(s) { return escapeHtml(s == null ? '' : s); }

/* ==========================================================================
   State
   ========================================================================== */
const state = {
  search: '', category: '', origin: '', parity: '', size: '', threads: '',
  viz: 'duration',      // duration | divergence | speedup
  sort: { col: '__algo', dir: 1 },
  showScores: false,
};

/* ==========================================================================
   Column ordering — keep DATA.columns order, contiguous language bands.
   The reference column (if present) is pinned first by the payload already.
   ========================================================================== */
function visibleColumns() {
  return COLS.slice();   // payload order already groups by language band
}

/* contiguous (language, group) runs for the band row */
function langBands(colMetas) {
  const bands = [];
  let cur = null;
  for (const c of colMetas) {
    if (c.id === REF_COL) { bands.push({ slug: 'ref', label: 'reference', count: 1, ref: true }); cur = null; continue; }
    const slug = langSlug(c.lang);
    const key = slug + '|' + (c.group || '');
    const isPls = (c.kind || '') === 'pls4all';
    const label = (isPls ? 'n4m · ' : 'ext · ') + c.lang;
    if (!cur || cur.key !== key) { cur = { slug, key, label, count: 0 }; bands.push(cur); }
    cur.count++;
  }
  return bands;
}

/* ==========================================================================
   Row filtering + sorting
   ========================================================================== */
function searchNorm(s) { return String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, ''); }
function algoDisplay(a) { return (DATA.algo_display && DATA.algo_display[a]) || a; }

function filteredRows() {
  const q = searchNorm(state.search.trim());
  const gmap = DATA.algo_to_group || {};
  const omap = DATA.algo_origin || {};
  return DATA.rows.filter(r => {
    if (state.category && gmap[r.algo] !== state.category) return false;
    if (state.origin && (omap[r.algo] || 'pls4all') !== state.origin) return false;
    if (state.size && (r.n + '×' + r.p) !== state.size) return false;
    if (state.threads && String(r.threads) !== state.threads) return false;
    if (state.parity) {
      const want = state.parity;
      let hit = false;
      for (const cid of Object.keys(r.cells)) {
        if (cid === REF_COL) continue;
        const v = effectiveParity(r.cells[cid], COL_BY_ID[cid]);
        if (want === 'attention' ? (v === 'divergent' || v === 'drift' || v === 'error')
                                 : v === want) { hit = true; break; }
      }
      if (!hit) return false;
    }
    if (q) {
      const hay = searchNorm([r.algo, algoDisplay(r.algo), (DATA.algo_fq || {})[r.algo] || '',
        GROUP_LABEL[gmap[r.algo]] || ''].join(' '));
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function rowCells(row, cols) {
  // timing cells for fastest/baseline math
  const out = [];
  for (const c of cols) {
    if (c.id === REF_COL) continue;
    const cell = row.cells[c.id];
    if (cell && cell.ok && isFinite(cell.ms)) out.push({ id: c.id, ms: cell.ms });
  }
  return out;
}

function sortRows(rows, cols) {
  const sc = state.sort;
  const dir = sc.dir;
  let cmp;
  if (sc.col === '__algo') cmp = (a, b) => algoDisplay(a.algo).localeCompare(algoDisplay(b.algo)) || (a.n * a.p - b.n * b.p);
  else if (sc.col === '__size') cmp = (a, b) => (a.n * a.p - b.n * b.p) || a.threads - b.threads;
  else if (sc.col === '__threads') cmp = (a, b) => a.threads - b.threads;
  else {
    const id = sc.col;
    const val = r => { const c = r.cells[id]; return (c && isFinite(c.ms)) ? c.ms : Infinity; };
    cmp = (a, b) => val(a) - val(b);
  }
  return rows.slice().sort((a, b) => dir * cmp(a, b));
}

/* ==========================================================================
   Cell rendering
   ========================================================================== */
function refCellHtml(cell, lcls) {
  if (!cell) return `<td class="${lcls}"><span class="mx-dash">—</span></td>`;
  const name = cell.ref_of || '–';
  const rp = verdict(cell.reference_parity || cell.parity);
  return `<td class="c-ref ${lcls}" title="canonical reference for this method">
    <span class="mx-cell"><span class="refname">${escapeHtml(name)}</span>
    <span class="mx-gly ${rp.cls}" aria-hidden="true">${rp.gly}</span></span></td>`;
}

function divModeCell(cell, meta, lcls, colCls) {
  const metric = cell.divergence_metric || 'rmse';
  const fmt = cell.divergence_fmt;
  if (fmt == null || fmt === '—') {
    return `<td class="${lcls} ${colCls}"><span class="mx-dash">—</span></td>`;
  }
  if (metric === 'jaccard') {
    const bd = (cell.divergence_quality === 'selection_bydesign');
    const v = parseFloat(cell.divergence) || 0;
    const cls = bd ? 'j-bydesign' : (v >= 0.9 ? 'j-good' : 'j-mid');
    const note = cell.divergence_note ? ' — ' + cell.divergence_note : '';
    const title = bd ? `By-design selector cross-check · Jaccard=${fmt}${note}`
                     : `Jaccard feature-set overlap: ${fmt} (1.00 = identical mask)${note}`;
    return `<td class="${lcls} ${colCls}"><span class="mx-div ${cls}" title="${escAttr(title)}">${bd ? 'BD ' : '∩ '}J ${escapeHtml(fmt)}</span></td>`;
  }
  // relative-RMSE δ; a failure verdict overrides the tolerance-band class
  let dq = cell.divergence_quality || 'unknown';
  const ev = effectiveParity(cell, meta);
  if (ev === 'divergent' || ev === 'error' || ev === 'drift') dq = ev;
  const basis = cell.divergence_basis || 'unknown';
  const lead = basis === 'self' ? '' : 'δ ';
  const note = cell.divergence_note ? ' — ' + cell.divergence_note : '';
  const title = basis === 'binding' ? `binding diff vs C++ core (gate 1): δ=${fmt}${note}`
              : basis === 'reference' ? `reference-gate rel-RMSE vs ${cell.reference_kind || 'external ref'} (gate 2, ${dq}): δ=${fmt}${note}`
              : basis === 'self' ? `this row is the canonical reference for this method${note}`
              : `divergence ${fmt}${note}`;
  return `<td class="${lcls} ${colCls}"><span class="mx-div q-${dq}" title="${escAttr(title)}">${lead}${escapeHtml(fmt)}</span></td>`;
}

function dataCellHtml(row, meta, baselineMs, fastestId) {
  const id = meta.id;
  const lcls = `lt-${langSlug(meta.lang)}`;
  const colCls = (meta.kind || '') === 'pls4all' ? 'col-pls4all' : '';
  const cell = row.cells[id];

  // missing cell: distinguish "C++ tier not run this snapshot" (NR) from n/a
  if (!cell) {
    if (id.startsWith('pls4all.cpp.') &&
        Object.keys(row.cells).some(c => c.startsWith('pls4all.cpp.'))) {
      return `<td class="${lcls} ${colCls}" title="C++ build tier not run in this snapshot"><span class="mx-nr">NR</span></td>`;
    }
    return `<td class="${lcls} ${colCls}" title="not available in lib"><span class="mx-gly na">–</span></td>`;
  }

  if (state.viz === 'divergence') return divModeCell(cell, meta, lcls, colCls);

  // build-insensitive sentinel (elementwise op, no BLAS/OMP path)
  if ((!cell.ok || !isFinite(cell.ms)) && cell.build_insensitive) {
    return `<td class="${lcls} ${colCls}" title="≡ native — no BLAS/OpenMP code path"><span class="mx-sentinel">≡</span></td>`;
  }

  // no usable timing: hard failure OR parity-validated-but-not-timed
  if (!cell.ok || !isFinite(cell.ms)) {
    const notTimed = cell.ok && !isFinite(cell.ms);
    const ev = effectiveParity(cell, meta);
    const reason = cell.reason ? escAttr(cell.reason)
                 : notTimed ? 'parity-validated (C++ fixture) · not timed' : '';
    const t = reason ? ` title="${reason}"` : '';
    if (notTimed && cell.is_canonical_reference) {
      return `<td class="${lcls} ${colCls}"${t}><span class="mx-gly ref" title="canonical reference · not timed">◆</span></td>`;
    }
    const v = verdict(ev);
    const cls = v.gly === 'NR' ? 'mx-nr' : `mx-gly ${v.cls}`;
    return `<td class="${lcls} ${colCls}"${t}><span class="${cls}">${v.gly}</span></td>`;
  }

  // timed cell — value depends on viz mode
  let valHtml, valCls = '';
  if (state.viz === 'speedup' && baselineMs != null) {
    const r = id === NATIVE_COL ? 1 : baselineMs / cell.ms;
    valCls = r > 1.03 ? 'faster' : r < 0.97 ? 'slower' : 'equal';
    valHtml = fmtSpeedup(r);
  } else {
    valHtml = cell.fmt || fmtMs(cell.ms);
  }
  const rowBest = (state.viz === 'duration' && id === fastestId) ? 'row-best' : '';
  const ev = effectiveParity(cell, meta);
  const gly = cell.is_canonical_reference
    ? `<span class="mx-gly ref" title="canonical reference">◆</span>`
    : (ev !== 'not_available' ? `<span class="mx-gly ${verdict(ev).cls}" title="${usesReferenceGate(meta) ? 'reference parity vs canonical external library' : 'binding parity vs C++ core'}">${verdict(ev).gly}</span>` : '');
  const tt = cell.timing_statistic ? ` title="timing: ${escAttr(cell.timing_statistic)}, ${cell.n_runs || '?'} run(s)"` : '';
  return `<td class="${lcls} ${colCls} ${rowBest}"${tt}><span class="mx-cell ${rowBest}"><span class="mx-val ${valCls}">${escapeHtml(valHtml)}</span>${gly}</span></td>`;
}

/* ==========================================================================
   Table render
   ========================================================================== */
function render() {
  const cols = visibleColumns();
  const dataCols = cols.filter(c => c.id !== REF_COL);
  const bands = langBands(cols);
  let rows = filteredRows();
  rows = sortRows(rows, cols);

  // header
  const sc = state.sort;
  const arrow = id => sc.col === id ? `<span class="arrow" aria-hidden="true">${sc.dir === 1 ? ' ↑' : ' ↓'}</span>` : '';
  let head = `<tr class="lang-band">
    <th class="spacer c-algo"></th><th class="spacer c-size"></th><th class="spacer c-thr"></th>`;
  for (const b of bands) head += `<th class="${b.ref ? 'c-ref' : ''}" colspan="${b.count}">${escapeHtml(b.label)}<span class="bcount">${b.count}</span></th>`;
  head += `</tr><tr class="colrow">
    <th class="c-algo lt-cpp" data-col="__algo" role="button" tabindex="0"><span class="col-label">algorithm${arrow('__algo')}</span></th>
    <th class="c-size" data-col="__size" role="button" tabindex="0"><span class="col-label">size${arrow('__size')}</span></th>
    <th class="c-thr" data-col="__threads" role="button" tabindex="0"><span class="col-label">t${arrow('__threads')}</span></th>`;
  for (const c of cols) {
    if (c.id === REF_COL) {
      head += `<th class="c-ref" title="canonical external reference per row"><span class="col-label">reference</span><span class="col-sub">per-method truth</span></th>`;
      continue;
    }
    const lt = `lt-${langSlug(c.lang)}`;
    head += `<th class="${lt}" data-col="${escAttr(c.id)}" role="button" tabindex="0" title="${escAttr([c.label, c.tier, c.lang, c.what].filter(Boolean).join(' — '))}">
      <span class="col-label">${escapeHtml(c.short || c.label)}${arrow(c.id)}</span>
      <span class="col-sub">${escapeHtml(c.tier || c.kind || '')}</span></th>`;
  }
  head += `</tr>`;

  // body
  let body = '';
  if (!rows.length) {
    body = `<tr><td colspan="${cols.length + 3}"><div class="mx-empty"><h3>nothing in this slice</h3>
      <p>your filters narrowed every row out. <span class="lnk" id="mxReset">reset all filters</span></p></div></td></tr>`;
  } else {
    const omap = DATA.algo_origin || {};
    const dmap = DATA.algo_has_doc || {};
    const fqmap = DATA.algo_fq || {};
    for (const row of rows) {
      const timed = rowCells(row, cols);
      const fastestId = timed.length ? timed.reduce((a, b) => a.ms <= b.ms ? a : b).id : null;
      const baseCell = NATIVE_COL ? row.cells[NATIVE_COL] : null;
      const baselineMs = baseCell && isFinite(baseCell.ms) ? baseCell.ms : null;
      const origin = omap[row.algo] || 'pls4all';
      const rowCls = origin === 'pls4all' ? 'row-pls4all' : 'row-donor';
      const disp = algoDisplay(row.algo);
      const fq = fqmap[row.algo];
      const docPage = dmap[row.algo];
      const algoHtml = docPage
        ? `<a href="methods/${escAttr(docPage)}.html" title="open ${escAttr(disp)} documentation">${escapeHtml(disp)}</a>`
        : `<span class="algo-nodoc" title="no dedicated documentation page yet">${escapeHtml(disp)}</span>`;
      let tr = `<tr class="${rowCls}">
        <td class="c-algo" title="${escAttr(fq || row.algo)}">${algoHtml}${fq ? `<span class="ns">${escapeHtml(fq)}</span>` : ''}</td>
        <td class="c-size">${row.n}×${row.p}</td>
        <td class="c-thr"><span class="thr-pill">${row.threads}t</span></td>`;
      for (const c of cols) {
        if (c.id === REF_COL) { tr += refCellHtml(row.cells[REF_COL], 'c-ref'); continue; }
        tr += dataCellHtml(row, c, baselineMs, fastestId);
      }
      tr += `</tr>`;
      body += tr;
    }
  }

  document.getElementById('mxHead').innerHTML = head;
  document.getElementById('mxBody').innerHTML = body;

  // meta line
  document.getElementById('mxMeta').innerHTML =
    `<span><b>${rows.length}</b> row${rows.length === 1 ? '' : 's'} shown · <b>${dataCols.length}</b> implementation columns</span>` +
    `<span>${DATA.rows.length} rows · <b>${(DATA.stats && DATA.stats.cells || 0).toLocaleString()}</b> timed cells total</span>` +
    `<span>generated <b>${escapeHtml(DATA.generated_at || '—')}</b></span>`;

  // wire sortable headers
  document.querySelectorAll('#mxHead [data-col]').forEach(th => {
    const act = () => {
      const col = th.dataset.col;
      if (state.sort.col === col) state.sort.dir *= -1;
      else state.sort = { col, dir: 1 };
      render();
    };
    th.addEventListener('click', act);
    th.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); } });
  });
  const rb = document.getElementById('mxReset');
  if (rb) rb.addEventListener('click', resetFilters);
}

/* ==========================================================================
   Per-method score cards
   ========================================================================== */
function histTotalReal(h) {
  return (h.exact || 0) + (h.cross_check || 0) + (h.divergent || 0) + (h.drift || 0) + (h.error || 0);
}
function histTotalValidated(h) {
  return (h.exact || 0) + (h.cross_check || 0);
}
function scoreBar(h) {
  const order = [['exact', 'exact'], ['cross_check', 'cross'], ['divergent', 'diverg'], ['drift', 'diverg'], ['error', 'diverg'], ['not_available', 'na'], ['not_run', 'nr']];
  const total = Object.values(h).reduce((s, v) => s + (typeof v === 'number' ? v : 0), 0) || 1;
  return order.map(([k, cls]) => h[k] ? `<i class="${cls}" style="width:${(h[k] / total) * 100}%" title="${k}: ${h[k]}"></i>` : '').join('');
}
function scoreRow(k, h) {
  const real = histTotalReal(h);
  const validated = histTotalValidated(h);
  const pct = real ? fmtPct(validated, real) + '%' : '—';
  const detail = real ? `${validated}/${real} validated` : 'no comparable gate';
  return `<div class="sc-row"><span class="sc-k">${k}</span><span class="sc-bar" role="img" aria-label="${k} parity">${scoreBar(h)}</span><span class="sc-pct" title="${detail}">${pct}</span></div>`;
}
function renderScores() {
  const wrap = document.getElementById('mxScores');
  const ms = DATA.method_scores || {};
  const gmap = DATA.algo_to_group || {};
  const fqmap = DATA.algo_fq || {};
  // show score cards only for the methods currently in the row filter
  const present = new Set(filteredRows().map(r => r.algo));
  const algos = Object.keys(ms).filter(a => present.has(a))
    .sort((a, b) => (GROUP_ORDER.indexOf(gmap[a]) - GROUP_ORDER.indexOf(gmap[b])) || algoDisplay(a).localeCompare(algoDisplay(b)));
  let html = '';
  for (const a of algos) {
    const sc = ms[a];
    const accent = GROUP_ACCENT[gmap[a]] || '#0d9488';
    const t = sc.timing || {};
    const dr = (sc.divergence && sc.divergence.reference) || {};
    const db = (sc.divergence && sc.divergence.binding) || {};
    const foot = [];
    if (t && t.median_ms != null) foot.push(`<span>⏱ <b>${fmtMs(t.median_ms)}</b> median</span>`);
    if (dr && dr.n) foot.push(`<span>ref Δ <b>${dr.max != null ? dr.max.toPrecision(2) : '—'}</b></span>`);
    if (db && db.n) foot.push(`<span>bind Δ <b>${db.max != null ? db.max.toPrecision(2) : '—'}</b></span>`);
    html += `<div class="mx-scorecard" style="--accent:${accent}">
      <h4>${escapeHtml(algoDisplay(a))}</h4>
      ${fqmap[a] ? `<div class="sc-ns">${escapeHtml(fqmap[a])}</div>` : ''}
      ${scoreRow('ref', sc.reference || {})}
      ${scoreRow('bind', sc.binding || {})}
      <div class="sc-foot">${foot.join('')}</div>
    </div>`;
  }
  wrap.innerHTML = `<div class="mx-scoregrid">${html || '<div class="mx-empty"><p>no methods in this slice</p></div>'}</div>`;
}

/* ==========================================================================
   Chrome: nav stats, filters
   ========================================================================== */
function resetFilters() {
  Object.assign(state, { search: '', category: '', origin: '', parity: '', size: '', threads: '' });
  document.getElementById('mxSearch').value = '';
  ['mxCategory', 'mxOrigin', 'mxParity', 'mxSize', 'mxThreads'].forEach(id => { const e = document.getElementById(id); if (e) e.value = ''; });
  refresh();
}
function refresh() { render(); if (state.showScores) renderScores(); }

function renderChrome() {
  const s = DATA.stats || {};
  document.getElementById('navAlgos').textContent = (s.algos ?? 0).toLocaleString();
  document.getElementById('navCells').textContent = (s.cells ?? 0).toLocaleString();
  document.getElementById('navGenerated').textContent = (DATA.generated_at || '').split(' ')[0] || '—';

  // host provenance strip
  const h = DATA.host || {};
  const prov = document.getElementById('provenance');
  if (prov) prov.innerHTML = [
    `<span class="prov-chip"><span class="dot"></span><b>${escapeHtml(h.cpu_model || 'host')}</b></span>`,
    `<span class="prov-chip"><b>${escapeHtml(String(h.cpu_count_physical || '?'))}</b>&nbsp;phys cores · <b>${escapeHtml(String(h.cpu_count_logical || '?'))}</b>&nbsp;threads</span>`,
    `<span class="prov-chip"><b>${escapeHtml(String(h.ram_gb || '?'))}</b>&nbsp;GB RAM</span>`,
    h.os ? `<span class="prov-chip">${escapeHtml(h.os)}</span>` : '',
    `<span class="prov-chip">generated <b>${escapeHtml(DATA.generated_at || '')}</b></span>`,
  ].join('');

  // category
  const cat = document.getElementById('mxCategory');
  cat.innerHTML = `<option value="">all categories</option>` +
    (DATA.algo_groups || []).map(g => `<option value="${escAttr(g.key)}">${escapeHtml(g.label)} (${g.algos.length})</option>`).join('');

  // origin
  const oc = {};
  for (const a of Object.keys(DATA.algo_origin || {})) oc[DATA.algo_origin[a]] = (oc[DATA.algo_origin[a]] || 0) + 1;
  const olab = { pls4all: 'pls4all native', n4m_donor: 'donor', external: 'external' };
  const orig = document.getElementById('mxOrigin');
  orig.innerHTML = `<option value="">all origins</option>` +
    Object.keys(oc).sort((a, b) => oc[b] - oc[a]).map(o => `<option value="${escAttr(o)}">${escapeHtml(olab[o] || o)} (${oc[o]})</option>`).join('');

  // size
  const sizes = [...new Set(DATA.rows.map(r => r.n + '×' + r.p))];
  const sz = document.getElementById('mxSize');
  sz.innerHTML = `<option value="">all sizes</option>` + sizes.map(s => `<option value="${escAttr(s)}">${escapeHtml(s)}</option>`).join('');

  // threads
  const thr = [...new Set(DATA.rows.map(r => r.threads))].sort((a, b) => a - b);
  const th = document.getElementById('mxThreads');
  th.innerHTML = `<option value="">all threads</option>` + thr.map(t => `<option value="${t}">${t} thread${t === 1 ? '' : 's'}</option>`).join('');
}

function wire() {
  const dbounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  document.getElementById('mxSearch').addEventListener('input', dbounce(e => { state.search = e.target.value.trim(); refresh(); }, 90));
  document.getElementById('mxCategory').addEventListener('change', e => { state.category = e.target.value; refresh(); });
  document.getElementById('mxOrigin').addEventListener('change', e => { state.origin = e.target.value; refresh(); });
  document.getElementById('mxParity').addEventListener('change', e => { state.parity = e.target.value; refresh(); });
  document.getElementById('mxSize').addEventListener('change', e => { state.size = e.target.value; refresh(); });
  document.getElementById('mxThreads').addEventListener('change', e => { state.threads = e.target.value; refresh(); });

  document.querySelectorAll('.mx-viz [data-viz]').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.mx-viz [data-viz]').forEach(x => x.setAttribute('aria-pressed', 'false'));
      b.setAttribute('aria-pressed', 'true');
      state.viz = b.dataset.viz;
      render();
    });
  });

  const sb = document.getElementById('mxScoreToggle');
  sb.addEventListener('click', () => {
    state.showScores = !state.showScores;
    sb.setAttribute('aria-pressed', String(state.showScores));
    sb.textContent = state.showScores ? 'hide score cards' : 'show score cards';
    const w = document.getElementById('mxScoreWrap');
    w.classList.toggle('collapsed', !state.showScores);
    if (state.showScores) renderScores();
  });
}

/* ==========================================================================
   Boot
   ========================================================================== */
renderChrome();
render();
wire();
