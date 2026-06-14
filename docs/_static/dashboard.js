/* ==========================================================================
   nirs4all-methods · parity & benchmark explorer (proposal v2)
   Vanilla JS, zero deps, zero network. Reads window.BENCH_DATA (never fetch()).
   Hand-rolled inline-SVG charts. file:// safe.
   ========================================================================== */
'use strict';

const DATA = window.BENCH_DATA;
if (!DATA) {
  document.getElementById('methods').innerHTML =
    '<div class="empty"><div class="big">⚠️</div><h3>benchmark data failed to load</h3>' +
    '<p>window.BENCH_DATA is undefined. The Sphinx build injects it inline before this script.</p></div>';
  throw new Error('BENCH_DATA missing');
}

/* ---------- verdict vocabulary: color + glyph + label, never color alone ---------- */
const VERDICT = {
  exact:        { cls: 'exact',  gly: '✓', label: 'exact' },
  cross_check:  { cls: 'cross',  gly: '◐', label: 'cross-check' },
  divergent:    { cls: 'diverg', gly: '▲', label: 'divergent' },
  drift:        { cls: 'diverg', gly: '▲', label: 'drift' },
  not_available:{ cls: 'na',     gly: '∅', label: 'n/a' },
  not_run:      { cls: 'nr',     gly: '·', label: 'NR' },
  error:        { cls: 'error',  gly: '✕', label: 'error' },
};
function verdict(v) {
  if (v == null) return { cls: 'na', gly: '∅', label: 'n/a' };
  return VERDICT[v] || { cls: 'na', gly: '∅', label: String(v) };
}

const LANG_CLASS = {
  'C++': 'lang-cpp', 'Python': 'lang-python', 'R': 'lang-r',
  'MATLAB/Octave': 'lang-matlab', 'external': 'lang-ext',
};
function langClass(l) { return LANG_CLASS[l] || 'lang-ext'; }

/* ---------- origin vocabulary (data values → human label + css token) ---------- */
const ORIGIN_LABEL = {
  pls4all: 'pls4all native',
  n4m_donor: 'donor',
  donor: 'donor',
  external: 'external',
};
function originLabel(o) { return ORIGIN_LABEL[o] || o; }
// css token: map the data value onto the styled classes that styles.css defines
const ORIGIN_CLASS = { pls4all: 'pls4all', n4m_donor: 'donor', donor: 'donor', external: 'external' };
function originClass(o) { return ORIGIN_CLASS[o] || 'external'; }

/* ---------- column lookup ---------- */
const COLS = DATA.columns;
const COL_BY_ID = Object.fromEntries(COLS.map(c => [c.id, c]));

/* ---------- category metadata ---------- */
const GROUP_LABEL = Object.fromEntries(DATA.algo_groups.map(g => [g.key, g.label]));
const GROUP_ORDER = DATA.algo_groups.map(g => g.key);
// Method doc pages are served from the site root as methods/<stem>.html.
// Relative (not the absolute methods.nirs4all.org URL) so the same build works
// at the custom domain, on gh-pages, and from a local file:// preview.
const DOCS_BASE = '.';
// per-category accent rotation through the brand palette
const ACCENTS = ['#0d9488', '#0891b2', '#4f46e5', '#d97706', '#10b981', '#0f766e', '#06b6d4', '#4338ca'];
const GROUP_ACCENT = {};
GROUP_ORDER.forEach((k, i) => { GROUP_ACCENT[k] = ACCENTS[i % ACCENTS.length]; });

/* ---------- build per-method model from rows + method_scores ---------- */
function buildMethods() {
  // index rows by algo
  const rowsByAlgo = {};
  for (const r of DATA.rows) {
    (rowsByAlgo[r.algo] || (rowsByAlgo[r.algo] = [])).push(r);
  }
  const methods = [];
  for (const algo of Object.keys(DATA.method_scores)) {
    const score = DATA.method_scores[algo];
    const group = DATA.algo_to_group[algo] || 'other';
    const origin = DATA.algo_origin[algo] || 'external';
    // ABI-2 namespace identity (replaces the legacy pp_/aug_/… registry id)
    const display = (DATA.algo_display && DATA.algo_display[algo]) || algo;
    const fq = (DATA.algo_fq && DATA.algo_fq[algo]) || null;
    const docPage = (DATA.algo_has_doc && DATA.algo_has_doc[algo]) || null;
    const docUrl = docPage ? `${DOCS_BASE}/methods/${docPage}.html` : null;  // ./methods/<stem>.html
    const symbol = docPage; // retained: doc-page handle
    const rows = rowsByAlgo[algo] || [];

    // distinct problem sizes for this method
    const sizes = [];
    const seen = new Set();
    for (const r of rows) {
      const key = r.n + 'x' + r.p + '@' + r.threads;
      if (!seen.has(key)) { seen.add(key); sizes.push({ n: r.n, p: r.p, threads: r.threads, key }); }
    }
    sizes.sort((a, b) => (a.n * a.p) - (b.n * b.p) || a.threads - b.threads);

    // which languages actually have any cell
    const langs = new Set();
    let anyJaccard = false;
    for (const r of rows) {
      for (const cid of Object.keys(r.cells)) {
        const cell = r.cells[cid];
        if (!cell) continue;
        const col = COL_BY_ID[cid];
        if (col) langs.add(col.lang);
        if (cell.divergence_metric === 'jaccard') anyJaccard = true;
      }
    }

    const refHist = score.reference || {};
    const bindHist = score.binding || {};
    const timing = score.timing || null;
    const divRef = (score.divergence && score.divergence.reference) || null;
    const divBind = (score.divergence && score.divergence.binding) || null;

    // an "attention" flag: any divergent / cross-check / error in either gate
    const attention = ['divergent', 'drift', 'cross_check', 'error']
      .reduce((s, k) => s + (refHist[k] || 0) + (bindHist[k] || 0), 0);

    methods.push({
      algo, display, fq, docPage, docUrl, group, origin, symbol, rows, sizes,
      langs: [...langs], anyJaccard,
      refHist, bindHist, timing, divRef, divBind, attention,
      hasRef: histTotalReal(refHist) > 0,
      hasDoc: !!docPage,
      // searchable haystack: new name + namespace + legacy id (so old IDs still find it)
      hay: (display + ' ' + (fq || '') + ' ' + algo + ' ' + group + ' ' + (GROUP_LABEL[group] || '') + ' ' + origin).toLowerCase(),
    });
  }
  return methods;
}
// "real" verdict total = excludes not_run/not_available so percentages mean something
function histTotalReal(h) {
  return (h.exact || 0) + (h.cross_check || 0) + (h.divergent || 0) + (h.drift || 0) + (h.error || 0);
}
function histTotalAll(h) {
  return Object.values(h).reduce((s, v) => s + (typeof v === 'number' ? v : 0), 0);
}

const METHODS = buildMethods();

/* ==========================================================================
   Fuzzy search — lightweight subsequence scorer (no deps).
   Matches name / symbol / category. Returns score; higher = better.
   ========================================================================== */
function fuzzyScore(query, target) {
  if (!query) return 1;
  query = query.toLowerCase();
  target = target.toLowerCase();
  if (target.includes(query)) {
    // contiguous match: strong, bonus for prefix / word-boundary
    let base = 1000 - target.indexOf(query);
    if (target.startsWith(query)) base += 600;
    return base;
  }
  // subsequence match
  let ti = 0, score = 0, streak = 0, matched = 0;
  for (let qi = 0; qi < query.length; qi++) {
    const ch = query[qi];
    let found = -1;
    for (let j = ti; j < target.length; j++) {
      if (target[j] === ch) { found = j; break; }
    }
    if (found === -1) return 0;
    matched++;
    streak = (found === ti) ? streak + 1 : 0;
    score += 10 + streak * 4 - (found - ti);
    ti = found + 1;
  }
  if (matched < query.length) return 0;
  return Math.max(1, score);
}

/* ==========================================================================
   State + filtering
   ========================================================================== */
const state = {
  query: '',
  category: '',
  origin: '',
  parity: '',
  sort: 'category',
};

function filtered() {
  let list = METHODS.slice();

  if (state.category) list = list.filter(m => m.group === state.category);
  if (state.origin)   list = list.filter(m => m.origin === state.origin);
  if (state.parity === 'clean')     list = list.filter(m => m.attention === 0 && (m.refHist.exact || m.bindHist.exact));
  if (state.parity === 'attention') list = list.filter(m => m.attention > 0);
  if (state.parity === 'hasref')    list = list.filter(m => m.hasRef);
  if (state.parity === 'hasdoc')    list = list.filter(m => m.hasDoc);

  if (state.query) {
    list = list.map(m => ({ m, s: fuzzyScore(state.query, m.hay) }))
               .filter(x => x.s > 0)
               .sort((a, b) => b.s - a.s)
               .map(x => x.m);
    // when searching, score order wins unless an explicit non-default sort is chosen
    if (state.sort === 'category') return list;
  }

  const cmp = {
    name: (a, b) => a.algo.localeCompare(b.algo),
    slowest: (a, b) => (b.timing ? b.timing.max_ms : -1) - (a.timing ? a.timing.max_ms : -1),
    attention: (a, b) => b.attention - a.attention || a.algo.localeCompare(b.algo),
    category: (a, b) => GROUP_ORDER.indexOf(a.group) - GROUP_ORDER.indexOf(b.group) || a.algo.localeCompare(b.algo),
  }[state.sort];
  if (cmp && !(state.query && state.sort === 'category')) list.sort(cmp);
  return list;
}

/* ==========================================================================
   Rendering — overview
   ========================================================================== */
const el = {
  methods: document.getElementById('methods'),
  count: document.getElementById('resultCount'),
};

function fmtPct(n, d) { return d ? Math.round((n / d) * 100) : 0; }
function fmtMs(ms) {
  if (ms == null) return '—';
  if (ms < 1) return ms.toFixed(2) + ' ms';
  if (ms < 100) return ms.toFixed(1) + ' ms';
  if (ms < 1000) return Math.round(ms) + ' ms';
  return (ms / 1000).toFixed(2) + ' s';
}
function fmtDiv(value, metric) {
  if (value == null) return '—';
  if (metric === 'jaccard') return value.toFixed(2);
  if (value === 0) return '0';
  const a = Math.abs(value);
  if (a < 1e-3 || a >= 1e4) return a.toExponential(0).replace('e+', 'e').replace('e-0', 'e-').replace('e+0', 'e');
  if (a < 1) return a.toPrecision(2);
  return a.toFixed(2);
}

// horizontal stacked verdict bar segments for a histogram
function meterSegments(h) {
  const order = [
    ['exact', 'exact'], ['cross_check', 'cross'], ['divergent', 'diverg'],
    ['drift', 'diverg'], ['error', 'diverg'], ['not_available', 'na'], ['not_run', 'nr'],
  ];
  const total = histTotalAll(h) || 1;
  let html = '';
  for (const [k, cls] of order) {
    const v = h[k] || 0;
    if (!v) continue;
    html += `<i class="${cls}" style="width:${(v / total) * 100}%" title="${k}: ${v}"></i>`;
  }
  return html;
}

function meterMeta(h) {
  const real = histTotalReal(h);
  const exact = h.exact || 0;
  const nr = h.not_run || 0;
  const na = h.not_available || 0;
  if (real === 0) {
    // nothing comparable — surface NR / NA honestly, never a fake %
    const bits = [];
    if (nr) bits.push(`${nr} NR`);
    if (na) bits.push(`${na} n/a`);
    return bits.length ? `<span style="color:var(--v-nr)">${bits.join(' · ')}</span>` : '<span style="color:var(--v-nr)">no gate</span>';
  }
  let s = `<b>${fmtPct(exact, real)}%</b> exact <span style="color:var(--text-3)">(${exact}/${real})</span>`;
  if (nr) s += ` <span style="color:var(--v-nr)">· ${nr} NR</span>`;
  return s;
}

// human-readable verdict tally for screen readers (color-independent)
function meterAriaLabel(label, h) {
  const order = [
    ['exact', 'exact'], ['cross_check', 'cross-check'], ['divergent', 'divergent'],
    ['drift', 'drift'], ['error', 'error'], ['not_available', 'not available'], ['not_run', 'not run'],
  ];
  const bits = order.filter(([k]) => h[k]).map(([k, name]) => `${h[k]} ${name}`);
  return `${label} parity: ${bits.length ? bits.join(', ') : 'no cells'}`;
}

function parityMeter(label, glyph, h) {
  return `
    <div class="pmeter">
      <div class="pm-label"><span aria-hidden="true">${glyph}</span>${label}</div>
      <div class="pm-bar" role="img" aria-label="${escapeAttr(meterAriaLabel(label, h))}">${meterSegments(h)}</div>
      <div class="pm-meta">${meterMeta(h)}</div>
    </div>`;
}

function topVerdictBadge(h) {
  // pick the most "interesting" non-clean verdict to surface, else exact
  for (const k of ['error', 'divergent', 'drift', 'cross_check']) {
    if (h[k]) { const v = verdict(k); return `<span class="badge ${v.cls} sm"><span class="gly" aria-hidden="true">${v.gly}</span>${v.label} ×${h[k]}</span>`; }
  }
  if (h.exact) { const v = verdict('exact'); return `<span class="badge ${v.cls} sm"><span class="gly" aria-hidden="true">${v.gly}</span>all exact</span>`; }
  if (h.not_available) { const v = verdict('not_available'); return `<span class="badge ${v.cls} sm"><span class="gly" aria-hidden="true">${v.gly}</span>${v.label}</span>`; }
  const v = verdict('not_run'); return `<span class="badge ${v.cls} sm"><span class="gly" aria-hidden="true">${v.gly}</span>not run</span>`;
}

function methodCard(m) {
  const accent = GROUP_ACCENT[m.group];
  const refBadge = topVerdictBadge(m.refHist);
  const bindBadge = topVerdictBadge(m.bindHist);
  const tMed = m.timing ? fmtMs(m.timing.median_ms) : '—';
  const sym = m.fq
    ? `<div class="m-sym" title="namespace">${escapeHtml(m.fq)}</div>`
    : `<div class="m-sym" style="opacity:.55">${escapeHtml(m.algo)}</div>`;
  const docLink = m.docUrl
    ? `<a class="m-doc" href="${escapeAttr(m.docUrl)}"
          title="Open method documentation"
          aria-label="${escapeAttr(m.display)} documentation"
          onclick="event.stopPropagation()">📄 docs</a>`
    : '';
  return `
    <div class="mcard" style="--accent:${accent}" data-algo="${escapeAttr(m.algo)}"
         role="button" tabindex="0" aria-label="Open details for ${escapeAttr(m.display)}">
      <div class="m-top">
        <div>
          <h3>${escapeHtml(m.display)}</h3>
          ${sym}
        </div>
        <span class="m-origin ${originClass(m.origin)}">${escapeHtml(originLabel(m.origin))}</span>
      </div>
      <div class="m-badges">
        <span style="font-size:.62rem;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;font-weight:700;align-self:center">ref</span>${refBadge}
        <span style="font-size:.62rem;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;font-weight:700;align-self:center;margin-left:4px">bind</span>${bindBadge}
      </div>
      <div class="parity-pair">
        ${parityMeter('reference', '⇄', m.refHist)}
        ${parityMeter('binding', '⊻', m.bindHist)}
      </div>
      <div class="m-foot">
        <span title="languages with at least one cell">${langDots(m.langs)}</span>
        <span title="median wall-clock across timed cells" style="margin-left:auto">⏱ <b>${tMed}</b></span>
        <span title="problem sizes benchmarked">▦ <b>${m.sizes.length}</b> size${m.sizes.length === 1 ? '' : 's'}</span>
        ${docLink}
      </div>
    </div>`;
}

function langDots(langs) {
  const order = ['C++', 'Python', 'R', 'MATLAB/Octave', 'external'];
  return order.filter(l => langs.includes(l)).map(l =>
    `<span class="lang-tag ${langClass(l)}" style="display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:3px" title="${l}"></span>`
  ).join('');
}

function render() {
  const list = filtered();
  el.count.textContent = `${list.length} of ${METHODS.length} methods`;

  if (list.length === 0) {
    el.methods.innerHTML = `
      <div class="empty">
        <div class="big" aria-hidden="true">🔍</div>
        <h3>No methods match your filters</h3>
        <p>Try a shorter search term or reset the filters.</p>
        <button class="ctl" id="resetFilters">Reset all filters</button>
      </div>`;
    document.getElementById('resetFilters').onclick = resetFilters;
    return;
  }

  // group rendering: only group when sort is by-category and not actively fuzzy-ranked
  const grouped = state.sort === 'category' && !state.query;
  let html = '';
  if (grouped) {
    const byGroup = {};
    for (const m of list) (byGroup[m.group] || (byGroup[m.group] = [])).push(m);
    for (const gk of GROUP_ORDER) {
      const ms = byGroup[gk];
      if (!ms || !ms.length) continue;
      const accent = GROUP_ACCENT[gk];
      // category mini-summary: aggregate exact ratio
      let exact = 0, real = 0;
      for (const m of ms) { exact += histTotalReal(m.refHist) ? (m.refHist.exact || 0) : 0; real += histTotalReal(m.refHist); }
      const miniRef = real ? `${fmtPct(exact, real)}% ref-exact` : '';
      html += `
        <section class="cat-section">
          <div class="cat-head">
            <span class="cat-bar" style="background:${accent}"></span>
            <h2>${escapeHtml(GROUP_LABEL[gk] || gk)}</h2>
            <span class="cat-count">${ms.length}</span>
            ${miniRef ? `<span class="cat-mini"><span class="badge exact sm"><span class="gly" aria-hidden="true">✓</span>${miniRef}</span></span>` : ''}
          </div>
          <div class="cards">${ms.map(methodCard).join('')}</div>
        </section>`;
    }
  } else {
    html = `<section class="cat-section"><div class="cards">${list.map(methodCard).join('')}</div></section>`;
  }
  el.methods.innerHTML = html;

  // wire cards
  el.methods.querySelectorAll('.mcard').forEach(btn => {
    btn.addEventListener('click', () => openDrawer(btn.dataset.algo));
    btn.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDrawer(btn.dataset.algo); }
    });
  });
}

function resetFilters() {
  state.query = ''; state.category = ''; state.origin = ''; state.parity = '';
  document.getElementById('search').value = '';
  document.getElementById('filterCategory').value = '';
  document.getElementById('filterOrigin').value = '';
  document.getElementById('filterParity').value = '';
  render();
}

/* ==========================================================================
   Drill-down drawer
   ========================================================================== */
const drawer = document.getElementById('drawer');
const overlay = document.getElementById('overlay');
let lastFocus = null;

function openDrawer(algo) {
  const m = METHODS.find(x => x.algo === algo);
  if (!m) return;
  lastFocus = document.activeElement;
  document.getElementById('drawerTitle').textContent = m.display;
  const _fq = m.fq
    ? `<span style="font-family:var(--mono)">${escapeHtml(m.fq)}</span>`
    : `<span style="opacity:.7">${escapeHtml(m.algo)} — no namespace mapping</span>`;
  const _doc = m.docUrl
    ? `  ·  <a href="${escapeAttr(m.docUrl)}">Open documentation</a>`
    : '';
  document.getElementById('drawerSym').innerHTML = _fq + _doc;
  document.getElementById('drawerMeta').innerHTML = drawerMetaHtml(m);
  document.getElementById('drawerBody').innerHTML = drawerBodyHtml(m);

  drawer.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';
  document.getElementById('drawerClose').focus();

  // post-render wiring (charts + size picker)
  wireDrawer(m);
  history.replaceState(null, '', '#m/' + encodeURIComponent(algo));
}

function closeDrawer() {
  drawer.classList.remove('open');
  drawer.setAttribute('aria-hidden', 'true');
  overlay.classList.remove('open');
  document.body.style.overflow = '';
  if (lastFocus) lastFocus.focus();
  history.replaceState(null, '', '#');
}

function drawerMetaHtml(m) {
  const accent = GROUP_ACCENT[m.group];
  let h = `<span class="badge" style="color:${accent};background:${accent}1a;border-color:${accent}40">${escapeHtml(GROUP_LABEL[m.group] || m.group)}</span>`;
  h += `<span class="m-origin ${originClass(m.origin)}" style="font-size:.66rem;padding:3px 9px">${escapeHtml(originLabel(m.origin))}</span>`;
  h += `<span class="prov-chip"><span class="dot" style="background:${accent}"></span>${m.langs.length} language${m.langs.length === 1 ? '' : 's'}</span>`;
  if (m.anyJaccard) h += `<span class="badge cross"><span class="gly" aria-hidden="true">∩</span>selector method · Jaccard parity</span>`;
  if (!m.hasDoc) h += `<span class="badge na"><span class="gly" aria-hidden="true">∅</span>undocumented</span>`;
  return h;
}

function scoreTile(label, value, sub) {
  return `<div class="tile"><div class="t-l">${label}</div><div class="t-v">${value}</div><div class="t-s">${sub || ''}</div></div>`;
}

function drawerBodyHtml(m) {
  let html = '';

  /* --- 1. score summary --- */
  const refReal = histTotalReal(m.refHist), bindReal = histTotalReal(m.bindHist);
  html += `<div class="dsection"><h3>Score card</h3><div class="dscore">`;
  html += scoreTile('Reference parity', refReal ? `${fmtPct(m.refHist.exact || 0, refReal)}%` : '—',
    refReal ? `${m.refHist.exact || 0}/${refReal} exact vs canonical lib` : `${m.refHist.not_run || 0} NR · ${m.refHist.not_available || 0} n/a`);
  html += scoreTile('Binding parity', bindReal ? `${fmtPct(m.bindHist.exact || 0, bindReal)}%` : '—',
    bindReal ? `${m.bindHist.exact || 0}/${bindReal} exact vs C++ core` : `${m.bindHist.not_run || 0} NR · ${m.bindHist.not_available || 0} n/a`);
  // selector methods report divergence as Jaccard set-overlap on BOTH gates; numeric
  // methods report relative-RMSE Δ. Label + format must follow the method's metric.
  const divMetric = m.anyJaccard ? 'jaccard' : 'rmse';
  const divLead = m.anyJaccard ? 'Jaccard overlap (1.00 = identical mask)' : 'max |Δrel-RMSE|';
  if (m.divRef) {
    html += scoreTile('Reference divergence', fmtDiv(m.divRef.max, divMetric),
      `${divLead} · n=${m.divRef.n} · median ${fmtDiv(m.divRef.median, divMetric)}`);
  }
  if (m.divBind) {
    html += scoreTile('Binding divergence', fmtDiv(m.divBind.max, divMetric),
      `${divLead} · n=${m.divBind.n} · median ${fmtDiv(m.divBind.median, divMetric)}`);
  }
  if (m.timing) {
    html += scoreTile('Timing', fmtMs(m.timing.median_ms),
      `median · ${fmtMs(m.timing.min_ms)} → ${fmtMs(m.timing.max_ms)} · ${m.timing.n} cells`);
  }
  html += `</div></div>`;

  /* --- 2. verdict histograms (full breakdown) --- */
  html += `<div class="dsection"><h3>Verdict breakdown</h3><div class="dscore" style="grid-template-columns:1fr 1fr">`;
  html += verdictBreakdownCard('Reference gate', '⇄', m.refHist, 'libn4m C++ core vs canonical external library');
  html += verdictBreakdownCard('Binding gate', '⊻', m.bindHist, 'each binding tier vs the C++ core');
  html += `</div></div>`;

  /* --- 3. timing curves (only when timed cells actually exist) --- */
  const hasTiming = m.timing && (m.timing.n || 0) > 0 && m.timing.median_ms != null;
  if (hasTiming && m.sizes.length >= 1) {
    html += `<div class="dsection"><h3>Timing vs problem size</h3>${timingChartHtml(m)}</div>`;
  } else {
    html += `<div class="dsection"><h3>Timing vs problem size</h3>
      <div class="chart-card" style="color:var(--text-3);font-size:.82rem">
        No wall-clock timing was recorded for this method — it is a parity-only / fixture method
        (the matrix still verifies its verdicts, but it carries no benchmarked <span class="mono">ms</span>).
      </div></div>`;
  }

  /* --- 4. multi-reference parity table --- */
  html += `<div class="dsection"><h3>Parity matrix · per implementation</h3>${parityTableHtml(m)}</div>`;

  /* --- 5. provenance --- */
  html += `<div class="dsection"><h3>Provenance</h3>${provenanceHtml(m)}</div>`;

  return html;
}

function verdictBreakdownCard(label, glyph, h, sub) {
  const order = ['exact', 'cross_check', 'divergent', 'drift', 'error', 'not_available', 'not_run'];
  let rows = '';
  for (const k of order) {
    const v = h[k]; if (!v) continue;
    const vd = verdict(k);
    rows += `<div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:.8rem">
      <span class="badge ${vd.cls} sm" style="min-width:96px"><span class="gly" aria-hidden="true">${vd.gly}</span>${vd.label}</span>
      <span class="mono" style="font-weight:700">${v}</span></div>`;
  }
  if (!rows) rows = '<div style="color:var(--text-3);font-size:.8rem">no cells</div>';
  return `<div class="tile"><div class="t-l">${glyph} ${label}</div>
    <div class="t-s" style="margin:2px 0 8px">${sub}</div>${rows}</div>`;
}

/* ---------- timing chart (hand-rolled SVG, linear/log toggle) ---------- */
function collectTimingSeries(m) {
  // Build per-column series of {sizeIndex, n, p, threads, ms} across sizes.
  // Use the smallest-threads (canonical, threads=1) cells for a clean curve,
  // falling back to whatever exists. One point per (column, size).
  const sizeKeys = m.sizes.map(s => s.key);
  const sizeIndex = Object.fromEntries(sizeKeys.map((k, i) => [k, i]));
  const series = {}; // colId -> [{x, ms, n, p, threads, size}]
  for (const r of m.rows) {
    const skey = r.n + 'x' + r.p + '@' + r.threads;
    const xi = sizeIndex[skey];
    if (xi == null) continue;
    for (const cid of Object.keys(r.cells)) {
      const cell = r.cells[cid];
      if (!cell || cell.ms == null) continue;
      (series[cid] || (series[cid] = [])).push({
        x: xi, ms: cell.ms, n: r.n, p: r.p, threads: r.threads,
        sizeLabel: `${r.n}×${r.p}` + (m.sizes.some(s => s.threads !== 1) ? ` · ${r.threads}t` : ''),
      });
    }
  }
  // keep only columns with >=1 point; sort points by x then threads
  const out = [];
  for (const cid of Object.keys(series)) {
    const pts = series[cid].sort((a, b) => a.x - b.x || a.threads - b.threads);
    const col = COL_BY_ID[cid];
    if (!col) continue;
    out.push({ id: cid, col, pts });
  }
  // surface the most informative columns first: cpp blas+omp, python, sklearn, R, reference
  const PRIORITY = ['pls4all.cpp.blas+omp', 'pls4all.cpp.native', 'pls4all.python', 'pls4all.sklearn', 'pls4all.R', '__reference'];
  out.sort((a, b) => {
    const pa = PRIORITY.indexOf(a.id), pb = PRIORITY.indexOf(b.id);
    return (pa === -1 ? 99 : pa) - (pb === -1 ? 99 : pb);
  });
  return { series: out, sizes: m.sizes };
}

const SERIES_COLORS = {
  'pls4all.cpp.blas+omp': '#0891b2', 'pls4all.cpp.native': '#0e7490', 'pls4all.cpp.blas': '#22d3ee',
  'pls4all.cpp.omp': '#155e75', 'pls4all.python': '#4f46e5', 'pls4all.sklearn': '#7c3aed',
  'pls4all.R': '#0d9488', 'pls4all.R.formula': '#14b8a6', 'pls4all.matlab': '#d97706',
  '__reference': '#b3261e',
};
function seriesColor(id, i) { return SERIES_COLORS[id] || ACCENTS[i % ACCENTS.length]; }

function timingChartHtml(m) {
  return `
    <div class="chart-card">
      <div class="chart-head">
        <span class="ch-title">wall-clock (median) across ${m.sizes.length} problem size${m.sizes.length === 1 ? '' : 's'}</span>
        <div class="seg" role="group" aria-label="Y axis scale">
          <button type="button" class="scale-btn" data-scale="linear" aria-pressed="true">linear</button>
          <button type="button" class="scale-btn" data-scale="log" aria-pressed="false">log</button>
        </div>
      </div>
      <svg class="chart-svg" id="timingSvg" viewBox="0 0 720 300" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Timing versus problem size line chart"></svg>
      <div class="chart-legend" id="timingLegend"></div>
    </div>`;
}

function drawTimingChart(m, scale) {
  const svg = document.getElementById('timingSvg');
  if (!svg) return;
  const { series, sizes } = collectTimingSeries(m);
  const W = 720, H = 300, padL = 56, padR = 16, padT = 14, padB = 46;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const allMs = series.flatMap(s => s.pts.map(p => p.ms)).filter(v => v > 0);
  if (allMs.length === 0) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle">no timing data</text>`; return; }
  let minY = Math.min(...allMs), maxY = Math.max(...allMs);
  const nX = Math.max(1, sizes.length);

  const xPos = i => padL + (nX === 1 ? plotW / 2 : (i / (nX - 1)) * plotW);
  let yPos;
  let ticks = [];
  if (scale === 'log') {
    const lmin = Math.log10(Math.max(minY, 1e-4)), lmax = Math.log10(maxY <= minY ? minY * 10 : maxY);
    const pad = (lmax - lmin) * 0.08 || 0.3;
    const a = lmin - pad, b = lmax + pad;
    yPos = v => padT + plotH - ((Math.log10(Math.max(v, 1e-6)) - a) / (b - a)) * plotH;
    for (let e = Math.floor(a); e <= Math.ceil(b); e++) {
      const val = Math.pow(10, e);
      if (val < minY / 5 || val > maxY * 5) continue;
      ticks.push({ v: val, y: yPos(val) });
    }
    if (ticks.length < 2) ticks = [{ v: minY, y: yPos(minY) }, { v: maxY, y: yPos(maxY) }];
  } else {
    minY = 0;
    const a = 0, b = maxY * 1.08 || 1;
    yPos = v => padT + plotH - ((v - a) / (b - a)) * plotH;
    for (let i = 0; i <= 4; i++) { const val = (b) * i / 4; ticks.push({ v: val, y: yPos(val) }); }
  }

  let g = '';
  // grid + y ticks
  for (const t of ticks) {
    g += `<line x1="${padL}" y1="${t.y.toFixed(1)}" x2="${W - padR}" y2="${t.y.toFixed(1)}" stroke="#e8e2d3" stroke-width="1"/>`;
    g += `<text x="${padL - 8}" y="${(t.y + 3.5).toFixed(1)}" text-anchor="end" font-size="10">${fmtMsShort(t.v)}</text>`;
  }
  // x axis labels (size)
  for (let i = 0; i < nX; i++) {
    const s = sizes[i];
    const lab = `${s.n}×${s.p}` + (sizes.some(z => z.threads !== 1) ? ` ${s.threads}t` : '');
    g += `<text x="${xPos(i).toFixed(1)}" y="${H - padB + 18}" text-anchor="middle" font-size="10">${lab}</text>`;
  }
  g += `<text x="${padL + plotW / 2}" y="${H - 6}" text-anchor="middle" font-size="10.5" fill="#64748b">problem size (n × p)</text>`;
  g += `<text transform="translate(13,${padT + plotH / 2}) rotate(-90)" text-anchor="middle" font-size="10.5" fill="#64748b">median wall-clock</text>`;

  // series paths + points
  let legend = '';
  series.forEach((s, i) => {
    const color = seriesColor(s.id, i);
    const pts = s.pts.filter(p => p.ms > 0);
    if (!pts.length) return;
    // line (only across distinct x; if a column has multiple thread points per x just plot points)
    const lineable = (() => {
      const byx = {};
      for (const p of pts) { if (!(p.x in byx)) byx[p.x] = p; }
      return Object.values(byx).sort((a, b) => a.x - b.x);
    })();
    if (lineable.length >= 2) {
      const d = lineable.map((p, k) => `${k ? 'L' : 'M'}${xPos(p.x).toFixed(1)} ${yPos(p.ms).toFixed(1)}`).join(' ');
      g += `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" opacity="0.92"/>`;
    }
    for (const p of pts) {
      g += `<circle class="tpt" cx="${xPos(p.x).toFixed(1)}" cy="${yPos(p.ms).toFixed(1)}" r="3.6" fill="${color}" stroke="#fff" stroke-width="1.3"
        data-label="${escapeAttr(s.col.label)}" data-ms="${p.ms}" data-size="${escapeAttr(p.sizeLabel)}" data-lang="${escapeAttr(s.col.lang)}"/>`;
    }
    legend += `<span class="lk"><span class="sw" style="background:${color}"></span>${escapeHtml(s.col.short || s.col.label)}</span>`;
  });

  svg.innerHTML = g;
  const lg = document.getElementById('timingLegend');
  if (lg) lg.innerHTML = legend || '<span style="color:var(--text-3)">no timed implementations</span>';

  // tooltips
  const tip = document.getElementById('chartTip');
  svg.querySelectorAll('.tpt').forEach(c => {
    c.addEventListener('mouseenter', (e) => {
      tip.innerHTML = `<b>${c.dataset.label}</b><br>${c.dataset.size} · ${c.dataset.lang}<br>median <b>${fmtMs(parseFloat(c.dataset.ms))}</b>`;
      tip.style.opacity = '1';
      moveTip(e);
    });
    c.addEventListener('mousemove', moveTip);
    c.addEventListener('mouseleave', () => { tip.style.opacity = '0'; });
  });
  function moveTip(e) {
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top = (e.clientY - 10) + 'px';
  }
}
function fmtMsShort(v) {
  if (v >= 1000) return (v / 1000).toFixed(v >= 10000 ? 0 : 1) + 's';
  if (v >= 10) return Math.round(v) + '';
  if (v >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

/* ---------- parity table (per implementation column, for a chosen size) ---------- */
function parityTableHtml(m) {
  // size picker: default to median-ish size (largest)
  const sizes = m.sizes;
  let picker = '';
  if (sizes.length > 1) {
    picker = `<div class="size-pick" role="group" aria-label="Choose problem size for parity table">` +
      sizes.map((s, i) =>
        `<button type="button" data-size="${s.key}" aria-pressed="${i === sizes.length - 1 ? 'true' : 'false'}">${s.n}×${s.p}${sizes.some(z => z.threads !== 1) ? ` · ${s.threads}t` : ''}</button>`
      ).join('') + `</div>`;
  }
  return `${picker}<div id="ptableMount" class="ptable-wrap"></div>`;
}

function drawParityTable(m, sizeKey) {
  const mount = document.getElementById('ptableMount');
  if (!mount) return;
  const row = m.rows.find(r => (r.n + 'x' + r.p + '@' + r.threads) === sizeKey) || m.rows[m.rows.length - 1];
  if (!row) { mount.innerHTML = '<div style="padding:18px;color:var(--text-3)">no rows for this method</div>'; return; }

  // Build one table row per implementation column that has a cell here.
  // Order: C++ tiers, python, R, matlab, then externals, reference last.
  const colIds = Object.keys(row.cells).filter(cid => row.cells[cid] && COL_BY_ID[cid]);
  const ORDER_GROUP = { cpp: 0, python: 1, r: 2, matlab: 3, 'ext-py': 4, 'ext-r': 5, 'ext-ml': 6, reference: 9 };
  colIds.sort((a, b) => {
    const ga = ORDER_GROUP[COL_BY_ID[a].group] ?? 7, gb = ORDER_GROUP[COL_BY_ID[b].group] ?? 7;
    return ga - gb || a.localeCompare(b);
  });

  let body = '';
  for (const cid of colIds) {
    const cell = row.cells[cid];
    const col = COL_BY_ID[cid];
    const lc = langClass(col.lang);

    const rp = verdict(cell.reference_parity);
    const bp = verdict(cell.binding_parity);

    // divergence cell — distinguish metric, NEVER fake NR/NA into a number
    let divHtml;
    const metric = cell.divergence_metric || 'rmse';
    if (cell.divergence == null) {
      // honest dash: explain what we DON'T have
      const why = cell.reference_parity === 'not_run' || cell.binding_parity === 'not_run' ? 'not run'
                : cell.reference_parity === 'not_available' ? 'n/a'
                : (cell.divergence_fmt === 'source' ? 'reference source' : '—');
      divHtml = `<span class="dash" title="${why}">—</span>`;
    } else {
      const tag = metric === 'jaccard'
        ? `<span class="metric-tag jaccard" title="set overlap of selected variables; 1.00 = identical mask">J</span>`
        : `<span class="metric-tag rmse" title="relative RMSE delta vs the gate basis">ΔRMSE</span>`;
      const basis = cell.divergence_basis ? `<span style="font-size:.6rem;color:var(--text-3)">${cell.divergence_basis}</span>` : '';
      divHtml = `<span class="div-cell">${tag}<span class="mono">${escapeHtml(cell.divergence_fmt ?? fmtDiv(cell.divergence, metric))}</span>${basis}</span>`;
    }

    const ms = cell.ms == null
      ? `<span class="dash">—</span>`
      : `<span class="mono">${cell.fmt || fmtMs(cell.ms)}</span>`;

    body += `<tr class="${lc}">
      <td class="col-impl"><span class="lang-tag"></span>${escapeHtml(col.label)}
        <span style="color:var(--text-3);font-size:.66rem;display:block;margin-left:15px">${escapeHtml(col.lang)} · ${escapeHtml(col.tier || col.group)}</span></td>
      <td><span class="badge ${rp.cls} sm"><span class="gly" aria-hidden="true">${rp.gly}</span>${rp.label}</span></td>
      <td><span class="badge ${bp.cls} sm"><span class="gly" aria-hidden="true">${bp.gly}</span>${bp.label}</span></td>
      <td>${divHtml}</td>
      <td class="num">${ms}</td>
    </tr>`;

    // surface the informational note whenever ANY gate is a cross-check (not just the
    // top-level verdict), but skip the terse `selection_exact` marker which carries no
    // explanation and would just add noise on already-exact selector rows.
    const isCross = cell.parity === 'cross_check'
      || cell.reference_parity === 'cross_check'
      || cell.binding_parity === 'cross_check';
    if (cell.divergence_note && cell.divergence_note !== 'selection_exact' && isCross) {
      body += `<tr class="note-row"><td colspan="5">◐ ${escapeHtml(cell.divergence_note)}</td></tr>`;
    }
  }

  mount.innerHTML = `
    <table class="ptable">
      <thead><tr>
        <th>implementation</th>
        <th title="libn4m core vs canonical external library">reference&nbsp;⇄</th>
        <th title="this binding vs the C++ core">binding&nbsp;⊻</th>
        <th title="ΔRMSE for numeric rows, Jaccard for selector rows">divergence</th>
        <th style="text-align:right">median&nbsp;ms</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table>`;
}

function provenanceHtml(m) {
  const h = DATA.host || {};
  const v = DATA.versions || {};
  // representative version strings
  const verBits = [];
  for (const k of ['pls4all.cpp.blas+omp', 'pls4all.python', 'pls4all.R', 'sklearn']) {
    if (v[k]) verBits.push(`<span class="prov-chip"><b>${escapeHtml(k.replace('pls4all.', 'n4m.'))}</b> ${escapeHtml(String(v[k]))}</span>`);
  }
  return `
    <div class="provenance" style="margin-top:0">
      <span class="prov-chip"><span class="dot"></span><b>host</b> ${escapeHtml(h.cpu_model || '—')}</span>
      <span class="prov-chip"><b>cores</b> ${escapeHtml(String(h.cpu_count_physical || '?'))}p / ${escapeHtml(String(h.cpu_count_logical || '?'))}l</span>
      <span class="prov-chip"><b>RAM</b> ${escapeHtml(String(h.ram_gb || '?'))} GB</span>
      <span class="prov-chip"><b>os</b> ${escapeHtml(h.os || '—')}</span>
      <span class="prov-chip"><b>python</b> ${escapeHtml(h.python || '—')}</span>
      <span class="prov-chip"><b>generated</b> ${escapeHtml(DATA.generated_at || '—')}</span>
    </div>
    <div class="provenance">${verBits.join('')}</div>
    <p style="font-size:.74rem;color:var(--text-3);margin:12px 0 0">
      Timing is the <b>median</b> wall-clock over repeated runs after warm-up; parity-only fixture cells and
      build-insensitive sentinels stay visible in the matrix but never inflate timing aggregates.
      <b>NR</b> (not run) and <b>n/a</b> (not available) are kept distinct and are never coerced into a divergence number.
    </p>`;
}

function wireDrawer(m) {
  // timing chart
  drawTimingChart(m, 'linear');
  document.querySelectorAll('.scale-btn').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.scale-btn').forEach(x => x.setAttribute('aria-pressed', 'false'));
      b.setAttribute('aria-pressed', 'true');
      drawTimingChart(m, b.dataset.scale);
    });
  });
  // parity table
  const defaultSize = m.sizes.length ? m.sizes[m.sizes.length - 1].key : (m.rows[0] ? (m.rows[0].n + 'x' + m.rows[0].p + '@' + m.rows[0].threads) : '');
  drawParityTable(m, defaultSize);
  document.querySelectorAll('.size-pick button').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.size-pick button').forEach(x => x.setAttribute('aria-pressed', 'false'));
      b.setAttribute('aria-pressed', 'true');
      drawParityTable(m, b.dataset.size);
    });
  });
}

/* ==========================================================================
   Top-level chrome: KPIs, provenance, nav stats
   ========================================================================== */
function renderChrome() {
  const s = DATA.stats || {};
  document.getElementById('navAlgos').textContent = (s.algos ?? METHODS.length).toLocaleString();
  document.getElementById('navCells').textContent = (s.cells ?? 0).toLocaleString();
  document.getElementById('navGenerated').textContent = (DATA.generated_at || '').split(' ')[0] || '—';

  // global verdict tallies across all cells
  let exact = 0, cross = 0, diverg = 0, nr = 0, na = 0, err = 0, timed = 0;
  for (const r of DATA.rows) {
    for (const cid of Object.keys(r.cells)) {
      const c = r.cells[cid]; if (!c) continue;
      switch (c.parity) {
        case 'exact': exact++; break;
        case 'cross_check': cross++; break;
        case 'divergent': case 'drift': diverg++; break;
        case 'not_run': nr++; break;
        case 'not_available': na++; break;
        case 'error': err++; break;
      }
      if (c.ms != null) timed++;
    }
  }
  const real = exact + cross + diverg + err;
  const kpis = [
    { v: METHODS.length, l: 'methods', sub: `${DATA.algo_groups.length} categories`, accent: '#0d9488' },
    { v: `${fmtPct(exact, real)}%`, l: 'cells exact', sub: `${exact.toLocaleString()} of ${real.toLocaleString()} comparable`, accent: '#0f766e' },
    { v: diverg, l: 'divergent', sub: 'flagged for attention', accent: diverg ? '#b3261e' : '#10b981' },
    { v: cross, l: 'cross-checks', sub: 'informational, not failures', accent: '#b45309' },
    { v: nr.toLocaleString(), l: 'not run', sub: 'kept honest, never faked', accent: '#7c7f86' },
    { v: timed.toLocaleString(), l: 'timed cells', sub: `${COLS.length} implementations`, accent: '#4f46e5' },
  ];
  document.getElementById('kpis').innerHTML = kpis.map(k =>
    `<div class="kpi" style="--accent:${k.accent}"><div class="v" style="color:${k.accent}">${k.v}</div><div class="l">${k.l}</div><div class="sub">${k.sub}</div></div>`
  ).join('');

  // provenance strip
  const h = DATA.host || {};
  document.getElementById('provenance').innerHTML = [
    `<span class="prov-chip"><span class="dot"></span><b>${escapeHtml(h.cpu_model || 'host')}</b></span>`,
    `<span class="prov-chip"><b>${escapeHtml(String(h.cpu_count_physical || '?'))}</b>&nbsp;phys cores · <b>${escapeHtml(String(h.cpu_count_logical || '?'))}</b>&nbsp;threads</span>`,
    `<span class="prov-chip"><b>${escapeHtml(String(h.ram_gb || '?'))}</b>&nbsp;GB RAM</span>`,
    `<span class="prov-chip">${escapeHtml(h.os || '')}</span>`,
    `<span class="prov-chip">generated <b>${escapeHtml(DATA.generated_at || '')}</b></span>`,
  ].join('');

  // category dropdown
  const catSel = document.getElementById('filterCategory');
  catSel.innerHTML = `<option value="">all categories (${DATA.algo_groups.length})</option>` +
    DATA.algo_groups.map(g => {
      const n = METHODS.filter(m => m.group === g.key).length;
      return n ? `<option value="${escapeAttr(g.key)}">${escapeHtml(g.label)} (${n})</option>` : '';
    }).join('');

  // origin dropdown — derived from the data so options always match real values
  const originCounts = {};
  for (const m of METHODS) originCounts[m.origin] = (originCounts[m.origin] || 0) + 1;
  const originKeys = Object.keys(originCounts).sort((a, b) => originCounts[b] - originCounts[a]);
  const originSel = document.getElementById('filterOrigin');
  originSel.innerHTML = `<option value="">all origins (${METHODS.length})</option>` +
    originKeys.map(o =>
      `<option value="${escapeAttr(o)}">${escapeHtml(originLabel(o))} (${originCounts[o]})</option>`
    ).join('');
}

/* ==========================================================================
   Events
   ========================================================================== */
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

function wireControls() {
  const search = document.getElementById('search');
  search.addEventListener('input', debounce(() => { state.query = search.value.trim(); render(); }, 90));

  document.getElementById('filterCategory').addEventListener('change', e => { state.category = e.target.value; render(); });
  document.getElementById('filterOrigin').addEventListener('change', e => { state.origin = e.target.value; render(); });
  document.getElementById('filterParity').addEventListener('change', e => { state.parity = e.target.value; render(); });

  document.querySelectorAll('.seg [data-sort]').forEach(b => {
    b.addEventListener('click', () => {
      document.querySelectorAll('.seg [data-sort]').forEach(x => x.setAttribute('aria-pressed', 'false'));
      b.setAttribute('aria-pressed', 'true');
      state.sort = b.dataset.sort;
      render();
    });
  });

  document.getElementById('drawerClose').addEventListener('click', closeDrawer);
  overlay.addEventListener('click', closeDrawer);

  // keyboard
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && drawer.classList.contains('open')) { closeDrawer(); return; }
    // focus trap inside drawer
    if (e.key === 'Tab' && drawer.classList.contains('open')) {
      const focusables = drawer.querySelectorAll('button, [href], input, select, [tabindex]:not([tabindex="-1"])');
      if (!focusables.length) return;
      const first = focusables[0], last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      return;
    }
    // "/" focuses search (when not already typing)
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'SELECT' && !drawer.classList.contains('open')) {
      e.preventDefault();
      search.focus();
    }
  });

  // deep-link: open drawer if hash names a method
  const hash = decodeURIComponent(location.hash.replace(/^#m\//, '').replace(/^#/, ''));
  if (hash && METHODS.some(m => m.algo === hash)) {
    // defer so chrome/render done
    setTimeout(() => openDrawer(hash), 30);
  }
}

/* ==========================================================================
   Utilities
   ========================================================================== */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function escapeAttr(s) { return escapeHtml(s); }

/* ==========================================================================
   Boot
   ========================================================================== */
renderChrome();
render();
wireControls();
