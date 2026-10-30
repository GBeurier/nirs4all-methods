/* Shared presentation policy for the explorer and full matrix. Raw benchmark
 * verdicts remain in BENCH_DATA; all views consume the same display copy. */
'use strict';

window.N4MParity = (() => {
  const MATCH_LIMIT = 1e-8;
  const comparable = new Set(['exact', 'cross_check', 'divergent', 'drift']);

  function usesReferenceGate(column) {
    return !!column && (['external', 'reference'].includes(column.kind) ||
      column.id.startsWith('pls4all.cpp.'));
  }

  function displayCell(cell, column) {
    if (!cell) return cell;
    const copy = { ...cell };
    const metric = cell.divergence_metric || 'rmse';
    const small = metric === 'rmse' && Number.isFinite(cell.divergence) &&
      Math.abs(cell.divergence) < MATCH_LIMIT;
    if (!small || cell.ok === false || cell.parity === 'error') return copy;

    // A binding delta cannot establish reference parity (and vice versa).
    const basis = cell.divergence_basis === 'cross_check'
      ? (usesReferenceGate(column) ? 'reference' : 'binding') : cell.divergence_basis;
    if (!['reference', 'binding'].includes(basis)) return copy;
    const key = `${basis}_parity`;
    if (!comparable.has(cell[key])) return copy;
    copy[key] = 'exact';
    copy.display_match_basis = basis;
    copy.divergence_quality = 'exact';
    if (comparable.has(cell.parity)) copy.parity = 'exact';
    return copy;
  }

  function prepare(data) {
    if (!data) return data;
    const publicColumns = data.columns.filter(c => !['nirs4all', 'pls4all.registry'].includes(c.id))
      .map(c => c.id === '__reference' ? { ...c, label: 'Reference', short: 'reference' } : c);
    const columns = Object.fromEntries(publicColumns.map(c => [c.id, c]));
    const scores = Object.fromEntries(Object.entries(data.method_scores || {}).map(([id, s]) =>
      [id, { ...s, reference: {}, binding: {} }]));
    const measures = Object.fromEntries(Object.keys(scores).map(id =>
      [id, { timing: [], reference: [], binding: [] }]));
    const rows = data.rows.map(row => {
      const reference = row.cells.__reference;
      const sourceId = reference && reference.ref_of;
      const cells = Object.fromEntries(Object.entries(row.cells)
        .filter(([id]) => columns[id] && (id === '__reference' || id !== sourceId))
        .map(([id, cell]) => [id, displayCell(cell, columns[id])]));
      if (reference) {
        // Keep the source once, in the dedicated reference row/column.
        cells.__reference = { ...row.cells[sourceId], ...reference };
      }
      for (const [id, cell] of Object.entries(cells)) {
        if (!cell || !columns[id] || id === '__reference' || id === 'pls4all.registry') continue;
        const basis = usesReferenceGate(columns[id]) ? 'reference' : 'binding';
        const status = cell[`${basis}_parity`];
        const score = scores[row.algo];
        if (score && status) score[basis][status] = (score[basis][status] || 0) + 1;
        const values = measures[row.algo];
        if (values && cell.ok && Number.isFinite(cell.ms)) values.timing.push(cell.ms);
        if (values && Number.isFinite(cell.divergence)) values[basis].push(Math.abs(cell.divergence));
      }
      return { ...row, cells };
    });
    for (const [id, values] of Object.entries(measures)) {
      const stats = xs => {
        const sorted = xs.slice().sort((a, b) => a - b), n = sorted.length;
        return { max: n ? sorted[n - 1] : null, min: n ? sorted[0] : null, n,
          median: n ? (sorted[Math.floor((n - 1) / 2)] + sorted[Math.floor(n / 2)]) / 2 : null };
      };
      const time = stats(values.timing);
      scores[id].timing = { min_ms: time.min, median_ms: time.median, max_ms: time.max, n: time.n };
      scores[id].divergence = { reference: stats(values.reference), binding: stats(values.binding) };
    }
    return { ...data, columns: publicColumns, rows, method_scores: scores };
  }

  function languageIcon(language) {
    const icon = { Python: 'python', R: 'r', 'MATLAB/Octave': 'matlab' }[language];
    if (!icon) return '';
    return `<img class="language-icon" src="_static/languages/${icon}.svg" width="22" height="22" alt="${language}" title="${language}">`;
  }

  function implementationLabel(column, origin) {
    if (column.kind === 'pls4all' && origin !== 'pls4all') {
      return column.label.replace(/^pls4all/, 'nirs4all-methods');
    }
    return column.label;
  }

  return { MATCH_LIMIT, prepare, usesReferenceGate, languageIcon, implementationLabel };
})();
