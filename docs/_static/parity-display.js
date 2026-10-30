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
    const columns = Object.fromEntries(data.columns.map(c => [c.id, c]));
    const scores = Object.fromEntries(Object.entries(data.method_scores || {}).map(([id, s]) =>
      [id, { ...s, reference: {}, binding: {} }]));
    const rows = data.rows.map(row => {
      const cells = Object.fromEntries(Object.entries(row.cells).map(([id, cell]) =>
        [id, displayCell(cell, columns[id])]));
      for (const [id, cell] of Object.entries(cells)) {
        if (!cell || !columns[id] || id === '__reference' || id === 'pls4all.registry') continue;
        const basis = usesReferenceGate(columns[id]) ? 'reference' : 'binding';
        const status = cell[`${basis}_parity`];
        const score = scores[row.algo];
        if (score && status) score[basis][status] = (score[basis][status] || 0) + 1;
      }
      return { ...row, cells };
    });
    return { ...data, rows, method_scores: scores };
  }

  return { MATCH_LIMIT, prepare, usesReferenceGate };
})();
