# Benchmark methodology

This page defines the policy for chemometrics4all benchmarks. The current
runner is a bootstrap matrix for Python tier-2 bindings; the same CSV contract
is intended for C++/R/MATLAB and external references.

## Policy

1. A method must be listed in `benchmarks/benchmark_registry.json`.
2. Every reference used by that method must be locked in
   `benchmarks/truth_sources.lock.json`.
3. `python benchmarks/check_truth_sources.py` must pass before results are
   rendered.
4. Result metadata must capture package versions, project commits, host details,
   thread settings, dataset dimensions, and seeds.
5. No dashboard or documentation page may report timing, speedup, parity, or
   "exact" counts without an actual generated result file.

## Reference policy

Prefer references in this order:

1. the local git checkout of `nirs4all` for chemometrics-specific homemade
   conventions such as splitters, outlier filters, EPO/OSC-style operators,
   resampling, and wavelet feature conventions;
2. pinned package references for numerical primitives or established algorithms
   (`scipy.signal`, `scipy.ndimage`, scikit-learn, NumPy, pybaselines, R base,
   R stats, and PyWavelets).

External references are allowed only when they match the operator being tested:
same parameters, same boundary convention, same output shape, same deterministic
RNG contract when exact indices are compared, and equivalent component
conventions for PCA/SVD-style methods.
PLS model references from sklearn, `pls`, `mixOmics`, `ropls`, or MATLAB
toolboxes are not valid references for this repository's non-model operators.

The cross-binding dashboard records the resolved external source in each CSV
row. For `nirs4all`, the runner inserts `/home/delete/nirs4all/nirs4all` ahead
of installed packages and stores the git short SHA plus a dirty marker in
`lib_build`.

Each method has one canonical external reference row
(`reference_role=canonical`). That row identifies the reference method and is
not itself rendered as a pass/fail gate. For credible same-contract references,
the benchmark runner stores the canonical external output in
`benchmarks/reference_snapshots/cross_binding/` together with package version or
git commit metadata. Native C++ rows carry the reference gate against that
stored snapshot; non-canonical external rows carry comparator parity against the
same snapshot; Python/R/MATLAB binding rows carry binding parity against the
native C++ route.

Rows that are useful for performance but not identical in contract are still
gated against the stored canonical snapshot. They therefore render as
`exact`/`drift`/`divergent`/`error`, not as timing-only entries. Examples
include wavelet transforms with different coefficient ordering/statistical
contracts, Savitzky-Golay calls with different edge modes, and randomized
splitters that do not share the same RNG contract. For
`nirs4all.WaveletFeatures`, the runner sets `n_coeffs_per_level=0` and selects
c4a's `histogram` entropy with `symmetric` boundary mode so the comparator row
uses the same contract.

Some external libraries expose a different computational surface even when the
name is close. `pybaselines` baseline references are applied row-by-row because
the public API is one-dimensional; `pybaselines.iasls` is now the c4a IAsLS
contract, while `pybaselines.beads(full)` remains a full external algorithm and
chemometrics4all still ships a compact NIRS-oriented BEADS variant.
`pybaselines.imodpoly` is called with
`mask_initial_peaks=False` to match chemometrics4all's disabled initial peak
masking before the row is gated like the other references.

## Parity tolerance

The initial binding-parity gate uses `1e-6`, matching the package-level
binding parity tests. External-reference tolerances must still be declared per
method before external rows are promoted into the dashboard.

PCA/SVD score comparisons are sign-invariant per component because eigenvector
and singular-vector signs are mathematically arbitrary. This keeps the gate on
the subspace/projection contract instead of flagging meaningless sign flips.

## Timing protocol

The cross-binding runner follows the pls4all timing convention: each callable
gets `max(1, min(3, repeat))` warmup calls that are not measured, then `repeat`
timed calls. The reported `median_ms` is the median per-call wall time across
the timed samples. The default is `--repeat 5`.

Very short calls are measured in adaptive batches: each timed sample may run
the callable several times and divide by `batch_loops`. This keeps sub-ms C++,
Python, and R cells from being dominated by clock quantization or a single R
runtime event. The CSV stores `warmup_runs`, `timed_runs`, `batch_loops`, and
semicolon-separated per-call `samples_ms` so suspicious cells can be audited
without rerunning the matrix.

Python-side BLAS/OpenMP threadpools are constrained with `threadpoolctl` in
addition to `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`, because
NumPy may already be imported before a benchmark thread setting is selected.
Datasets are keyed by size and seed only, not by thread count, so thread sweeps
measure the same input matrix.

Use `--write-reference-snapshots` only when deliberately creating or refreshing
the stored external-reference outputs. Normal gate runs load the stored
snapshots and compare against them.

## Result status

The committed dashboard matrix is generated by
`benchmarks/cross_binding/orchestrator.py`. The small default preset follows
pls4all's NIRS smoke sizes (`100x50`, `100x500`, `100x2500`); the full preset
uses the pls4all production size matrix. If `full_matrix.csv` is absent, the
dashboard builder falls back to an explicit placeholder.
