# Cross-Binding Benchmarks

This directory contains the executable benchmark matrix for chemometrics4all.
It follows the same broad contract as pls4all:

* one CSV row per `(algorithm, backend, n, p, threads)` cell;
* `median_ms` is wall-clock median over timed runs after an unmeasured warmup;
  `warmup_runs`, `timed_runs`, and `samples_ms` expose the exact protocol and
  raw per-run samples used for each cell;
* internal rows cover direct libc4a C ABI (`cpp`), Python sklearn-style
  bindings (`python`), and installed public R bindings (`r`);
* external rows cover the local git checkout of `nirs4all`, NumPy, SciPy
  (`scipy.signal` and `scipy.ndimage` where appropriate), scikit-learn,
  pybaselines, R base, and R stats comparators where a credible reference
  exists;
* `binding_parity_ok` records whether the method is covered by the binding
  parity suite and whether the binding output matched the direct ABI output;
* `reference_parity_ok` compares native C++ and comparator rows against the
  stored output snapshot from the canonical external reference method.

Run the default dashboard matrix from the repository root:

```bash
CHEMOMETRICS4ALL_LIB_PATH=$PWD/build/dev-release/cpp/src/libc4a.so \
PYTHONPATH=$PWD/bindings/python/src \
python benchmarks/cross_binding/orchestrator.py \
  --repeat 5 --size-preset small --threads 1 \
  --write-reference-snapshots
python docs/_extras/build_landing.py \
  --results benchmarks/cross_binding/results \
  --out docs/_static/bench-data.json
```

`--size-preset small` uses the pls4all NIRS smoke block: `100x50`,
`100x500`, and `100x2500`. `--size-preset pls4all` expands to the full
pls4all matrix: `100x50`, `100x500`, `100x2500`, `500x50`, `500x500`,
`500x2500`, `2500x50`, `2500x500`, `2500x2500`, `10000x50`, and
`10000x500`.

The current generated matrix covers 48 methods, 3 dataset sizes, 1 thread
setting, and dashboard columns for `cpp`, `python`, `r`, `nirs4all`, NumPy,
SciPy, scikit-learn, pybaselines, PyWavelets, R base, and R stats. Every
registered method has direct C ABI, Python binding, and R binding timing.
Every dashboard method now also has at least one external reference row.
External rows are always surfaced with a parity verdict. When a comparator has
a loose contract (different boundary convention, output shape, RNG contract, or
dimensionality convention), it is still compared against the canonical snapshot
and will render as divergent/error if the outputs do not match.
`nirs4all.WaveletFeatures` is timed with `n_coeffs_per_level=0`, histogram
entropy, and the symmetric boundary contract so its output feature count and
contract match the chemometrics4all dashboard surface.

Canonical reference rows are marked with `reference_role=canonical` and render
with a reference-method icon, not a pass/fail gate icon. When a canonical
external contract is credible, the runner stores its output under
`benchmarks/reference_snapshots/cross_binding/` with the external package
version or git commit in metadata. Reference gates compare native C++ rows and
non-canonical external comparator rows against that stored snapshot, not against
freshly recomputed reference output.

Timing follows the pls4all cross-binding convention: each backend performs
`max(1, min(3, repeat))` unmeasured warmup calls, then `repeat` measured calls.
The dashboard reports the median per call, not process startup time. For
sub-ms calls, the runner uses adaptive per-sample batching and records
`batch_loops` in the CSV; `samples_ms` remains a per-call value after division
by that batch count. This is especially important for R, whose wall clock is
too coarse for single-call measurements on tiny transforms.
For Python libraries that use BLAS/OpenMP under the hood, the runner applies
`threadpoolctl` in addition to the usual thread environment variables so
`--threads 1` also constrains already-imported NumPy/SciPy/scikit-learn/nirs4all
threadpools.
Thread sweeps reuse the same synthetic dataset for a given size and seed; the
seed does not depend on the thread count.

The `ref.nirs4all` column is loaded from `/home/delete/nirs4all/nirs4all` when
that checkout exists next to this repository. The CSV captures the exact git
revision and dirty marker in `lib_build`, for example `nirs4all@<sha>+dirty`.

## EPO

The `epo` C++ timing row uses `c4a_pp_epo_transform_with_d(...)`, so it is a
real binding-parity baseline for Python `EPO.fit_transform(X, d)`.
