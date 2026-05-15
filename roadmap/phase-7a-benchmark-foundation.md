# Phase 7a - Benchmark Foundation

Status: shipped locally as `phase-7a-benchmark-foundation`.

Goal: stand up a deterministic, locally-runnable benchmark suite over the
public C ABI shipped in Phase 6f. First runner: AOM-PLS global selection
against the `nirs4all/bench/AOM_v0/aompls` oracle.

Delivered:

- `benchmarks/` directory with a small Python driver:
  - `benchmarks/run.py` — driver script; supports `--check` (gates the
    committed accuracy CSV) and `--repeats N` (timing repeats).
  - `benchmarks/runners/_harness.py` — `AccuracyRecord`, `TimingRecord`,
    `median_time_ms`, summary-table formatter, host-info collector.
  - `benchmarks/runners/aom_global.py` — drives `p4a_aom_global_select`
    through the Python ctypes wrapper, mirrors the same dataset and fold
    indices into the bench oracle, and reports both numerical and timing
    deltas.
- Four committed test cases (`9x6/12x8/16x10` smooth-monotonic cases plus
  a `14x9` detrend-favoured case that selects operator index 1) covering
  the identity and detrend operators in the AOM bank.
- Two output channels:
  - `benchmarks/results/aom_global_accuracy.csv` — gated by `--check`;
    numerical deltas, selected operator / component matches, accuracy
    pass/fail. All four current cases hit 0.0 absolute RMSE delta against
    the oracle.
  - `benchmarks/results/aom_global_timing.csv` and
    `benchmarks/results/aom_global_summary.md` — informational only, NOT
    gated. Records median + min wall times for one full `fit + predict`
    call on both sides, plus host info.
- `benchmarks/README.md` documenting the run command, the gated vs
  informational split, the timing caveat (ctypes marshalling overhead is
  included), and the contract for adding a new runner.

Reviewer notes (post-codex):

- Accuracy record drops `pls4all_version`, `bench_version` and
  `coefficient_l2_abs_delta`. Version metadata now lives in the summary
  markdown only; AOM coefficient deltas land once the ABI exposes them.
- The validation plan is constructed once per case and passed identically
  to both sides via the explicit-fold splitter mechanism on each side.

Local gate (informational only — benchmarks are not part of the C++ build):

- `python benchmarks/run.py` produces 4/4 `accuracy_pass=True` rows.
- `python benchmarks/run.py --check` succeeds bit-for-bit.
- Timing CSV / summary markdown regenerated on every run.

Deferred:

- PLS regression benchmark (NIPALS / SIMPLS / SVD vs sklearn) — needs
  ctypes wrappers for `p4a_model_fit`, `p4a_model_predict_alloc`,
  `p4a_model_get_array`, `p4a_array_*` first. Belongs with the Phase 2
  Python NumPy zero-copy work.
- POP per-component benchmark — same harness, additional runner.
- Wider AOM operator bank (Savitzky-Golay, Norris-Williams, Whittaker,
  FCK) — needs the same operator-parameter table the parity generator
  uses; track alongside Phase 6g.
- Larger-scale benchmark variants (500x100+) — gated on NumPy zero-copy
  matrix views in the Python binding.
