# Benchmark overview

The benchmark system is a scaffold, not a results matrix.

The first committed contract is the registry in
`benchmarks/benchmark_registry.json`. It lists a small set of non-model methods,
the intended operation type, basic dataset dimensions, and the truth sources
that future runners must use before publishing parity or timing claims.

## Registry shape

Each method entry records:

- `method_id`: stable method key used by future runners.
- `family`: preprocessing, baseline, splitter, diagnostic, or another
  non-model operator family.
- `operation`: transform, fit/transform, split, or diagnostic operation.
- `truth_sources`: references that must exist in
  `benchmarks/truth_sources.lock.json`.
- `metrics`: comparison checks expected for that method.
- `default_dataset`: deterministic starter dimensions and seed.

## Truth sources

The lockfile distinguishes:

- package references such as NumPy, SciPy, and pybaselines;
- project references such as nirs4all modules;
- local frozen references under the existing parity generator tree.

The lockfile is not a benchmark result. It is a source-of-truth inventory used
to prevent untraceable dashboard claims.

## Dashboard status

The landing dashboard currently renders a placeholder row only. It must stay
that way until a real chemometrics4all benchmark runner emits a result matrix
from this registry.
