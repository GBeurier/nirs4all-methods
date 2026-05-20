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

- external package references such as SciPy, scikit-learn, PyWavelets, and pybaselines;
- project comparators such as pinned nirs4all modules;
- literature sources for methods that have no credible external executable comparator.

Internal fixtures under `parity/python_generator/src/n4m_parity_*_ref/` are
development oracles for deterministic C/binding parity. They are not external
benchmark references and must not appear as `ref.*` benchmark backends.

The lockfile is not a benchmark result. It is a source-of-truth inventory used
to prevent untraceable dashboard claims.

## Dashboard status

The landing dashboard currently renders a placeholder row only. It must stay
that way until a real nirs4all-methods benchmark runner emits a result matrix
from this registry.
