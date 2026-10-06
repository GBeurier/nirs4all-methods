# `ragged_summary` — n4m.utilities.ragged_summary

_Namespace_: **`n4m.utilities`** · _Fully-qualified_: `n4m.utilities.ragged_summary` · _Catalog id_: `utilities.ragged_summary`

## API surface

**C ABI:** no standalone exported symbol is declared for this method.

**Python:** catalog binding is not currently an AST-verified public `n4m` re-export. See the implementation source below.

**R:** no current source-verified entry point was found for this catalog method.

**MATLAB / Octave:** no current source-verified entry point was found for this catalog method.

## Explanations

### Bibliographic source

Descriptive population statistics and explicit temporal support. This operator is specified by its native sequence contract rather than a separate learned model. NumPy documents the same population-standard-deviation convention (ddof=0): https://numpy.org/doc/stable/reference/generated/numpy.std.html.

### Mathematical principle

For each observed sequence and channel, emit the arithmetic mean, population standard deviation (ddof=0), minimum and maximum. Append the sequence length, last-minus-first physical time (or index span when times are absent), and presence indicator one. Explicitly absent sequences receive zeros and indicator zero only under the opt-in zero_with_indicator policy; the default reject policy refuses absence.

### Appropriate uses

Creating a fixed-width tabular view of variable-duration, multichannel records before fold-scoped predictive fitting, while retaining source presence evidence.

### Limits and validation

Summaries discard detailed temporal ordering and cannot replace a sequence model. No resampling, padding or learned imputation occurs. Offsets must be integral and bound the packed rows; present sequences must be nonempty, absent sequences empty, and optional finite times strictly increasing per sequence. The shared procedure input contract requires at least one observed packed row. Nonfinite output is refused.

### Implementation

The native utilities.ragged_summary procedure uses the existing n4m_procedure_run ABI. cpp/src/core/estimator/procedures.cpp accumulates means and squared deviations in extended precision and checks the resulting double features. Python/R/JS/Rust bindings marshal typed parameters and matrices without recomputing statistics. Independent NumPy population reductions and malformed-boundary cases qualify it.

### Sources and provenance

https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/include/n4m/estimator.h; https://github.com/GBeurier/nirs4all-methods/blob/main/cpp/src/core/estimator/procedures.cpp; https://github.com/GBeurier/nirs4all-methods/blob/main/catalog/methods/utilities.ragged_summary.yaml; https://github.com/GBeurier/nirs4all-methods/blob/main/bindings/python/tests/test_ragged_summary.py

## Catalog note

Packed ragged series summary: per-channel mean/population std/min/max, length, duration and presence. Default refuses absence; explicit zero_with_indicator emits zeros for absent sequences. No resampling or learned imputation.


_See also_: [methods index](index.md).