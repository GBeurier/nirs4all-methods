# Ordered search-space contract

`OrderedSearchSpaceSpec` is the portable, fingerprinted description of a tuning
space shared by DAG-ML hosts and the native n4m optimizer. It is a vector of
axes: array order, categorical-choice order, activation and the structured
`ParameterPatch` target are all semantic.

The normative JSON shape is
[`docs/contracts/ordered_search_space.v1.schema.json`](../contracts/ordered_search_space.v1.schema.json).
The first golden contract is
[`parity/hpo/contracts/ordered_search_space.v1.json`](../../parity/hpo/contracts/ordered_search_space.v1.json).

## Ownership boundary

- A binding or DAG host owns the mapping from an axis to its structured patch.
- libn4m receives the ordered axis definitions and returns trial values in that
  same order.
- DAG-ML evaluates the patched pipeline and reports the score through ask/tell.
- DAG-ML does not link to libn4m, and n4m does not execute a host DAG.

## Canonical values

Raw JSON numbers are excluded from fingerprinted parameter domains:

- signed integers use canonical decimal `{"i64":"-12"}`;
- floats use their IEEE-754 binary64 bits as 16 lowercase hexadecimal digits,
  for example `1.0` is `{"f64_bits":"3ff0000000000000"}`;
- `-0.0` is normalized to positive zero;
- NaN and infinities are invalid;
- strings must be non-empty, NFC-normalized UTF-8 (the empty label is reserved
  by the current C ABI as the no-label sentinel);
- booleans remain typed booleans.

This prevents host number parsers and locale formatting from changing a study
fingerprint. V1 restricts every integer domain bound, step, tuple bound, and
integer choice to `[-2^53, 2^53]`, because the current native representation
must marshal it through binary64 without silent rounding. Choice domains are
homogeneous: one categorical or ordinal axis cannot mix scalar kinds. Integer
steps are positive, log-integer steps are exactly one, and log-float domains
are continuous (`step = 0`).

## Activation and constraints

V1 activation is `always`, `when`, or `when_not`, with a parent choice index.
Parents must be categorical or ordinal, all parents must exist, and the
activation graph must be acyclic. Inactive axes use `sample_then_omit`, matching
the current native sampler trajectory while preventing inactive values from
becoming `ParameterPatch` entries.

Hard constraints are `mutex_group`, `requires`, and `exclude`. A constraint
reference can address axis presence or one categorical/ordinal choice index.
Reference identity is the pair `(axis, choice_index-or-bare)`: two distinct
choices of one axis are distinct atoms, while an exact repeated pair is invalid.
Support still depends on the selected sampler: unsupported sampler × constraint
combinations fail before the first trial; they never degrade to best-effort
sampling.

## Current C ABI projection

The current builder API is a projection of this host contract, not an alternate
wire format:

- axis array order and names become `n4m_search_space_add_*` call order;
- numeric and choice domains map to the matching builder, subject to the exact
  binary64 restrictions above;
- `when` and `when_not` become `CONDITION_IN` and `CONDITION_NOT_IN` records;
- a choice index is resolved to the native choice label before adding a
  condition or hard constraint: the label is the string itself, canonical
  decimal integer, `true`/`false`, or the locale-independent binary64
  round-trip text used by the C++ builder;
- `target` and `inactive_policy` remain host coordination metadata and are not
  discarded from the full contract fingerprint.

The projected native `SearchSpace` therefore cannot reproduce the full
fingerprint by itself. The normative fingerprint belongs to this complete
ordered contract; DAG-ML stores it as an opaque study input and must not
recompute it with DAG-ML's unrelated internal JSON fingerprint helper. A future
native JSON entry point may validate and fingerprint the complete contract, but
must match the published vectors byte for byte.

## Fingerprint

The fingerprint is `sha256:` followed by SHA-256 of the RFC 8785/JCS canonical
UTF-8 representation. Object properties are ordered by UTF-16 code units;
arrays are never reordered. Because all domain numbers are tagged strings, the
only numeric JSON tokens in V1 are non-negative structural integers such as
`schema_version`, tuple length, and choice indices. This restricted value domain
avoids host-dependent floating-point JSON rendering. Structural integers stay
within the interoperable JSON safe-integer range; tuple length is additionally
bounded by the current C ABI's signed 32-bit argument.

Display names, documentation and UI metadata are deliberately outside the
fingerprinted object. Changing axis order, choice order, target, activation or
constraint always changes the fingerprint.

The test-only oracle in `parity/hpo/ordered_space_contract.py` validates the V1
invariants and pins the golden fingerprint. Production bindings must use the
eventual native validator/fingerprint API; they must not fork this Python oracle.
