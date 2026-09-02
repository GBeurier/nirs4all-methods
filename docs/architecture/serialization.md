# Fitted-model wire serialization

The C ABI implements two backward-readable fitted-model wire versions behind:

- `n4m_model_export_size`
- `n4m_model_export_to_buffer`
- `n4m_model_import_from_buffer`
- `n4m_serialization_inspect`
- `n4m_serialization_inspect_model_v1`

The format is little-endian and starts with:

```text
magic[4] = "N4MM"
u32 format_version = 1 or 2
u32 writer_abi_major
u32 writer_abi_minor
u32 writer_abi_patch
```

The shared payload stores model metadata, PLS centring/scaling statistics,
coefficients, latent matrices and optional training scores. A trailing FNV-1a
64-bit checksum covers every byte before the checksum field. Imports reject bad
magic, truncated payloads, impossible dimensions, length mismatches and checksum
failures with `N4M_ERR_CORRUPT_BUFFER`; unsupported format versions return
`N4M_ERR_VERSION_INCOMPATIBLE`.

Format 1 remains byte-for-byte unchanged and is still emitted for every model
without a pipeline. Format 2 is emitted only for a PLS-regression model fitted
with the exact `SNV(no params) -> Savitzky-Golay smooth(window, poly)` pipeline.
After the eleven format-1 vectors it appends, in order:

```text
u32 operator_count = 2
u32 first_kind = N4M_OP_SNV
u32 first_param_count = 0
u32 second_kind = N4M_OP_SAVGOL_SMOOTH
u32 second_param_count = 4
f64 window                 # odd integer, 3..501
f64 polynomial_degree      # integer, 0 <= degree < window
f64 derivative_order = 0
f64 delta = 1
u64 fnv1a64 checksum
```

This fixed block is deliberately not a general pipeline envelope. Fit rejects
different orders, operators and algorithms. Inspection/import reject altered
tags, counts or parameters even when the non-cryptographic checksum is
recomputed. The restored model owns the fitted stateless pipeline and reapplies
it before prediction and latent transformation.

The ABI fields are provenance, not a strict load gate in either version. The
current importer accepts a structurally valid payload written by a different
ABI version and records a compatibility warning on the context. Callers must
therefore not describe an ABI-major or newer-minor mismatch as a guaranteed
rejection.

`N4MM` is a raw fitted-model payload. It has no canonical filename extension
yet and is not the nirs4all `.n4a` pipeline bundle. The possible `.n4am`
envelope remains a future contract and is not part of either N4MM version.
The FNV-1a trailer detects corruption; it is not a content-address or a
training-data fingerprint.

`n4m_serialization_inspect_model_v1` is the authoritative, allocation-free
inspection gate for a complete fitted-model payload. It validates the header,
supported format, checksum, algorithm/solver/deflation recipe, dimensions,
flags, all eleven length-delimited array sections, the required format-2
pipeline block when present, and exact end-of-payload
before filling `n4m_serialized_model_info_v1_t`. The output is zero on every
failure. Its schema version is 1 and its capability bits are derived from the
validated N4MM bytes:

- native format-1 fitted models report `PREDICT | TRANSFORM`;
- the bounded format-2 pipeline model reports `PREDICT | TRANSFORM | PIPELINE`;
- `N4M_ALGO_IMPORTED_LINEAR_PREDICTOR` reports `PREDICT | AFFINE` and never
  reports `TRANSFORM`.

Unknown model recipes return `N4M_ERR_UNSUPPORTED`; future wire versions return
`N4M_ERR_VERSION_INCOMPATIBLE`; malformed or checksum-invalid payloads return
`N4M_ERR_CORRUPT_BUFFER`. External JSON, manifests and filename conventions are
not capability authorities.

## Optimizer checkpoint wire format

The optimizer uses a separate `N4MOPT` format behind
`n4m_optimizer_save` / `n4m_optimizer_load`. Version 1 is a little-endian,
self-contained study checkpoint:

```text
magic[8] = "N4MOPT\r\n"
u32 format_version = 1
u32 header_size = 32
u64 total_size                 # exact; includes zero padding + checksum
u64 payload_size               # excludes padding + checksum
payload[payload_size]
zero padding to an 8-byte total
u64 fnv1a64(header || payload || padding)
```

The payload starts with independently hashed, length-delimited canonical
encodings of the ordered `SearchSpace` and normalized
`n4m_optimizer_options_t`. It then stores base state (SplitMix64 state,
elapsed-time offsets, next ids/sequences, full trial lifecycle, intermediates,
structured errors and enqueue FIFO) and a sampler-tagged state block. Sampler
blocks contain only portable scalars/vectors: Sobol Gray-code state, GA
population, PSO particle/best state, separable CMA-ES distribution/population,
or GP-EI decoded proposals. LHS designs and ternary/TPE state are deterministically
derived from the encoded options, space, RNG and history. Pruner decisions are
likewise reconstructed from options and intermediate history.

The decoder is transactional and applies a 64 MiB envelope, per-string and
per-collection limits, and remaining-buffer checks before allocation. It rejects
unknown versions, non-zero padding, trailing/truncated data, invalid UTF-8,
non-canonical option/space fingerprints, invalid lifecycle event graphs and
sampler-state shape mismatches. It never reads or writes a pointer, `size_t`,
C++ object representation, vtable, clock epoch, or host endianness-dependent
scalar. The FNV checksum is corruption detection, not a security MAC.

The public save symbol predates byte dtypes. It therefore returns an owning
`N4M_DTYPE_I64` array whose underlying bytes are the aligned checkpoint; bindings
must treat the storage as bytes, not numeric word values. Python presents those
bytes directly through `Optimizer.save()` / `Optimizer.load()`.
