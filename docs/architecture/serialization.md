# Serialization

The C ABI implements the first binary fitted-model format behind:

- `c4a_model_export_size`
- `c4a_model_export_to_buffer`
- `c4a_model_import_from_buffer`
- `c4a_serialization_inspect`

The format is little-endian and starts with:

```text
magic[4] = "C4AM"
u32 format_version = 1
u32 writer_abi_major
u32 writer_abi_minor
u32 writer_abi_patch
```

The payload stores model metadata, preprocessing statistics, coefficients,
latent matrices and optional training scores. A trailing FNV-1a
64-bit checksum covers every byte before the checksum field. Imports reject bad
magic, truncated payloads, impossible dimensions, length mismatches and checksum
failures with `C4A_ERR_CORRUPT_BUFFER`; unsupported format versions return
`C4A_ERR_VERSION_INCOMPATIBLE`.
