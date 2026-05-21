# Serialization

The C ABI implements the first binary fitted-model format behind:

- `n4m_model_export_size`
- `n4m_model_export_to_buffer`
- `n4m_model_import_from_buffer`
- `n4m_serialization_inspect`

The format is little-endian and starts with:

```text
magic[4] = "N4MM"
u32 format_version = 1
u32 writer_abi_major
u32 writer_abi_minor
u32 writer_abi_patch
```

The payload stores model metadata, preprocessing statistics, coefficients,
latent matrices and optional training scores. A trailing FNV-1a
64-bit checksum covers every byte before the checksum field. Imports reject bad
magic, truncated payloads, impossible dimensions, length mismatches and checksum
failures with `N4M_ERR_CORRUPT_BUFFER`; unsupported format versions return
`N4M_ERR_VERSION_INCOMPATIBLE`.
