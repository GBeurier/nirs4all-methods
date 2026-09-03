# N4MM hostile-input harness

`n4m_fuzz_n4mm` exercises the existing public C ABI only:

1. header inspection with `n4m_serialization_inspect`;
2. complete allocation-free inspection with
   `n4m_serialization_inspect_model_v1` and
   `n4m_serialization_inspect_pipeline_v1`;
3. `n4m_model_import_from_buffer` only after the authoritative complete
   inspector accepts the payload.

Inputs larger than 1 MiB are refused before any ABI call. The driver neither
copies nor decodes the input and does not repair its checksum, so it cannot
become a second N4MM parser. `n4m_fuzz_n4mm_smoke` feeds a finite set of hostile
prefixes through the same entry point and is available on non-Clang developer
machines as a build-contract check.

## Build contract

The declared sanitizer configuration uses the existing CMake options and the
repository's Clang 16 CI toolchain contract:

```console
cmake --preset ci-fuzz-n4mm
cmake --build --preset ci-fuzz-n4mm --target n4m_fuzz_n4mm
```

That target links libFuzzer; ASAN and UBSAN instrument the Methods library and
the harness. This commit does not add a corpus, run a campaign, or schedule the
target in CI. SEC-001 therefore remains open pending sustained fuzzing, hostile
corpus coverage across the other release surfaces, finding triage, and the
complete security qualification matrix.
