# ABI — Stability Policy

The installed headers under `cpp/include/n4m/` and the exported-symbol
snapshots under `cpp/abi/` are the normative `libn4m` C ABI. The current ABI is
`2.5.0`; it is versioned independently from the Methods project and Rust crate.

## Compatibility rules

- A major ABI change may rename/remove symbols or change layouts and changes
  the SONAME. ABI 2 therefore uses `libn4m.so.2` and the `N4M_2` version node.
- A minor ABI change is additive. Existing symbols, enum values and structure
  prefixes keep their meaning and layout.
- A patch ABI change fixes behavior without changing the public surface.
- Callers zero-initialize public structures, set their `struct_size` when the
  type defines one, and call
  `n4m_check_abi_compatibility(header_major, header_minor)` before any other
  ABI operation.
- Handles and core-allocated arrays are released only by their matching
  `n4m_*_destroy` or `n4m_array_free` function. Memory is never freed across a
  different language/runtime allocator boundary.
- Every binding remains a translation layer over this ABI; numerical or
  parsing logic is not reimplemented in Python, R, MATLAB/Octave, JavaScript or
  Rust.

An intentional public change must update all platform symbol snapshots and the
[ABI changes log](changes_log.md) in the same change. Header/runtime skew,
missing symbols and unsupported serialized formats fail closed. The full
surface and current version are documented in the [ABI reference](reference.md).
