# Codex review · Phase 0 · PR 4 · status / version / context / config

- **Date**: 2026-05-14
- **Codex CLI**: v0.125.0
- **Model**: gpt-5.5 (reasoning_effort=medium)
- **Subject**: PR 4 — `cpp/src/core/{status,version,context,config}.{hpp,cpp}` and `cpp/src/c_api/c_api_{version,context,config}.cpp`. 60 exported `n4m_*` symbols.
- **Outcome**: 0 critical + 5 important + 3 minor. All addressed.

## Codex findings

### Verdict (round 1): changes requested.

### Important
1. Most exported wrappers had no try/catch boundary guard — only `create/destroy/clone` paths were protected. → fixed by wrapping every `N4M_API` body in a try/catch (via inline blocks in c_api_version.cpp and c_api_context.cpp, and the `N4M_CFG_TRY_BEGIN/END` macro in c_api_config.cpp). The catch-all returns the appropriate fallback (`N4M_ERR_INTERNAL` for status, `nullptr` / `""` / `0` for other types).
2. `n4m_context_set_num_threads` silently accepted any `int32_t`. → semantics now documented in `p4a.h`: `n_threads <= 0` means "library picks the default"; positive values pin the count; no upper cap.
3. Getters returned `N4M_ERR_NULL_POINTER` for null out-params without writing `last_error` when `ctx` was non-NULL. → fixed in `c_api_context.cpp`: when `ctx` is non-NULL but the out-pointer is NULL, the getter now writes a `"null out pointer in n4m_context_get_*"` message to the context's error buffer.
4. Undocumented hard caps `n_components <= 10000` and `max_iter <= 10000000`. → caps removed; setters now only reject non-positive values. The Phase 1 `n4m_model_fit` will surface a fit-time `N4M_ERR_INVALID_ARGUMENT` if `n_components > rank(X)`. Header documentation added accordingly.
5. Exported definitions used `std::size_t` / `std::uint*_t` / `std::int32_t`. → switched to C headers (`<stddef.h>`, `<stdint.h>`) and unqualified types (`size_t`, `uint32_t`, `int32_t`) to keep the public ABI surface free of C++-flavoured spellings.

### Minor
6. `n4m_config_set_max_iter` parameter named `n` instead of `max_iter`. → renamed.
7. ABI version values hardcoded in `version.hpp` (`1u/0u/0u`) instead of derived from `n4m_version.h`. → `version.hpp` now `static_cast`s the `N4M_ABI_VERSION_*` macros from `pls4all/n4m_version.h`, eliminating the drift risk.
8. `n4m_context_get_user_data(NULL)` silently returns `NULL`, indistinguishable from a stored `NULL`. → documented explicitly in `p4a.h`; if a binding needs to distinguish, store a sentinel.

### Confirmed correct
- 60 exported `n4m_*` symbols on Linux x86_64 (gcc-11); CMake includes the 7 new TUs.
- `extern "C"` linkage on every wrapper.
- `n4m_check_abi_compatibility` truth table.
- 4096-byte error buffer; safe truncation; NUL-terminated.
- `n4m_context_destroy` / `n4m_config_destroy` NULL-safe.
- `n4m_context_set_backend` returns `BACKEND_UNAVAILABLE` + writes message; state unchanged on rejection.
- `n4m_config_clone` deep-copies all 16 members.
- PR 3 static-asserts already pin every public enum to 4 bytes.

## Local smoke (after fixes)
- `cmake --build --preset dev-release` → 13 steps, no warnings.
- 60 `n4m_*` symbols exported; no forbidden runtime deps.
- Smoke test passes: version, status_to_string, dtype_size, backend_is_available,
  context lifecycle (seed/backend/threads/user_data), last_error on
  backend-unavailable, NULL ctx returns "", config lifecycle with
  validation (NaN tol rejected, negative n_components rejected, clone
  independent of source).
