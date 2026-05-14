# Codex review · Phase 0 · PR 3 · p4a-header

- **Date**: 2026-05-14
- **Codex CLI**: v0.125.0
- **Model**: gpt-5.5 (reasoning_effort=medium)
- **Subject**: PR 3 — `cpp/include/pls4all/p4a.h` (96 P4A_API declarations) and `cpp/include/pls4all/p4a_version.h`.
- **Outcome**: 0 critical + 4 important + 5 minor. All addressed in the same PR.

## Round 1 findings

### Verdict: doc/ABI-clarity fixes required before merge.

### Important
1. Per-function status contracts: only broad group comments existed; bindings need to know which `P4A_ERR_*` each function can return. → fixed by adding a uniform "Status-code contracts" preamble in the header doc-block that documents the rules for `_create`, `_destroy`, `_clone`, `_set_*`, `_get_*`, context-aware functions, and matrix-view-consuming functions. Per-function overrides only where needed.
2. Stride contract was internally ambiguous. The header doc said "positive only" but `_validate` allowed zero stride when the corresponding dim was ≤ 1. → fixed by spelling out the rules explicitly in the §4 prelude.
3. Public ABI relies on enum width but no static assertion guards against `-fshort-enums`. → fixed by adding a `P4A_STATIC_ASSERT(sizeof(p4a_X_t) == 4, ...)` block at the end of the header for every public enum, plus a 48-byte size assertion on `p4a_matrix_view_t`. Verified locally: `-fshort-enums` now fails fast with a clear message.
4. `params` ownership for `_operator_bank_add` and `_pipeline_add_operator` was unspecified. → fixed by documenting "the caller's buffer may be released immediately after the call".

### Minor
5. `p4a_pipeline_size` declared in header but absent from roadmap §4.6. → roadmap updated to match.
6. Model section wording "every function in this section returns NOT_IMPLEMENTED" but `p4a_model_destroy` is void. → wording fixed: "every status-returning function returns NOT_IMPLEMENTED; `p4a_model_destroy` is a void no-op".
7. `p4a_context_last_error(NULL)` / `clear_error(NULL)` behavior undocumented. → documented: `last_error` returns `""`, `clear_error` is no-op; both never crash.
8. `P4A_SERIALIZATION_MAGIC` as a C string literal is 5 bytes (incl NUL) — the format spec says 4-byte magic. → documented "compare or memcpy exactly 4 bytes; the trailing NUL is not part of the format".
9. Header comment said "LP64 (Linux/macOS/Windows-64)" — Windows-64 is LLP64. → fixed to "LP64 (Linux, macOS) and LLP64 (Windows-64)".

## Confirmed correct (no action)
- 96 `P4A_API` declarations present; no missing roadmap symbol found.
- Enum names and values match roadmap for status/backend/dtype/algorithm/solver/deflation/operator_kind/gating_mode/model_array.
- `extern "C"` wraps the full declaration block.
- No STL/Eigen/C++ leakage; comments-only mention of no-exception rule.
- Serialization constants and `p4a_serialization_inspect` present.
- C11 and C++17 syntax checks pass locally.

## Local validation after fixes
- `gcc -Wall -Wextra -Wpedantic -std=c11 -c` (with the canonical p4a_export.h): OK.
- `g++ -Wall -Wextra -Wpedantic -std=c++17 -c`: OK.
- `gcc -Wall -Wextra -Wpedantic -std=c11 -fshort-enums -c`: fails fast at the static assert (`"p4a_status_t must be 4 bytes"`), as designed.
- Build (`cmake --build --preset dev-release`): `ninja: no work to do` — header-only change, no source rebuild needed.
- Header line counts: p4a.h 641 lines, p4a_version.h 43 lines.
