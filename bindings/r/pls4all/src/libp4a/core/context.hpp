// SPDX-License-Identifier: CECILL-2.1
//
// Internal Context — the C++ object pointed to by the opaque n4m_context_s
// handle. Owns the seed, backend choice, thread-count hint, the 4 KiB
// per-context error buffer, and a binding-side user-pointer.
//
// Threading model:
//   - A single Context instance is documented as single-thread-only at the C
//     ABI level (see p4a.h §6). Internally that means we DO NOT need locks
//     for read-modify-write on most members.
//   - The error buffer is fixed-capacity, written with a snprintf-style
//     helper that NUL-terminates safely. No allocations across the write
//     path → no exceptions from error reporting.

#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>

#include "n4m/n4m.h"

#if defined(__GNUC__) || defined(__clang__)
#  define N4M_PRINTF_LIKE(fmt_index, first_arg) __attribute__((format(printf, fmt_index, first_arg)))
#else
#  define N4M_PRINTF_LIKE(fmt_index, first_arg)
#endif

namespace n4m::core {

class Context {
  public:
    Context() noexcept;
    ~Context() noexcept = default;

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    void set_seed(std::uint64_t seed) noexcept { seed_ = seed; }
    [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }

    // Returns N4M_OK on success, N4M_ERR_INVALID_ARGUMENT for an
    // out-of-range backend value, or N4M_ERR_BACKEND_UNAVAILABLE if the
    // backend is not compiled into this build.
    n4m_status_t set_backend(n4m_backend_t backend) noexcept;
    [[nodiscard]] n4m_backend_t backend() const noexcept { return backend_; }

    // Negative or zero means "library default" (interpreted later by the
    // accelerated backends). Phase 0 just stores the value.
    void set_num_threads(std::int32_t n) noexcept { num_threads_ = n; }
    [[nodiscard]] std::int32_t num_threads() const noexcept { return num_threads_; }

    void set_user_data(void* user) noexcept { user_data_ = user; }
    [[nodiscard]] void* user_data() const noexcept { return user_data_; }

    // Error-buffer access. The pointer is stable for the Context's lifetime;
    // its CONTENT is invalidated on the next set_error / clear_error call.
    [[nodiscard]] const char* last_error() const noexcept { return error_buf_; }
    void clear_error() noexcept { error_buf_[0] = '\0'; }

    // Writes a NUL-terminated, safely-truncated message into the error
    // buffer. Always succeeds. Pre-existing content is overwritten.
    void set_error(const char* msg) noexcept;
    void set_errorf(const char* fmt, ...) noexcept N4M_PRINTF_LIKE(2, 3);

  private:
    std::uint64_t seed_;
    n4m_backend_t backend_;
    std::int32_t  num_threads_;
    void*         user_data_;
    char          error_buf_[N4M_ERROR_BUFFER_BYTES];
};

}  // namespace n4m::core

// Opaque struct alias. p4a.h declares `struct n4m_context_s; typedef … n4m_context_t;`;
// we attach the C++ Context implementation directly to that tag.
struct n4m_context_s : public ::n4m::core::Context {};

#undef N4M_PRINTF_LIKE
