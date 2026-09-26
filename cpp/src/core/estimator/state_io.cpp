// SPDX-License-Identifier: CECILL-2.1
//
// Kernel fitted-state writer / reader (see state_io.h).

#include "core/estimator/state_io.h"

#include <cstdlib>
#include <cstring>

#include "core/estimator/state_io.hpp"

namespace {

void put(std::vector<unsigned char>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<unsigned char>(v >> (8 * i)));
}

std::uint64_t f64_bits(double v) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, 8);
    return bits;
}

template <typename T>
n4m_status_t read_array_new(n4m_state_reader_t* r, int64_t expected, T** out,
                            int (*read)(n4m_state_reader_t*, T*, int64_t)) {
    *out = nullptr;
    int64_t n = 0;
    if (expected < 1 || !n4m_state_peek_array_length(r, expected, &n) || n != expected) {
        return N4M_ERR_CORRUPT_BUFFER;
    }
    auto* v = static_cast<T*>(std::malloc(sizeof(T) * static_cast<std::size_t>(n)));
    if (v == nullptr) return N4M_ERR_OUT_OF_MEMORY;
    if (!read(r, v, n)) {
        std::free(v);
        return N4M_ERR_CORRUPT_BUFFER;
    }
    *out = v;
    return N4M_OK;
}

}  // namespace

extern "C" {

void n4m_state_write_i64(n4m_state_writer_t* w, int64_t v) {
    put(w->bytes, static_cast<std::uint64_t>(v));
}

void n4m_state_write_f64(n4m_state_writer_t* w, double v) { put(w->bytes, f64_bits(v)); }

void n4m_state_write_f64_array(n4m_state_writer_t* w, const double* v, int64_t n) {
    put(w->bytes, static_cast<std::uint64_t>(n));
    for (int64_t i = 0; i < n; ++i) put(w->bytes, f64_bits(v[i]));
}

void n4m_state_write_i64_array(n4m_state_writer_t* w, const int64_t* v, int64_t n) {
    put(w->bytes, static_cast<std::uint64_t>(n));
    for (int64_t i = 0; i < n; ++i) put(w->bytes, static_cast<std::uint64_t>(v[i]));
}

int n4m_state_read_i64(n4m_state_reader_t* r, int64_t* out) {
    std::uint64_t v = 0;
    if (!r->take(v)) return 0;
    *out = static_cast<int64_t>(v);
    return 1;
}

int n4m_state_read_f64(n4m_state_reader_t* r, double* out) {
    std::uint64_t v = 0;
    if (!r->take(v)) return 0;
    std::memcpy(out, &v, 8);
    return 1;
}

int n4m_state_read_f64_array(n4m_state_reader_t* r, double* out, int64_t expected) {
    int64_t n = 0;
    if (!n4m_state_read_i64(r, &n) || n != expected) return 0;
    for (int64_t i = 0; i < n; ++i) {
        if (!n4m_state_read_f64(r, &out[i])) return 0;
    }
    return 1;
}

int n4m_state_read_i64_array(n4m_state_reader_t* r, int64_t* out, int64_t expected) {
    int64_t n = 0;
    if (!n4m_state_read_i64(r, &n) || n != expected) return 0;
    for (int64_t i = 0; i < n; ++i) {
        if (!n4m_state_read_i64(r, &out[i])) return 0;
    }
    return 1;
}

int n4m_state_peek_array_length(n4m_state_reader_t* r, int64_t max_len, int64_t* out) {
    std::uint64_t v = 0;
    if (!r->peek(v)) return 0;
    const auto n = static_cast<int64_t>(v);
    if (n < 0 || n > max_len || static_cast<std::uint64_t>(n) > r->remaining() / 8) return 0;
    *out = n;
    return 1;
}

n4m_status_t n4m_state_read_f64_array_new(n4m_state_reader_t* r, int64_t expected,
                                          double** out) {
    return read_array_new(r, expected, out, n4m_state_read_f64_array);
}

n4m_status_t n4m_state_read_i64_array_new(n4m_state_reader_t* r, int64_t expected,
                                          int64_t** out) {
    return read_array_new(r, expected, out, n4m_state_read_i64_array);
}

}  // extern "C"
