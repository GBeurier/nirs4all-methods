// SPDX-License-Identifier: CECILL-2.1
// Exact signed accumulator of sums of products of binary64 values (a Kulisch
// fixed-point accumulator). Every product a*b of finite doubles is an integer
// multiple of 2^-2148 below 2^2048; the accumulator holds such sums exactly
// for up to 2^60 terms, so the sign of an expression like
// sum_j (q_j - r_j)^2 - n t^2 is decided exactly, without rounding, overflow
// or underflow. Used only to resolve threshold decisions that a fast
// floating-point estimate cannot settle within its error bound.
#ifndef N4M_CORE_COMMON_EXACT_SUM_HPP
#define N4M_CORE_COMMON_EXACT_SUM_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace n4m::core {

class ExactSum {
public:
    /// Adds sign * a * b exactly (a, b finite).
    void add_product(double a, double b, bool negative) {
        std::uint64_t ma = 0, mb = 0;
        int ea = 0, eb = 0;
        if (!decompose(a, ma, ea) || !decompose(b, mb, eb)) return;
        negative = negative != ((a < 0) != (b < 0));
        // ma, mb < 2^53: split into 32-bit halves so every partial product
        // fits in 64 bits (portable, no 128-bit integers).
        const std::uint64_t a1 = ma >> 32, a0 = ma & 0xffffffffu;
        const std::uint64_t b1 = mb >> 32, b0 = mb & 0xffffffffu;
        const int bit = ea + eb + kOffset;
        add_shifted(a0 * b0, bit, negative);
        add_shifted(a0 * b1, bit + 32, negative);
        add_shifted(a1 * b0, bit + 32, negative);
        add_shifted(a1 * b1, bit + 64, negative);
    }
    /// -1, 0 or +1: the exact sign of the accumulated sum.
    int sign() const {
        if (words_[kWords - 1] >> 63) return -1;
        for (std::uint64_t w : words_)
            if (w) return 1;
        return 0;
    }

private:
    // Products span bits -2148 .. 2048; 2^60 terms add 60 more. Bit 0 of the
    // accumulator is 2^-kOffset.
    static constexpr int kOffset = 2160;
    static constexpr int kWords = 72;  // 4608 bits, two's complement

    // |x| = m 2^e with integer m < 2^53; false for zero.
    static bool decompose(double x, std::uint64_t& m, int& e) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &x, sizeof bits);
        const auto biased = static_cast<int>((bits >> 52) & 0x7ff);
        m = bits & ((std::uint64_t{1} << 52) - 1);
        if (biased == 0) {
            e = -1074;
        } else {
            m |= std::uint64_t{1} << 52;
            e = biased - 1075;
        }
        return m != 0;
    }
    void add_shifted(std::uint64_t v, int bit, bool negative) {
        if (!v) return;
        const auto index = static_cast<std::size_t>(bit >> 6);
        const int shift = bit & 63;
        const std::uint64_t lo = v << shift;
        const std::uint64_t hi = shift ? v >> (64 - shift) : 0;
        if (negative) {
            sub_at(index, lo);
            if (hi) sub_at(index + 1, hi);
        } else {
            add_at(index, lo);
            if (hi) add_at(index + 1, hi);
        }
    }
    void add_at(std::size_t i, std::uint64_t v) {
        for (; i < kWords && v; ++i) {
            const std::uint64_t before = words_[i];
            words_[i] = before + v;
            v = words_[i] < before ? 1 : 0;
        }
    }
    void sub_at(std::size_t i, std::uint64_t v) {
        for (; i < kWords && v; ++i) {
            const std::uint64_t before = words_[i];
            words_[i] = before - v;
            v = before < v ? 1 : 0;
        }
    }
    std::array<std::uint64_t, kWords> words_{};
};

}  // namespace n4m::core

#endif  // N4M_CORE_COMMON_EXACT_SUM_HPP
