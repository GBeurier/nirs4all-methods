// SPDX-License-Identifier: CECILL-2.1
//
// Internal-core unit tests for the shared dense linear-algebra helpers in
// core/common/linalg.c (Householder QR + Q^T apply + back-substitution).
// These functions are hidden in libn4m, so this TU is linked into
// n4m_internal_tests (static archive) rather than n4m_tests.
//
// Covered:
//   - A well-conditioned least-squares solve is correct (and is unchanged by
//     the cancellation-free ||v||^2 reformulation in n4m_householder_qr).
//   - A rank-deficient design (two identical columns) is flagged by
//     n4m_back_solve_R with N4M_ERR_NUMERICAL_FAILURE rather than silently
//     returning a garbage huge solution.
//   - A near-axis-aligned column (the case the cancellation-free reformulation
//     targets) factorises to an orthonormal Q to high accuracy.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

extern "C" {
#include "core/common/linalg.h"
}

namespace {

int g_linalg_failures = 0;

#define LIN_CHECK(cond)                                                       \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);     \
            ++g_linalg_failures;                                              \
        }                                                                     \
    } while (0)

// Solve a (rows x cols) least-squares problem A x ~= b via the shared QR
// helpers, leaving the caller to inspect the status. A and b are copied so the
// in-place factorisation does not clobber the originals.
n4m_status_t qr_lstsq(const std::vector<double>& A_in, int64_t rows,
                      int64_t cols, const std::vector<double>& b_in,
                      std::vector<double>* x_out) {
    std::vector<double> A = A_in;
    std::vector<double> b = b_in;
    std::vector<double> tau(static_cast<size_t>(cols), 0.0);
    n4m_status_t st = n4m_householder_qr(A.data(), rows, cols, tau.data());
    if (st != N4M_OK) return st;
    st = n4m_apply_qt(A.data(), rows, cols, tau.data(), b.data());
    if (st != N4M_OK) return st;
    x_out->assign(static_cast<size_t>(cols), 0.0);
    return n4m_back_solve_R(A.data(), rows, cols, b.data(), x_out->data());
}

// Well-conditioned overdetermined system with an exact solution:
//   x = [2, -1, 0.5]; b = A x exactly, so the least-squares fit recovers x.
void test_wellconditioned_solve() {
    const int64_t rows = 5;
    const int64_t cols = 3;
    // A row-major (5 x 3); deliberately well-spread columns.
    const std::vector<double> A = {
        1.0,  0.0,  2.0,
        0.0,  1.0,  3.0,
        1.0,  1.0, -1.0,
        2.0, -1.0,  0.0,
       -1.0,  2.0,  1.0,
    };
    const std::vector<double> x_true = {2.0, -1.0, 0.5};
    std::vector<double> b(static_cast<size_t>(rows), 0.0);
    for (int64_t i = 0; i < rows; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            s += A[static_cast<size_t>(i) * static_cast<size_t>(cols) +
                   static_cast<size_t>(j)] * x_true[static_cast<size_t>(j)];
        }
        b[static_cast<size_t>(i)] = s;
    }
    std::vector<double> x;
    n4m_status_t st = qr_lstsq(A, rows, cols, b, &x);
    LIN_CHECK(st == N4M_OK);
    if (st == N4M_OK) {
        for (int64_t j = 0; j < cols; ++j) {
            LIN_CHECK(std::fabs(x[static_cast<size_t>(j)] -
                                x_true[static_cast<size_t>(j)]) < 1e-12);
        }
    }
}

// Rank-deficient design: columns 0 and 2 are identical, so R has a
// (numerically) zero pivot. The back-substitution must flag this rather than
// return a garbage huge solution.
void test_rankdeficient_flagged() {
    const int64_t rows = 4;
    const int64_t cols = 3;
    const std::vector<double> A = {
        1.0,  0.5, 1.0,
        2.0, -1.0, 2.0,
       -1.0,  3.0, -1.0,
        0.5,  2.0, 0.5,
    };
    const std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> x;
    n4m_status_t st = qr_lstsq(A, rows, cols, b, &x);
    LIN_CHECK(st == N4M_ERR_NUMERICAL_FAILURE);
}

// Near-axis-aligned first column (Akk ~= sigma): the historical formula
// computed ||v||^2 = vk0^2 + (sigma^2 - Akk^2) where the parenthesised term is
// a catastrophic-cancellation difference of two near-equal numbers. The
// cancellation-free form keeps Q orthonormal to ~machine precision. We check
// orthonormality of the reconstructed Q columns (Q^T Q = I).
void test_near_axis_orthonormal() {
    const int64_t rows = 4;
    const int64_t cols = 2;
    // Column 0 is dominated by its first entry (near axis-aligned).
    const std::vector<double> A_in = {
        1.0e8, 1.0,
        1.0,   2.0,
        2.0,  -1.0,
       -1.0,   3.0,
    };
    std::vector<double> A = A_in;
    std::vector<double> tau(static_cast<size_t>(cols), 0.0);
    LIN_CHECK(n4m_householder_qr(A.data(), rows, cols, tau.data()) == N4M_OK);

    // Reconstruct the explicit Q columns by applying Q to the identity basis.
    // Q e_j = (apply each reflection in reverse order). We instead form Q^T's
    // action and reuse it: Q's columns q_j satisfy q_j = Q e_j, so we apply the
    // reflections to e_j in reverse. Simpler: build Q^T columns then check
    // orthonormality of the rows of Q^T (== columns of Q). We apply Q^T to each
    // standard basis vector, giving the rows of Q^T == columns of Q.
    std::vector<std::vector<double>> qcols(static_cast<size_t>(rows));
    for (int64_t j = 0; j < rows; ++j) {
        std::vector<double> e(static_cast<size_t>(rows), 0.0);
        e[static_cast<size_t>(j)] = 1.0;
        // Apply Q to e_j: reflections in reverse order H_{cols-1}...H_0.
        for (int64_t k = cols - 1; k >= 0; --k) {
            if (tau[static_cast<size_t>(k)] == 0.0) continue;
            double dot = e[static_cast<size_t>(k)];
            for (int64_t i = k + 1; i < rows; ++i) {
                dot += A[static_cast<size_t>(i) * static_cast<size_t>(cols) +
                         static_cast<size_t>(k)] * e[static_cast<size_t>(i)];
            }
            const double coef = tau[static_cast<size_t>(k)] * dot;
            e[static_cast<size_t>(k)] -= coef;
            for (int64_t i = k + 1; i < rows; ++i) {
                e[static_cast<size_t>(i)] -=
                    coef * A[static_cast<size_t>(i) * static_cast<size_t>(cols) +
                             static_cast<size_t>(k)];
            }
        }
        qcols[static_cast<size_t>(j)] = e;  // Q's column j (= Q e_j).
    }
    // Q is rows x rows orthogonal; check the first `cols` columns are
    // orthonormal: <q_a, q_b> == delta_ab to ~1e-12.
    for (int64_t a = 0; a < cols; ++a) {
        for (int64_t b = 0; b < cols; ++b) {
            double dot = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                dot += qcols[static_cast<size_t>(a)][static_cast<size_t>(i)] *
                       qcols[static_cast<size_t>(b)][static_cast<size_t>(i)];
            }
            const double expect = (a == b) ? 1.0 : 0.0;
            LIN_CHECK(std::fabs(dot - expect) < 1e-12);
        }
    }
}

// CGS2 reorthogonalisation gate for the SIMPLS deflation basis. SIMPLS deflates
// the cross-covariance by projecting out an orthonormal basis V built, one
// component at a time, by Gram-Schmidt against the prior basis columns (see
// fit_pls_regression_simpls in model.cpp). A SINGLE classical Gram-Schmidt pass
// loses orthogonality on near-collinear designs as the component count grows;
// the production fix runs the projection twice (CGS2), which restores
// orthonormality to ~machine precision.
//
// We can't observe the internal basis_v through the model API, so this gate
// reproduces the exact deflation recurrence on a deliberately near-collinear
// loading sequence and demonstrates the numerical contract the production code
// relies on: single-pass CGS drifts (>1e-7) while CGS2 stays orthonormal
// (<1e-12). It locks in the rationale for the two-pass loop in model.cpp.
void test_simpls_cgs2_reorthogonalisation() {
    const int p = 50;       // feature dimension
    const int k = 30;       // components / basis columns
    const int latent = 4;   // true rank of the loading generator (collinear)

    // Deterministic LCG (no RNG dependency): map state -> [-1, 1).
    std::uint64_t s = 0x9E3779B97F4A7C15ULL;
    auto next = [&s]() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<double>(s >> 11) / 9007199254740992.0) * 2.0 - 1.0;
    };

    // A sequence of k raw loading vectors that is deliberately ill-conditioned:
    // each vector is a near-duplicate of a small set of `latent` base directions
    // plus a shrinking perturbation, so successive vectors are nearly parallel.
    // This is the classic regime where a single classical Gram-Schmidt pass
    // loses orthogonality (the residual after one projection still has a large
    // relative component along the existing basis).
    std::vector<double> basedir(static_cast<size_t>(latent * p));
    for (auto& b : basedir) b = next();
    std::vector<std::vector<double>> raw(static_cast<size_t>(k));
    for (int c = 0; c < k; ++c) {
        std::vector<double> col(static_cast<size_t>(p), 0.0);
        // Dominant near-parallel part: heavily weight one base direction so the
        // vectors cluster, with a geometrically shrinking off-direction so the
        // independent content of later components is tiny relative to the bulk.
        const int dom = c % latent;
        const double shrink = std::pow(1e-1, c);  // 1, 0.1, 0.01, ... -> ill-cond
        for (int f = 0; f < p; ++f) {
            col[static_cast<size_t>(f)] =
                basedir[static_cast<size_t>(dom * p + f)] +
                shrink * basedir[static_cast<size_t>(((c + 1) % latent) * p + f)] +
                1e-9 * next();
        }
        raw[static_cast<size_t>(c)] = col;
    }

    // Orthonormalise the raw loadings with `passes` Gram-Schmidt passes per
    // component (1 = single classical GS, 2 = CGS2), exactly mirroring the
    // production fit_pls_regression_simpls deflation, and report the worst
    // off-diagonal of V^T V.
    auto build_and_measure = [&](int passes) -> double {
        std::vector<std::vector<double>> V;
        for (int c = 0; c < k; ++c) {
            std::vector<double> v = raw[static_cast<size_t>(c)];
            for (int pass = 0; pass < passes; ++pass) {
                for (const auto& q : V) {
                    double proj = 0.0;
                    for (int f = 0; f < p; ++f) proj += q[static_cast<size_t>(f)] *
                                                        v[static_cast<size_t>(f)];
                    for (int f = 0; f < p; ++f) v[static_cast<size_t>(f)] -=
                        q[static_cast<size_t>(f)] * proj;
                }
            }
            double nrm = 0.0;
            for (int f = 0; f < p; ++f) nrm += v[static_cast<size_t>(f)] *
                                               v[static_cast<size_t>(f)];
            nrm = std::sqrt(nrm);
            if (nrm <= 1e-300) break;  // basis vanished (rank exhausted)
            for (int f = 0; f < p; ++f) v[static_cast<size_t>(f)] /= nrm;
            V.push_back(std::move(v));
        }
        double max_off = 0.0;
        for (size_t a = 0; a < V.size(); ++a) {
            for (size_t b = a + 1; b < V.size(); ++b) {
                double dot = 0.0;
                for (int f = 0; f < p; ++f) dot += V[a][static_cast<size_t>(f)] *
                                                   V[b][static_cast<size_t>(f)];
                if (std::fabs(dot) > max_off) max_off = std::fabs(dot);
            }
        }
        return max_off;
    };

    const double off_single = build_and_measure(1);
    const double off_double = build_and_measure(2);
    // Single-pass must visibly drift (the bug the audit flagged) and CGS2 must
    // restore orthonormality to ~machine precision. If single-pass did NOT drift
    // this design is not stressful enough to be a meaningful gate.
    LIN_CHECK(off_single > 1e-7);
    LIN_CHECK(off_double < 1e-12);
}

}  // namespace

// Entry point invoked from the n4m_internal_tests main (test_internal_rng_choice.cpp).
int run_internal_linalg_tests() {
    g_linalg_failures = 0;
    test_wellconditioned_solve();
    test_rankdeficient_flagged();
    test_near_axis_orthonormal();
    test_simpls_cgs2_reorthogonalisation();
    if (g_linalg_failures == 0) {
        std::printf("[internal] linalg QR/back-solve ... ok\n");
    } else {
        std::printf("[internal] linalg QR/back-solve ... %d failure(s)\n",
                    g_linalg_failures);
    }
    return g_linalg_failures;
}
