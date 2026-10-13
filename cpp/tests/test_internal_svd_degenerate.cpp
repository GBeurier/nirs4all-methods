// SPDX-License-Identifier: CECILL-2.1
//
// Internal-core tests for the shared SVD / symmetric-eigen helpers
// (core/common/svd.c) on degenerate and near-degenerate inputs: equal
// eigenvalues, repeated singular values, rank-deficient and +-1
// sign-symmetric designs with opposite rows, already diagonal matrices and
// tiny / denormal off-diagonals.
//
// Before the fix, n4m_svd_compact (one-sided Jacobi) returned
// N4M_ERR_CONVERGENCE_FAILED on +-1 designs with an opposite row pair (the
// null-space column stayed parallel to a significant column), and the QL
// eigen-solver behind n4m_symmetric_eigh / the Gram SVD paths failed on large
// clusters of equal eigenvalues (local-only deflation test).
//
// References are NumPy/LAPACK goldens (svd_degenerate_goldens.inc, generated
// by parity/scripts/svd_degenerate_reference.py) plus closed-form spectra.
// Comparisons are basis-free: eigen/singular values, cluster / row-space /
// column-space projectors and reconstruction, never raw vector signs.
//
// The helpers are hidden in libn4m, so this TU is linked into
// n4m_internal_tests (static archive).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

extern "C" {
#include "core/common/pca_helper.h"
#include "core/common/svd.h"
}

#include "svd_degenerate_goldens.inc"

int run_internal_svd_degenerate_tests();

namespace {

int g_svd_failures = 0;

#define SVD_CHECK(name, cond)                                                  \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL %s:%d [%s] %s\n", __FILE__, __LINE__, (name),  \
                        #cond);                                                \
            ++g_svd_failures;                                                  \
        }                                                                      \
    } while (0)

using Mat = std::vector<double>;

size_t at(int64_t i, int64_t j, int64_t cols) {
    return static_cast<size_t>(i) * static_cast<size_t>(cols) + static_cast<size_t>(j);
}

double max_abs_diff(const double* a, const double* b, size_t n) {
    double d = 0.0;
    for (size_t i = 0; i < n; ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

// Columns [c0, c1) of the row-major (rows x cols) matrix M -> M_c M_c^T.
Mat column_projector(const Mat& M, int64_t rows, int64_t cols, int64_t c0, int64_t c1) {
    Mat P(static_cast<size_t>(rows * rows), 0.0);
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < rows; ++j) {
            double s = 0.0;
            for (int64_t c = c0; c < c1; ++c) s += M[at(i, c, cols)] * M[at(j, c, cols)];
            P[at(i, j, rows)] = s;
        }
    return P;
}

// Rows [r0, r1) of the row-major (rows x cols) matrix M -> M_r^T M_r.
Mat row_projector(const Mat& M, int64_t cols, int64_t r0, int64_t r1) {
    Mat P(static_cast<size_t>(cols * cols), 0.0);
    for (int64_t i = 0; i < cols; ++i)
        for (int64_t j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int64_t r = r0; r < r1; ++r) s += M[at(r, i, cols)] * M[at(r, j, cols)];
            P[at(i, j, cols)] = s;
        }
    return P;
}

// max |M_r M_r^T - I| over rows [0, r) of a row-major (rows x cols) matrix.
double row_orthonormality_error(const Mat& M, int64_t cols, int64_t r) {
    double err = 0.0;
    for (int64_t a = 0; a < r; ++a)
        for (int64_t b = 0; b < r; ++b) {
            double s = 0.0;
            for (int64_t j = 0; j < cols; ++j) s += M[at(a, j, cols)] * M[at(b, j, cols)];
            err = std::max(err, std::fabs(s - (a == b ? 1.0 : 0.0)));
        }
    return err;
}

Mat transpose(const Mat& M, int64_t rows, int64_t cols) {
    Mat T(M.size());
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j) T[at(j, i, rows)] = M[at(i, j, cols)];
    return T;
}

// Run n4m_symmetric_eigh and return eigenpairs sorted ascending (eigenvectors
// as columns of the row-major n x n matrix).
n4m_status_t eigh_sorted(const double* A, int64_t n, std::vector<double>* w, Mat* V) {
    std::vector<double> w_raw(static_cast<size_t>(n));
    Mat V_raw(static_cast<size_t>(n * n));
    const n4m_status_t st = n4m_symmetric_eigh(A, n, w_raw.data(), V_raw.data());
    if (st != N4M_OK) return st;
    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return w_raw[static_cast<size_t>(a)] < w_raw[static_cast<size_t>(b)];
    });
    w->assign(static_cast<size_t>(n), 0.0);
    V->assign(static_cast<size_t>(n * n), 0.0);
    for (int64_t c = 0; c < n; ++c) {
        const int64_t src = order[static_cast<size_t>(c)];
        (*w)[static_cast<size_t>(c)] = w_raw[static_cast<size_t>(src)];
        for (int64_t r = 0; r < n; ++r) (*V)[at(r, c, n)] = V_raw[at(r, src, n)];
    }
    return N4M_OK;
}

// Shared eigen checks: sorted eigenvalues vs reference, V orthonormal, and
// V diag(w) V^T reconstructs A.
void check_eigh(const char* name, const double* A, int64_t n, const double* w_ref,
                const std::vector<double>& w, const Mat& V) {
    double anorm = 0.0;
    for (int64_t i = 0; i < n; ++i) anorm = std::max(anorm, std::fabs(w_ref[i]));
    const double tol = 1e-12 * anorm + 1e-300;
    SVD_CHECK(name, max_abs_diff(w.data(), w_ref, static_cast<size_t>(n)) <= tol);
    SVD_CHECK(name, row_orthonormality_error(transpose(V, n, n), n, n) <= 1e-12);
    double rec = 0.0;
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (int64_t c = 0; c < n; ++c)
                s += V[at(i, c, n)] * w[static_cast<size_t>(c)] * V[at(j, c, n)];
            rec = std::max(rec, std::fabs(s - A[at(i, j, n)]));
        }
    SVD_CHECK(name, rec <= tol);
}

void test_symmetric_goldens() {
    for (const SymGolden& g : kSymGoldens) {
        std::vector<double> w;
        Mat V;
        const n4m_status_t st = eigh_sorted(g.A, g.n, &w, &V);
        SVD_CHECK(g.name, st == N4M_OK);
        if (st != N4M_OK) continue;
        check_eigh(g.name, g.A, g.n, g.evals_asc, w, V);
        if (g.projectors == nullptr) continue;
        size_t offset = 0;
        for (int c = 0; c < g.n_clusters; ++c) {
            const Mat P = column_projector(V, g.n, g.n, g.cluster_start[c],
                                           g.cluster_start[c + 1]);
            SVD_CHECK(g.name, max_abs_diff(P.data(), g.projectors + offset, P.size()) <= 1e-10);
            offset += P.size();
        }
    }
}

// Shared SVD checks against a golden: singular values, reconstruction,
// orthonormal leading factors and rank-r row / column projectors.
void check_svd(const char* name, const SvdGolden& g, int64_t k, const Mat& U, const Mat& S,
               const Mat& Vt, double sv_rtol, double proj_tol, bool full) {
    const int64_t r = std::min<int64_t>(g.rank, k);
    const double s0 = g.sv_desc[0];
    SVD_CHECK(name, max_abs_diff(S.data(), g.sv_desc, static_cast<size_t>(r)) <= sv_rtol * s0);
    SVD_CHECK(name, row_orthonormality_error(Vt, g.n, r) <= 1e-10);
    SVD_CHECK(name, row_orthonormality_error(transpose(U, g.m, k), g.m, r) <= 1e-10);
    if (full) {
        double rec = 0.0;
        for (int64_t i = 0; i < g.m; ++i)
            for (int64_t j = 0; j < g.n; ++j) {
                double s = 0.0;
                for (int64_t c = 0; c < k; ++c)
                    s += U[at(i, c, k)] * S[static_cast<size_t>(c)] * Vt[at(c, j, g.n)];
                rec = std::max(rec, std::fabs(s - g.A[at(i, j, g.n)]));
            }
        SVD_CHECK(name, rec <= 1e-12 * s0);
    }
    if (r == g.rank) {
        const Mat Prow = row_projector(Vt, g.n, 0, r);
        const Mat Pcol = column_projector(U, g.m, k, 0, r);
        SVD_CHECK(name, max_abs_diff(Prow.data(), g.row_projector, Prow.size()) <= proj_tol);
        SVD_CHECK(name, max_abs_diff(Pcol.data(), g.col_projector, Pcol.size()) <= proj_tol);
    }
}

void test_svd_goldens() {
    for (const SvdGolden& g : kSvdGoldens) {
        const int64_t k = std::min(g.m, g.n);
        Mat U(static_cast<size_t>(g.m * k)), S(static_cast<size_t>(k)),
            Vt(static_cast<size_t>(k * g.n));
        Mat A(g.A, g.A + g.m * g.n);
        const n4m_status_t st = n4m_svd_compact(A.data(), g.m, g.n, U.data(), S.data(), Vt.data());
        SVD_CHECK(g.name, st == N4M_OK);
        if (st == N4M_OK) check_svd(g.name, g, k, U, S, Vt, 1e-12, 1e-10, true);

        // Truncated paths at the numerical rank: randomized (Jacobi on the
        // sketch) and the Gram-matrix QL paths.
        const int64_t r = g.rank;
        Mat Ur(static_cast<size_t>(g.m * r)), Sr(static_cast<size_t>(r)),
            Vtr(static_cast<size_t>(r * g.n));
        SVD_CHECK(g.name, n4m_svd_truncated_randomized(g.A, g.m, g.n, r, Ur.data(), Sr.data(),
                                                       Vtr.data()) == N4M_OK);
        check_svd(g.name, g, r, Ur, Sr, Vtr, 1e-12, 1e-10, false);
        if (g.m <= g.n) {
            SVD_CHECK(g.name, n4m_svd_truncated_dual_wide(g.A, g.m, g.n, r, Ur.data(), Sr.data(),
                                                          Vtr.data()) == N4M_OK);
            check_svd(g.name, g, r, Ur, Sr, Vtr, 1e-10, 1e-8, false);
        }
        if (g.m >= g.n) {
            SVD_CHECK(g.name, n4m_svd_truncated_tall_gram(g.A, g.m, g.n, r, Ur.data(), Sr.data(),
                                                          Vtr.data()) == N4M_OK);
            check_svd(g.name, g, r, Ur, Sr, Vtr, 1e-10, 1e-8, false);
        }
    }
}

// Mirrors pm1_opposite_rows / pm1_opposite_cols in the reference script.
Mat pca_pattern(int pattern, int64_t rows, int64_t cols) {
    Mat X(static_cast<size_t>(rows * cols));
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j) {
            double v;
            if (pattern == 0) {
                switch (i % 4) {
                    case 0: v = (j % 3 == 0) ? 1.0 : -1.0; break;
                    case 1: v = ((j / 2) % 2) ? 1.0 : -1.0; break;
                    case 2: v = (j % 3 == 0) ? -1.0 : 1.0; break;
                    default: v = (j % 5 < 2) ? 1.0 : -1.0; break;
                }
            } else {
                const int64_t half = cols / 2;
                const int64_t jj = j % half;
                v = ((7 * i + 3 * jj) % 5 < 2) ? 1.0 : -1.0;
                if (j >= half) v = -v;
            }
            X[at(i, j, cols)] = v;
        }
    return X;
}

// n4m_pca_fit through its three dispatch paths (Jacobi, dual-wide Gram QL,
// tall Gram QL) on +-1 designs with opposite rows / columns.
void test_pca_goldens() {
    for (const PcaGolden& g : kPcaGoldens) {
        const Mat X = pca_pattern(g.pattern, g.rows, g.cols);
        n4m_pca_fit_t fit{};
        const n4m_status_t st = n4m_pca_fit(X.data(), g.rows, g.cols, g.k, 1, &fit);
        SVD_CHECK(g.name, st == N4M_OK);
        if (st != N4M_OK) continue;
        const double ev0 = g.explained_var[0];
        SVD_CHECK(g.name, max_abs_diff(fit.explained_var, g.explained_var,
                                       static_cast<size_t>(g.k)) <= 1e-10 * ev0);
        int64_t r = 0;
        while (r < g.k && g.explained_var[r] > 1e-10 * ev0) ++r;
        const Mat C(fit.components, fit.components + g.k * g.cols);
        SVD_CHECK(g.name, row_orthonormality_error(C, g.cols, r) <= 1e-10);
        n4m_pca_fit_free(&fit);
    }
}

// Deterministic sweep over small +-1 designs with one row the opposite of
// another, raw and column-centred: every one must factorise and reconstruct.
void test_pm1_opposite_sweep() {
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    auto next = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 33);
    };
    int failures = 0;
    double worst_rec = 0.0;
    for (int64_t m = 3; m <= 6; ++m)
        for (int64_t n = 3; n <= 10; ++n)
            for (int trial = 0; trial < 25; ++trial)
                for (int center = 0; center < 2; ++center) {
                    Mat A(static_cast<size_t>(m * n));
                    for (double& v : A) v = (next() & 1U) ? 1.0 : -1.0;
                    const int64_t a = static_cast<int64_t>(next() % static_cast<uint32_t>(m));
                    const int64_t b =
                        (a + 1 + static_cast<int64_t>(next() % static_cast<uint32_t>(m - 1))) % m;
                    for (int64_t j = 0; j < n; ++j) A[at(b, j, n)] = -A[at(a, j, n)];
                    if (center) {
                        for (int64_t j = 0; j < n; ++j) {
                            double mu = 0.0;
                            for (int64_t i = 0; i < m; ++i) mu += A[at(i, j, n)];
                            mu /= static_cast<double>(m);
                            for (int64_t i = 0; i < m; ++i) A[at(i, j, n)] -= mu;
                        }
                    }
                    const int64_t k = std::min(m, n);
                    Mat W = A, U(static_cast<size_t>(m * k)), S(static_cast<size_t>(k)),
                        Vt(static_cast<size_t>(k * n));
                    if (n4m_svd_compact(W.data(), m, n, U.data(), S.data(), Vt.data()) != N4M_OK) {
                        ++failures;
                        continue;
                    }
                    for (int64_t i = 0; i < m; ++i)
                        for (int64_t j = 0; j < n; ++j) {
                            double s = 0.0;
                            for (int64_t c = 0; c < k; ++c)
                                s += U[at(i, c, k)] * S[static_cast<size_t>(c)] * Vt[at(c, j, n)];
                            worst_rec = std::max(worst_rec, std::fabs(s - A[at(i, j, n)]));
                        }
                }
    SVD_CHECK("pm1_opposite_sweep", failures == 0);
    SVD_CHECK("pm1_opposite_sweep", worst_rec <= 1e-12 * 10.0);
}

// Closed-form references for sizes too large to embed: Householder
// reflections H diag(lambda) H of highly repeated spectra, and the all-ones
// matrix (eigenvalues n, 0, ..., 0). Several of these exhausted the old QL
// iteration cap.
void test_large_repeated_spectra() {
    const int64_t sizes[] = {64, 128, 200};
    for (const int64_t n : sizes) {
        for (int kind = 0; kind < 3; ++kind) {
            std::vector<double> lam(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                double v = 0.0;
                if (kind == 0) v = (i < n / 2) ? 0.0 : 1.0;
                if (kind == 1) v = (i % 3 == 0) ? 2.0 : ((i % 3 == 1) ? -2.0 : 0.0);
                if (kind == 2) v = (i == 0) ? static_cast<double>(n) : 0.0;
                lam[static_cast<size_t>(i)] = v;
            }
            Mat A(static_cast<size_t>(n * n), 0.0);
            if (kind == 2) {
                std::fill(A.begin(), A.end(), 1.0);
            } else {
                std::vector<double> h(static_cast<size_t>(n));
                double hh = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    h[static_cast<size_t>(i)] = std::sin(0.7 * static_cast<double>(i) + 0.3) + 0.1;
                    hh += h[static_cast<size_t>(i)] * h[static_cast<size_t>(i)];
                }
                // H = I - 2 h h^T / (h^T h); A = H diag(lam) H.
                Mat H(static_cast<size_t>(n * n));
                for (int64_t i = 0; i < n; ++i)
                    for (int64_t j = 0; j < n; ++j)
                        H[at(i, j, n)] = (i == j ? 1.0 : 0.0) -
                                         2.0 * h[static_cast<size_t>(i)] *
                                             h[static_cast<size_t>(j)] / hh;
                for (int64_t i = 0; i < n; ++i)
                    for (int64_t j = 0; j <= i; ++j) {
                        double s = 0.0;
                        for (int64_t c = 0; c < n; ++c)
                            s += H[at(i, c, n)] * lam[static_cast<size_t>(c)] * H[at(j, c, n)];
                        A[at(i, j, n)] = s;
                        A[at(j, i, n)] = s;
                    }
            }
            std::sort(lam.begin(), lam.end());
            std::vector<double> w;
            Mat V;
            const n4m_status_t st = eigh_sorted(A.data(), n, &w, &V);
            SVD_CHECK("large_repeated_spectra", st == N4M_OK);
            if (st != N4M_OK) continue;
            double anorm = 0.0;
            for (double v : lam) anorm = std::max(anorm, std::fabs(v));
            // The analytic A is itself rounded at ~n * eps * ||A||.
            SVD_CHECK("large_repeated_spectra",
                      max_abs_diff(w.data(), lam.data(), lam.size()) <= 1e-12 * anorm);
            SVD_CHECK("large_repeated_spectra",
                      row_orthonormality_error(transpose(V, n, n), n, n) <= 1e-12);
        }
    }
}

}  // namespace

// Entry point invoked from the n4m_internal_tests main (test_internal_rng_choice.cpp).
int run_internal_svd_degenerate_tests() {
    g_svd_failures = 0;
    test_symmetric_goldens();
    test_svd_goldens();
    test_pca_goldens();
    test_pm1_opposite_sweep();
    test_large_repeated_spectra();
    if (g_svd_failures == 0) {
        std::printf("[internal] degenerate SVD/eigh vs numpy %s ... ok\n", kSvdGoldenNumpyVersion);
    } else {
        std::printf("[internal] degenerate SVD/eigh ... %d failure(s)\n", g_svd_failures);
    }
    return g_svd_failures;
}
