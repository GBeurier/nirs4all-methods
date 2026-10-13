// SPDX-License-Identifier: CECILL-2.1
//
// Internal-core tests for the shared SVD / symmetric-eigen helpers
// (core/common/svd.c) on degenerate, near-degenerate and extreme-magnitude
// inputs: equal eigenvalues, repeated singular values, rank-deficient and
// +-1 sign-symmetric designs with opposite rows, diagonal matrices, tiny or
// subnormal off-diagonals, weak blocks decoupled from strong ones (in both
// orders and at scales 2^+-600), and inputs scaled by 2^s.
//
// Regressions covered: the Jacobi SVD never converged on +-1 designs with an
// opposite row pair; the QL eigensolver exhausted its per-eigenvalue cap on
// large clusters of equal eigenvalues; a matrix-scale deflation test erased
// the coupling of an independent small block; an absolute DBL_MIN guard
// erased subnormal spectra; zeroing "negligible" Jacobi columns erased weak
// but full-rank directions; extreme magnitudes overflowed or underflowed.
//
// References are NumPy/LAPACK goldens (svd_degenerate_goldens.inc, generated
// by parity/scripts/svd_degenerate_reference.py) plus closed-form spectra.
// Comparisons are basis-free: eigen/singular values, cluster / row-space /
// column-space projectors, orthonormality and per-component residuals, never
// raw vector signs. "Relative" goldens (diagonal, block-decoupled, graded,
// subnormal) compare every value to itself with no absolute floor; coupled
// goldens compare to eps-level of the matrix norm over ALL components.
//
// The helpers are hidden in libn4m, so this TU is linked into
// n4m_internal_tests (static archive).

#include <algorithm>
#include <cfloat>
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

// Every |a_i - b_i| <= rtol * |b_i| (an exact zero reference needs an exact 0).
bool all_relative_close(const double* a, const double* b, size_t n, double rtol) {
    for (size_t i = 0; i < n; ++i)
        if (!(std::fabs(a[i] - b[i]) <= rtol * std::fabs(b[i]))) return false;
    return true;
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

// max |<m_a, m_b> - delta_ab| over the selected rows of a row-major matrix.
double row_orthonormality_error(const Mat& M, int64_t cols, const std::vector<int64_t>& rows) {
    double err = 0.0;
    for (int64_t a : rows)
        for (int64_t b : rows) {
            double s = 0.0;
            for (int64_t j = 0; j < cols; ++j) s += M[at(a, j, cols)] * M[at(b, j, cols)];
            err = std::max(err, std::fabs(s - (a == b ? 1.0 : 0.0)));
        }
    return err;
}

std::vector<int64_t> first_n(int64_t n) {
    std::vector<int64_t> v(static_cast<size_t>(n));
    std::iota(v.begin(), v.end(), 0);
    return v;
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
    std::vector<int64_t> order = first_n(n);
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

// Eigen checks: sorted eigenvalues vs reference, V orthonormal, and the
// residual ||A v_c - w_c v_c|| of EVERY eigenpair. Relative mode bounds each
// value and residual by 1e-12 * |w_c| (an exact zero must be exact);
// coupled mode by 1e-12 * ||A||.
void check_eigh(const char* name, const double* A, int64_t n, const double* w_ref,
                bool relative, const std::vector<double>& w, const Mat& V) {
    double anorm = 0.0;
    for (int64_t i = 0; i < n; ++i) anorm = std::max(anorm, std::fabs(w_ref[i]));
    if (relative) {
        SVD_CHECK(name, all_relative_close(w.data(), w_ref, static_cast<size_t>(n), 1e-12));
    } else {
        SVD_CHECK(name, max_abs_diff(w.data(), w_ref, static_cast<size_t>(n)) <= 1e-12 * anorm);
    }
    SVD_CHECK(name, row_orthonormality_error(transpose(V, n, n), n, first_n(n)) <= 1e-12);
    for (int64_t c = 0; c < n; ++c) {
        const double lambda = w[static_cast<size_t>(c)];
        double res = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < n; ++j) s += A[at(i, j, n)] * V[at(j, c, n)];
            res = std::max(res, std::fabs(s - lambda * V[at(i, c, n)]));
        }
        SVD_CHECK(name, res <= 1e-12 * (relative ? std::fabs(lambda) : anorm));
    }
}

void test_symmetric_goldens() {
    for (const SymGolden& g : kSymGoldens) {
        std::vector<double> w;
        Mat V;
        const n4m_status_t st = eigh_sorted(g.A, g.n, &w, &V);
        SVD_CHECK(g.name, st == N4M_OK);
        if (st != N4M_OK) continue;
        check_eigh(g.name, g.A, g.n, g.evals_asc, g.relative != 0, w, V);
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

enum class Accuracy { Relative, Absolute };

// SVD checks of a k-component factorisation against a golden, over ALL k
// components: singular values, orthonormal U columns and Vt rows of every
// component with sigma > 0, the residuals ||A v_i - s_i u_i|| and
// ||A^T u_i - s_i v_i|| of each such component, and the rank projectors.
void check_svd(const char* name, const SvdGolden& g, int64_t k, const Mat& U, const Mat& S,
               const Mat& Vt, Accuracy acc, double tol, double proj_tol) {
    const double s0 = g.sv_desc[0];
    if (acc == Accuracy::Relative) {
        SVD_CHECK(name, all_relative_close(S.data(), g.sv_desc, static_cast<size_t>(k), tol));
    } else {
        SVD_CHECK(name, max_abs_diff(S.data(), g.sv_desc, static_cast<size_t>(k)) <= tol * s0);
    }
    std::vector<int64_t> live;
    for (int64_t c = 0; c < k; ++c)
        if (S[static_cast<size_t>(c)] > 0.0) live.push_back(c);
    SVD_CHECK(name, row_orthonormality_error(Vt, g.n, live) <= 1e-10);
    SVD_CHECK(name, row_orthonormality_error(transpose(U, g.m, k), g.m, live) <= 1e-10);
    for (int64_t c : live) {
        const double sigma = S[static_cast<size_t>(c)];
        const double bound = tol * (acc == Accuracy::Relative ? sigma : s0);
        double res = 0.0;
        for (int64_t i = 0; i < g.m; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < g.n; ++j) s += g.A[at(i, j, g.n)] * Vt[at(c, j, g.n)];
            res = std::max(res, std::fabs(s - sigma * U[at(i, c, k)]));
        }
        for (int64_t j = 0; j < g.n; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < g.m; ++i) s += g.A[at(i, j, g.n)] * U[at(i, c, k)];
            res = std::max(res, std::fabs(s - sigma * Vt[at(c, j, g.n)]));
        }
        SVD_CHECK(name, res <= bound);
    }
    if (k >= g.rank) {
        const Mat Prow = row_projector(Vt, g.n, 0, g.rank);
        const Mat Pcol = column_projector(U, g.m, k, 0, g.rank);
        SVD_CHECK(name, max_abs_diff(Prow.data(), g.row_projector, Prow.size()) <= proj_tol);
        SVD_CHECK(name, max_abs_diff(Pcol.data(), g.col_projector, Pcol.size()) <= proj_tol);
    }
}

void test_svd_goldens() {
    for (const SvdGolden& g : kSvdGoldens) {
        const Accuracy mode = g.relative ? Accuracy::Relative : Accuracy::Absolute;
        const int64_t k = std::min(g.m, g.n);
        Mat U(static_cast<size_t>(g.m * k)), S(static_cast<size_t>(k)),
            Vt(static_cast<size_t>(k * g.n));
        Mat A(g.A, g.A + g.m * g.n);
        const n4m_status_t st = n4m_svd_compact(A.data(), g.m, g.n, U.data(), S.data(), Vt.data());
        SVD_CHECK(g.name, st == N4M_OK);
        if (st == N4M_OK) check_svd(g.name, g, k, U, S, Vt, mode, 1e-12, 1e-10);

        // Truncated paths at the golden rank. The Gram paths keep relative
        // accuracy on decoupled blocks; the randomized sketch resolves values
        // only to O(eps * ||A||) (documented), so it is checked absolutely.
        const int64_t r = g.rank;
        Mat Ur(static_cast<size_t>(g.m * r)), Sr(static_cast<size_t>(r)),
            Vtr(static_cast<size_t>(r * g.n));
        if (n4m_svd_truncated_randomized(g.A, g.m, g.n, r, Ur.data(), Sr.data(), Vtr.data()) ==
            N4M_OK) {
            SVD_CHECK(g.name, max_abs_diff(Sr.data(), g.sv_desc, static_cast<size_t>(r)) <=
                                  1e-12 * g.sv_desc[0]);
        } else {
            SVD_CHECK(g.name, false);
        }
        const double gram_tol = g.relative ? 1e-12 : 1e-10;
        if (g.m <= g.n) {
            SVD_CHECK(g.name, n4m_svd_truncated_dual_wide(g.A, g.m, g.n, r, Ur.data(), Sr.data(),
                                                          Vtr.data()) == N4M_OK);
            check_svd(g.name, g, r, Ur, Sr, Vtr, mode, gram_tol, 1e-8);
        }
        if (g.m >= g.n) {
            SVD_CHECK(g.name, n4m_svd_truncated_tall_gram(g.A, g.m, g.n, r, Ur.data(), Sr.data(),
                                                          Vtr.data()) == N4M_OK);
            check_svd(g.name, g, r, Ur, Sr, Vtr, mode, gram_tol, 1e-8);
        }
    }
}

const double kWeakTall[9] = {1.0, 0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, 1e-9, 1e-9};

// Mirrors the PCA patterns of the reference script.
Mat pca_pattern(int pattern, int64_t rows, int64_t cols) {
    Mat X(static_cast<size_t>(rows * cols), 0.0);
    if (pattern == 2) {
        for (int64_t i = 0; i < 3; ++i)
            for (int64_t j = 0; j < 3; ++j) {
                X[at(i, j, cols)] = kWeakTall[at(i, j, 3)];
                X[at(i + 3, j, cols)] = -kWeakTall[at(i, j, 3)];
            }
        return X;
    }
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
// tall Gram QL) on +-1 designs and on a weak decoupled variance.
void test_pca_goldens() {
    for (const PcaGolden& g : kPcaGoldens) {
        const Mat X = pca_pattern(g.pattern, g.rows, g.cols);
        n4m_pca_fit_t fit{};
        const n4m_status_t st = n4m_pca_fit(X.data(), g.rows, g.cols, g.k, 1, &fit);
        SVD_CHECK(g.name, st == N4M_OK);
        if (st != N4M_OK) continue;
        const double ev0 = g.explained_var[0];
        const auto kk = static_cast<size_t>(g.k);
        if (g.relative) {
            SVD_CHECK(g.name, all_relative_close(fit.explained_var, g.explained_var, kk, 1e-12));
        } else {
            SVD_CHECK(g.name, max_abs_diff(fit.explained_var, g.explained_var, kk) <= 1e-10 * ev0);
        }
        std::vector<int64_t> live;
        for (int64_t a = 0; a < g.k; ++a)
            if (g.relative || g.explained_var[a] > 1e-10 * ev0) live.push_back(a);
        const Mat C(fit.components, fit.components + g.k * g.cols);
        SVD_CHECK(g.name, row_orthonormality_error(C, g.cols, live) <= 1e-10);
        n4m_pca_fit_free(&fit);
    }
}

// Deterministic sweep over small +-1 designs with one row the opposite of
// another, raw and column-centred: every one must factorise with orthonormal
// factors (completed directions included) and reconstruct.
void test_pm1_opposite_sweep() {
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    auto next = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 33);
    };
    int failures = 0;
    double worst_rec = 0.0, worst_orth = 0.0;
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
                    std::vector<int64_t> live;
                    for (int64_t c = 0; c < k; ++c)
                        if (S[static_cast<size_t>(c)] > 0.0) live.push_back(c);
                    worst_orth = std::max(worst_orth, row_orthonormality_error(Vt, n, live));
                    worst_orth = std::max(worst_orth,
                                          row_orthonormality_error(transpose(U, m, k), m, live));
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
    SVD_CHECK("pm1_opposite_sweep", worst_orth <= 1e-12);
}

// Closed-form references for sizes too large to embed: Householder
// reflections H diag(lambda) H of highly repeated spectra, and the all-ones
// matrix (eigenvalues n, 0, ..., 0).
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
            check_eigh("large_repeated_spectra", A.data(), n, lam.data(), false, w, V);
        }
    }
}

// Magnitude sweep of every helper: A * 2^s must give singular values /
// eigenvalues exactly scaled by 2^s (inputs are exact dyadic integers), or an
// explicit N4M_ERR_NUMERICAL_FAILURE when a published value is not
// representable. Non-finite input is an explicit failure too.
void test_magnitude_sweep() {
    const double base[24] = {3, 1, -2, 0.5, 1, 4, 0, -1, -2, 0, 5, 1,
                             0.5, -1, 1, 2, 1, 1, 1, 1, -1, 2, 0.25, 3};
    const double sym_base[16] = {2, 0.5, 0, 0, 0.5, 2, 0.5, 0, 0, 0.5, 2, 0.5, 0, 0, 0.5, 2};
    Mat S0(4), U(36), Vt(36), S(6);
    Mat W(base, base + 24);
    SVD_CHECK("sweep", n4m_svd_compact(W.data(), 6, 4, U.data(), S0.data(), Vt.data()) == N4M_OK);
    std::vector<double> e0(4);
    Mat V(16);
    SVD_CHECK("sweep", n4m_symmetric_eigh(sym_base, 4, e0.data(), V.data()) == N4M_OK);
    const int scales[] = {-1060, -1022, -1000, -500, -100, 0, 100, 500, 1000, 1020};
    for (int s : scales) {
        Mat A(24), G(16);
        for (size_t i = 0; i < 24; ++i) A[i] = std::ldexp(base[i], s);
        for (size_t i = 0; i < 16; ++i) G[i] = std::ldexp(sym_base[i], s);
        // Expected: every scaled value finite and >= 2^-1042.
        bool sv_ok = true, ev_ok = true;
        for (double v : S0) {
            const double x = std::ldexp(v, s);
            sv_ok = sv_ok && std::isfinite(x) && x >= std::ldexp(1.0, -1042);
        }
        for (double v : e0) {
            const double x = std::ldexp(v, s);
            ev_ok = ev_ok && std::isfinite(x) && x >= std::ldexp(1.0, -1042);
        }
        const n4m_status_t want_sv = sv_ok ? N4M_OK : N4M_ERR_NUMERICAL_FAILURE;
        Mat Wc = A;
        SVD_CHECK("sweep_compact",
                  n4m_svd_compact(Wc.data(), 6, 4, U.data(), S.data(), Vt.data()) == want_sv);
        if (sv_ok)
            for (size_t i = 0; i < 4; ++i) SVD_CHECK("sweep_compact", S[i] == std::ldexp(S0[i], s));
        SVD_CHECK("sweep_tall", n4m_svd_truncated_tall_gram(A.data(), 6, 4, 4, U.data(), S.data(),
                                                            Vt.data()) == want_sv);
        if (sv_ok)
            for (size_t i = 0; i < 4; ++i)
                SVD_CHECK("sweep_tall", std::fabs(S[i] / std::ldexp(S0[i], s) - 1.0) <= 1e-13);
        SVD_CHECK("sweep_dual", n4m_svd_truncated_dual_wide(A.data(), 4, 6, 4, U.data(), S.data(),
                                                            Vt.data()) ==
                                    (sv_ok ? N4M_OK : N4M_ERR_NUMERICAL_FAILURE));
        SVD_CHECK("sweep_rand", n4m_svd_truncated_randomized(A.data(), 6, 4, 2, U.data(),
                                                             S.data(), Vt.data()) == want_sv);
        if (sv_ok)
            SVD_CHECK("sweep_rand", std::fabs(S[1] / std::ldexp(S0[1], s) - 1.0) <= 1e-13);
        std::vector<double> e(4);
        SVD_CHECK("sweep_eigh", n4m_symmetric_eigh(G.data(), 4, e.data(), V.data()) ==
                                    (ev_ok ? N4M_OK : N4M_ERR_NUMERICAL_FAILURE));
        if (ev_ok)
            for (size_t i = 0; i < 4; ++i) SVD_CHECK("sweep_eigh", e[i] == std::ldexp(e0[i], s));
    }
    // n4m_pca_fit squares singular values: exact 4^s scaling while the
    // variances are representable, an explicit failure otherwise.
    n4m_pca_fit_t fit0{};
    SVD_CHECK("pca_sweep", n4m_pca_fit(base, 6, 4, 4, 1, &fit0) == N4M_OK);
    for (int s : {-600, -200, 0, 200, 600}) {
        Mat A(24);
        for (size_t i = 0; i < 24; ++i) A[i] = std::ldexp(base[i], s);
        n4m_pca_fit_t fit{};
        const n4m_status_t st = n4m_pca_fit(A.data(), 6, 4, 4, 1, &fit);
        if (s == -600 || s == 600) {
            SVD_CHECK("pca_sweep", st == N4M_ERR_NUMERICAL_FAILURE);
            continue;
        }
        SVD_CHECK("pca_sweep", st == N4M_OK);
        if (st != N4M_OK) continue;
        for (size_t i = 0; i < 4; ++i)
            SVD_CHECK("pca_sweep", std::fabs(fit.explained_var[i] /
                                             std::ldexp(fit0.explained_var[i], 2 * s) - 1.0) <=
                                       1e-14);
        n4m_pca_fit_free(&fit);
    }
    n4m_pca_fit_free(&fit0);
    Mat bad(base, base + 24);
    bad[5] = std::nan("");
    SVD_CHECK("nonfinite", n4m_svd_compact(bad.data(), 6, 4, U.data(), S.data(), Vt.data()) ==
                               N4M_ERR_NUMERICAL_FAILURE);
    bad[5] = HUGE_VAL;
    SVD_CHECK("nonfinite", n4m_svd_truncated_tall_gram(bad.data(), 6, 4, 2, U.data(), S.data(),
                                                       Vt.data()) == N4M_ERR_NUMERICAL_FAILURE);
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
    test_magnitude_sweep();
    if (g_svd_failures == 0) {
        std::printf("[internal] degenerate SVD/eigh vs numpy %s ... ok\n", kSvdGoldenNumpyVersion);
    } else {
        std::printf("[internal] degenerate SVD/eigh ... %d failure(s)\n", g_svd_failures);
    }
    return g_svd_failures;
}
