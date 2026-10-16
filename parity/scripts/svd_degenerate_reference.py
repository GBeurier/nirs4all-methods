#!/usr/bin/env python3
"""NumPy/LAPACK goldens for the degenerate SVD / symmetric-eigen tests.

Writes ``cpp/tests/svd_degenerate_goldens.inc`` (test-only data consumed by
``cpp/tests/test_internal_svd_degenerate.cpp``). The cases are degenerate or
near-degenerate on purpose: equal eigenvalues, repeated singular values,
rank-deficient designs, +-1 sign-symmetric designs with opposite rows, already
diagonal matrices and tiny / denormal off-diagonals.

Eigen and singular vectors of a repeated value are not unique, so the goldens
are basis-free: sorted eigenvalues / singular values, the orthogonal projector
onto each eigenvalue cluster, and the projectors onto the row and column
spaces of the numerical rank. Vector signs are never compared.

Run from the repository root:

    python3 parity/scripts/svd_degenerate_reference.py
"""

from __future__ import annotations

import pathlib
import platform

import mpmath as mp
import numpy as np

OUT = (
    pathlib.Path(__file__).resolve().parents[2] / "cpp/tests/svd_degenerate_goldens.inc"
)
CLUSTER_RTOL = 1e-8
# Numerical rank of the coupled cases; "relative" cases (diagonal,
# block-decoupled, graded or subnormal) use their exact rank instead, and
# every eigen/singular value of theirs is compared relative to itself.
RANK_RTOL = 1e-10
# Larger cases keep eigenvalue, reconstruction and orthonormality checks only.
PROJECTOR_MAX_N = 8


def pm1_opposite_rows(rows: int, cols: int) -> np.ndarray:
    """+-1 design; row 2 is the exact opposite of row 0 (mirrored in C++)."""
    x = np.empty((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if i % 4 == 0:
                v = 1.0 if j % 3 == 0 else -1.0
            elif i % 4 == 1:
                v = 1.0 if (j // 2) % 2 else -1.0
            elif i % 4 == 2:
                v = -(1.0 if j % 3 == 0 else -1.0)
            else:
                v = 1.0 if j % 5 < 2 else -1.0
            x[i, j] = v
    return x


def pm1_opposite_cols(rows: int, half: int) -> np.ndarray:
    """Tall +-1 design whose column j + half is minus column j."""
    x = np.empty((rows, 2 * half))
    for i in range(rows):
        for j in range(half):
            v = 1.0 if (7 * i + 3 * j) % 5 < 2 else -1.0
            x[i, j] = v
            x[i, j + half] = -v
    return x


def astra_sym_block(small_first: bool, scale_exp: int = 0) -> np.ndarray:
    """diag(1, 1e-18 [[2,1],[1,2]]): a small block decoupled from a large one."""
    small = 1e-18 * np.array([[2.0, 1.0], [1.0, 2.0]])
    t = np.zeros((3, 3))
    if small_first:
        t[:2, :2] = small
        t[2, 2] = 1.0
    else:
        t[0, 0] = 1.0
        t[1:, 1:] = small
    return np.ldexp(t, scale_exp)


def astra_weak_block(weak_first: bool, scale_exp: int = 0) -> np.ndarray:
    """diag(1, 1e-17 B): full rank, singular values 1 and 1e-17 (2+-sqrt2, 2)."""
    b = 1e-17 * np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    a = np.zeros((4, 4))
    if weak_first:
        a[:3, :3] = b
        a[3, 3] = 1.0
    else:
        a[0, 0] = 1.0
        a[1:, 1:] = b
    return np.ldexp(a, scale_exp)


ASTRA_TALL = np.array([[1.0, 0.0, 0.0], [0.0, 1e-9, 0.0], [0.0, 1e-9, 1e-9]])


def clusters(values: np.ndarray, relative: bool) -> list[tuple[int, int]]:
    """Eigenvalue clusters: gaps measured against max(1, |lambda|max), or,
    for relative cases, against the two neighbouring eigenvalues."""
    scale = max(1.0, float(np.max(np.abs(values))))
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i < len(values):
            ref = max(abs(values[i]), abs(values[i - 1])) if relative else scale
            if values[i] - values[i - 1] <= CLUSTER_RTOL * ref:
                continue
        out.append((start, i))
        start = i
    return out


def sym_cases() -> list[tuple[str, np.ndarray, bool]]:
    cases = []
    x = pm1_opposite_rows(4, 6)
    xc = x - x.mean(axis=0)
    cases.append(("pm1_opposite_centered_gram_4", xc @ xc.T, False))
    cases.append(("scaled_identity_4", 4.0 * np.eye(4), True))
    cases.append(("all_ones_6", np.ones((6, 6)), False))
    cases.append(("centering_5", np.eye(5) - 0.2, False))
    cases.append(("diagonal_repeated_6", np.diag([2.0, 0.0, 1.0, 2.0, 0.0, 1.0]), True))
    t = np.diag([1.0, 1.0, 1.0, 1.0])
    for i in range(3):
        t[i, i + 1] = t[i + 1, i] = 1e-120
    cases.append(("tiny_offdiag_equal_diag_4", t, True))
    t = np.zeros((4, 4))
    for i in range(3):
        t[i, i + 1] = t[i + 1, i] = 1e-310
    cases.append(("denormal_offdiag_zero_diag_4", t, True))
    t = np.diag([1.0, 1.0 + 1e-14, 1.0 + 2e-14, 3.0])
    for i in range(3):
        t[i, i + 1] = t[i + 1, i] = 1e-15
    cases.append(("near_degenerate_cluster_4", t, True))
    w = np.diag(np.abs(np.arange(-10.0, 11.0)))
    for i in range(20):
        w[i, i + 1] = w[i + 1, i] = 1.0
    cases.append(("wilkinson_plus_21", w, False))
    xt = pm1_opposite_cols(40, 8)
    xtc = xt - xt.mean(axis=0)
    cases.append(("pm1_opposite_cols_centered_gram_16", xtc.T @ xtc, False))
    # Review counter-examples: independent small block after / before a large
    # one, at several scales, and subnormal spectra.
    for small_first in (False, True):
        for e in (-600, 0, 600):
            order = "first" if small_first else "last"
            cases.append(
                (f"small_block_{order}_2e{e}", astra_sym_block(small_first, e), True)
            )
    for v in (1e-308, 1e-310):
        cases.append((f"subnormal_pair_{v:g}", np.array([[0.0, v], [v, 0.0]]), True))
    return cases


def svd_cases() -> list[tuple[str, np.ndarray, bool]]:
    h2 = np.array([[1.0, 1.0], [1.0, -1.0]])
    cases = [
        (
            "pm1_opposite_3x3",
            np.array([[-1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
            False,
        ),
        ("pm1_opposite_4x6", pm1_opposite_rows(4, 6), False),
        (
            "pm1_opposite_centered_4x6",
            pm1_opposite_rows(4, 6) - pm1_opposite_rows(4, 6).mean(axis=0),
            False,
        ),
        (
            "pm1_opposite_centered_4x8",
            pm1_opposite_rows(4, 8) - pm1_opposite_rows(4, 8).mean(axis=0),
            False,
        ),
        ("pm1_opposite_cols_10x6", pm1_opposite_cols(10, 3), False),
        ("hadamard_4_equal_sv", np.kron(h2, h2), True),
        (
            "hadamard_4_near_equal_sv",
            np.kron(h2, h2) + np.diag([1e-12, 0.0, 0.0, 0.0]),
            True,
        ),
        (
            "duplicate_columns_5x4",
            np.array(
                [
                    [1.0, 2.0, 1.0, 2.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [3.0, -1.0, 3.0, -1.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [-1.0, 0.5, -1.0, 0.5],
                ]
            ),
            False,
        ),
        ("diagonal_repeated_4x4", np.diag([3.0, 3.0, 1.0, 0.0]), True),
        (
            "tiny_offdiag_3x3",
            np.array([[1.0, 1e-120, 0.0], [1e-120, 1.0, 1e-120], [0.0, 1e-120, 1.0]]),
            True,
        ),
        ("weak_tall_gram_3x3", ASTRA_TALL, True),
    ]
    for weak_first in (False, True):
        for e in (-600, 0, 600):
            order = "first" if weak_first else "last"
            cases.append(
                (f"weak_block_{order}_2e{e}", astra_weak_block(weak_first, e), True)
            )
    return cases


def bidiag(n: int, t) -> list[list]:
    """B_n(t): t on the diagonal, 1 on the superdiagonal (det = t^n)."""
    return [
        [t if i == j else (1 if j == i + 1 else 0) for j in range(n)] for i in range(n)
    ]


def stacked_opposite(a: list[list], scale) -> list[list]:
    """scale * [A; -A]: exactly centred, so the PCA sees it unchanged."""
    return [[scale * v for v in row] for row in a] + [
        [-scale * v for v in row] for row in a
    ]


def kron_b6(t) -> list[list]:
    """[A; -A], A = B_6(t) (x) [[1,3/4],[1/2,1]], columns reversed."""
    kron = [[1, mp.mpf(3) / 4], [mp.mpf(1) / 2, 1]]
    b6 = bidiag(6, t)
    a12 = [
        [b6[i // 2][j // 2] * kron[i % 2][j % 2] for j in range(12)] for i in range(12)
    ]
    return [list(reversed(row)) for row in stacked_opposite(a12, 1)]


def weak_row(t) -> list[list]:
    """[A; -A], A = [[-1,1,-1],[-1,-1,-1],[1,-1,1],[-t,t,t]]."""
    a = [[-1, 1, -1], [-1, -1, -1], [1, -1, 1], [-t, t, t]]
    return stacked_opposite(a, 1)


def multiprecision_cases() -> list[tuple[str, list[list], bool]]:
    """Review families whose small singular values come from cancellation.

    The flag tells whether the matrix lies in the preflight PCA domain
    (nonzero |x| in [2^-128, 2^128], max/min nonzero |x| <= 2^32).
    """
    two = mp.mpf(2)
    return [
        # Review round 3, outside the preflight domain (spreads 2^63, 2^60,
        # 2^51): kernel tests only; the preflight PCA refuses them.
        ("b8_plateau", stacked_opposite(bidiag(8, two**-63), two**63), False),
        ("b8_tiny", stacked_opposite(bidiag(8, two**-60), two**-67), False),
        ("b6_kron", kron_b6(two**-50), False),
        # Same families inside the domain (spread <= 2^31).
        ("b8_plateau_d", stacked_opposite(bidiag(8, two**-30), 1), True),
        ("b8_tiny_d", stacked_opposite(bidiag(8, two**-30), two**-97), True),
        ("b6_kron_d", kron_b6(two**-30), True),
        # Weak but resolved third variance (4t^2/7 ~ 3e-13 lambda_1).
        ("weak_resolved_d", weak_row(two**-20), True),
    ]


def multiprecision_singular_values(rows: list[list]) -> list:
    return sorted(mp.svd_r(mp.matrix(rows), compute_uv=False), reverse=True)


def fmt_mp(values) -> str:
    """Exact-to-double rendering (values below the double range become 0)."""
    return ", ".join(repr(float(v)) for v in values)


def fmt(values) -> str:
    return ", ".join(repr(float(v)) for v in np.asarray(values, dtype=float).ravel())


def lapack_backend() -> str:
    lapack = np.show_config(mode="dicts")["Build Dependencies"]["lapack"]
    return f"{lapack['name']} {lapack['version']}"


def main() -> None:
    lines = [
        "// GENERATED by parity/scripts/svd_degenerate_reference.py -- do not edit.",
        "// Test-only NumPy/LAPACK goldens for test_internal_svd_degenerate.cpp.",
        (
            f"// numpy {np.__version__} ({lapack_backend()}); python {platform.python_version()};"
            " numpy.linalg eigh -> dsyevd, svd -> dgesdd."
        ),
        f'static const char* const kSvdGoldenNumpyVersion = "{np.__version__}";',
        "",
    ]
    sym_entries = []
    for idx, (name, a, relative) in enumerate(sym_cases()):
        w, v = np.linalg.eigh(a)
        cl = clusters(w, relative)
        proj = [v[:, s:e] @ v[:, s:e].T for s, e in cl]
        n = a.shape[0]
        lines.append(f"static const double kSymA{idx}[] = {{{fmt(a)}}};")
        lines.append(f"static const double kSymW{idx}[] = {{{fmt(w)}}};")
        lines.append(
            f"static const int kSymC{idx}[] = {{{', '.join(str(s) for s, _ in cl)}, {n}}};"
        )
        if n <= PROJECTOR_MAX_N:
            lines.append(
                f"static const double kSymP{idx}[] = {{{fmt(np.concatenate([p.ravel() for p in proj]))}}};"
            )
        p_ref = f"kSymP{idx}" if n <= PROJECTOR_MAX_N else "nullptr"
        sym_entries.append(
            f'    {{"{name}", {n}, kSymA{idx}, kSymW{idx}, {len(cl)}, kSymC{idx}, {p_ref}, {int(relative)}}},'
        )
    lines.append("")
    lines.append(
        "struct SymGolden { const char* name; int n; const double* A; const double* evals_asc;"
    )
    lines.append("                   int n_clusters; const int* cluster_start;")
    lines.append(
        "                   const double* projectors;  /* nullptr when n > 8 */ int relative; };"
    )
    lines.append("static const SymGolden kSymGoldens[] = {")
    lines.extend(sym_entries)
    lines.append("};")
    lines.append("")

    svd_entries = []
    for idx, (name, a, relative) in enumerate(svd_cases()):
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        cut = 0.0 if relative else RANK_RTOL * s[0]
        rank = int(np.sum(s > cut))
        row_proj = vt[:rank].T @ vt[:rank]
        col_proj = u[:, :rank] @ u[:, :rank].T
        m, n = a.shape
        lines.append(f"static const double kSvdA{idx}[] = {{{fmt(a)}}};")
        lines.append(f"static const double kSvdS{idx}[] = {{{fmt(s)}}};")
        lines.append(f"static const double kSvdRow{idx}[] = {{{fmt(row_proj)}}};")
        lines.append(f"static const double kSvdCol{idx}[] = {{{fmt(col_proj)}}};")
        svd_entries.append(
            f'    {{"{name}", {m}, {n}, kSvdA{idx}, kSvdS{idx}, {rank}, kSvdRow{idx}, kSvdCol{idx}, {int(relative)}}},'
        )
    lines.append("")
    lines.append(
        "struct SvdGolden { const char* name; int m; int n; const double* A; const double* sv_desc;"
    )
    lines.append(
        "                   int rank; const double* row_projector; const double* col_projector;"
    )
    lines.append("                   int relative; };")
    lines.append("static const SvdGolden kSvdGoldens[] = {")
    lines.extend(svd_entries)
    lines.append("};")
    lines.append("")

    # PCA through n4m_pca_fit's three dispatch paths (Jacobi, dual-wide QL,
    # tall-Gram QL). Matrices are rebuilt in C++ from the same formulas.
    pca_entries = []
    for idx, (name, pattern, rows, cols, k, relative) in enumerate(
        [
            ("pca_jacobi_pm1_opposite_4x8", 0, 4, 8, 4, False),
            ("pca_dual_wide_pm1_opposite_4x8192", 0, 4, 8192, 3, False),
            ("pca_tall_gram_pm1_opposite_cols_40x16", 1, 40, 16, 12, False),
            # [A; -A; 26 zero rows], A = ASTRA_TALL: decoupled weak variance.
            ("pca_tall_gram_weak_block_32x3", 2, 32, 3, 2, True),
        ]
    ):
        if pattern == 0:
            x = pm1_opposite_rows(rows, cols)
        elif pattern == 1:
            x = pm1_opposite_cols(rows, cols // 2)
        else:
            x = np.vstack([ASTRA_TALL, -ASTRA_TALL, np.zeros((rows - 6, cols))])
        xc = x - x.mean(axis=0)
        ev = (np.linalg.svd(xc, compute_uv=False) ** 2 / (rows - 1))[:k]
        lines.append(f"static const double kPcaEv{idx}[] = {{{fmt(ev)}}};")
        pca_entries.append(
            f'    {{"{name}", {pattern}, {rows}, {cols}, {k}, kPcaEv{idx}, {int(relative)}}},'
        )
    lines.append("")
    lines.append(
        "struct PcaGolden { const char* name; int pattern; int rows; int cols; int k;"
    )
    lines.append("                   const double* explained_var; int relative; };")
    lines.append("static const PcaGolden kPcaGoldens[] = {")
    lines.extend(pca_entries)
    lines.append("};")
    # Multiprecision (mpmath, 600 digits) references: matrix entries (exact
    # dyadic values), singular values and centred variances sigma^2/(n-1).
    mp.mp.dps = 600
    lines.append(f"// mpmath {mp.__version__}, {mp.mp.dps} digits.")
    mp_entries = []
    for idx, (name, rows, in_domain) in enumerate(multiprecision_cases()):
        s = multiprecision_singular_values(rows)
        var = [v * v / (len(rows) - 1) for v in s]
        m, n = len(rows), len(rows[0])
        lines.append(
            f"static const double kMpA{idx}[] = {{{fmt_mp(v for r in rows for v in r)}}};"
        )
        lines.append(f"static const double kMpS{idx}[] = {{{fmt_mp(s)}}};")
        lines.append(f"static const double kMpVar{idx}[] = {{{fmt_mp(var)}}};")
        mp_entries.append(
            f'    {{"{name}", {m}, {n}, kMpA{idx}, kMpS{idx}, kMpVar{idx}, {int(in_domain)}}},'
        )
    lines.append(
        "struct MpGolden { const char* name; int m; int n; const double* A; const double* sv_desc;"
    )
    lines.append("                  const double* variances; int preflight_domain; };")
    lines.append("static const MpGolden kMpGoldens[] = {")
    lines.extend(mp_entries)
    lines.append("};")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
