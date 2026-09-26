/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * C translation of FITPACK curfit/fpcurf/fpback/fpbspl/fpdisc/fpgivs/fpknot/
 * fprati/fprota/splev (P. Dierckx), from SciPy 1.17.1
 * scipy/interpolate/fitpack/. Copyright (c) 2001-2002 Enthought, Inc.
 * 2003, SciPy Developers; see LICENSE.scipy.txt in this directory.
 *
 * The translation keeps Fortran's 1-based index variables and arithmetic
 * order so results match the Fortran build bit for bit: every access is
 * written as array[index - 1], and 2-D column-major arrays go through FP2.
 */
#include "fitpack.h"

#include <math.h>

#define FP2(arr, ld, i, j) (arr)[((i) - 1) + ((j) - 1) * (ld)]

static void fpbspl(const double* t, int k, double x, int l, double* h) {
    double hh[19];
    h[0] = 1.0;
    for (int j = 1; j <= k; ++j) {
        for (int i = 1; i <= j; ++i) hh[i - 1] = h[i - 1];
        h[0] = 0.0;
        for (int i = 1; i <= j; ++i) {
            const int li = l + i;
            const int lj = li - j;
            if (t[li - 1] == t[lj - 1]) {
                h[i] = 0.0;
                continue;
            }
            const double f = hh[i - 1] / (t[li - 1] - t[lj - 1]);
            h[i - 1] = h[i - 1] + f * (t[li - 1] - x);
            h[i] = f * (x - t[lj - 1]);
        }
    }
}

static void fpgivs(double piv, double* ww, double* co, double* si) {
    const double store = fabs(piv);
    double dd;
    if (store >= *ww) {
        const double r = *ww / piv;
        dd = store * sqrt(1.0 + r * r);
    } else {
        const double r = piv / *ww;
        dd = *ww * sqrt(1.0 + r * r);
    }
    *co = *ww / dd;
    *si = piv / dd;
    *ww = dd;
}

static void fprota(double co, double si, double* a, double* b) {
    const double stor1 = *a;
    const double stor2 = *b;
    *b = co * stor2 + si * stor1;
    *a = co * stor1 - si * stor2;
}

/* Solves a*c = z, a upper triangular of bandwidth k; z and c may alias. */
static void fpback(const double* a, const double* z, int n, int k, double* c,
                   int nest) {
    const int k1 = k - 1;
    c[n - 1] = z[n - 1] / FP2(a, nest, n, 1);
    int i = n - 1;
    for (int j = 2; j <= n; ++j) {
        double store = z[i - 1];
        const int i1 = j <= k1 ? j - 1 : k1;
        int m = i;
        for (int l = 1; l <= i1; ++l) {
            m = m + 1;
            store = store - c[m - 1] * FP2(a, nest, i, l + 1);
        }
        c[i - 1] = store / FP2(a, nest, i, 1);
        i = i - 1;
    }
}

static void fpdisc(const double* t, int n, int k2, double* b, int nest) {
    double h[12];
    const int k1 = k2 - 1;
    const int k = k1 - 1;
    const int nk1 = n - k1;
    const double an = nk1 - k;
    const double fac = an / (t[nk1] - t[k1 - 1]);
    for (int l = k2; l <= nk1; ++l) {
        const int lmk = l - k1;
        for (int j = 1; j <= k1; ++j) {
            const int ik = j + k1;
            const int lj = l + j;
            const int lk = lj - k2;
            h[j - 1] = t[l - 1] - t[lk - 1];
            h[ik - 1] = t[l - 1] - t[lj - 1];
        }
        int lp = lmk;
        for (int j = 1; j <= k2; ++j) {
            int jk = j;
            double prod = h[j - 1];
            for (int i = 1; i <= k; ++i) {
                jk = jk + 1;
                prod = prod * h[jk - 1] * fac;
            }
            const int lk = lp + k1;
            FP2(b, nest, lmk, j) = (t[lk - 1] - t[lp - 1]) / prod;
            lp = lp + 1;
        }
    }
}

static void fpknot(const double* x, double* t, int* n, double* fpint,
                   int* nrdata, int* nrint, int istart) {
    const int k = (*n - *nrint - 1) / 2;
    int iserr = 1;
    double fpmax = 0.0;
    int number = 0;
    int maxpt = 0;
    int maxbeg = 0;
    int jbegin = istart;
    for (int j = 1; j <= *nrint; ++j) {
        const int jpoint = nrdata[j - 1];
        if (!(fpmax >= fpint[j - 1] || jpoint == 0)) {
            iserr = 0;
            fpmax = fpint[j - 1];
            number = j;
            maxpt = jpoint;
            maxbeg = jbegin;
        }
        jbegin = jbegin + jpoint + 1;
    }
    if (!iserr) {
        const int ihalf = maxpt / 2 + 1;
        const int nrx = maxbeg + ihalf;
        const int next = number + 1;
        for (int j = next; j <= *nrint; ++j) {
            const int jj = next + *nrint - j;
            fpint[jj] = fpint[jj - 1];
            nrdata[jj] = nrdata[jj - 1];
            const int jk = jj + k;
            t[jk] = t[jk - 1];
        }
        nrdata[number - 1] = ihalf - 1;
        nrdata[next - 1] = maxpt - ihalf;
        const double am = maxpt;
        double an = nrdata[number - 1];
        fpint[number - 1] = fpmax * an / am;
        an = nrdata[next - 1];
        fpint[next - 1] = fpmax * an / am;
        t[next + k - 1] = x[nrx - 1];
    }
    *n = *n + 1;
    *nrint = *nrint + 1;
}

static double fprati(double* p1, double* f1, double p2, double f2, double* p3,
                     double* f3) {
    double p;
    if (*p3 > 0.0) {
        const double h1 = *f1 * (f2 - *f3);
        const double h2 = f2 * (*f3 - *f1);
        const double h3 = *f3 * (*f1 - f2);
        p = -(*p1 * p2 * h3 + p2 * *p3 * h1 + *p3 * *p1 * h2) /
            (*p1 * h1 + p2 * h2 + *p3 * h3);
    } else {
        p = (*p1 * (*f1 - *f3) * f2 - p2 * (f2 - *f3) * *f1) /
            ((*f1 - f2) * *f3);
    }
    if (f2 < 0.0) {
        *p3 = p2;
        *f3 = f2;
    } else {
        *p1 = p2;
        *f1 = f2;
    }
    return p;
}

/* Interior knots of the interpolating spline (fpcurf label 10). */
static void fpcurf_interp_knots(const double* x, int m, int k, double* t) {
    const int k1 = k + 1;
    const int mk1 = m - k1;
    const int k3 = k / 2;
    int i = k1 + 1;
    int j = k3 + 2;
    for (int l = 1; l <= mk1; ++l) {
        t[i - 1] = k3 * 2 == k ? (x[j - 1] + x[j - 2]) * 0.5 : x[j - 1];
        i = i + 1;
        j = j + 1;
    }
}

static int fpcurf(int iopt, const double* x, const double* y, const double* w,
                  int m, double xb, double xe, int k, double s, int nest,
                  double tol, int maxit, int k1, int k2, int* n_io, double* t,
                  double* c, double* fp_out, double* fpint, double* z,
                  double* a, double* b, double* g, double* q, int* nrdata,
                  int ier) {
    const double con1 = 0.1;
    const double con9 = 0.9;
    const double con4 = 0.04;
    double h[7];
    int n = *n_io;
    double fp = 0.0;
    double fp0 = 0.0;
    double fpold = 0.0;
    double fpms = 0.0;
    int nplus = 0;
    int nk1 = 0;
    const int nmin = 2 * k1;
    const double acc = tol * s;
    const int nmax = m + k1;

    if (iopt != 0 && n != nmin) {
        fp0 = fpint[n - 1];
        fpold = fpint[n - 2];
        nplus = nrdata[n - 1];
        if (fp0 > s) goto knot_loop;
    }
    n = nmin;
    fpold = 0.0;
    nplus = 0;
    nrdata[0] = m - 2;

knot_loop:
    for (int iter = 1; iter <= m; ++iter) {
        if (n == nmin) ier = -2;
        int nrint = n - nmin + 1;
        nk1 = n - k1;
        int i = n;
        for (int j = 1; j <= k1; ++j) {
            t[j - 1] = xb;
            t[i - 1] = xe;
            i = i - 1;
        }
        fp = 0.0;
        for (i = 1; i <= nk1; ++i) {
            z[i - 1] = 0.0;
            for (int j = 1; j <= k1; ++j) FP2(a, nest, i, j) = 0.0;
        }
        int l = k1;
        for (int it = 1; it <= m; ++it) {
            const double xi = x[it - 1];
            const double wi = w[it - 1];
            double yi = y[it - 1] * wi;
            while (!(xi < t[l] || l == nk1)) l = l + 1;
            fpbspl(t, k, xi, l, h);
            for (i = 1; i <= k1; ++i) {
                FP2(q, m, it, i) = h[i - 1];
                h[i - 1] = h[i - 1] * wi;
            }
            int j = l - k1;
            for (i = 1; i <= k1; ++i) {
                j = j + 1;
                const double piv = h[i - 1];
                if (piv == 0.0) continue;
                double co;
                double si;
                fpgivs(piv, &FP2(a, nest, j, 1), &co, &si);
                fprota(co, si, &yi, &z[j - 1]);
                int i2 = 1;
                for (int i1 = i + 1; i1 <= k1; ++i1) {
                    i2 = i2 + 1;
                    fprota(co, si, &h[i1 - 1], &FP2(a, nest, j, i2));
                }
            }
            fp = fp + yi * yi;
        }
        if (ier == -2) fp0 = fp;
        fpint[n - 1] = fp0;
        fpint[n - 2] = fpold;
        nrdata[n - 1] = nplus;
        fpback(a, z, nk1, k1, c, nest);
        fpms = fp - s;
        if (fabs(fpms) < acc) goto done;
        if (fpms < 0.0) goto smoothing;
        if (n == nmax) {
            ier = -1;
            goto done;
        }
        if (n == nest) {
            ier = 1;
            goto done;
        }
        if (ier != 0) {
            nplus = 1;
            ier = 0;
        } else {
            int npl1 = nplus * 2;
            const double rn = nplus;
            if (fpold - fp > acc) {
                /* Fortran truncates the real ratio; clamping at nplus*2
                 * keeps the min0 below unchanged and avoids int overflow. */
                const double ratio = rn * fpms / (fpold - fp);
                npl1 = ratio < (double)npl1 ? (int)ratio : npl1;
            }
            int lower = nplus / 2;
            if (npl1 > lower) lower = npl1;
            if (lower < 1) lower = 1;
            nplus = nplus * 2 < lower ? nplus * 2 : lower;
        }
        fpold = fp;
        double fpart = 0.0;
        i = 1;
        l = k2;
        int new_knot = 0;
        for (int it = 1; it <= m; ++it) {
            if (!(x[it - 1] < t[l - 1] || l > nk1)) {
                new_knot = 1;
                l = l + 1;
            }
            double term = 0.0;
            int l0 = l - k2;
            for (int j = 1; j <= k1; ++j) {
                l0 = l0 + 1;
                term = term + c[l0 - 1] * FP2(q, m, it, j);
            }
            const double r = w[it - 1] * (term - y[it - 1]);
            term = r * r;
            fpart = fpart + term;
            if (new_knot == 0) continue;
            const double store = term * 0.5;
            fpint[i - 1] = fpart - store;
            i = i + 1;
            fpart = store;
            new_knot = 0;
        }
        fpint[nrint - 1] = fpart;
        for (l = 1; l <= nplus; ++l) {
            fpknot(x, t, &n, fpint, nrdata, &nrint, 1);
            if (n == nmax) {
                fpcurf_interp_knots(x, m, k, t);
                goto knot_loop;
            }
            if (n == nest) break;
        }
    }

smoothing:
    if (ier == -2) goto done;
    fpdisc(t, n, k2, b, nest);
    {
        double p1 = 0.0;
        double f1 = fp0 - s;
        double p3 = -1.0;
        double f3 = fpms;
        double p = 0.0;
        for (int i = 1; i <= nk1; ++i) p = p + FP2(a, nest, i, 1);
        const double rn = nk1;
        p = rn / p;
        int ich1 = 0;
        int ich3 = 0;
        const int n8 = n - nmin;
        for (int iter = 1; iter <= maxit; ++iter) {
            const double pinv = 1.0 / p;
            for (int i = 1; i <= nk1; ++i) {
                c[i - 1] = z[i - 1];
                FP2(g, nest, i, k2) = 0.0;
                for (int j = 1; j <= k1; ++j)
                    FP2(g, nest, i, j) = FP2(a, nest, i, j);
            }
            for (int it = 1; it <= n8; ++it) {
                for (int i = 1; i <= k2; ++i) h[i - 1] = FP2(b, nest, it, i) * pinv;
                double yi = 0.0;
                for (int j = it; j <= nk1; ++j) {
                    const double piv = h[0];
                    double co;
                    double si;
                    fpgivs(piv, &FP2(g, nest, j, 1), &co, &si);
                    fprota(co, si, &yi, &c[j - 1]);
                    if (j == nk1) break;
                    const int i2 = j > n8 ? nk1 - j : k1;
                    for (int i = 1; i <= i2; ++i) {
                        const int i1 = i + 1;
                        fprota(co, si, &h[i1 - 1], &FP2(g, nest, j, i1));
                        h[i - 1] = h[i1 - 1];
                    }
                    h[i2] = 0.0;
                }
            }
            fpback(g, c, nk1, k2, c, nest);
            fp = 0.0;
            int l = k2;
            for (int it = 1; it <= m; ++it) {
                if (!(x[it - 1] < t[l - 1] || l > nk1)) l = l + 1;
                int l0 = l - k2;
                double term = 0.0;
                for (int j = 1; j <= k1; ++j) {
                    l0 = l0 + 1;
                    term = term + c[l0 - 1] * FP2(q, m, it, j);
                }
                const double r = w[it - 1] * (term - y[it - 1]);
                fp = fp + r * r;
            }
            fpms = fp - s;
            if (fabs(fpms) < acc) goto done;
            if (iter == maxit) {
                ier = 3;
                goto done;
            }
            const double p2 = p;
            const double f2 = fpms;
            if (ich3 == 0) {
                if (!(f2 - f3 > acc)) {
                    /* initial p too large */
                    p3 = p2;
                    f3 = f2;
                    p = p * con4;
                    if (p <= p1) p = p1 * con9 + p2 * con1;
                    continue;
                }
                if (f2 < 0.0) ich3 = 1;
            }
            if (ich1 == 0) {
                if (!(f1 - f2 > acc)) {
                    /* initial p too small */
                    p1 = p2;
                    f1 = f2;
                    p = p / con4;
                    if (p3 < 0.0) continue;
                    if (p >= p3) p = p2 * con1 + p3 * con9;
                    continue;
                }
                if (f2 > 0.0) ich1 = 1;
            }
            if (f2 >= f1 || f2 <= f3) {
                ier = 2;
                goto done;
            }
            p = fprati(&p1, &f1, p2, f2, &p3, &f3);
        }
    }

done:
    *n_io = n;
    *fp_out = fp;
    return ier;
}

int n4m_fitpack_curfit(int iopt, int m, const double* x, const double* y,
                       const double* w, double xb, double xe, int k, double s,
                       int nest, int* n, double* t, double* c, double* fp,
                       double* wrk, int* iwrk) {
    const int k1 = k + 1;
    const int k2 = k1 + 1;
    double* fpint = wrk;
    double* z = fpint + nest;
    double* a = z + nest;
    double* b = a + nest * k1;
    double* g = b + nest * k2;
    double* q = g + nest * k2;
    /* curfit presets ier = 10 before its input checks; fpcurf reads it. */
    return fpcurf(iopt, x, y, w, m, xb, xe, k, s, nest, 0.001, 20, k1, k2, n,
                  t, c, fp, fpint, z, a, b, g, q, iwrk, 10);
}

void n4m_fitpack_splev(const double* t, int n, const double* c, int k,
                       const double* x, double* y, int m) {
    double h[20];
    const int k1 = k + 1;
    const int k2 = k1 + 1;
    const int nk1 = n - k1;
    int l = k1;
    int l1 = l + 1;
    for (int i = 1; i <= m; ++i) {
        const double arg = x[i - 1];
        while (!(arg >= t[l - 1] || l1 == k2)) {
            l1 = l;
            l = l - 1;
        }
        while (!(arg < t[l1 - 1] || l == nk1)) {
            l = l1;
            l1 = l + 1;
        }
        fpbspl(t, k, arg, l, h);
        double sp = 0.0;
        int ll = l - k1;
        for (int j = 1; j <= k1; ++j) {
            ll = ll + 1;
            sp = sp + c[ll - 1] * h[j - 1];
        }
        y[i - 1] = sp;
    }
}
