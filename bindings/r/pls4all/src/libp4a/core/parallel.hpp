// SPDX-License-Identifier: CECILL-2.1
//
// Internal OpenMP dispatch macros (Phase 44).
//
// Annotates inner-loop work in cpp/src/core/model.cpp without
// littering the call sites with `#ifdef _OPENMP` blocks.
//
//   N4M_PARALLEL_FOR_STATIC
//     -> `#pragma omp parallel for schedule(static)` when
//        N4M_USE_OPENMP is defined.
//     -> nothing when it is not (the loop runs sequentially and
//        the compiled binary is bit-identical to pre-Phase-44).
//
// The schedule is `static` everywhere: every annotated loop has
// equal work per iteration (rank-1 deflation, dot-product loading,
// fill_prediction row), so static partitioning is the lowest-
// overhead choice. If a future site needs `dynamic` or `guided`,
// add a sibling macro rather than extending this one — keeping
// the surface narrow keeps the audit story short.

#ifndef PLS4ALL_CORE_PARALLEL_HPP
#define PLS4ALL_CORE_PARALLEL_HPP

#if defined(N4M_USE_OPENMP)
#  include <omp.h>
#  if defined(_MSC_VER)
//   MSVC's pragma-in-macro mechanism. Recent VS toolchains (16.6+) also
//   accept _Pragma, but __pragma is the MSVC-native form and works on
//   older releases.
#    define N4M_PARALLEL_FOR_STATIC __pragma(omp parallel for schedule(static))
#  else
#    define N4M_PARALLEL_FOR_STATIC _Pragma("omp parallel for schedule(static)")
#  endif
#else
#  define N4M_PARALLEL_FOR_STATIC
#endif

#endif  // PLS4ALL_CORE_PARALLEL_HPP
