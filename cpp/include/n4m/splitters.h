/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * cpp/include/n4m/splitters.h — M6 scaffold.
 *
 * Category: splitters
 * Source of truth (post-M5 rename): donor's n4m.h §splitter
 * Contents: KennardStone, SPXY, SPXYFold, SPXYGFold, KBinsStratified, KMeans, SystematicCircular, BinnedStratGroupKFold, SplitSplitter
 *
 * STATUS: scaffold only. The per-category declarations will be lifted
 * from the renamed umbrella headers in the focused M5+M6 session per
 * /home/delete/nirs4all/M5/STATUS.md and /home/delete/nirs4all/M6/STATUS.md.
 *
 * After M5+M6 lands:
 *   - cpp/include/n4m/n4m.h is the umbrella header that #includes each
 *     of these per-category headers, deduplicating common-core
 *     declarations (those live in context.h alone).
 *   - cpp/include/pls4all/p4a.h is deleted (its content was the
 *     PLS surface that moves into models.h, selection.h, diagnostics.h,
 *     aom_pop.h, transfer.h).
 *
 * Until that lands, the public ABI surface remains the existing
 * pls4all/p4a.h + n4m/n4m.h pair.
 */
#ifndef N4M_SPLITTERS_H_INCLUDED
#define N4M_SPLITTERS_H_INCLUDED

#include "n4m/n4m.h"  /* TEMP: pull existing umbrella. M6 final step removes this. */

#ifdef __cplusplus
extern "C" {
#endif

/* TODO: lift the splitters category declarations from the umbrella here. */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* N4M_SPLITTERS_H_INCLUDED */
