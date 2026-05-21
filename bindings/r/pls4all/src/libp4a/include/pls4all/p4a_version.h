/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * pls4all — compile-time ABI and project version macros.
 *
 * These macros are the canonical source of truth for the *header* version.
 * The *library* version is queried at runtime via the n4m_get_abi_version_*
 * and n4m_get_version_string functions declared in p4a.h. Bindings should
 * call n4m_check_abi_compatibility(N4M_ABI_VERSION_MAJOR, N4M_ABI_VERSION_MINOR)
 * to detect header / library skew before using libn4m.
 *
 * ABI versioning rules (semver of the C ABI surface, independent of the
 * project's semver):
 *   - MAJOR  : breaking change (signature, layout, removed symbol).
 *   - MINOR  : additive change (new function, new enum value).
 *   - PATCH  : bugfix only (no surface change).
 */
#ifndef PLS4ALL_P4A_VERSION_H
#define PLS4ALL_P4A_VERSION_H

#define N4M_ABI_VERSION_MAJOR 1
#define N4M_ABI_VERSION_MINOR 16
#define N4M_ABI_VERSION_PATCH 0
#define N4M_ABI_VERSION_INT  (N4M_ABI_VERSION_MAJOR * 10000 + \
                              N4M_ABI_VERSION_MINOR * 100   + \
                              N4M_ABI_VERSION_PATCH)

#define N4M_PROJECT_VERSION_MAJOR  0
#define N4M_PROJECT_VERSION_MINOR  97
#define N4M_PROJECT_VERSION_PATCH  3
#define N4M_PROJECT_VERSION_STRING "0.97.3"

/* Canonical error-buffer capacity for n4m_context_t. The 4 KiB figure is part
 * of the ABI contract — see docs/architecture/error_model.md. Bindings may
 * allocate a small copy of this size when they need to retain a last_error
 * value across subsequent n4m_* calls. */
#define N4M_ERROR_BUFFER_BYTES 4096

/* Serialization magic + format version. Kept here so consumers can write
 * import-compatibility checks without including the full p4a.h surface. */
#define N4M_SERIALIZATION_MAGIC          "P4AM"
#define N4M_SERIALIZATION_FORMAT_VERSION 1u

#endif /* PLS4ALL_P4A_VERSION_H */
