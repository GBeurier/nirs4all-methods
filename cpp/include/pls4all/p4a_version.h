/* SPDX-License-Identifier: CeCILL-2.1 */
/*
 * pls4all — compile-time ABI and project version macros.
 *
 * These macros are the canonical source of truth for the *header* version.
 * The *library* version is queried at runtime via the p4a_get_abi_version_*
 * and p4a_get_version_string functions declared in p4a.h. Bindings should
 * call p4a_check_abi_compatibility(P4A_ABI_VERSION_MAJOR, P4A_ABI_VERSION_MINOR)
 * to detect header / library skew before using libp4a.
 *
 * ABI versioning rules (semver of the C ABI surface, independent of the
 * project's semver):
 *   - MAJOR  : breaking change (signature, layout, removed symbol).
 *   - MINOR  : additive change (new function, new enum value).
 *   - PATCH  : bugfix only (no surface change).
 */
#ifndef PLS4ALL_P4A_VERSION_H
#define PLS4ALL_P4A_VERSION_H

#define P4A_ABI_VERSION_MAJOR 1
#define P4A_ABI_VERSION_MINOR 0
#define P4A_ABI_VERSION_PATCH 0
#define P4A_ABI_VERSION_INT  (P4A_ABI_VERSION_MAJOR * 10000 + \
                              P4A_ABI_VERSION_MINOR * 100   + \
                              P4A_ABI_VERSION_PATCH)

#define P4A_PROJECT_VERSION_MAJOR  0
#define P4A_PROJECT_VERSION_MINOR  46
#define P4A_PROJECT_VERSION_PATCH  0
#define P4A_PROJECT_VERSION_STRING "0.46.0"

/* Canonical error-buffer capacity for p4a_context_t. The 4 KiB figure is part
 * of the ABI contract — see docs/architecture/error_model.md. Bindings may
 * allocate a small copy of this size when they need to retain a last_error
 * value across subsequent p4a_* calls. */
#define P4A_ERROR_BUFFER_BYTES 4096

/* Serialization magic + format version. Kept here so consumers can write
 * import-compatibility checks without including the full p4a.h surface. */
#define P4A_SERIALIZATION_MAGIC          "P4AM"
#define P4A_SERIALIZATION_FORMAT_VERSION 1u

#endif /* PLS4ALL_P4A_VERSION_H */
