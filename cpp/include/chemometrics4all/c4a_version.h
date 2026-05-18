/* SPDX-License-Identifier: CECILL-2.1 */
/*
 * chemometrics4all — compile-time ABI and project version macros.
 *
 * These macros are the canonical source of truth for the *header* version.
 * The *library* version is queried at runtime via the c4a_get_abi_version_*
 * and c4a_get_version_string functions declared in c4a.h. Bindings should
 * call c4a_check_abi_compatibility(C4A_ABI_VERSION_MAJOR, C4A_ABI_VERSION_MINOR)
 * to detect header / library skew before using libc4a.
 *
 * ABI versioning rules (semver of the C ABI surface, independent of the
 * project's semver):
 *   - MAJOR  : breaking change (signature, layout, removed symbol).
 *   - MINOR  : additive change (new function, new enum value).
 *   - PATCH  : bugfix only (no surface change).
 */
#ifndef CHEMOMETRICS4ALL_C4A_VERSION_H
#define CHEMOMETRICS4ALL_C4A_VERSION_H

#define C4A_ABI_VERSION_MAJOR 1
#define C4A_ABI_VERSION_MINOR 3
#define C4A_ABI_VERSION_PATCH 0
#define C4A_ABI_VERSION_INT  (C4A_ABI_VERSION_MAJOR * 10000 + \
                              C4A_ABI_VERSION_MINOR * 100   + \
                              C4A_ABI_VERSION_PATCH)

#define C4A_PROJECT_VERSION_MAJOR  0
#define C4A_PROJECT_VERSION_MINOR  1
#define C4A_PROJECT_VERSION_PATCH  0
#define C4A_PROJECT_VERSION_STRING "0.1.0"

/* Canonical error-buffer capacity for c4a_context_t. The 4 KiB figure is part
 * of the ABI contract — see docs/architecture/error_model.md. Bindings may
 * allocate a small copy of this size when they need to retain a last_error
 * value across subsequent c4a_* calls. */
#define C4A_ERROR_BUFFER_BYTES 4096

/* Serialization magic + format version. Kept here so consumers can write
 * import-compatibility checks without including the full c4a.h surface. */
#define C4A_SERIALIZATION_MAGIC          "C4AM"
#define C4A_SERIALIZATION_FORMAT_VERSION 1u

#endif /* CHEMOMETRICS4ALL_C4A_VERSION_H */
