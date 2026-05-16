/* SPDX-License-Identifier: CeCILL-2.1
 *
 * JNI shim that bridges io.github.pls4all.Pls4all to the libp4a
 * public C ABI. Compiles into libp4a_jni.so which links against
 * libp4a (the same .so used by every other binding).
 *
 * The Java arrays are row-major, same as the C ABI, so we can
 * use GetPrimitiveArrayCritical to pin them without copying.
 */

#include "io_github_pls4all_Pls4all.h"
#include "pls4all/p4a.h"

#include <stdlib.h>
#include <string.h>

JNIEXPORT jstring JNICALL
Java_io_github_pls4all_Pls4all_version(JNIEnv *env, jclass cls) {
    (void)cls;
    const char *v = p4a_get_version_string();
    return (*env)->NewStringUTF(env, v == NULL ? "" : v);
}

JNIEXPORT jintArray JNICALL
Java_io_github_pls4all_Pls4all_abiVersion(JNIEnv *env, jclass cls) {
    (void)cls;
    jint vals[3];
    vals[0] = (jint)p4a_get_abi_version_major();
    vals[1] = (jint)p4a_get_abi_version_minor();
    vals[2] = (jint)p4a_get_abi_version_patch();
    jintArray out = (*env)->NewIntArray(env, 3);
    if (out == NULL) {
        return NULL;
    }
    (*env)->SetIntArrayRegion(env, out, 0, 3, vals);
    return out;
}

JNIEXPORT jint JNICALL
Java_io_github_pls4all_Pls4all_plsFitNative(
        JNIEnv *env, jclass cls,
        jdoubleArray j_x, jdoubleArray j_y,
        jint n, jint p, jint q, jint n_components,
        jdoubleArray j_coefs, jdoubleArray j_x_mean,
        jdoubleArray j_y_mean, jdoubleArray j_preds) {
    (void)cls;

    /* Pin the Java buffers. GetPrimitiveArrayCritical avoids the
     * intermediate copy that GetDoubleArrayElements would make; in
     * exchange we must not call back into the JVM, throw, or block
     * between Get and Release. We respect that here. */
    jdouble *x = (*env)->GetPrimitiveArrayCritical(env, j_x, NULL);
    jdouble *y = (*env)->GetPrimitiveArrayCritical(env, j_y, NULL);
    jdouble *coefs = (*env)->GetPrimitiveArrayCritical(env, j_coefs, NULL);
    jdouble *xm = (*env)->GetPrimitiveArrayCritical(env, j_x_mean, NULL);
    jdouble *ym = (*env)->GetPrimitiveArrayCritical(env, j_y_mean, NULL);
    jdouble *pr = (*env)->GetPrimitiveArrayCritical(env, j_preds, NULL);

    p4a_status_t status = -1;
    if (x != NULL && y != NULL && coefs != NULL && xm != NULL &&
        ym != NULL && pr != NULL) {
        status = p4a_pls_fit_simple(
            (const double *)x, (const double *)y,
            (int32_t)n, (int32_t)p, (int32_t)q, (int32_t)n_components,
            (double *)coefs, (double *)xm, (double *)ym, (double *)pr);
    }

    if (pr  != NULL) (*env)->ReleasePrimitiveArrayCritical(env, j_preds,  pr,  0);
    if (ym  != NULL) (*env)->ReleasePrimitiveArrayCritical(env, j_y_mean, ym,  0);
    if (xm  != NULL) (*env)->ReleasePrimitiveArrayCritical(env, j_x_mean, xm,  0);
    if (coefs != NULL) (*env)->ReleasePrimitiveArrayCritical(env, j_coefs, coefs, 0);
    if (y   != NULL) (*env)->ReleasePrimitiveArrayCritical(env, j_y, y, JNI_ABORT);
    if (x   != NULL) (*env)->ReleasePrimitiveArrayCritical(env, j_x, x, JNI_ABORT);

    return (jint)status;
}
