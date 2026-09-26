// SPDX-License-Identifier: CECILL-2.1
// Shared one-shot boundary over already-qualified X->X augmentation kernels.
#include <cmath>
#include <cstdint>
#include <limits>

#include "n4m/n4m.h"

namespace {

struct Layout { int32_t count; uint32_t integer_mask; };

Layout layout_for(int32_t kind) noexcept {
    switch (kind) {
        case N4M_AUG_GAUSSIAN_NOISE: return {1, 0};
        case N4M_AUG_MULTIPLICATIVE_NOISE: return {1, 0};
        case N4M_AUG_SPIKE_NOISE: return {4, 0x3};
        case N4M_AUG_HETERO_NOISE: return {2, 0};
        case N4M_AUG_LINEAR_DRIFT: return {4, 0};
        case N4M_AUG_PATH_LENGTH: return {2, 0};
        case N4M_AUG_BAND_PERTURB: return {7, 0x7};
        case N4M_AUG_BAND_MASK: return {5, 0x1f};
        case N4M_AUG_CHANNEL_DROPOUT: return {2, 0x2};
        case N4M_AUG_GAUSS_JITTER: return {3, 0x4};
        case N4M_AUG_UNSHARP_MASK: return {4, 0x8};
        case N4M_AUG_LOCAL_CLIP: return {3, 0x7};
        case N4M_AUG_ROTATE_TRANSLATE: return {2, 0};
        case N4M_AUG_RANDOM_X_OP: return {3, 0x1};
        case N4M_AUG_SCATTER_SIM_MSC: return {4, 0};
        case N4M_AUG_DEAD_BAND: return {6, 0x27};
        case N4M_AUG_BATCH_EFFECT: return {4, 0x8};
        case N4M_AUG_SPLINE_SMOOTHING: return {0, 0};
        case N4M_AUG_SPLINE_X_PERTURB: return {4, 0x1};
        case N4M_AUG_SPLINE_Y_PERTURB: return {2, 0x1};
        case N4M_AUG_SPLINE_X_SIMPLIFY: return {2, 0x3};
        case N4M_AUG_SPLINE_CURVE_SIMPLIFY: return {2, 0x3};
        default: return {-1, 0};
    }
}

template <typename Handle, typename Create, typename Apply, typename Destroy>
n4m_status_t run_one(uint64_t seed, Create create, Apply apply, Destroy destroy,
                     n4m_matrix_view_t X, n4m_matrix_view_t out) {
    n4m_rng_pcg64_state_t* rng = nullptr;
    n4m_status_t status = n4m_rng_pcg64_create(seed, &rng);
    if (status != N4M_OK) return status;
    Handle* handle = nullptr;
    try {
        status = create(&handle, rng);
        if (status == N4M_OK) status = apply(handle, X, out);
    } catch (...) {
        status = N4M_ERR_INTERNAL;
    }
    if (handle != nullptr) destroy(handle);
    n4m_rng_pcg64_destroy(rng);
    return status;
}

}  // namespace

#define AUG_RUN(KIND, HANDLE, NAME, ...) \
    case KIND: \
        return run_one<HANDLE>(seed, \
            [&](HANDLE** h, n4m_rng_pcg64_state_t* rng) { \
                return n4m_augmentation_##NAME##_create(h, rng, __VA_ARGS__); \
            }, n4m_augmentation_##NAME##_apply, \
            n4m_augmentation_##NAME##_destroy, X, out)

extern "C" N4M_API n4m_status_t n4m_augmentation_run(
    int32_t kind, const double* params, int32_t n_params, uint64_t seed,
    n4m_matrix_view_t X, n4m_matrix_view_t out) {
    const Layout layout = layout_for(kind);
    if (layout.count < 0 || n_params != layout.count) return N4M_ERR_INVALID_ARGUMENT;
    if (n_params > 0 && params == nullptr) return N4M_ERR_NULL_POINTER;
    if (X.data == nullptr || out.data == nullptr) return N4M_ERR_NULL_POINTER;
    if (X.dtype != N4M_DTYPE_F64 || out.dtype != N4M_DTYPE_F64)
        return N4M_ERR_DTYPE_MISMATCH;
    if (X.rows < 1 || X.cols < 1 || X.rows != out.rows || X.cols != out.cols)
        return N4M_ERR_SHAPE_MISMATCH;
    if (X.col_stride != 1 || out.col_stride != 1 ||
        X.row_stride != X.cols || out.row_stride != out.cols)
        return N4M_ERR_STRIDE_INVALID;
    if (X.data == out.data) return N4M_ERR_INVALID_ARGUMENT;
    for (int32_t i = 0; i < n_params; ++i) {
        if (!std::isfinite(params[i])) return N4M_ERR_INVALID_ARGUMENT;
        if ((layout.integer_mask & (1u << i)) &&
            (std::trunc(params[i]) != params[i] ||
             params[i] < std::numeric_limits<int32_t>::min() ||
             params[i] > std::numeric_limits<int32_t>::max()))
            return N4M_ERR_INVALID_ARGUMENT;
    }
    const double* p = params;
    switch (kind) {
        AUG_RUN(N4M_AUG_GAUSSIAN_NOISE, n4m_aug_gaussian_noise_handle_t,
                gaussian_noise, p[0]);
        AUG_RUN(N4M_AUG_MULTIPLICATIVE_NOISE, n4m_aug_multiplicative_noise_handle_t,
                multiplicative_noise, p[0]);
        AUG_RUN(N4M_AUG_SPIKE_NOISE, n4m_aug_spike_noise_handle_t,
                spike_noise, static_cast<int32_t>(p[0]), static_cast<int32_t>(p[1]), p[2], p[3]);
        AUG_RUN(N4M_AUG_HETERO_NOISE, n4m_aug_hetero_noise_handle_t,
                hetero_noise, p[0], p[1]);
        AUG_RUN(N4M_AUG_LINEAR_DRIFT, n4m_aug_linear_drift_handle_t,
                linear_drift, p[0], p[1], p[2], p[3]);
        AUG_RUN(N4M_AUG_PATH_LENGTH, n4m_aug_path_length_handle_t,
                path_length, p[0], p[1]);
        AUG_RUN(N4M_AUG_BAND_PERTURB, n4m_aug_band_perturb_handle_t,
                band_perturb, static_cast<int32_t>(p[0]), static_cast<int32_t>(p[1]), static_cast<int32_t>(p[2]),
                p[3], p[4], p[5], p[6]);
        AUG_RUN(N4M_AUG_BAND_MASK, n4m_aug_band_mask_handle_t,
                band_mask, static_cast<int32_t>(p[0]), static_cast<int32_t>(p[1]), static_cast<int32_t>(p[2]),
                static_cast<int32_t>(p[3]), static_cast<int32_t>(p[4]));
        AUG_RUN(N4M_AUG_CHANNEL_DROPOUT, n4m_aug_channel_dropout_handle_t,
                channel_dropout, p[0], static_cast<int32_t>(p[1]));
        AUG_RUN(N4M_AUG_GAUSS_JITTER, n4m_aug_gauss_jitter_handle_t,
                gauss_jitter, p[0], p[1], static_cast<int32_t>(p[2]));
        AUG_RUN(N4M_AUG_UNSHARP_MASK, n4m_aug_unsharp_mask_handle_t,
                unsharp_mask, p[0], p[1], p[2], static_cast<int32_t>(p[3]));
        AUG_RUN(N4M_AUG_LOCAL_CLIP, n4m_aug_local_clip_handle_t,
                local_clip, static_cast<int32_t>(p[0]), static_cast<int32_t>(p[1]), static_cast<int32_t>(p[2]));
        AUG_RUN(N4M_AUG_ROTATE_TRANSLATE, n4m_aug_rotate_translate_handle_t,
                rotate_translate, p[0], p[1]);
        AUG_RUN(N4M_AUG_RANDOM_X_OP, n4m_aug_random_x_op_handle_t,
                random_x_op, static_cast<int32_t>(p[0]), p[1], p[2]);
        AUG_RUN(N4M_AUG_SCATTER_SIM_MSC, n4m_aug_scatter_sim_handle_t,
                scatter_sim_msc, p[0], p[1], p[2], p[3]);
        AUG_RUN(N4M_AUG_DEAD_BAND, n4m_aug_dead_band_handle_t,
                dead_band, static_cast<int32_t>(p[0]), static_cast<int32_t>(p[1]), static_cast<int32_t>(p[2]),
                p[3], p[4], static_cast<int32_t>(p[5]));
        AUG_RUN(N4M_AUG_BATCH_EFFECT, n4m_aug_batch_effect_handle_t,
                batch_effect, p[0], p[1], p[2], static_cast<int32_t>(p[3]), nullptr, 0);
        case N4M_AUG_SPLINE_SMOOTHING:
            return run_one<n4m_aug_spline_smooth_handle_t>(seed,
                [](auto** h, auto* rng) {
                    return n4m_augmentation_spline_smoothing_create(h, rng);
                }, n4m_augmentation_spline_smoothing_apply,
                n4m_augmentation_spline_smoothing_destroy, X, out);
        AUG_RUN(N4M_AUG_SPLINE_X_PERTURB, n4m_aug_spline_x_perturb_handle_t,
                spline_x_perturbations, static_cast<int32_t>(p[0]), p[1], p[2], p[3]);
        AUG_RUN(N4M_AUG_SPLINE_Y_PERTURB, n4m_aug_spline_y_perturb_handle_t,
                spline_y_perturbations, static_cast<int32_t>(p[0]), p[1]);
        AUG_RUN(N4M_AUG_SPLINE_X_SIMPLIFY, n4m_aug_spline_x_simplify_handle_t,
                spline_x_simplification, static_cast<int32_t>(p[0]), static_cast<int32_t>(p[1]));
        AUG_RUN(N4M_AUG_SPLINE_CURVE_SIMPLIFY, n4m_aug_spline_curve_simplify_handle_t,
                spline_curve_simplification, static_cast<int32_t>(p[0]), static_cast<int32_t>(p[1]));
        default: return N4M_ERR_INVALID_ARGUMENT;
    }
}

#undef AUG_RUN
