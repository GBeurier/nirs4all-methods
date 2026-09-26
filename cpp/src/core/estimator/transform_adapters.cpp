// SPDX-License-Identifier: CECILL-2.1
//
// Transformer role adapters over the per-method preprocessing kernels.
//
// Stateless transformers keep no fitted data: the fitted state is the input
// width (the method is rebuilt from its N4ME parameters). The kernels need
// contiguous row-major F64 views, so strided inputs and outputs go through a
// private contiguous copy.

#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core/estimator/generated_factories.hpp"
#include "core/estimator/spec.hpp"
#include "core/estimator/state_io.hpp"

namespace n4m::estimator {

namespace {

constexpr std::uint32_t kTagWidth = 0x31445457u;  // "WTD1"

bool row_major(const n4m_matrix_view_t& v) noexcept {
    return v.dtype == N4M_DTYPE_F64 && v.col_stride == 1 && v.row_stride == v.cols;
}

}  // namespace

n4m_matrix_view_t contiguous_view(const n4m_matrix_view_t& X, std::vector<double>& storage) {
    if (row_major(X)) return X;
    storage.resize(static_cast<std::size_t>(X.rows * X.cols));
    const auto* src = static_cast<const double*>(X.data);
    for (std::int64_t i = 0; i < X.rows; ++i) {
        for (std::int64_t j = 0; j < X.cols; ++j) {
            storage[static_cast<std::size_t>(i * X.cols + j)] = src[i * X.row_stride + j * X.col_stride];
        }
    }
    n4m_matrix_view_t v{};
    n4m_matrix_view_init_rowmajor(&v, storage.data(), X.rows, X.cols, N4M_DTYPE_F64);
    return v;
}

std::int32_t to_i32(std::int64_t v) {
    if (v > std::numeric_limits<std::int32_t>::max() ||
        v < std::numeric_limits<std::int32_t>::min()) {
        throw std::out_of_range("integer parameter exceeds int32");
    }
    return static_cast<std::int32_t>(v);
}

// Runs a contiguous-only kernel on possibly strided views.
n4m_status_t contiguous_call(
    const n4m_matrix_view_t& X, n4m_matrix_view_t& out,
    const std::function<n4m_status_t(n4m_matrix_view_t, n4m_matrix_view_t)>& kernel) {
    if (X.dtype != N4M_DTYPE_F64 || out.dtype != N4M_DTYPE_F64) return N4M_ERR_DTYPE_MISMATCH;
    std::vector<double> x_copy, out_copy;
    const n4m_matrix_view_t x_view = contiguous_view(X, x_copy);
    n4m_matrix_view_t out_view = out;
    if (!row_major(out)) {
        out_copy.resize(static_cast<std::size_t>(out.rows * out.cols));
        n4m_matrix_view_init_rowmajor(&out_view, out_copy.data(), out.rows, out.cols,
                                      N4M_DTYPE_F64);
    }
    const n4m_status_t st = kernel(x_view, out_view);
    if (st == N4M_OK && !out_copy.empty()) {
        auto* dst = static_cast<double*>(out.data);
        for (std::int64_t i = 0; i < out.rows; ++i) {
            for (std::int64_t j = 0; j < out.cols; ++j) {
                dst[i * out.row_stride + j * out.col_stride] =
                    out_copy[static_cast<std::size_t>(i * out.cols + j)];
            }
        }
    }
    return st;
}

namespace {

std::int32_t i32(std::int64_t v) { return to_i32(v); }

// A stateless kernel: create from parameters, transform, destroy, and the
// output width for a given input width.
template <typename H>
struct Kernel {
    std::function<n4m_status_t(H**, const Params&)> create;
    n4m_status_t (*transform)(const H*, n4m_matrix_view_t, n4m_matrix_view_t);
    void (*destroy)(H*);
    std::function<n4m_status_t(const H*, std::int64_t, std::int64_t*)> output_cols;
};

template <typename H>
class StatelessTransformAdapter final : public Adapter {
  public:
    explicit StatelessTransformAdapter(Kernel<H> kernel) : kernel_(std::move(kernel)) {}

    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_TRANSFORM | N4M_CAP_SERIALIZABLE;
    }
    std::int64_t n_features_in() const noexcept override { return n_features_; }
    std::int64_t n_outputs() const noexcept override { return 0; }
    std::int64_t transform_cols() const noexcept override { return out_cols_; }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        return build(ctx, params, in.X->cols);
    }

    n4m_status_t transform(n4m_context_t*, const n4m_matrix_view_t& X,
                           n4m_matrix_view_t& out) const override {
        if (out.cols != out_cols_) return N4M_ERR_SHAPE_MISMATCH;
        return contiguous_call(X, out, [this](n4m_matrix_view_t x, n4m_matrix_view_t o) {
            return kernel_.transform(handle_.get(), x, o);
        });
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        StateBlock block;
        block.tag = kTagWidth;
        block.bytes.resize(8);
        const auto width = static_cast<std::uint64_t>(n_features_);
        for (int i = 0; i < 8; ++i) block.bytes[static_cast<std::size_t>(i)] =
            static_cast<unsigned char>(width >> (8 * i));
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params& params,
                            const std::vector<StateBlock>& blocks) override {
        if (blocks.size() != 1 || blocks[0].tag != kTagWidth || blocks[0].bytes.size() != 8) {
            set_error(ctx, "transformer state must be one WTD1 block");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        std::uint64_t width = 0;
        for (int i = 0; i < 8; ++i) {
            width |= static_cast<std::uint64_t>(blocks[0].bytes[static_cast<std::size_t>(i)])
                     << (8 * i);
        }
        if (width == 0 || width > (std::uint64_t{1} << 31)) return N4M_ERR_CORRUPT_BUFFER;
        return build(ctx, params, static_cast<std::int64_t>(width));
    }

  private:
    n4m_status_t build(n4m_context_t* ctx, const Params& params, std::int64_t width) {
        handle_.reset();
        n_features_ = 0;
        H* raw = nullptr;
        n4m_status_t st = kernel_.create(&raw, params);
        if (st != N4M_OK) {
            set_error(ctx, "invalid transformer parameters");
            return st;
        }
        handle_.reset(raw);
        std::int64_t cols = width;
        if (kernel_.output_cols) {
            st = kernel_.output_cols(raw, width, &cols);
            if (st != N4M_OK || cols <= 0) {
                set_error(ctx, "transformer produces no output column for this input width");
                return st != N4M_OK ? st : N4M_ERR_INVALID_ARGUMENT;
            }
        }
        n_features_ = width;
        out_cols_ = cols;
        return N4M_OK;
    }

    struct Destroy {
        void (*destroy)(H*);
        void operator()(H* h) const noexcept { destroy(h); }
    };
    Kernel<H> kernel_;
    std::unique_ptr<H, Destroy> handle_{nullptr, Destroy{kernel_.destroy}};
    std::int64_t n_features_ = 0;
    std::int64_t out_cols_ = 0;
};

template <typename H>
std::unique_ptr<Adapter> stateless(Kernel<H> kernel) {
    return std::make_unique<StatelessTransformAdapter<H>>(std::move(kernel));
}

}  // namespace

// ---- baselines --------------------------------------------------------------

std::unique_ptr<Adapter> make_tr_airpls(const MethodSpec&) {
    return stateless<n4m_pp_airpls_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_airpls_create(h, p.get_double("lam"), i32(p.get_int("max_iter")),
                                                p.get_double("tol"));
         },
         n4m_transform_airpls_transform, n4m_transform_airpls_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_arpls(const MethodSpec&) {
    return stateless<n4m_pp_arpls_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_arpls_create(h, p.get_double("lam"), i32(p.get_int("max_iter")),
                                               p.get_double("tol"));
         },
         n4m_transform_arpls_transform, n4m_transform_arpls_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_asls(const MethodSpec&) {
    return stateless<n4m_pp_asls_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_asls_create(h, p.get_double("lam"), p.get_double("p"),
                                              i32(p.get_int("max_iter")), p.get_double("tol"));
         },
         n4m_transform_asls_transform, n4m_transform_asls_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_beads(const MethodSpec&) {
    return stateless<n4m_pp_beads_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_beads_create(h, p.get_double("lam_0"), p.get_double("lam_1"),
                                               p.get_double("lam_2"), i32(p.get_int("max_iter")),
                                               p.get_double("tol"));
         },
         n4m_transform_beads_transform, n4m_transform_beads_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_detrend(const MethodSpec&) {
    return stateless<n4m_pp_detrend_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_detrend_create(h, i32(p.get_int("polyorder")));
         },
         n4m_transform_detrend_transform, n4m_transform_detrend_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_imodpoly(const MethodSpec&) {
    return stateless<n4m_pp_imodpoly_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_imodpoly_create(h, i32(p.get_int("polyorder")),
                                                  i32(p.get_int("max_iter")), p.get_double("tol"));
         },
         n4m_transform_imodpoly_transform, n4m_transform_imodpoly_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_modpoly(const MethodSpec&) {
    return stateless<n4m_pp_modpoly_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_modpoly_create(h, i32(p.get_int("polyorder")),
                                                 i32(p.get_int("max_iter")), p.get_double("tol"));
         },
         n4m_transform_modpoly_transform, n4m_transform_modpoly_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_rolling_ball(const MethodSpec&) {
    return stateless<n4m_pp_rolling_ball_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_rolling_ball_create(h, i32(p.get_int("half_window")),
                                                      i32(p.get_int("smooth_half_window")));
         },
         n4m_transform_rolling_ball_transform, n4m_transform_rolling_ball_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_snip(const MethodSpec&) {
    return stateless<n4m_pp_snip_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_snip_create(h, i32(p.get_int("max_half_window")));
         },
         n4m_transform_snip_transform, n4m_transform_snip_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_iasls(const MethodSpec&) {
    return stateless<n4m_pp_iasls_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_iasls_create_ex(
                 h, p.get_double("lam"), p.get_double("p"), p.get_double("lam_1"),
                 i32(p.get_int("polyorder")), i32(p.get_int("diff_order")),
                 i32(p.get_int("max_iter")), p.get_double("tol"));
         },
         n4m_transform_iasls_transform, n4m_transform_iasls_destroy, {}});
}

// ---- derivatives and smoothing ---------------------------------------------

std::unique_ptr<Adapter> make_tr_first_derivative(const MethodSpec&) {
    return stateless<n4m_pp_first_derivative_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_first_derivative_create(h, p.get_double("delta"),
                                                          i32(p.get_int("edge_order")));
         },
         n4m_transform_first_derivative_transform, n4m_transform_first_derivative_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_second_derivative(const MethodSpec&) {
    return stateless<n4m_pp_second_derivative_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_second_derivative_create(h, p.get_double("delta"),
                                                           i32(p.get_int("edge_order")));
         },
         n4m_transform_second_derivative_transform, n4m_transform_second_derivative_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_norris_williams(const MethodSpec&) {
    return stateless<n4m_pp_norris_williams_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_norris_williams_create(
                 h, i32(p.get_int("segment")), i32(p.get_int("gap")),
                 i32(p.get_int("derivative_order")), p.get_double("delta"));
         },
         n4m_transform_norris_williams_transform, n4m_transform_norris_williams_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_savitzky_golay(const MethodSpec&) {
    return stateless<n4m_pp_savgol_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_savitzky_golay_create(
                 h, i32(p.get_int("window_length")), i32(p.get_int("polyorder")),
                 i32(p.get_int("deriv")), p.get_double("delta"),
                 static_cast<n4m_pp_savgol_mode_t>(p.get_int("mode")), p.get_double("cval"));
         },
         n4m_transform_savitzky_golay_transform, n4m_transform_savitzky_golay_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_gaussian(const MethodSpec&) {
    return stateless<n4m_pp_gaussian_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_gaussian_create(
                 h, p.get_double("sigma"), i32(p.get_int("order")),
                 static_cast<n4m_pp_gaussian_mode_t>(p.get_int("mode")), p.get_double("cval"),
                 p.get_double("truncate"));
         },
         n4m_transform_gaussian_transform, n4m_transform_gaussian_destroy, {}});
}

// ---- resampling ------------------------------------------------------------

std::unique_ptr<Adapter> make_tr_crop(const MethodSpec&) {
    return stateless<n4m_pp_crop_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_crop_create(h, p.get_int("start"), p.get_int("end"));
         },
         n4m_transform_crop_transform, n4m_transform_crop_destroy,
         [](const n4m_pp_crop_handle_t* h, std::int64_t in, std::int64_t* out) {
             *out = n4m_transform_crop_output_cols(h, in);
             return N4M_OK;
         }});
}

std::unique_ptr<Adapter> make_tr_resample(const MethodSpec&) {
    return stateless<n4m_pp_resample_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_resample_transformer_create(h, p.get_int("num_samples"));
         },
         n4m_transform_resample_transformer_transform, n4m_transform_resample_transformer_destroy,
         [](const n4m_pp_resample_handle_t* h, std::int64_t in, std::int64_t* out) {
             *out = n4m_transform_resample_transformer_output_cols(h, in);
             return N4M_OK;
         }});
}

// ---- scatter ----------------------------------------------------------------

std::unique_ptr<Adapter> make_tr_area_normalization(const MethodSpec&) {
    return stateless<n4m_pp_area_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_area_normalization_create(h, i32(p.get_int("method")));
         },
         n4m_transform_area_normalization_transform, n4m_transform_area_normalization_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_local_snv(const MethodSpec&) {
    return stateless<n4m_pp_lsnv_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_local_snv_create(h, i32(p.get_int("window")),
                                                   i32(p.get_int("pad_mode")),
                                                   p.get_double("constant_value"));
         },
         n4m_transform_local_snv_transform, n4m_transform_local_snv_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_robust_snv(const MethodSpec&) {
    return stateless<n4m_pp_rnv_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_robust_snv_create(h, p.get_bool("with_center") ? 1 : 0,
                                                    p.get_bool("with_scale") ? 1 : 0,
                                                    p.get_double("k"));
         },
         n4m_transform_robust_snv_transform, n4m_transform_robust_snv_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_snv(const MethodSpec&) {
    return stateless<n4m_pp_snv_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_snv_create(h, p.get_bool("with_mean") ? 1 : 0,
                                             p.get_bool("with_std") ? 1 : 0,
                                             i32(p.get_int("ddof")));
         },
         n4m_transform_snv_transform, n4m_transform_snv_destroy, {}});
}

// ---- signal conversion -----------------------------------------------------

std::unique_ptr<Adapter> make_tr_fraction_to_percent(const MethodSpec&) {
    return stateless<n4m_pp_frac_to_pct_handle_t>(
        {[](auto** h, const Params&) { return n4m_transform_fraction_to_percent_create(h); },
         n4m_transform_fraction_to_percent_transform, n4m_transform_fraction_to_percent_destroy,
         {}});
}

std::unique_ptr<Adapter> make_tr_percent_to_fraction(const MethodSpec&) {
    return stateless<n4m_pp_pct_to_frac_handle_t>(
        {[](auto** h, const Params&) { return n4m_transform_percent_to_fraction_create(h); },
         n4m_transform_percent_to_fraction_transform, n4m_transform_percent_to_fraction_destroy,
         {}});
}

std::unique_ptr<Adapter> make_tr_from_absorbance(const MethodSpec&) {
    return stateless<n4m_pp_from_absorbance_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_from_absorbance_create(h, p.get_bool("is_percent") ? 1 : 0);
         },
         n4m_transform_from_absorbance_transform, n4m_transform_from_absorbance_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_kubelka_munk(const MethodSpec&) {
    return stateless<n4m_pp_kubelka_munk_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_kubelka_munk_create(h, p.get_bool("is_percent") ? 1 : 0,
                                                      p.get_double("epsilon"));
         },
         n4m_transform_kubelka_munk_transform, n4m_transform_kubelka_munk_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_to_absorbance(const MethodSpec&) {
    return stateless<n4m_pp_to_absorbance_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_to_absorbance_create(h, p.get_bool("is_percent") ? 1 : 0,
                                                       p.get_double("epsilon"),
                                                       p.get_bool("clip_negative") ? 1 : 0);
         },
         n4m_transform_to_absorbance_transform, n4m_transform_to_absorbance_destroy, {}});
}

// ---- wavelets ---------------------------------------------------------------

std::unique_ptr<Adapter> make_tr_haar(const MethodSpec&) {
    return stateless<n4m_pp_haar_handle_t>(
        {[](auto** h, const Params&) { return n4m_transform_haar_create(h); },
         n4m_transform_haar_transform, n4m_transform_haar_destroy,
         [](const n4m_pp_haar_handle_t*, std::int64_t in, std::int64_t* out) {
             return n4m_transform_haar_output_cols(in, out);
         }});
}

std::unique_ptr<Adapter> make_tr_wavelet(const MethodSpec&) {
    return stateless<n4m_pp_wavelet_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_wavelet_create(
                 h, static_cast<n4m_pp_wavelet_family_t>(p.get_int("family")),
                 static_cast<n4m_pp_wavelet_boundary_t>(p.get_int("mode")));
         },
         n4m_transform_wavelet_transform, n4m_transform_wavelet_destroy,
         n4m_transform_wavelet_output_cols});
}

std::unique_ptr<Adapter> make_tr_wavelet_denoise(const MethodSpec&) {
    return stateless<n4m_pp_wavelet_denoise_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_wavelet_denoise_create(
                 h, static_cast<n4m_pp_wavelet_family_t>(p.get_int("family")),
                 static_cast<n4m_pp_wavelet_boundary_t>(p.get_int("mode")),
                 i32(p.get_int("level")),
                 static_cast<n4m_pp_wavelet_threshold_t>(p.get_int("threshold_mode")),
                 static_cast<n4m_pp_wavelet_noise_t>(p.get_int("noise_estimator")));
         },
         n4m_transform_wavelet_denoise_transform, n4m_transform_wavelet_denoise_destroy, {}});
}

std::unique_ptr<Adapter> make_tr_wavelet_features(const MethodSpec&) {
    return stateless<n4m_pp_wavelet_features_handle_t>(
        {[](auto** h, const Params& p) {
             return n4m_transform_wavelet_features_create_ex(
                 h, static_cast<n4m_pp_wavelet_family_t>(p.get_int("family")),
                 static_cast<n4m_pp_wavelet_boundary_t>(p.get_int("mode")),
                 i32(p.get_int("max_level")),
                 static_cast<n4m_pp_wavelet_features_entropy_t>(p.get_int("entropy")));
         },
         n4m_transform_wavelet_features_transform, n4m_transform_wavelet_features_destroy,
         n4m_transform_wavelet_features_output_cols});
}

}  // namespace n4m::estimator
