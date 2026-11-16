// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
#include "core/estimator/block_rms_adapter.hpp"

#include <cmath>
#include <utility>

#include "core/estimator/state_io.hpp"

namespace n4m::estimator {
namespace {

constexpr std::uint32_t kTagBlockRms = 0x31535242u;  // "BRS1"

double read(const n4m_matrix_view_t& view, std::int64_t row, std::int64_t column) {
    const auto offset = row * view.row_stride + column * view.col_stride;
    return view.dtype == N4M_DTYPE_F64 ? static_cast<const double*>(view.data)[offset]
                                     : static_cast<const float*>(view.data)[offset];
}

class BlockRmsAdapter final : public Adapter {
  public:
    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_TRANSFORM | N4M_CAP_SERIALIZABLE;
    }
    std::int64_t n_features_in() const noexcept override { return width_; }
    std::int64_t n_outputs() const noexcept override { return 0; }
    std::int64_t transform_cols() const noexcept override { return width_; }

    n4m_status_t fit(n4m_context_t* ctx, const Params&, const FitInputs& in) override {
        if (in.X == nullptr || in.X->rows < 1 || in.X->cols < 1) {
            set_error(ctx, "block RMS scaling requires a nonempty training matrix");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        const auto& x = *in.X;
        if (x.dtype != N4M_DTYPE_F64 && x.dtype != N4M_DTYPE_F32)
            return N4M_ERR_DTYPE_MISMATCH;
        // Scaled sum of squares avoids overflowing x*x for finite inputs.
        double scale = 0.0;
        double sum = 1.0;
        for (std::int64_t row = 0; row < x.rows; ++row) {
            for (std::int64_t column = 0; column < x.cols; ++column) {
                const double value = std::abs(read(x, row, column));
                if (!std::isfinite(value)) {
                    set_error(ctx, "block RMS scaling requires finite training values");
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                if (value == 0.0) continue;
                if (scale < value) {
                    const double ratio = scale / value;
                    sum = 1.0 + sum * ratio * ratio;
                    scale = value;
                } else {
                    const double ratio = value / scale;
                    sum += ratio * ratio;
                }
            }
        }
        const double fitted = scale == 0.0 ? 1.0 : scale * std::sqrt(sum / static_cast<double>(x.rows));
        if (!std::isfinite(fitted) || fitted <= 0.0) {
            set_error(ctx, "block RMS scale is outside the finite numeric range");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        width_ = x.cols;
        scale_ = fitted;
        return N4M_OK;
    }

    n4m_status_t transform(n4m_context_t* ctx, const n4m_matrix_view_t& x,
                            n4m_matrix_view_t& out) const override {
        if (width_ == 0) return N4M_ERR_NOT_FITTED;
        if (x.cols != width_ || out.cols != width_ || x.rows != out.rows)
            return N4M_ERR_SHAPE_MISMATCH;
        if ((x.dtype != N4M_DTYPE_F64 && x.dtype != N4M_DTYPE_F32) ||
            (out.dtype != N4M_DTYPE_F64 && out.dtype != N4M_DTYPE_F32))
            return N4M_ERR_DTYPE_MISMATCH;
        for (std::int64_t row = 0; row < x.rows; ++row) {
            for (std::int64_t column = 0; column < x.cols; ++column) {
                const double value = read(x, row, column) / scale_;
                if (!std::isfinite(value) ||
                    (out.dtype == N4M_DTYPE_F32 && !std::isfinite(static_cast<float>(value)))) {
                    set_error(ctx, "block RMS transformed value is outside the finite numeric range");
                    return N4M_ERR_INVALID_ARGUMENT;
                }
                const auto offset = row * out.row_stride + column * out.col_stride;
                if (out.dtype == N4M_DTYPE_F64) static_cast<double*>(out.data)[offset] = value;
                else static_cast<float*>(out.data)[offset] = static_cast<float>(value);
            }
        }
        return N4M_OK;
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        if (width_ == 0) return N4M_ERR_NOT_FITTED;
        n4m_state_writer_t writer;
        n4m_state_write_i64(&writer, width_);
        n4m_state_write_f64(&writer, scale_);
        out.push_back({kTagBlockRms, std::move(writer.bytes)});
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params&,
                             const std::vector<StateBlock>& blocks) override {
        if (blocks.size() != 1 || blocks[0].tag != kTagBlockRms) {
            set_error(ctx, "block RMS state requires one BRS1 block");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_state_reader_t reader(blocks[0].bytes.data(), blocks[0].bytes.size());
        std::int64_t width = 0;
        double scale = 0.0;
        if (!n4m_state_read_i64(&reader, &width) || !n4m_state_read_f64(&reader, &scale) ||
            reader.remaining() != 0 || width <= 0 || !std::isfinite(scale) || scale <= 0.0) {
            set_error(ctx, "invalid fitted block RMS width or scale");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        width_ = width;
        scale_ = scale;
        return N4M_OK;
    }

    n4m_status_t check_params(n4m_context_t*, const Params&) const override { return N4M_OK; }

  private:
    std::int64_t width_{0};
    double scale_{1.0};
};

}  // namespace

std::unique_ptr<Adapter> make_tr_block_rms(const MethodSpec&) {
    return std::make_unique<BlockRmsAdapter>();
}

}  // namespace n4m::estimator
