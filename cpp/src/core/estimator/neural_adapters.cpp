// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
//
// Estimator adapter of models.neural.multibranch_regressor: resolves the
// declared branches against the structural `blocks` fit input, runs the
// core network (core/neural/multibranch_network.hpp) on a row-major copy of
// possibly strided views, and frames its complete fitted state (architecture,
// training-only standardization, learned weights) as one N4ME block.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"
#include "core/method_result.hpp"
#include "core/neural/multibranch_network.hpp"

namespace n4m::estimator {

namespace {

namespace nn = ::n4m::core::neural;

constexpr std::uint32_t kTagMultibranch = 0x314E424Du;  // "MBN1"

// Branch declarations for the given block widths (block order). Throws
// std::invalid_argument naming the inconsistency.
std::vector<nn::BranchShape> resolve_branches(const Params& params,
                                              const std::vector<std::int64_t>& widths) {
    const std::vector<std::int64_t> kinds = params.get_ints("branch_kinds");
    const std::vector<std::int64_t> dims = params.get_ints("branch_shapes");
    const std::size_t n = widths.size();
    if (!kinds.empty() && kinds.size() != n) {
        throw std::invalid_argument("branch_kinds has " + std::to_string(kinds.size()) +
                                    " entries for " + std::to_string(n) + " blocks");
    }
    if (!dims.empty() && dims.size() != 3 * n) {
        throw std::invalid_argument("branch_shapes needs 3 integers per block (" +
                                    std::to_string(3 * n) + "), got " +
                                    std::to_string(dims.size()));
    }
    std::vector<nn::BranchShape> out(n);
    for (std::size_t b = 0; b < n; ++b) {
        nn::BranchShape& s = out[b];
        s.kind = kinds.empty() ? nn::kKindDense : kinds[b];
        if (dims.empty()) {
            if (s.kind == nn::kKindConv2d) {
                throw std::invalid_argument("block " + std::to_string(b) +
                                            ": a conv2d branch needs its [H, W, C] in "
                                            "branch_shapes");
            }
            s.d0 = widths[b];
            s.d1 = 1;
            s.d2 = 1;
        } else {
            s.d0 = dims[3 * b];
            s.d1 = dims[3 * b + 1];
            s.d2 = dims[3 * b + 2];
        }
        // Each factor is at most 2^31 here (otherwise refused) so the product
        // fits in 64 bits after the first check.
        const std::int64_t limit = std::int64_t{1} << 31;
        if (s.d0 < 1 || s.d1 < 1 || s.d2 < 1 || s.d0 > limit || s.d1 > limit || s.d2 > limit) {
            throw std::invalid_argument("block " + std::to_string(b) +
                                        ": shape dimensions must be positive");
        }
        const std::int64_t hw = s.d0 * s.d1;
        if (hw > limit || hw * s.d2 != widths[b]) {
            throw std::invalid_argument("block " + std::to_string(b) + ": shape product " +
                                        "does not equal the block width " +
                                        std::to_string(widths[b]));
        }
    }
    return out;
}

nn::Hyper hyper_of(const Params& params) {
    nn::Hyper h;
    h.filters = params.get_int("filters");
    h.kernel_size = params.get_int("kernel_size");
    h.pool_size = params.get_int("pool_size");
    h.embedding_dim = params.get_int("embedding_dim");
    h.head_hidden = params.get_int("head_hidden");
    return h;
}

bool finite_view(const n4m_matrix_view_t& v) {
    const auto* d = static_cast<const double*>(v.data);
    for (std::int64_t i = 0; i < v.rows; ++i) {
        for (std::int64_t j = 0; j < v.cols; ++j) {
            if (!std::isfinite(d[i * v.row_stride + j * v.col_stride])) return false;
        }
    }
    return true;
}

// Row-major values into a possibly strided output view.
void write_rows(const std::vector<double>& values, n4m_matrix_view_t& out) {
    auto* dst = static_cast<double*>(out.data);
    for (std::int64_t i = 0; i < out.rows; ++i) {
        for (std::int64_t j = 0; j < out.cols; ++j) {
            dst[i * out.row_stride + j * out.col_stride] =
                values[static_cast<std::size_t>(i * out.cols + j)];
        }
    }
}

class MultibranchRegressor final : public Adapter {
  public:
    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_PREDICT | N4M_CAP_TRANSFORM | N4M_CAP_SERIALIZABLE;
    }
    std::int64_t n_features_in() const noexcept override {
        return fitted_ ? static_cast<std::int64_t>(net_.n_features()) : 0;
    }
    std::int64_t n_outputs() const noexcept override {
        return fitted_ ? static_cast<std::int64_t>(net_.n_targets()) : 0;
    }
    std::int64_t transform_cols() const noexcept override {
        return fitted_ ? static_cast<std::int64_t>(net_.embedding_width()) : 0;
    }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        fitted_ = false;
        result_.reset();
        if (in.Y == nullptr) {
            set_error(ctx, "multibranch regressor: y is required");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        // Complete-data profile: missing values are refused, never imputed.
        if (!finite_view(*in.X)) {
            set_error(ctx, "multibranch regressor: X contains NaN or Inf; training needs "
                           "complete data (no imputation is performed)");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        if (!finite_view(*in.Y)) {
            set_error(ctx, "multibranch regressor: y contains NaN or Inf");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::vector<std::int64_t> widths;
        if (in.block_sizes != nullptr) {
            widths.assign(in.block_sizes, in.block_sizes + in.n_blocks);
        } else {
            widths.push_back(in.X->cols);
        }
        try {
            const std::vector<nn::BranchShape> shapes = resolve_branches(params, widths);
            net_.configure(in.X->cols, in.Y->cols, shapes, hyper_of(params));
            nn::Training t;
            t.epochs = params.get_int("epochs");
            t.batch_size = params.get_int("batch_size");
            t.learning_rate = params.get_double("learning_rate");
            t.weight_decay = params.get_double("weight_decay");
            // The int64 parameter carries the u64 seed bit for bit.
            t.seed = static_cast<std::uint64_t>(params.get_int("seed"));
            std::vector<double> x_store, y_store;
            const n4m_matrix_view_t X = contiguous_view(*in.X, x_store);
            const n4m_matrix_view_t Y = contiguous_view(*in.Y, y_store);
            const std::vector<double> history =
                net_.fit(static_cast<const double*>(X.data), static_cast<const double*>(Y.data),
                         static_cast<std::size_t>(X.rows), t);
            auto result = std::make_unique<n4m_method_result_s>();
            result->set_double_matrix("loss_history", history,
                                      static_cast<std::int64_t>(history.size()), 1);
            result->set_scalar("initial_loss", history.front());
            result->set_scalar("final_loss", history.back());
            result->set_scalar("n_parameters", static_cast<double>(net_.n_parameters()));
            result->set_scalar("epochs", static_cast<double>(t.epochs));
            result_ = std::move(result);
        } catch (const std::invalid_argument& e) {
            set_error(ctx, e.what());
            return N4M_ERR_INVALID_ARGUMENT;
        } catch (const std::domain_error& e) {
            set_error(ctx, e.what());
            return N4M_ERR_NUMERICAL_FAILURE;
        }
        fitted_ = true;
        return N4M_OK;
    }

    n4m_status_t predict(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                         n4m_matrix_view_t& out) const override {
        return run(ctx, X, out, net_.n_targets(), false);
    }

    n4m_status_t transform(n4m_context_t* ctx, const n4m_matrix_view_t& X,
                           n4m_matrix_view_t& out) const override {
        return run(ctx, X, out, net_.embedding_width(), true);
    }

    const n4m_method_result_t* fit_result() const noexcept override { return result_.get(); }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        n4m_state_writer_t w;
        net_.save(&w);
        StateBlock block;
        block.tag = kTagMultibranch;
        block.bytes = std::move(w.bytes);
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params&,
                            const std::vector<StateBlock>& blocks) override {
        fitted_ = false;
        result_.reset();
        if (blocks.size() != 1 || blocks[0].tag != kTagMultibranch) {
            set_error(ctx, "multibranch regressor state must be one MBN1 block");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_state_reader_t r(blocks[0].bytes.data(), blocks[0].bytes.size());
        if (!net_.load(&r) || r.remaining() != 0) {
            set_error(ctx, "multibranch regressor state does not match the method");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        fitted_ = true;
        return N4M_OK;
    }

    // The architecture parameters must describe the restored network; the
    // optimization parameters and the seed are provenance.
    n4m_status_t check_params(n4m_context_t* ctx, const Params& params) const override {
        const nn::Hyper& h = net_.hyper();
        const std::pair<const char*, std::int64_t> recorded[] = {
            {"filters", h.filters},
            {"kernel_size", h.kernel_size},
            {"pool_size", h.pool_size},
            {"embedding_dim", h.embedding_dim},
            {"head_hidden", h.head_hidden}};
        for (const auto& [name, value] : recorded) {
            const n4m_status_t st = check_int(ctx, params, name, value);
            if (st != N4M_OK) return st;
        }
        std::vector<std::int64_t> widths;
        for (const nn::BranchLayout& L : net_.branches()) {
            widths.push_back(static_cast<std::int64_t>(L.width));
        }
        std::vector<nn::BranchShape> declared;
        try {
            declared = resolve_branches(params, widths);
        } catch (const std::invalid_argument&) {
            return contradicts(ctx, "branch_shapes");
        }
        for (std::size_t b = 0; b < declared.size(); ++b) {
            const nn::BranchShape& s = net_.branches()[b].shape;
            if (declared[b].kind != s.kind) return contradicts(ctx, "branch_kinds");
            if (declared[b].d0 != s.d0 || declared[b].d1 != s.d1 || declared[b].d2 != s.d2) {
                return contradicts(ctx, "branch_shapes");
            }
        }
        return N4M_OK;
    }

  private:
    n4m_status_t run(n4m_context_t* ctx, const n4m_matrix_view_t& X, n4m_matrix_view_t& out,
                     std::size_t width, bool embeddings) const {
        if (static_cast<std::size_t>(out.cols) != width) return N4M_ERR_SHAPE_MISMATCH;
        if (!finite_view(X)) {
            set_error(ctx, "multibranch regressor: X contains NaN or Inf");
            return N4M_ERR_INVALID_ARGUMENT;
        }
        std::vector<double> storage;
        const n4m_matrix_view_t Xc = contiguous_view(X, storage);
        const auto n = static_cast<std::size_t>(Xc.rows);
        std::vector<double> values(n * width);
        const auto* data = static_cast<const double*>(Xc.data);
        if (embeddings) {
            net_.transform(data, n, values.data());
        } else {
            net_.predict(data, n, values.data());
        }
        write_rows(values, out);
        return N4M_OK;
    }

    nn::MultibranchNetwork net_;
    std::unique_ptr<n4m_method_result_s> result_;
    bool fitted_ = false;
};

}  // namespace

std::unique_ptr<Adapter> make_multibranch_regressor(const MethodSpec&) {
    return std::make_unique<MultibranchRegressor>();
}

}  // namespace n4m::estimator
