// SPDX-License-Identifier: CECILL-2.1
//
// C++ side of the kernel fitted-state writer / reader, and the adapter for
// transformers whose kernels learn a state at fit (see state_io.h).

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "core/estimator/spec.hpp"
#include "core/estimator/state_io.h"

struct n4m_state_writer_s {
    std::vector<unsigned char> bytes;
};

struct n4m_state_reader_s {
    n4m_state_reader_s(const unsigned char* p, std::size_t n) : data(p), size(n) {}
    bool take(std::uint64_t& v) {
        if (!peek(v)) return false;
        pos += 8;
        return true;
    }
    bool peek(std::uint64_t& v) const {
        if (size - pos < 8) return false;
        v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(data[pos + static_cast<std::size_t>(i)]) << (8 * i);
        return true;
    }
    std::size_t remaining() const { return size - pos; }
    const unsigned char* data;
    std::size_t size;
    std::size_t pos = 0;
};

namespace n4m::estimator {

// Runs a contiguous-only kernel on possibly strided views (defined in
// transform_adapters.cpp).
n4m_status_t contiguous_call(
    const n4m_matrix_view_t& X, n4m_matrix_view_t& out,
    const std::function<n4m_status_t(n4m_matrix_view_t, n4m_matrix_view_t)>& kernel);

// Row-major contiguous copy of a view when the kernel needs one; `storage`
// keeps the copy alive.
n4m_matrix_view_t contiguous_view(const n4m_matrix_view_t& X, std::vector<double>& storage);

std::int32_t to_i32(std::int64_t v);

// A transformer kernel with a fitted state: create from parameters, fit,
// transform, destroy, output width, and state save / load.
template <typename H>
struct FittedKernel {
    std::function<n4m_status_t(H**, const Params&)> create;
    std::function<n4m_status_t(n4m_context_t*, H*, const FitInputs&)> fit;
    std::function<n4m_status_t(const H*, n4m_matrix_view_t, n4m_matrix_view_t)> transform;
    void (*destroy)(H*);
    // Empty: the output width is the input width.
    std::function<n4m_status_t(const H*, std::int64_t, std::int64_t*)> output_cols;
    std::function<n4m_status_t(const H*, n4m_state_writer_t*)> save;
    std::function<n4m_status_t(H*, n4m_state_reader_t*, std::int64_t n_features)> load;
};

template <typename H>
class FittedTransformAdapter final : public Adapter {
  public:
    explicit FittedTransformAdapter(FittedKernel<H> kernel) : kernel_(std::move(kernel)) {}

    std::uint64_t capabilities() const noexcept override {
        return N4M_CAP_TRANSFORM | N4M_CAP_SERIALIZABLE;
    }
    std::int64_t n_features_in() const noexcept override { return n_features_; }
    std::int64_t n_outputs() const noexcept override { return 0; }
    std::int64_t transform_cols() const noexcept override { return out_cols_; }

    n4m_status_t fit(n4m_context_t* ctx, const Params& params, const FitInputs& in) override {
        reset();
        n4m_status_t st = create(ctx, params);
        if (st == N4M_OK) st = kernel_.fit(ctx, handle_.get(), in);
        if (st == N4M_OK) st = finish(ctx, in.X->cols);
        if (st != N4M_OK) reset();
        return st;
    }

    n4m_status_t transform(n4m_context_t*, const n4m_matrix_view_t& X,
                           n4m_matrix_view_t& out) const override {
        if (out.cols != out_cols_) return N4M_ERR_SHAPE_MISMATCH;
        return contiguous_call(X, out, [this](n4m_matrix_view_t x, n4m_matrix_view_t o) {
            return kernel_.transform(handle_.get(), x, o);
        });
    }

    n4m_status_t save_state(n4m_context_t*, std::vector<StateBlock>& out) const override {
        n4m_state_writer_t writer;
        n4m_state_write_i64(&writer, n_features_);
        n4m_status_t st = kernel_.save(handle_.get(), &writer);
        if (st != N4M_OK) return st;
        StateBlock block;
        block.tag = 0x31545354u;  // "TST1"
        block.bytes = std::move(writer.bytes);
        out.push_back(std::move(block));
        return N4M_OK;
    }

    n4m_status_t load_state(n4m_context_t* ctx, const Params& params,
                            const std::vector<StateBlock>& blocks) override {
        reset();
        if (blocks.size() != 1 || blocks[0].tag != 0x31545354u) {
            set_error(ctx, "transformer state must be one TST1 block");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_state_reader_t reader(blocks[0].bytes.data(), blocks[0].bytes.size());
        std::int64_t width = 0;
        if (!n4m_state_read_i64(&reader, &width) || width <= 0 || width > (std::int64_t{1} << 31)) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_status_t st = create(ctx, params);
        if (st == N4M_OK) st = kernel_.load(handle_.get(), &reader, width);
        if (st == N4M_OK && reader.remaining() != 0) st = N4M_ERR_CORRUPT_BUFFER;
        if (st == N4M_OK) st = finish(ctx, width);
        if (st != N4M_OK) {
            reset();
            set_error(ctx, "transformer state does not match the method");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        return N4M_OK;
    }

  private:
    n4m_status_t create(n4m_context_t* ctx, const Params& params) {
        H* raw = nullptr;
        const n4m_status_t st = kernel_.create(&raw, params);
        if (st != N4M_OK) {
            set_error(ctx, "invalid transformer parameters");
            return st;
        }
        handle_.reset(raw);
        return N4M_OK;
    }

    n4m_status_t finish(n4m_context_t* ctx, std::int64_t width) {
        std::int64_t cols = width;
        if (kernel_.output_cols) {
            const n4m_status_t st = kernel_.output_cols(handle_.get(), width, &cols);
            if (st != N4M_OK || cols <= 0) {
                set_error(ctx, "transformer produces no output column for this input width");
                return st != N4M_OK ? st : N4M_ERR_INVALID_ARGUMENT;
            }
        }
        n_features_ = width;
        out_cols_ = cols;
        return N4M_OK;
    }

    void reset() noexcept {
        handle_.reset();
        n_features_ = 0;
        out_cols_ = 0;
    }

    struct Destroy {
        void (*destroy)(H*);
        void operator()(H* h) const noexcept { destroy(h); }
    };
    FittedKernel<H> kernel_;
    std::unique_ptr<H, Destroy> handle_{nullptr, Destroy{kernel_.destroy}};
    std::int64_t n_features_ = 0;
    std::int64_t out_cols_ = 0;
};

template <typename H>
std::unique_ptr<Adapter> fitted(FittedKernel<H> kernel) {
    return std::make_unique<FittedTransformAdapter<H>>(std::move(kernel));
}

}  // namespace n4m::estimator
