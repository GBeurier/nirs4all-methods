// SPDX-License-Identifier: CeCILL-2.1

#include "core/validation.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace {

[[nodiscard]] bool validate_sample_count(::pls4all::core::Context& ctx,
                                         std::int64_t n_samples,
                                         std::int64_t minimum) noexcept {
    if (n_samples < minimum) {
        ctx.set_errorf("n_samples must be >= %lld; got %lld",
                       static_cast<long long>(minimum),
                       static_cast<long long>(n_samples));
        return false;
    }
    return true;
}

void fill_fold(std::int64_t n_samples,
               std::int64_t test_start,
               std::int64_t test_stop,
               ::pls4all::core::ValidationFold& fold) {
    fold.train_indices.clear();
    fold.test_indices.clear();
    fold.train_indices.reserve(static_cast<std::size_t>(n_samples - (test_stop - test_start)));
    fold.test_indices.reserve(static_cast<std::size_t>(test_stop - test_start));
    for (std::int64_t sample = 0; sample < n_samples; ++sample) {
        if (sample >= test_start && sample < test_stop) {
            fold.test_indices.push_back(sample);
        } else {
            fold.train_indices.push_back(sample);
        }
    }
}

}  // namespace

namespace pls4all::core {

p4a_status_t make_kfold_validation_plan(Context& ctx,
                                        std::int64_t n_samples,
                                        std::int32_t n_splits,
                                        ValidationPlan& out) {
    try {
        out = ValidationPlan{};
        if (!validate_sample_count(ctx, n_samples, 2)) {
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (n_splits < 2) {
            ctx.set_errorf("n_splits must be >= 2; got %d", static_cast<int>(n_splits));
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (static_cast<std::int64_t>(n_splits) > n_samples) {
            ctx.set_errorf("n_splits (%d) must be <= n_samples (%lld)",
                           static_cast<int>(n_splits),
                           static_cast<long long>(n_samples));
            return P4A_ERR_INVALID_ARGUMENT;
        }

        out.n_samples = n_samples;
        out.folds.resize(static_cast<std::size_t>(n_splits));
        const std::int64_t base = n_samples / static_cast<std::int64_t>(n_splits);
        const std::int64_t remainder = n_samples % static_cast<std::int64_t>(n_splits);
        std::int64_t start = 0;
        for (std::int32_t fold_idx = 0; fold_idx < n_splits; ++fold_idx) {
            const std::int64_t size = base + ((static_cast<std::int64_t>(fold_idx) < remainder) ? 1 : 0);
            const std::int64_t stop = start + size;
            fill_fold(n_samples, start, stop, out.folds[static_cast<std::size_t>(fold_idx)]);
            start = stop;
        }
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while generating k-fold validation plan");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while generating k-fold validation plan");
        return P4A_ERR_INTERNAL;
    }
}

p4a_status_t make_leave_one_out_validation_plan(Context& ctx,
                                               std::int64_t n_samples,
                                               ValidationPlan& out) {
    if (n_samples > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        out = ValidationPlan{};
        ctx.set_errorf("leave-one-out n_samples is too large: %lld",
                       static_cast<long long>(n_samples));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return make_kfold_validation_plan(ctx,
                                      n_samples,
                                      static_cast<std::int32_t>(n_samples),
                                      out);
}

p4a_status_t make_holdout_validation_plan(Context& ctx,
                                         std::int64_t n_samples,
                                         std::int64_t test_start,
                                         std::int64_t test_count,
                                         ValidationPlan& out) {
    try {
        out = ValidationPlan{};
        if (!validate_sample_count(ctx, n_samples, 2)) {
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (test_start < 0) {
            ctx.set_errorf("holdout test_start must be >= 0; got %lld",
                           static_cast<long long>(test_start));
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (test_count <= 0) {
            ctx.set_errorf("holdout test_count must be > 0; got %lld",
                           static_cast<long long>(test_count));
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (test_start > n_samples || test_count > n_samples - test_start) {
            ctx.set_errorf("holdout test interval [%lld, %lld) exceeds n_samples (%lld)",
                           static_cast<long long>(test_start),
                           static_cast<long long>(test_start + test_count),
                           static_cast<long long>(n_samples));
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (test_count >= n_samples) {
            ctx.set_error("holdout split must leave at least one training sample");
            return P4A_ERR_INVALID_ARGUMENT;
        }

        out.n_samples = n_samples;
        out.folds.resize(1U);
        fill_fold(n_samples, test_start, test_start + test_count, out.folds[0]);
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while generating holdout validation plan");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while generating holdout validation plan");
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
