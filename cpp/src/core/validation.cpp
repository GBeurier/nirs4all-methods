// SPDX-License-Identifier: CeCILL-2.1

#include "core/validation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "core/matrix_view.hpp"

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

[[nodiscard]] bool checked_square_size(std::int64_t n,
                                       std::size_t& out) noexcept {
    if (n < 0) {
        return false;
    }
    const auto un = static_cast<std::size_t>(n);
    if (un != 0U && un > std::numeric_limits<std::size_t>::max() / un) {
        return false;
    }
    out = un * un;
    return true;
}

[[nodiscard]] bool checked_fold_count(::pls4all::core::Context& ctx,
                                      std::int32_t left,
                                      std::int32_t right) noexcept {
    const auto uleft = static_cast<std::size_t>(left);
    const auto uright = static_cast<std::size_t>(right);
    if (uleft != 0U &&
        uright > std::numeric_limits<std::size_t>::max() / uleft) {
        ctx.set_error("validation plan fold count is too large");
        return false;
    }
    return true;
}

[[nodiscard]] double read_value(const p4a_matrix_view_t& view,
                                std::size_t row,
                                std::size_t col) noexcept {
    const std::int64_t off =
        static_cast<std::int64_t>(row) * view.row_stride +
        static_cast<std::int64_t>(col) * view.col_stride;
    const auto uoff = static_cast<std::size_t>(off);
    if (view.dtype == P4A_DTYPE_F64) {
        const auto* ptr = static_cast<const double*>(view.data);
        return ptr[uoff];
    }
    const auto* ptr = static_cast<const float*>(view.data);
    return static_cast<double>(ptr[uoff]);
}

[[nodiscard]] p4a_status_t validate_numeric_matrix(::pls4all::core::Context& ctx,
                                                   const p4a_matrix_view_t& view,
                                                   const char* name) noexcept {
    const p4a_status_t status = ::pls4all::core::validate_nonnull_view(view);
    if (status != P4A_OK) {
        ctx.set_errorf("%s matrix view is invalid", name);
        return status;
    }
    if (view.dtype != P4A_DTYPE_F64 && view.dtype != P4A_DTYPE_F32) {
        ctx.set_errorf("%s dtype must be f64 or f32", name);
        return P4A_ERR_DTYPE_MISMATCH;
    }
    if (view.rows < 1 || view.cols < 1) {
        ctx.set_errorf("%s matrix must be non-empty", name);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

[[nodiscard]] std::uint64_t splitmix64_next(std::uint64_t& state) noexcept {
    state += UINT64_C(0x9E3779B97F4A7C15);
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31U);
}

[[nodiscard]] std::size_t random_bounded(std::uint64_t& state,
                                         std::size_t bound) noexcept {
    return static_cast<std::size_t>(splitmix64_next(state) %
                                    static_cast<std::uint64_t>(bound));
}

void shuffle_indices(std::vector<std::int64_t>& indices,
                     std::uint64_t& state) noexcept {
    for (std::size_t i = indices.size(); i > 1U; --i) {
        const std::size_t j = random_bounded(state, i);
        std::swap(indices[i - 1U], indices[j]);
    }
}

void fill_fold_from_test_indices(std::int64_t n_samples,
                                 const std::vector<std::int64_t>& test_indices,
                                 ::pls4all::core::ValidationFold& fold) {
    fold.train_indices.clear();
    fold.test_indices = test_indices;
    fold.train_indices.reserve(static_cast<std::size_t>(
        n_samples - static_cast<std::int64_t>(test_indices.size())));
    std::vector<unsigned char> is_test(static_cast<std::size_t>(n_samples), 0U);
    for (const std::int64_t sample : test_indices) {
        is_test[static_cast<std::size_t>(sample)] = 1U;
    }
    for (std::int64_t sample = 0; sample < n_samples; ++sample) {
        if (is_test[static_cast<std::size_t>(sample)] == 0U) {
            fold.train_indices.push_back(sample);
        }
    }
}

void fill_fold_from_train_indices(std::int64_t n_samples,
                                  const std::vector<std::int64_t>& train_indices,
                                  ::pls4all::core::ValidationFold& fold) {
    fold.train_indices = train_indices;
    fold.test_indices.clear();
    fold.test_indices.reserve(static_cast<std::size_t>(
        n_samples - static_cast<std::int64_t>(train_indices.size())));
    std::vector<unsigned char> is_train(static_cast<std::size_t>(n_samples), 0U);
    for (const std::int64_t sample : train_indices) {
        is_train[static_cast<std::size_t>(sample)] = 1U;
    }
    for (std::int64_t sample = 0; sample < n_samples; ++sample) {
        if (is_train[static_cast<std::size_t>(sample)] == 0U) {
            fold.test_indices.push_back(sample);
        }
    }
}

[[nodiscard]] p4a_status_t compute_pairwise_squared_distances(
    ::pls4all::core::Context& ctx,
    const p4a_matrix_view_t& matrix,
    std::vector<double>& out) {
    std::size_t n_values = 0;
    if (!checked_square_size(matrix.rows, n_values)) {
        ctx.set_error("pairwise distance matrix is too large");
        return P4A_ERR_INVALID_ARGUMENT;
    }
    out.assign(n_values, 0.0);
    const auto n_samples = static_cast<std::size_t>(matrix.rows);
    const auto n_cols = static_cast<std::size_t>(matrix.cols);
    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = i + 1U; j < n_samples; ++j) {
            double dist = 0.0;
            for (std::size_t col = 0; col < n_cols; ++col) {
                const double delta = read_value(matrix, i, col) - read_value(matrix, j, col);
                dist += delta * delta;
            }
            out[i * n_samples + j] = dist;
            out[j * n_samples + i] = dist;
        }
    }
    return P4A_OK;
}

[[nodiscard]] double max_distance(const std::vector<double>& distances) noexcept {
    double best = 0.0;
    for (const double value : distances) {
        if (value > best) {
            best = value;
        }
    }
    return best;
}

[[nodiscard]] p4a_status_t select_representative_indices(
    ::pls4all::core::Context& ctx,
    const std::vector<double>& distances,
    std::int64_t n_samples,
    std::int64_t train_count,
    std::vector<std::int64_t>& selected) {
    if (!validate_sample_count(ctx, n_samples, 3)) {
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (train_count < 2) {
        ctx.set_errorf("train_count must be >= 2; got %lld",
                       static_cast<long long>(train_count));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (train_count >= n_samples) {
        ctx.set_error("train_count must leave at least one validation sample");
        return P4A_ERR_INVALID_ARGUMENT;
    }

    const auto n = static_cast<std::size_t>(n_samples);
    std::int64_t first = 0;
    std::int64_t second = 1;
    double best_pair = distances[1U];
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1U; j < n; ++j) {
            const double dist = distances[i * n + j];
            if (dist > best_pair) {
                best_pair = dist;
                first = static_cast<std::int64_t>(i);
                second = static_cast<std::int64_t>(j);
            }
        }
    }

    selected.clear();
    selected.reserve(static_cast<std::size_t>(train_count));
    std::vector<unsigned char> is_selected(n, 0U);
    selected.push_back(first);
    selected.push_back(second);
    is_selected[static_cast<std::size_t>(first)] = 1U;
    is_selected[static_cast<std::size_t>(second)] = 1U;

    while (static_cast<std::int64_t>(selected.size()) < train_count) {
        std::int64_t best_sample = -1;
        double best_nearest = -1.0;
        for (std::size_t sample = 0; sample < n; ++sample) {
            if (is_selected[sample] != 0U) {
                continue;
            }
            double nearest = std::numeric_limits<double>::infinity();
            for (const std::int64_t chosen : selected) {
                const double dist = distances[sample * n + static_cast<std::size_t>(chosen)];
                if (dist < nearest) {
                    nearest = dist;
                }
            }
            if (nearest > best_nearest) {
                best_nearest = nearest;
                best_sample = static_cast<std::int64_t>(sample);
            }
        }
        if (best_sample < 0) {
            ctx.set_error("representative sample selection failed");
            return P4A_ERR_INTERNAL;
        }
        selected.push_back(best_sample);
        is_selected[static_cast<std::size_t>(best_sample)] = 1U;
    }
    return P4A_OK;
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

p4a_status_t make_external_folds_validation_plan(Context& ctx,
                                                std::int64_t n_samples,
                                                const std::int64_t* fold_ids,
                                                std::int32_t n_folds,
                                                ValidationPlan& out) {
    try {
        out = ValidationPlan{};
        if (!validate_sample_count(ctx, n_samples, 2)) {
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (fold_ids == nullptr) {
            ctx.set_error("external fold_ids pointer must not be null");
            return P4A_ERR_NULL_POINTER;
        }
        if (n_folds < 2) {
            ctx.set_errorf("n_folds must be >= 2; got %d", static_cast<int>(n_folds));
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (static_cast<std::int64_t>(n_folds) > n_samples) {
            ctx.set_errorf("n_folds (%d) must be <= n_samples (%lld)",
                           static_cast<int>(n_folds),
                           static_cast<long long>(n_samples));
            return P4A_ERR_INVALID_ARGUMENT;
        }

        const auto ufolds = static_cast<std::size_t>(n_folds);
        std::vector<std::int64_t> fold_counts(ufolds, 0);
        for (std::int64_t sample = 0; sample < n_samples; ++sample) {
            const std::int64_t fold = fold_ids[static_cast<std::size_t>(sample)];
            if (fold < 0 || fold >= static_cast<std::int64_t>(n_folds)) {
                ctx.set_errorf("external fold id at sample %lld is out of range: %lld",
                               static_cast<long long>(sample),
                               static_cast<long long>(fold));
                return P4A_ERR_INVALID_ARGUMENT;
            }
            fold_counts[static_cast<std::size_t>(fold)] += 1;
        }
        for (std::size_t fold = 0; fold < ufolds; ++fold) {
            if (fold_counts[fold] == 0) {
                ctx.set_errorf("external fold %llu has no test samples",
                               static_cast<unsigned long long>(fold));
                return P4A_ERR_INVALID_ARGUMENT;
            }
            if (fold_counts[fold] >= n_samples) {
                ctx.set_errorf("external fold %llu leaves no training samples",
                               static_cast<unsigned long long>(fold));
                return P4A_ERR_INVALID_ARGUMENT;
            }
        }

        out.n_samples = n_samples;
        out.folds.resize(ufolds);
        for (std::size_t fold = 0; fold < ufolds; ++fold) {
            auto& target = out.folds[fold];
            target.train_indices.reserve(static_cast<std::size_t>(n_samples - fold_counts[fold]));
            target.test_indices.reserve(static_cast<std::size_t>(fold_counts[fold]));
            for (std::int64_t sample = 0; sample < n_samples; ++sample) {
                if (fold_ids[static_cast<std::size_t>(sample)] == static_cast<std::int64_t>(fold)) {
                    target.test_indices.push_back(sample);
                } else {
                    target.train_indices.push_back(sample);
                }
            }
        }
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while generating external-fold validation plan");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while generating external-fold validation plan");
        return P4A_ERR_INTERNAL;
    }
}

p4a_status_t make_repeated_kfold_validation_plan(Context& ctx,
                                                std::int64_t n_samples,
                                                std::int32_t n_splits,
                                                std::int32_t n_repeats,
                                                std::uint64_t seed,
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
        if (n_repeats < 1) {
            ctx.set_errorf("n_repeats must be >= 1; got %d", static_cast<int>(n_repeats));
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (!checked_fold_count(ctx, n_splits, n_repeats)) {
            return P4A_ERR_INVALID_ARGUMENT;
        }

        out.n_samples = n_samples;
        out.folds.resize(static_cast<std::size_t>(n_splits) *
                         static_cast<std::size_t>(n_repeats));
        std::vector<std::int64_t> indices(static_cast<std::size_t>(n_samples));
        for (std::int64_t sample = 0; sample < n_samples; ++sample) {
            indices[static_cast<std::size_t>(sample)] = sample;
        }

        std::uint64_t state = seed;
        std::size_t out_fold = 0;
        for (std::int32_t repeat = 0; repeat < n_repeats; ++repeat) {
            std::vector<std::int64_t> shuffled = indices;
            shuffle_indices(shuffled, state);
            const std::int64_t base = n_samples / static_cast<std::int64_t>(n_splits);
            const std::int64_t remainder = n_samples % static_cast<std::int64_t>(n_splits);
            std::int64_t start = 0;
            for (std::int32_t fold = 0; fold < n_splits; ++fold) {
                const std::int64_t size = base + ((static_cast<std::int64_t>(fold) < remainder) ? 1 : 0);
                const std::int64_t stop = start + size;
                std::vector<std::int64_t> test(
                    shuffled.begin() + static_cast<std::ptrdiff_t>(start),
                    shuffled.begin() + static_cast<std::ptrdiff_t>(stop));
                fill_fold_from_test_indices(n_samples, test, out.folds[out_fold]);
                ++out_fold;
                start = stop;
            }
        }
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while generating repeated k-fold validation plan");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while generating repeated k-fold validation plan");
        return P4A_ERR_INTERNAL;
    }
}

p4a_status_t make_monte_carlo_validation_plan(Context& ctx,
                                             std::int64_t n_samples,
                                             std::int64_t test_count,
                                             std::int32_t n_repeats,
                                             std::uint64_t seed,
                                             ValidationPlan& out) {
    try {
        out = ValidationPlan{};
        if (!validate_sample_count(ctx, n_samples, 2)) {
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (test_count <= 0) {
            ctx.set_errorf("test_count must be > 0; got %lld",
                           static_cast<long long>(test_count));
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (test_count >= n_samples) {
            ctx.set_error("Monte-Carlo split must leave at least one training sample");
            return P4A_ERR_INVALID_ARGUMENT;
        }
        if (n_repeats < 1) {
            ctx.set_errorf("n_repeats must be >= 1; got %d", static_cast<int>(n_repeats));
            return P4A_ERR_INVALID_ARGUMENT;
        }

        out.n_samples = n_samples;
        out.folds.resize(static_cast<std::size_t>(n_repeats));
        std::vector<std::int64_t> indices(static_cast<std::size_t>(n_samples));
        for (std::int64_t sample = 0; sample < n_samples; ++sample) {
            indices[static_cast<std::size_t>(sample)] = sample;
        }

        std::uint64_t state = seed;
        for (std::int32_t repeat = 0; repeat < n_repeats; ++repeat) {
            std::vector<std::int64_t> shuffled = indices;
            shuffle_indices(shuffled, state);
            std::vector<std::int64_t> test(
                shuffled.begin(),
                shuffled.begin() + static_cast<std::ptrdiff_t>(test_count));
            fill_fold_from_test_indices(n_samples,
                                        test,
                                        out.folds[static_cast<std::size_t>(repeat)]);
        }
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while generating Monte-Carlo validation plan");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while generating Monte-Carlo validation plan");
        return P4A_ERR_INTERNAL;
    }
}

p4a_status_t make_kennard_stone_validation_plan(Context& ctx,
                                               const p4a_matrix_view_t& X,
                                               std::int64_t train_count,
                                               ValidationPlan& out) {
    try {
        out = ValidationPlan{};
        const p4a_status_t status = validate_numeric_matrix(ctx, X, "X");
        if (status != P4A_OK) {
            return status;
        }

        std::vector<double> distances;
        p4a_status_t dist_status = compute_pairwise_squared_distances(ctx, X, distances);
        if (dist_status != P4A_OK) {
            return dist_status;
        }
        std::vector<std::int64_t> selected;
        dist_status = select_representative_indices(ctx,
                                                    distances,
                                                    X.rows,
                                                    train_count,
                                                    selected);
        if (dist_status != P4A_OK) {
            return dist_status;
        }

        out.n_samples = X.rows;
        out.folds.resize(1U);
        fill_fold_from_train_indices(X.rows, selected, out.folds[0]);
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while generating Kennard-Stone validation plan");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while generating Kennard-Stone validation plan");
        return P4A_ERR_INTERNAL;
    }
}

p4a_status_t make_spxy_validation_plan(Context& ctx,
                                      const p4a_matrix_view_t& X,
                                      const p4a_matrix_view_t& Y,
                                      std::int64_t train_count,
                                      ValidationPlan& out) {
    try {
        out = ValidationPlan{};
        p4a_status_t status = validate_numeric_matrix(ctx, X, "X");
        if (status != P4A_OK) {
            return status;
        }
        status = validate_numeric_matrix(ctx, Y, "Y");
        if (status != P4A_OK) {
            return status;
        }
        if (Y.rows != X.rows) {
            ctx.set_errorf("SPXY X/Y row counts differ: %lld vs %lld",
                           static_cast<long long>(X.rows),
                           static_cast<long long>(Y.rows));
            return P4A_ERR_INVALID_ARGUMENT;
        }

        std::vector<double> x_distances;
        std::vector<double> y_distances;
        status = compute_pairwise_squared_distances(ctx, X, x_distances);
        if (status != P4A_OK) {
            return status;
        }
        status = compute_pairwise_squared_distances(ctx, Y, y_distances);
        if (status != P4A_OK) {
            return status;
        }

        const double max_x = max_distance(x_distances);
        const double max_y = max_distance(y_distances);
        std::vector<double> combined(x_distances.size(), 0.0);
        for (std::size_t i = 0; i < combined.size(); ++i) {
            combined[i] = (max_x > 0.0 ? x_distances[i] / max_x : 0.0) +
                          (max_y > 0.0 ? y_distances[i] / max_y : 0.0);
        }

        std::vector<std::int64_t> selected;
        status = select_representative_indices(ctx,
                                               combined,
                                               X.rows,
                                               train_count,
                                               selected);
        if (status != P4A_OK) {
            return status;
        }

        out.n_samples = X.rows;
        out.folds.resize(1U);
        fill_fold_from_train_indices(X.rows, selected, out.folds[0]);
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while generating SPXY validation plan");
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while generating SPXY validation plan");
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
