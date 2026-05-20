// SPDX-License-Identifier: CECILL-2.1

#include "core/emcuve_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#include "core/uve_selection.hpp"

namespace {

[[nodiscard]] std::vector<std::int64_t> rank_selected(
    const std::vector<double>& vote_frequencies,
    const std::vector<double>& mean_scores,
    double vote_threshold) {
    std::vector<std::int64_t> selected;
    selected.reserve(vote_frequencies.size());
    for (std::size_t feature = 0; feature < vote_frequencies.size(); ++feature) {
        if (vote_frequencies[feature] >= vote_threshold) {
            selected.push_back(static_cast<std::int64_t>(feature));
        }
    }
    std::stable_sort(selected.begin(),
                     selected.end(),
                     [&vote_frequencies, &mean_scores](std::int64_t lhs, std::int64_t rhs) {
                         const auto left = static_cast<std::size_t>(lhs);
                         const auto right = static_cast<std::size_t>(rhs);
                         if (vote_frequencies[left] == vote_frequencies[right]) {
                             if (mean_scores[left] == mean_scores[right]) {
                                 return lhs < rhs;
                             }
                             return mean_scores[left] > mean_scores[right];
                         }
                         return vote_frequencies[left] > vote_frequencies[right];
                     });
    return selected;
}

[[nodiscard]] p4a_status_t validate_request(::pls4all::core::Context& ctx,
                                            std::int32_t n_ensembles,
                                            double vote_threshold) {
    if (n_ensembles <= 0) {
        ctx.set_errorf("n_ensembles must be >= 1; got %d",
                       static_cast<int>(n_ensembles));
        return P4A_ERR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(vote_threshold) || vote_threshold <= 0.0 || vote_threshold > 1.0) {
        ctx.set_errorf("vote_threshold must be finite and in (0, 1]; got %.17g",
                       vote_threshold);
        return P4A_ERR_INVALID_ARGUMENT;
    }
    return P4A_OK;
}

}  // namespace

namespace pls4all::core {

p4a_status_t select_by_emcuve(Context& ctx,
                              const Config& cfg,
                              const p4a_matrix_view_t& X,
                              const p4a_matrix_view_t& Y,
                              const ValidationPlan& plan,
                              std::int32_t noise_features,
                              std::uint64_t noise_seed,
                              std::int32_t n_ensembles,
                              double vote_threshold,
                              EmcuveSelectionResult& out) {
    try {
        out = EmcuveSelectionResult{};
        p4a_status_t status = validate_request(ctx, n_ensembles, vote_threshold);
        if (status != P4A_OK) {
            return status;
        }

        const std::size_t ensemble_count = static_cast<std::size_t>(n_ensembles);
        std::vector<double> stability_sums;
        std::vector<double> vote_counts;
        out.noise_thresholds.assign(ensemble_count, 0.0);

        for (std::size_t ensemble = 0; ensemble < ensemble_count; ++ensemble) {
            const std::uint64_t member_seed =
                noise_seed + static_cast<std::uint64_t>(ensemble) + UINT64_C(1);
            UveSelectionResult member;
            status = select_by_uve(ctx,
                                   cfg,
                                   X,
                                   Y,
                                   plan,
                                   noise_features,
                                   member_seed,
                                   member);
            if (status != P4A_OK) {
                out = EmcuveSelectionResult{};
                return status;
            }

            if (ensemble == 0U) {
                stability_sums.assign(member.real_stability_scores.size(), 0.0);
                vote_counts.assign(member.real_stability_scores.size(), 0.0);
                out.n_features = member.n_features;
                out.n_targets = member.n_targets;
                out.n_repeats = member.n_repeats;
                out.n_noise_features = member.n_noise_features;
            } else if (member.real_stability_scores.size() != stability_sums.size() ||
                       member.n_features != out.n_features ||
                       member.n_targets != out.n_targets ||
                       member.n_repeats != out.n_repeats ||
                       member.n_noise_features != out.n_noise_features) {
                ctx.set_error("EMCUVE member result dimensions are inconsistent");
                out = EmcuveSelectionResult{};
                return P4A_ERR_INTERNAL;
            }

            out.noise_thresholds[ensemble] = member.noise_threshold;
            for (std::size_t feature = 0; feature < stability_sums.size(); ++feature) {
                stability_sums[feature] += member.real_stability_scores[feature];
            }
            for (std::int64_t feature : member.selected_indices) {
                if (feature < 0 ||
                    static_cast<std::size_t>(feature) >= vote_counts.size()) {
                    ctx.set_error("EMCUVE member selected index is out of range");
                    out = EmcuveSelectionResult{};
                    return P4A_ERR_INTERNAL;
                }
                vote_counts[static_cast<std::size_t>(feature)] += 1.0;
            }
        }

        const double inv_ensemble = 1.0 / static_cast<double>(ensemble_count);
        out.mean_real_stability_scores.assign(stability_sums.size(), 0.0);
        out.vote_frequencies.assign(vote_counts.size(), 0.0);
        for (std::size_t feature = 0; feature < stability_sums.size(); ++feature) {
            out.mean_real_stability_scores[feature] = stability_sums[feature] * inv_ensemble;
            out.vote_frequencies[feature] = vote_counts[feature] * inv_ensemble;
        }

        out.selected_indices =
            rank_selected(out.vote_frequencies, out.mean_real_stability_scores, vote_threshold);
        out.selected_count = static_cast<std::int32_t>(out.selected_indices.size());
        out.n_ensembles = n_ensembles;
        out.noise_seed = noise_seed;
        out.vote_threshold = vote_threshold;
        ctx.clear_error();
        return P4A_OK;
    } catch (const std::bad_alloc&) {
        ctx.set_error("out of memory while running EMCUVE selection");
        out = EmcuveSelectionResult{};
        return P4A_ERR_OUT_OF_MEMORY;
    } catch (...) {
        ctx.set_error("unexpected exception while running EMCUVE selection");
        out = EmcuveSelectionResult{};
        return P4A_ERR_INTERNAL;
    }
}

}  // namespace pls4all::core
