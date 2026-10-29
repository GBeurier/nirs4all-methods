// SPDX-License-Identifier: CECILL-2.1
// Shared descriptive protocols. Bindings only marshal named native outputs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/diagnostics/preflight_pca_budget.hpp"
#include "core/estimator/generated_factories.hpp"
#include "core/estimator/state_io.hpp"
#include "core/filters/_vendored/lof.h"
#include "core/method_result.hpp"
#include "core/preprocessing/feature_selection/flexible_pca.h"

namespace n4m::estimator {
namespace {
constexpr std::int32_t OK = 0, TOO_FEW = 1, ZERO_NORM = 2, DEGENERATE = 3, UNDEFINED = 4,
                       MISSING_PARAMETER = 5;

double quantile(std::vector<double> x, double q) {
    std::sort(x.begin(), x.end());
    const double at = static_cast<double>(x.size() - 1) * q;
    const auto lo = static_cast<std::size_t>(std::floor(at));
    const auto hi = static_cast<std::size_t>(std::ceil(at));
    return x[lo] * (1.0 - (at - static_cast<double>(lo))) + x[hi] * (at - static_cast<double>(lo));
}
struct Signal {
    std::string name;
    std::vector<double> values;
    std::vector<std::int32_t> rows, status;
    Signal(std::string key, std::int64_t n)
        : name(std::move(key)), status(static_cast<std::size_t>(n), TOO_FEW) {
    }
    void set(std::int64_t original_row, double value, std::int32_t state = OK) {
        const auto row = static_cast<std::size_t>(original_row);
        status[row] = state == OK && !std::isfinite(value) ? UNDEFINED : state;
        if (status[row] == OK) {
            values.push_back(value);
            rows.push_back(static_cast<std::int32_t>(row));
        }
    }
    void write(n4m_method_result_s& out) const {
        out.set_int_vector(name + "_status", status);
        if (!values.empty()) {
            out.set_double_matrix(
                name + "_values", values, static_cast<std::int64_t>(values.size()), 1);
            out.set_int_vector(name + "_rows", rows);
            out.set_scalar(name + "_median", quantile(values, .5));
            out.set_scalar(name + "_p95", quantile(values, .95));
        }
        out.set_scalar(name + "_n", static_cast<double>(values.size()));
        out.set_int_vector(name + "_aggregate_status", {values.empty() ? TOO_FEW : OK});
    }
};
void scalar(n4m_method_result_s& out,
            const std::string& key,
            double value,
            std::int32_t status = OK) {
    if (status == OK && !std::isfinite(value))
        status = UNDEFINED;
    if (status == OK)
        out.set_scalar(key, value);
    out.set_int_vector(key + "_status", {status});
}
struct Data {
    std::int64_t n, p;
    std::vector<double> x;
    std::vector<std::int64_t> mask;
    Data(const Params& params, const FitInputs& in) : n(in.X->rows), p(in.X->cols) {
        if (n < 1 || p < 1 || n > INT32_MAX || p > INT32_MAX || n > INT64_MAX / p)
            throw std::invalid_argument("diagnostic dimensions exceed supported range");
        if (in.X->dtype != N4M_DTYPE_F64)
            throw std::invalid_argument("X must be F64");
        std::vector<double> storage;
        const auto v = contiguous_view(*in.X, storage);
        const auto* ptr = static_cast<const double*>(v.data);
        x.assign(ptr, ptr + n * p);
        for (double z : x)
            if (!std::isfinite(z))
                throw std::invalid_argument(
                    "X must be finite; use zero-fill and explicit valid_mask");
        mask = params.get_ints("valid_mask");
        if (mask.empty())
            mask.assign(static_cast<std::size_t>(n * p), 1);
        if (static_cast<std::int64_t>(mask.size()) != n * p)
            throw std::invalid_argument("valid_mask must have n*p values");
    }
    bool valid(std::int64_t i, std::int64_t j) const {
        return mask[static_cast<std::size_t>(i * p + j)] != 0;
    }
    double at(std::int64_t i, std::int64_t j) const {
        return x[static_cast<std::size_t>(i * p + j)];
    }
};
std::unique_ptr<n4m_method_result_s> result(const Data& d) {
    auto out = std::make_unique<n4m_method_result_s>();
    out->set_scalar("protocol_version", 1);
    out->set_scalar("n_rows", static_cast<double>(d.n));
    return out;
}
template<class F>
n4m_status_t guarded(n4m_context_t* ctx, n4m_method_result_t** out, F fn) {
    try {
        *out = fn().release();
        return N4M_OK;
    } catch (const std::invalid_argument& e) {
        set_error(ctx, e.what());
        return N4M_ERR_INVALID_ARGUMENT;
    }
}
}  // namespace

n4m_status_t run_spectral_reference_qc(n4m_context_t* ctx,
                                       const Params& params,
                                       const FitInputs& in,
                                       n4m_method_result_t** output) {
    return guarded(ctx, output, [&] {
        const Data d(params, in);
        auto out = result(d);
        std::vector<double> ref(static_cast<std::size_t>(d.p), 0);
        std::vector<std::int32_t> counts(static_cast<std::size_t>(d.p), 0), valid(counts);
        if (params.get_int("reference_mode") == 0) {
            if (in.X_target)
                throw std::invalid_argument("dataset_mean does not accept X_target");
            for (std::int64_t j = 0; j < d.p; ++j) {
                // Incremental mean avoids overflow in a sum of large finite signals.
                for (std::int64_t i = 0; i < d.n; ++i)
                    if (d.valid(i, j)) {
                        ++counts[static_cast<std::size_t>(j)];
                        const double weight = 1.0 / counts[static_cast<std::size_t>(j)];
                        ref[static_cast<std::size_t>(j)] =
                            ref[static_cast<std::size_t>(j)] * (1.0 - weight) + d.at(i, j) * weight;
                    }
                valid[static_cast<std::size_t>(j)] =
                    counts[static_cast<std::size_t>(j)] > 0 ? 1 : 0;
            }
        } else {
            if (!in.X_target || in.X_target->rows != 1 || in.X_target->cols != d.p
                || in.X_target->dtype != N4M_DTYPE_F64)
                throw std::invalid_argument("supplied requires finite F64 X_target with shape 1*p");
            std::vector<double> storage;
            const auto v = contiguous_view(*in.X_target, storage);
            const auto* ptr = static_cast<const double*>(v.data);
            ref.assign(ptr, ptr + d.p);
            auto rm = params.get_ints("reference_valid_mask");
            if (rm.empty())
                rm.assign(static_cast<std::size_t>(d.p), 1);
            if (static_cast<std::int64_t>(rm.size()) != d.p)
                throw std::invalid_argument("reference_valid_mask must have p values");
            for (std::int64_t j = 0; j < d.p; ++j) {
                if (!std::isfinite(ref[static_cast<std::size_t>(j)]))
                    throw std::invalid_argument("reference must be finite");
                valid[static_cast<std::size_t>(j)] = counts[static_cast<std::size_t>(j)] =
                    static_cast<std::int32_t>(rm[static_cast<std::size_t>(j)]);
                if (!valid[static_cast<std::size_t>(j)])
                    ref[static_cast<std::size_t>(j)] = 0;
            }
        }
        Signal rms("rms", d.n), sam("sam", d.n), offset("affine_offset", d.n),
            gain("affine_gain", d.n), abs_offset("affine_abs_offset", d.n),
            abs_delta("affine_abs_gain_delta", d.n), residual("affine_residual_rms", d.n);
        std::vector<std::int32_t> common(static_cast<std::size_t>(d.n));
        for (std::int64_t i = 0; i < d.n; ++i) {
            std::vector<double> x, r;
            for (std::int64_t j = 0; j < d.p; ++j)
                if (d.valid(i, j) && valid[static_cast<std::size_t>(j)]) {
                    x.push_back(d.at(i, j));
                    r.push_back(ref[static_cast<std::size_t>(j)]);
                }
            common[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(x.size());
            if (x.empty())
                continue;
            double squared = 0, xx = 0, rr = 0, dot = 0, xm = 0, rm = 0;
            const double size = static_cast<double>(x.size());
            for (std::size_t j = 0; j < x.size(); ++j) {
                const double diff = x[j] - r[j];
                squared += diff * diff;
                xx += x[j] * x[j];
                rr += r[j] * r[j];
                dot += x[j] * r[j];
                xm += x[j] / size;
                rm += r[j] / size;
            }
            rms.set(i, std::sqrt(squared / size));
            if (x.size() < 2)
                continue;
            sam.set(i,
                    (xx > 0 && rr > 0)
                        ? std::acos(std::clamp(dot / std::sqrt(xx) / std::sqrt(rr), -1.0, 1.0))
                        : 0,
                    xx == 0 || rr == 0                                                  ? ZERO_NORM
                    : (!std::isfinite(xx) || !std::isfinite(rr) || !std::isfinite(dot)) ? UNDEFINED
                                                                                        : OK);
            double numerator = 0, denominator = 0;
            for (std::size_t j = 0; j < x.size(); ++j) {
                numerator += (x[j] - xm) * (r[j] - rm);
                denominator += (r[j] - rm) * (r[j] - rm);
            }
            const auto status = denominator == 0              ? DEGENERATE
                                : !std::isfinite(denominator) ? UNDEFINED
                                                              : OK;
            const double b = status == OK ? numerator / denominator : 0, a = xm - b * rm;
            double res = 0;
            for (std::size_t j = 0; j < x.size(); ++j) {
                const double v = x[j] - a - b * r[j];
                res += v * v;
            }
            offset.set(i, a, status);
            gain.set(i, b, status);
            abs_offset.set(i, std::abs(a), status);
            abs_delta.set(i, std::abs(b - 1), status);
            residual.set(i, std::sqrt(res / size), status);
        }
        out->set_double_matrix("reference", ref, 1, d.p);
        out->set_int_vector("reference_valid_mask", valid);
        out->set_int_vector("reference_counts", counts);
        out->set_int_vector("common_valid_counts", common);
        for (const auto* s : {&rms, &sam, &offset, &gain, &abs_offset, &abs_delta, &residual})
            s->write(*out);
        return out;
    });
}

n4m_status_t run_spectral_events_qc(n4m_context_t* ctx,
                                    const Params& params,
                                    const FitInputs& in,
                                    n4m_method_result_t** output) {
    return guarded(ctx, output, [&] {
        const Data d(params, in);
        auto out = result(d);
        const auto axis = params.get_doubles("axis_values");
        const bool physical = params.get_int("axis_basis") == 1;
        if (physical) {
            if (static_cast<std::int64_t>(axis.size()) != d.p)
                throw std::invalid_argument("physical axis needs p values");
            int direction = 0;
            for (std::int64_t j = 1; j < d.p; ++j) {
                const double delta =
                    axis[static_cast<std::size_t>(j)] - axis[static_cast<std::size_t>(j - 1)];
                if (!std::isfinite(delta) || delta == 0)
                    throw std::invalid_argument("axis must be finite and strictly monotonic");
                const int sign = delta > 0 ? 1 : -1;
                if (direction && direction != sign)
                    throw std::invalid_argument("axis must be strictly monotonic");
                direction = sign;
            }
        }
        const double threshold = params.get_double("outlier_threshold");
        if (!(threshold > 0))
            throw std::invalid_argument("threshold must be positive");
        const double lower = params.get_double("clip_lower"),
                     upper = params.get_double("clip_upper"),
                     tolerance = params.get_double("clip_tolerance");
        const bool bounded = std::isfinite(lower) || std::isfinite(upper);
        if (std::isfinite(lower) && std::isfinite(upper) && lower >= upper)
            throw std::invalid_argument("clip_lower must be less than clip_upper");
        Signal jumps("jump_count", d.n), jump_frac("jump_fraction", d.n),
            spikes("spike_count", d.n), spike_frac("spike_fraction", d.n),
            clips("declared_clip_count", d.n), clip_frac("declared_clip_fraction", d.n);
        std::vector<std::int32_t> jd(static_cast<std::size_t>(d.n)), sd(jd), cd(jd);
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (std::int64_t i = 0; i < d.n; ++i) {
            std::vector<double> diff, second;
            double clipped = 0;
            for (std::int64_t j = 0; j < d.p; ++j)
                if (d.valid(i, j)) {
                    lo = std::min(lo, d.at(i, j));
                    hi = std::max(hi, d.at(i, j));
                    ++cd[static_cast<std::size_t>(i)];
                    if ((std::isfinite(lower) && d.at(i, j) <= lower + tolerance)
                        || (std::isfinite(upper) && d.at(i, j) >= upper - tolerance))
                        ++clipped;
                    if (j + 1 < d.p && d.valid(i, j + 1)) {
                        const double slope = (d.at(i, j + 1) - d.at(i, j))
                                             / (physical ? axis[static_cast<std::size_t>(j + 1)]
                                                               - axis[static_cast<std::size_t>(j)]
                                                         : 1.0);
                        diff.push_back(slope);
                        if (j + 2 < d.p && d.valid(i, j + 2)) {
                            const double next =
                                (d.at(i, j + 2) - d.at(i, j + 1))
                                / (physical ? axis[static_cast<std::size_t>(j + 2)]
                                                  - axis[static_cast<std::size_t>(j + 1)]
                                            : 1.0);
                            second.push_back(physical ? 2 * (next - slope)
                                                            / (axis[static_cast<std::size_t>(j + 2)]
                                                               - axis[static_cast<std::size_t>(j)])
                                                      : next - slope);
                        }
                    }
                }
            auto count = [&](const std::vector<double>& values,
                             Signal& cnt,
                             Signal& frac,
                             std::vector<std::int32_t>& den) {
                den[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(values.size());
                if (values.size() < 3)
                    return;
                if (std::any_of(values.begin(), values.end(), [](double v) {
                        return !std::isfinite(v);
                    })) {
                    cnt.set(i, 0, UNDEFINED);
                    frac.set(i, 0, UNDEFINED);
                    return;
                }
                const double center = quantile(values, .5);
                std::vector<double> absolute;
                for (double v : values)
                    absolute.push_back(std::abs(v - center));
                double scale = 1.4826 * quantile(absolute, .5);
                if (scale == 0) {
                    const double mean = std::accumulate(values.begin(), values.end(), 0.0)
                                        / static_cast<double>(values.size());
                    double sum = 0;
                    for (double v : values)
                        sum += (v - mean) * (v - mean);
                    scale = std::sqrt(sum / static_cast<double>(values.size() - 1));
                }
                if (!std::isfinite(scale)) {
                    cnt.set(i, 0, UNDEFINED);
                    frac.set(i, 0, UNDEFINED);
                    return;
                }
                double c = 0;
                if (scale > 0)
                    for (double v : values)
                        if (std::abs(v - center) > threshold * scale)
                            ++c;
                cnt.set(i, c);
                frac.set(i, c / static_cast<double>(values.size()));
            };
            count(diff, jumps, jump_frac, jd);
            count(second, spikes, spike_frac, sd);
            const auto status = !bounded                               ? MISSING_PARAMETER
                                : cd[static_cast<std::size_t>(i)] == 0 ? TOO_FEW
                                                                       : OK;
            clips.set(i, clipped, status);
            clip_frac.set(
                i,
                cd[static_cast<std::size_t>(i)] ? clipped / cd[static_cast<std::size_t>(i)] : 0,
                status);
        }
        const double eps = std::max(1e-12, std::max(std::abs(lo), std::abs(hi)) * 1e-8);
        double extrema = 0, total = 0;
        for (std::int64_t i = 0; i < d.n; ++i)
            for (std::int64_t j = 0; j < d.p; ++j)
                if (d.valid(i, j)) {
                    ++total;
                    if (std::abs(d.at(i, j) - lo) <= eps || std::abs(d.at(i, j) - hi) <= eps)
                        ++extrema;
                }
        scalar(*out,
               "observed_extrema_occupancy_fraction",
               total > 0 ? extrema / total : 0,
               total > 0 ? OK : TOO_FEW);
        out->set_scalar("observed_extrema_occupancy_count", extrema);
        out->set_scalar("observed_extrema_occupancy_n", total);
        out->set_int_vector("observed_extrema_occupancy_status", {total > 0 ? OK : TOO_FEW});
        for (const auto* s : {&jumps, &jump_frac, &spikes, &spike_frac, &clips, &clip_frac})
            s->write(*out);
        out->set_int_vector("jump_eligible_counts", jd);
        out->set_int_vector("spike_eligible_counts", sd);
        out->set_int_vector("clip_eligible_counts", cd);
        auto pooled = [&](const std::string& key,
                          const Signal& counts_signal,
                          const std::vector<std::int32_t>& eligible) {
            double count = 0, denominator = 0;
            for (std::size_t row = 0; row < counts_signal.rows.size(); ++row) {
                count += counts_signal.values[row];
                denominator += eligible[static_cast<std::size_t>(counts_signal.rows[row])];
            }
            const auto status = denominator > 0                ? OK
                                : counts_signal.status.empty() ? TOO_FEW
                                                               : counts_signal.status[0];
            scalar(*out, "global_" + key + "_count", count, status);
            scalar(*out, "global_" + key + "_eligible", denominator, status);
            scalar(*out,
                   "global_" + key + "_fraction",
                   denominator > 0 ? count / denominator : 0,
                   status);
        };
        pooled("jump", jumps, jd);
        pooled("spike", spikes, sd);
        pooled("declared_clip", clips, cd);

        return out;
    });
}

n4m_status_t run_spectral_noise_proxy(n4m_context_t* ctx,
                                      const Params& params,
                                      const FitInputs& in,
                                      n4m_method_result_t** output) {
    return guarded(ctx, output, [&] {
        const Data d(params, in);
        auto out = result(d);
        Signal noise("noise_sigma", d.n), signal("signal_mean_abs", d.n),
            ratio("signal_noise_ratio", d.n), db("signal_noise_db", d.n);
        double global_signal = 0, total = 0;
        for (std::int64_t i = 0; i < d.n; ++i) {
            std::vector<double> second;
            double amplitude = 0, count = 0;
            for (std::int64_t j = 0; j < d.p; ++j)
                if (d.valid(i, j)) {
                    amplitude += std::abs(d.at(i, j));
                    ++count;
                    if (j + 2 < d.p && d.valid(i, j + 1) && d.valid(i, j + 2))
                        second.push_back(d.at(i, j + 2) - 2 * d.at(i, j + 1) + d.at(i, j));
                }
            global_signal += amplitude;
            total += count;
            if (count == 0)
                continue;
            signal.set(i, amplitude / count);
            if (second.empty())
                continue;
            if (std::any_of(second.begin(), second.end(), [](double v) {
                    return !std::isfinite(v);
                })) {
                noise.set(i, 0, UNDEFINED);
                ratio.set(i, 0, UNDEFINED);
                db.set(i, 0, UNDEFINED);
                continue;
            }
            const double center = quantile(second, .5);
            for (double& v : second)
                v = std::abs(v - center);
            const double sigma = quantile(second, .5) * 1.4826 / std::sqrt(6.0),
                         snr = sigma > 0 ? (amplitude / count) / sigma : 0;
            noise.set(i, sigma);
            const auto status = sigma > 0 && amplitude > 0 ? OK : UNDEFINED;
            ratio.set(i, snr, status);
            db.set(i, snr > 0 ? 20 * std::log10(snr) : 0, status);
        }
        const auto noise_status = noise.values.empty() ? TOO_FEW : OK;
        const double sigma = noise.values.empty() ? 0 : quantile(noise.values, .5),
                     amplitude = total > 0 ? global_signal / total : 0,
                     snr = sigma > 0 ? amplitude / sigma : 0;
        scalar(*out, "global_noise_sigma", sigma, noise_status);
        scalar(*out, "global_signal_mean_abs", amplitude, total > 0 ? OK : TOO_FEW);
        const auto status = noise_status != OK           ? noise_status
                            : sigma > 0 && amplitude > 0 ? OK
                                                         : UNDEFINED;
        scalar(*out, "global_signal_noise_ratio", snr, status);
        scalar(*out, "global_signal_noise_db", snr > 0 ? 20 * std::log10(snr) : 0, status);
        for (const auto* s : {&noise, &signal, &ratio, &db})
            s->write(*out);
        return out;
    });
}

n4m_status_t run_local_density(n4m_context_t* ctx,
                               const Params& params,
                               const FitInputs& in,
                               n4m_method_result_t** output) {
    return guarded(ctx, output, [&] {
        const Data d(params, in);
        auto out = result(d);
        if (std::any_of(d.mask.begin(), d.mask.end(), [](std::int64_t v) {
                return v != 1;
            }))
            throw std::invalid_argument("LOF requires complete data; no implicit imputation");
        const bool novelty = params.get_int("mode") == 1;
        const n4m_matrix_view_t* reference = novelty ? in.X_target : in.X;
        if (!reference || reference->dtype != N4M_DTYPE_F64 || reference->cols != d.p)
            throw std::invalid_argument(
                "novelty LOF requires finite X_target with matching features");
        if (!novelty && in.X_target)
            throw std::invalid_argument("training LOF does not accept X_target");
        Signal scores("lof", d.n);
        if (reference->rows < 2) {
            scores.write(*out);
            out->set_scalar("effective_neighbors", 0);
            out->set_scalar("reference_rows", static_cast<double>(reference->rows));
            return out;
        }
        if (reference->rows > INT32_MAX)
            throw std::invalid_argument("LOF reference row count exceeds supported range");
        std::vector<double> storage;
        const auto view = contiguous_view(*reference, storage);
        const auto* x = static_cast<const double*>(view.data);
        for (std::int64_t j = 0; j < reference->rows * d.p; ++j)
            if (!std::isfinite(x[j]))
                throw std::invalid_argument("LOF reference must be finite");
        const auto k =
            static_cast<int>(std::min(params.get_int("n_neighbors"), reference->rows - 1));
        std::unique_ptr<n4m_lof_t, decltype(&n4m_lof_free)> lof(n4m_lof_new(), n4m_lof_free);
        if (!lof)
            throw std::bad_alloc();
        auto st = n4m_lof_fit(lof.get(), x, reference->rows, d.p, k);
        if (st != N4M_OK)
            throw std::invalid_argument("native LOF fit failed");
        std::vector<double> values(static_cast<std::size_t>(d.n));
        st = novelty ? n4m_lof_score(lof.get(), d.x.data(), d.n, d.p, values.data())
                     : n4m_lof_training_scores(lof.get(), values.data());
        if (st != N4M_OK)
            throw std::invalid_argument("native LOF score failed");
        for (std::int64_t i = 0; i < d.n; ++i)
            scores.set(i, values[static_cast<std::size_t>(i)]);
        scores.write(*out);
        out->set_scalar("effective_neighbors", static_cast<double>(k));
        out->set_scalar("reference_rows", static_cast<double>(reference->rows));
        return out;
    });
}

n4m_status_t run_pca_reference_qc(n4m_context_t* ctx,
                                  const Params& params,
                                  const FitInputs& in,
                                  n4m_method_result_t** output) {
    // Bound the original input and its Gram-size computation before allocations.
    if (!in.X_target)
        return N4M_ERR_INVALID_ARGUMENT;
    const auto nr = in.X_target->rows, p = in.X_target->cols, n = in.X->rows;
    const auto requested = params.get_int("n_components");
    if (nr >= 2 && p >= 1 && n >= 1) {
        const auto budget = ::n4m::core::preflight_pca_budget(nr, p, requested);
        const long double cells = (static_cast<long double>(nr) + static_cast<long double>(n))
                                  * static_cast<long double>(p);
        const long double extra_work =
            8.0L * (static_cast<long double>(nr) + static_cast<long double>(n))
            * static_cast<long double>(requested) * static_cast<long double>(p) / 16.0L;
        const long double extra_doubles =
            4.0L * cells
            + 6.0L * (static_cast<long double>(nr) + static_cast<long double>(n))
                  * static_cast<long double>(requested);
        if (cells > 50000000.0L || budget.work + extra_work > ::n4m::core::preflight_pca_max_work
            || budget.doubles + extra_doubles > ::n4m::core::preflight_pca_max_doubles) {
            set_error(
                ctx,
                "PCA reference exceeds the whole-computation budget (2.4 billion multiply-add equivalents including eigensolver ceiling, 1 GiB buffers, 50 million cells); reduce reference/query rows, features or components; exact full moments were not requested");
            return N4M_ERR_UNSUPPORTED;
        }
    }
    return guarded(ctx, output, [&] {
        const Data query(params, in);
        auto out = result(query);
        if (nr < 2 || p != query.p || in.X_target->dtype != N4M_DTYPE_F64)
            throw std::invalid_argument(
                "PCA requires at least two finite reference rows and equal features");
        if (std::any_of(query.mask.begin(), query.mask.end(), [](std::int64_t v) {
                return v != 1;
            }))
            throw std::invalid_argument(
                "PCA requires a declared upstream train-fitted missingness policy; no implicit imputation");
        if (requested > std::min(nr - 1, p))
            throw std::invalid_argument("n_components exceeds centered reference rank bound");
        std::vector<double> storage;
        const auto reference = contiguous_view(*in.X_target, storage);
        const auto* x = static_cast<const double*>(reference.data);
        for (std::int64_t j = 0; j < nr * p; ++j)
            if (!std::isfinite(x[j]))
                throw std::invalid_argument("PCA reference must be finite");
        using State = n4m_pp_flex_pca_state_t;
        std::unique_ptr<State, decltype(&n4m_pp_flex_pca_state_free)> state(
            n4m_pp_flex_pca_state_new(static_cast<double>(requested)), n4m_pp_flex_pca_state_free);
        if (!state)
            throw std::bad_alloc();
        const auto fitted = n4m_pp_flex_pca_state_fit_exact(state.get(), x, nr, p);
        if (fitted != N4M_OK)
            throw std::invalid_argument("native FlexiblePCA fit failed");
        const auto k = n4m_pp_flex_pca_state_n_components(state.get());
        const double* mean = nullptr;
        const double* loadings = nullptr;
        if (n4m_pp_flex_pca_state_reference(state.get(), &mean, &loadings) != N4M_OK)
            throw std::logic_error("fitted PCA reference unavailable");
        std::vector<double> train_scores(static_cast<std::size_t>(nr * k)),
            query_scores(static_cast<std::size_t>(n * k)), eigen(static_cast<std::size_t>(k));
        if (n4m_pp_flex_pca_state_apply(state.get(), x, nr, p, k, train_scores.data()) != N4M_OK
            || n4m_pp_flex_pca_state_apply(
                   state.get(), query.x.data(), n, p, k, query_scores.data())
                   != N4M_OK)
            throw std::invalid_argument("native PCA projection failed");
        for (std::int64_t a = 0; a < k; ++a) {
            double sum = 0;
            for (std::int64_t i = 0; i < nr; ++i) {
                const double value = train_scores[static_cast<std::size_t>(i * k + a)];
                sum += value * value;
            }
            eigen[static_cast<std::size_t>(a)] = sum / static_cast<double>(nr - 1);
        }
        const double resolution = 16 * static_cast<double>(std::max(nr, p))
                                  * std::numeric_limits<double>::epsilon() * eigen[0];
        std::vector<std::int32_t> resolved(static_cast<std::size_t>(k));
        std::int32_t effective = 0;
        for (std::int64_t a = 0; a < k; ++a)
            if (std::isfinite(eigen[static_cast<std::size_t>(a)])
                && eigen[static_cast<std::size_t>(a)] > resolution) {
                resolved[static_cast<std::size_t>(a)] = 1;
                ++effective;
            }
        Signal q("pca_q", n), t2("pca_t2", n), train_q("reference_q", nr),
            train_t2("reference_t2", nr);
        auto diagnostics = [&](const double* values,
                               std::int64_t count,
                               const std::vector<double>& scores,
                               Signal& qout,
                               Signal& tout) {
            for (std::int64_t i = 0; i < count; ++i) {
                double qq = 0, tt = 0;
                for (std::int64_t j = 0; j < p; ++j) {
                    double delta = values[i * p + j] - mean[j];
                    for (std::int64_t a = 0; a < k; ++a)
                        if (resolved[static_cast<std::size_t>(a)])
                            delta -=
                                scores[static_cast<std::size_t>(i * k + a)] * loadings[a * p + j];
                    qq += delta * delta;
                }
                for (std::int64_t a = 0; a < k; ++a)
                    if (resolved[static_cast<std::size_t>(a)]) {
                        const double value = scores[static_cast<std::size_t>(i * k + a)];
                        tt += value * value / eigen[static_cast<std::size_t>(a)];
                    }
                qout.set(i, qq);
                tout.set(i, tt, effective ? OK : DEGENERATE);
            }
        };
        diagnostics(query.x.data(), n, query_scores, q, t2);
        diagnostics(x, nr, train_scores, train_q, train_t2);
        for (const auto* signal : {&q, &t2, &train_q, &train_t2})
            signal->write(*out);
        out->set_double_matrix("reference_mean", std::vector<double>(mean, mean + p), 1, p);
        out->set_double_matrix("loadings", std::vector<double>(loadings, loadings + k * p), k, p);
        out->set_double_matrix("eigenvalues", eigen, 1, k);
        out->set_int_vector("resolved_components", resolved);
        out->set_double_matrix("reference_scores", train_scores, nr, k);
        out->set_double_matrix("scores", query_scores, n, k);
        out->set_scalar("effective_components", effective);
        scalar(*out, "eigenvalue_resolution", resolution);
        out->set_int_vector("exact_full_moments_status", {6});
        out->set_scalar("solver_code", 1);  // exact native Gram/Jacobi; no randomized approximation
        for (const auto* v : {&train_scores, &query_scores, &eigen})
            if (std::any_of(v->begin(), v->end(), [](double value) {
                    return !std::isfinite(value);
                }))
                throw std::invalid_argument("PCA output outside finite numerical domain");
        // This operation deliberately never estimates full moments from retained eigenvalues.
        return out;
    });
}
}  // namespace n4m::estimator
