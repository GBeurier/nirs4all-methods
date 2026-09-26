// SPDX-License-Identifier: CECILL-2.1
//
// Manifest-driven conformance suite for procedures (n4m_procedure_run).
// Every procedure in the compiled manifest runs with its default parameters
// (and a non-default seed), is deterministic, gives the same result on a
// column-major X, refuses missing or unused inputs and unknown or foreign
// parameters, and reproduces its direct C entry point bitwise: splitters
// n4m_splitter_run, the ABI 2.11 augmentations n4m_augmentation_run, the
// other procedures their own C functions. Target-mixing augmenters also
// return "Y", mixed with the same draw as "X"; augmenters with nanometre
// constants refuse an axis that is not finite and strictly increasing.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/method_result.hpp"
#include "n4m/n4m.h"

#define CHECK(expr)                                                                   \
    do {                                                                              \
        if (!(expr)) throw std::runtime_error(std::string(#expr) + " @" + current_); \
    } while (0)

namespace {

std::string current_;

constexpr int64_t kRows = 36;
constexpr int64_t kCols = 24;
constexpr int64_t kTargetRows = 20;
constexpr int64_t kComponents = 6;  // one-SE rule input: components x folds
constexpr int64_t kFolds = 4;

n4m_matrix_view_t view(const double* data, int64_t rows, int64_t cols) {
    n4m_matrix_view_t v{};
    CHECK(n4m_matrix_view_init_rowmajor(&v, const_cast<double*>(data), rows, cols,
                                        N4M_DTYPE_F64) == N4M_OK);
    return v;
}

struct ResultPtr {
    n4m_method_result_t* p = nullptr;
    ResultPtr() = default;
    ResultPtr(const ResultPtr&) = delete;
    ResultPtr& operator=(const ResultPtr&) = delete;
    ~ResultPtr() { n4m_method_result_destroy(p); }
};

// Smooth positive spectra, a linear response, groups of two rows, a
// shifted target domain, fold ids, and the special matrices of the
// regression-metric and one-SE procedures.
struct Data {
    std::vector<double> x, y, x_target, axis, predictions, fold_rmse;
    std::vector<int64_t> groups, fold_ids;
    Data() {
        auto spectrum = [](int64_t i, int64_t j) {
            const double t = static_cast<double>(j) / kCols;
            const double a = 1.0 + 0.3 * std::sin(0.37 * static_cast<double>(i) + 0.11);
            const double b = 0.5 + 0.2 * std::cos(0.23 * static_cast<double>(i * i % 17));
            return 0.2 + a * std::exp(-8.0 * (t - 0.3) * (t - 0.3)) +
                   b * std::exp(-10.0 * (t - 0.7) * (t - 0.7)) +
                   0.01 * std::sin(static_cast<double>(i) * 1.7 + static_cast<double>(j));
        };
        for (int64_t i = 0; i < kRows; ++i) {
            double response = 0.0;
            for (int64_t j = 0; j < kCols; ++j) {
                x.push_back(spectrum(i, j));
                response += (j % 3 == 0 ? 1.5 : -0.4) * x.back();
            }
            y.push_back(response + 0.01 * std::sin(3.1 * static_cast<double>(i)));
            predictions.push_back(y.back() + 0.05 * std::cos(1.3 * static_cast<double>(i)));
            groups.push_back(i / 2);
            fold_ids.push_back(i % 4);
        }
        for (int64_t i = 0; i < kTargetRows; ++i) {
            for (int64_t j = 0; j < kCols; ++j) x_target.push_back(spectrum(i + 101, j) * 1.1);
        }
        // A NIR axis in nm, 1000-2380: covers the water and C-H bands and the
        // detector roll-off ranges of the augmenters with nanometre constants.
        for (int64_t j = 0; j < kCols; ++j) axis.push_back(1000.0 + 60.0 * static_cast<double>(j));
        for (int64_t k = 0; k < kComponents; ++k) {
            for (int64_t f = 0; f < kFolds; ++f) {
                fold_rmse.push_back(1.0 / static_cast<double>(k + 1) + 0.05 * static_cast<double>(k) +
                                    0.01 * static_cast<double>(f));
            }
        }
    }
};

struct Inputs {
    Data data;
    n4m_matrix_view_t X{}, Y{}, Xt{}, P{}, R{};

    Inputs() {
        X = view(data.x.data(), kRows, kCols);
        Y = view(data.y.data(), kRows, 1);
        Xt = view(data.x_target.data(), kTargetRows, kCols);
        P = view(data.predictions.data(), kRows, 1);
        R = view(data.fold_rmse.data(), kComponents, kFolds);
    }
    // The X a procedure reads: predictions for the regression metrics, the
    // fold RMSE matrix for the one-SE rule, spectra otherwise.
    const n4m_matrix_view_t* x_for(const char* method_id) const {
        if (std::strcmp(method_id, "diagnostics.regression_metrics") == 0) return &P;
        if (std::strcmp(method_id, "diagnostics.model_selection") == 0) return &R;
        return &X;
    }
    // Every input the method declares (required or optional).
    n4m_fit_inputs_v1_t for_method(const n4m_method_info_v1_t& info) const {
        n4m_fit_inputs_v1_t in{};
        in.struct_size = sizeof(in);
        in.X = x_for(info.method_id);
        auto wants = [&](n4m_fit_input_t k) { return info.inputs[k] != N4M_INPUT_NONE; };
        if (wants(N4M_FIT_INPUT_Y)) in.Y = &Y;
        if (wants(N4M_FIT_INPUT_GROUPS)) {
            in.groups = data.groups.data();
            in.n_groups = kRows;
        }
        if (wants(N4M_FIT_INPUT_AXIS)) {
            in.axis = data.axis.data();
            in.n_axis = kCols;
        }
        if (wants(N4M_FIT_INPUT_TARGET_DOMAIN)) in.X_target = &Xt;
        if (wants(N4M_FIT_INPUT_FOLD_IDS)) {
            in.fold_ids = data.fold_ids.data();
            in.n_fold_ids = kRows;
        }
        for (int k = 0; k < N4M_FIT_INPUT_COUNT; ++k) {
            const bool supplied = k == N4M_FIT_INPUT_Y || k == N4M_FIT_INPUT_GROUPS ||
                                  k == N4M_FIT_INPUT_AXIS || k == N4M_FIT_INPUT_TARGET_DOMAIN ||
                                  k == N4M_FIT_INPUT_FOLD_IDS;
            if (info.inputs[k] != N4M_INPUT_NONE && !supplied) {
                throw std::runtime_error("no test input for a declared input @" + current_);
            }
        }
        return in;
    }
};

// ---- Resolved parameter values ------------------------------------------

bool has_param(const n4m_params_t* p, const char* name) {
    int64_t n = 0;
    return n4m_params_get_int(p, name, nullptr, 0, &n) == N4M_OK ||
           n4m_params_get_double(p, name, nullptr, 0, &n) == N4M_OK;
}

std::vector<int64_t> ints(const n4m_params_t* p, const char* name) {
    int64_t n = 0;
    CHECK(n4m_params_get_int(p, name, nullptr, 0, &n) == N4M_OK);
    std::vector<int64_t> v(static_cast<size_t>(n));
    CHECK(n4m_params_get_int(p, name, v.data(), n, &n) == N4M_OK);
    return v;
}

std::vector<double> doubles(const n4m_params_t* p, const char* name) {
    int64_t n = 0;
    CHECK(n4m_params_get_double(p, name, nullptr, 0, &n) == N4M_OK);
    std::vector<double> v(static_cast<size_t>(n));
    CHECK(n4m_params_get_double(p, name, v.data(), n, &n) == N4M_OK);
    return v;
}

int64_t pint(const n4m_params_t* p, const char* name) { return ints(p, name).at(0); }
int32_t pi32(const n4m_params_t* p, const char* name) {
    return static_cast<int32_t>(pint(p, name));
}
double pdbl(const n4m_params_t* p, const char* name) { return doubles(p, name).at(0); }
// The dispatcher's positional double of a named parameter.
double positional(const n4m_params_t* p, const char* name) {
    int64_t n = 0;
    if (n4m_params_get_double(p, name, nullptr, 0, &n) == N4M_OK) return pdbl(p, name);
    return static_cast<double>(pint(p, name));
}

// ---- Bitwise comparisons -------------------------------------------------

bool same_bits(const double* a, const double* b, size_t n) {
    return n == 0 || std::memcmp(a, b, n * sizeof(double)) == 0;
}

// Every named output of two results, bitwise.
bool same_result(const n4m_method_result_t* a, const n4m_method_result_t* b) {
    if (a->double_arrays.size() != b->double_arrays.size() ||
        a->int_arrays != b->int_arrays || a->int64_arrays != b->int64_arrays ||
        a->double_shapes != b->double_shapes || a->scalars.size() != b->scalars.size()) {
        return false;
    }
    for (const auto& [name, values] : a->double_arrays) {
        const auto it = b->double_arrays.find(name);
        if (it == b->double_arrays.end() || it->second.size() != values.size() ||
            !same_bits(values.data(), it->second.data(), values.size())) {
            return false;
        }
    }
    for (const auto& [name, value] : a->scalars) {
        const auto it = b->scalars.find(name);
        if (it == b->scalars.end() || !same_bits(&value, &it->second, 1)) return false;
    }
    return true;
}

bool scalar_is(const n4m_method_result_t* r, const char* name, double expected) {
    double v = 0.0;
    return n4m_method_result_get_scalar(r, name, &v) == N4M_OK && same_bits(&v, &expected, 1);
}

// ---- Splitters -----------------------------------------------------------

struct SplitterCase {
    const char* method_id;
    int32_t kind;
    bool uses_x;
};

constexpr SplitterCase kSplitters[] = {
    {"splitters.kennard_stone", N4M_SPLITTER_KENNARD_STONE, true},
    {"splitters.spxy", N4M_SPLITTER_SPXY, true},
    {"splitters.spxy_fold", N4M_SPLITTER_SPXY_FOLD, true},
    {"splitters.spxy_g_fold", N4M_SPLITTER_SPXY_GROUP_FOLD, true},
    {"splitters.kmeans", N4M_SPLITTER_KMEANS, true},
    {"splitters.kbins_stratified", N4M_SPLITTER_KBINS_STRATIFIED, false},
    {"splitters.binned_strat_group_kfold", N4M_SPLITTER_BINNED_STRAT_GROUP_FOLD, false},
    {"splitters.systematic_circular", N4M_SPLITTER_SYSTEMATIC_CIRCULAR, false},
    {"splitters.split_splitter", N4M_SPLITTER_DATA_TWINNING, true},
};

void check_splitter(const n4m_params_t* params, const n4m_fit_inputs_v1_t& in,
                    const n4m_method_result_t* result) {
    const SplitterCase* c = nullptr;
    for (const auto& s : kSplitters) {
        if (current_ == s.method_id) c = &s;
    }
    CHECK(c != nullptr);
    n4m_splitter_spec_t spec{};
    spec.kind = c->kind;
    auto opt = [&](const char* name) { return has_param(params, name) ? pi32(params, name) : 0; };
    spec.n_splits = opt("n_splits");
    spec.y_metric = opt("y_metric");
    spec.aggregation = opt("aggregation");
    spec.n_bins = opt("n_bins");
    spec.strategy = opt("strategy");
    spec.shuffle = opt("shuffle");
    spec.max_iter = opt("max_iter");
    spec.test_size = has_param(params, "test_size") ? pdbl(params, "test_size") : 0.0;
    spec.seed = has_param(params, "seed") ? static_cast<uint64_t>(pint(params, "seed")) : 0;

    int32_t n_folds = 0;
    CHECK(n4m_method_result_get_n_folds(result, &n_folds) == N4M_OK);
    CHECK(n_folds == (spec.n_splits > 0 ? spec.n_splits : 1));
    const int64_t* none = nullptr;
    int64_t n0 = 0;
    CHECK(n4m_method_result_get_fold(result, n_folds, &none, &n0, &none, &n0) ==
          N4M_ERR_INVALID_ARGUMENT);
    CHECK(n4m_method_result_get_fold(result, -1, &none, &n0, &none, &n0) ==
          N4M_ERR_INVALID_ARGUMENT);
    for (int32_t fold = 0; fold < n_folds; ++fold) {
        const int64_t *train = nullptr, *test = nullptr;
        int64_t n_train = 0, n_test = 0;
        CHECK(n4m_method_result_get_fold(result, fold, &train, &n_train, &test, &n_test) ==
              N4M_OK);
        CHECK(n_train > 0 && n_test > 0);
        std::set<int64_t> seen;
        for (int64_t i = 0; i < n_train; ++i) CHECK(train[i] >= 0 && train[i] < kRows);
        for (int64_t i = 0; i < n_test; ++i) CHECK(test[i] >= 0 && test[i] < kRows);
        seen.insert(train, train + n_train);
        CHECK(static_cast<int64_t>(seen.size()) == n_train);
        seen.insert(test, test + n_test);
        CHECK(static_cast<int64_t>(seen.size()) == n_train + n_test);  // disjoint

        n4m_split_result_t direct{};
        CHECK(n4m_splitter_run(&spec, c->uses_x ? in.X : nullptr, in.Y, in.groups, in.n_groups,
                               fold, &direct) == N4M_OK);
        const bool same = direct.n_train == n_train && direct.n_test == n_test &&
                          std::memcmp(direct.train_idx, train,
                                      static_cast<size_t>(n_train) * sizeof(int64_t)) == 0 &&
                          std::memcmp(direct.test_idx, test,
                                      static_cast<size_t>(n_test) * sizeof(int64_t)) == 0;
        n4m_split_result_destroy(&direct);
        CHECK(same);
    }
}

// ---- Augmenters ------------------------------------------------------------

struct DispatchCase {
    const char* method_id;
    int32_t kind;
    std::vector<const char*> params;
};

const std::vector<DispatchCase>& dispatch_cases() {
    static const std::vector<DispatchCase> cases = {
        {"augmentation.noise.gaussian_noise", N4M_AUG_GAUSSIAN_NOISE, {"sigma"}},
        {"augmentation.noise.multiplicative_noise", N4M_AUG_MULTIPLICATIVE_NOISE, {"sigma_gain"}},
        {"augmentation.noise.spike_noise", N4M_AUG_SPIKE_NOISE,
         {"n_spikes_min", "n_spikes_max", "amplitude_min", "amplitude_max"}},
        {"augmentation.noise.hetero_noise", N4M_AUG_HETERO_NOISE,
         {"noise_base", "noise_signal_dep"}},
        {"augmentation.drift.linear_drift", N4M_AUG_LINEAR_DRIFT,
         {"offset_min", "offset_max", "slope_min", "slope_max"}},
        {"augmentation.drift.path_length", N4M_AUG_PATH_LENGTH,
         {"path_length_std", "min_path_length"}},
        {"augmentation.spectral.band_perturb", N4M_AUG_BAND_PERTURB,
         {"n_bands", "bw_lo", "bw_hi", "gain_lo", "gain_hi", "offset_lo", "offset_hi"}},
        {"augmentation.spectral.band_mask", N4M_AUG_BAND_MASK,
         {"n_bands_lo", "n_bands_hi", "bw_lo", "bw_hi", "mode"}},
        {"augmentation.spectral.channel_dropout", N4M_AUG_CHANNEL_DROPOUT,
         {"dropout_prob", "mode"}},
        {"augmentation.spectral.gauss_jitter", N4M_AUG_GAUSS_JITTER,
         {"sigma_lo", "sigma_hi", "kernel_width"}},
        {"augmentation.spectral.unsharp_mask", N4M_AUG_UNSHARP_MASK,
         {"amount_lo", "amount_hi", "sigma", "kernel_width"}},
        {"augmentation.spectral.local_clip", N4M_AUG_LOCAL_CLIP,
         {"n_regions", "width_lo", "width_hi"}},
        {"augmentation.random.rotate_translate", N4M_AUG_ROTATE_TRANSLATE,
         {"p_range", "y_factor"}},
        {"augmentation.random.random_x_op", N4M_AUG_RANDOM_X_OP,
         {"op_kind", "operator_range_min", "operator_range_max"}},
        {"augmentation.scattering.scatter_sim_msc", N4M_AUG_SCATTER_SIM_MSC,
         {"a_low", "a_high", "b_low", "b_high"}},
        {"augmentation.scattering.dead_band", N4M_AUG_DEAD_BAND,
         {"n_bands", "width_low", "width_high", "noise_std", "probability", "variation_scope"}},
        {"augmentation.scattering.batch_effect", N4M_AUG_BATCH_EFFECT,
         {"offset_std", "slope_std", "gain_std", "variation_scope"}},
        {"augmentation.splines.spline_smoothing", N4M_AUG_SPLINE_SMOOTHING, {}},
        {"augmentation.splines.spline_x_perturbations", N4M_AUG_SPLINE_X_PERTURB,
         {"spline_degree", "perturbation_density", "perturbation_range_min",
          "perturbation_range_max"}},
        {"augmentation.splines.spline_y_perturbations", N4M_AUG_SPLINE_Y_PERTURB,
         {"spline_points", "perturbation_intensity"}},
        {"augmentation.splines.spline_x_simplification", N4M_AUG_SPLINE_X_SIMPLIFY,
         {"spline_points", "uniform"}},
        {"augmentation.splines.spline_curve_simplification", N4M_AUG_SPLINE_CURVE_SIMPLIFY,
         {"spline_points", "uniform"}},
    };
    return cases;
}

// Seeded create / apply / destroy through the per-method C functions.
template <typename H>
void direct_apply(const n4m_params_t* params,
                  const std::function<n4m_status_t(H**, n4m_rng_pcg64_state_t*)>& create,
                  const std::function<n4m_status_t(const H*, n4m_matrix_view_t,
                                                   n4m_matrix_view_t)>& apply,
                  void (*destroy)(H*), n4m_matrix_view_t X, n4m_matrix_view_t out) {
    n4m_rng_pcg64_state_t* rng = nullptr;
    CHECK(n4m_rng_pcg64_create(static_cast<uint64_t>(pint(params, "seed")), &rng) == N4M_OK);
    H* h = nullptr;
    const n4m_status_t created = create(&h, rng);
    const n4m_status_t applied = created == N4M_OK ? apply(h, X, out) : created;
    destroy(h);
    n4m_rng_pcg64_destroy(rng);
    CHECK(applied == N4M_OK);
}

// The augmented rows the direct C entry point produces.
std::vector<double> direct_augmentation(const n4m_params_t* params,
                                        const n4m_fit_inputs_v1_t& in) {
    std::vector<double> out(static_cast<size_t>(kRows * kCols));
    const n4m_matrix_view_t O = view(out.data(), kRows, kCols);
    const n4m_matrix_view_t X = *in.X;
    for (const auto& c : dispatch_cases()) {
        if (current_ != c.method_id) continue;
        std::vector<double> values;
        for (const char* name : c.params) values.push_back(positional(params, name));
        CHECK(n4m_augmentation_run(c.kind, values.data(), static_cast<int32_t>(values.size()),
                                   static_cast<uint64_t>(pint(params, "seed")), X, O) == N4M_OK);
        return out;
    }
    const double* axis = in.axis;
    const int64_t n_axis = in.n_axis;
    if (current_ == "augmentation.drift.poly_drift") {
        const auto lo = doubles(params, "coeff_min");
        const auto hi = doubles(params, "coeff_max");
        direct_apply<n4m_aug_poly_drift_handle_t>(
            params,
            [&](n4m_aug_poly_drift_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_poly_drift_create(
                    h, rng, static_cast<int32_t>(lo.size()) - 1, lo.data(), hi.data());
            },
            n4m_augmentation_poly_drift_apply, n4m_augmentation_poly_drift_destroy, X, O);
    } else if (current_ == "augmentation.wavelength.wavelength_shift") {
        direct_apply<n4m_aug_wavelength_shift_handle_t>(
            params,
            [&](n4m_aug_wavelength_shift_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_wavelength_shift_create(
                    h, rng, pdbl(params, "shift_lo"), pdbl(params, "shift_hi"), axis, n_axis);
            },
            n4m_augmentation_wavelength_shift_apply, n4m_augmentation_wavelength_shift_destroy, X,
            O);
    } else if (current_ == "augmentation.wavelength.wavelength_stretch") {
        direct_apply<n4m_aug_wavelength_stretch_handle_t>(
            params,
            [&](n4m_aug_wavelength_stretch_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_wavelength_stretch_create(
                    h, rng, pdbl(params, "stretch_lo"), pdbl(params, "stretch_hi"), axis, n_axis);
            },
            n4m_augmentation_wavelength_stretch_apply,
            n4m_augmentation_wavelength_stretch_destroy, X, O);
    } else if (current_ == "augmentation.wavelength.local_warp") {
        direct_apply<n4m_aug_local_warp_handle_t>(
            params,
            [&](n4m_aug_local_warp_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_local_warp_create(h, rng, pi32(params, "n_control_points"),
                                                          pdbl(params, "max_shift"), axis, n_axis);
            },
            n4m_augmentation_local_warp_apply, n4m_augmentation_local_warp_destroy, X, O);
    } else if (current_ == "augmentation.spectral.magnitude_warp") {
        direct_apply<n4m_aug_magnitude_warp_handle_t>(
            params,
            [&](n4m_aug_magnitude_warp_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_magnitude_warp_create(
                    h, rng, pi32(params, "n_control_points"), pdbl(params, "gain_lo"),
                    pdbl(params, "gain_hi"), axis, n_axis);
            },
            n4m_augmentation_magnitude_warp_apply, n4m_augmentation_magnitude_warp_destroy, X, O);
    } else if (current_ == "augmentation.scattering.emsc_distort") {
        direct_apply<n4m_aug_emsc_distort_handle_t>(
            params,
            [&](n4m_aug_emsc_distort_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_emsc_distort_create(
                    h, rng, pdbl(params, "mult_low"), pdbl(params, "mult_high"),
                    pdbl(params, "add_low"), pdbl(params, "add_high"),
                    pi32(params, "polynomial_order"), pdbl(params, "polynomial_strength"),
                    pdbl(params, "correlation"), axis, n_axis);
            },
            n4m_augmentation_emsc_distort_apply, n4m_augmentation_emsc_distort_destroy, X, O);
    } else if (current_ == "augmentation.edge_artifacts.edge_curvature") {
        const n4m_matrix_view_t wl = view(axis, 1, n_axis);
        direct_apply<n4m_aug_edge_curve_handle_t>(
            params,
            [&](n4m_aug_edge_curve_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_edge_curvature_create(
                    h, rng, pdbl(params, "curvature_strength"), pi32(params, "curvature_type"),
                    pdbl(params, "asymmetry"), pdbl(params, "edge_focus"));
            },
            [&](const n4m_aug_edge_curve_handle_t* h, n4m_matrix_view_t x, n4m_matrix_view_t o) {
                return n4m_augmentation_edge_curvature_apply(h, x, wl, o);
            },
            n4m_augmentation_edge_curvature_destroy, X, O);
    } else if (current_ == "augmentation.scattering.instrument_broaden") {
        direct_apply<n4m_aug_instrument_broaden_handle_t>(
            params,
            [&](n4m_aug_instrument_broaden_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_instrument_broaden_create(
                    h, rng, pdbl(params, "fwhm"), pi32(params, "use_fwhm_range"),
                    pdbl(params, "fwhm_low"), pdbl(params, "fwhm_high"),
                    pi32(params, "variation_scope"), axis, n_axis);
            },
            n4m_augmentation_instrument_broaden_apply,
            n4m_augmentation_instrument_broaden_destroy, X, O);
    } else if (current_ == "augmentation.edge_artifacts.truncated_peak") {
        const n4m_matrix_view_t wl = view(axis, 1, n_axis);
        direct_apply<n4m_aug_truncated_peak_handle_t>(
            params,
            [&](n4m_aug_truncated_peak_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_truncated_peak_create(
                    h, rng, pdbl(params, "peak_probability"), pdbl(params, "amplitude_min"),
                    pdbl(params, "amplitude_max"), pdbl(params, "width_min"),
                    pdbl(params, "width_max"), pi32(params, "left_edge"),
                    pi32(params, "right_edge"));
            },
            [&](const n4m_aug_truncated_peak_handle_t* h, n4m_matrix_view_t x, n4m_matrix_view_t o) {
                return n4m_augmentation_truncated_peak_apply(h, x, wl, o);
            },
            n4m_augmentation_truncated_peak_destroy, X, O);
    } else if (current_ == "augmentation.mixup.mixup") {
        direct_apply<n4m_aug_mixup_handle_t>(
            params,
            [&](n4m_aug_mixup_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_mixup_create(h, rng, pdbl(params, "alpha"));
            },
            n4m_augmentation_mixup_apply, n4m_augmentation_mixup_destroy, X, O);
    } else if (current_ == "augmentation.mixup.local_mixup") {
        direct_apply<n4m_aug_local_mixup_handle_t>(
            params,
            [&](n4m_aug_local_mixup_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_local_mixup_create(h, rng, pdbl(params, "alpha"),
                                                           pi32(params, "k_neighbors"));
            },
            n4m_augmentation_local_mixup_apply, n4m_augmentation_local_mixup_destroy, X, O);
    } else if (current_ == "augmentation.environmental.temperature") {
        direct_apply<n4m_aug_temperature_handle_t>(
            params,
            [&](n4m_aug_temperature_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_temperature_create(
                    h, rng, pdbl(params, "temperature_delta"), pi32(params, "use_temp_range"),
                    pdbl(params, "temp_low"), pdbl(params, "temp_high"),
                    pi32(params, "enable_shift"), pi32(params, "enable_intensity"),
                    pi32(params, "enable_broadening"), pi32(params, "region_specific"), axis,
                    n_axis);
            },
            n4m_augmentation_temperature_apply, n4m_augmentation_temperature_destroy, X, O);
    } else if (current_ == "augmentation.environmental.moisture") {
        direct_apply<n4m_aug_moisture_handle_t>(
            params,
            [&](n4m_aug_moisture_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_moisture_create(
                    h, rng, pdbl(params, "water_activity_delta"), pi32(params, "use_aw_range"),
                    pdbl(params, "aw_low"), pdbl(params, "aw_high"),
                    pdbl(params, "reference_water_activity"), pdbl(params, "free_water_fraction"),
                    pdbl(params, "bound_water_shift"), pdbl(params, "moisture_content"),
                    pi32(params, "enable_shift"), pi32(params, "enable_intensity"), axis, n_axis);
            },
            n4m_augmentation_moisture_apply, n4m_augmentation_moisture_destroy, X, O);
    } else if (current_ == "augmentation.scattering.particle_size") {
        direct_apply<n4m_aug_particle_size_handle_t>(
            params,
            [&](n4m_aug_particle_size_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_particle_size_create(
                    h, rng, pdbl(params, "mean_size_um"), pdbl(params, "size_variation_um"),
                    pi32(params, "use_size_range"), pdbl(params, "size_range_low_um"),
                    pdbl(params, "size_range_high_um"), pdbl(params, "reference_size_um"),
                    pdbl(params, "wavelength_exponent"), pdbl(params, "size_effect_strength"),
                    pi32(params, "include_path_length"), pdbl(params, "path_length_sensitivity"),
                    axis, n_axis);
            },
            n4m_augmentation_particle_size_apply, n4m_augmentation_particle_size_destroy, X, O);
    } else if (current_ == "augmentation.edge_artifacts.detector_rolloff") {
        const n4m_matrix_view_t wl = view(axis, 1, n_axis);
        direct_apply<n4m_aug_detector_rolloff_handle_t>(
            params,
            [&](n4m_aug_detector_rolloff_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_detector_rolloff_create(
                    h, rng, pi32(params, "detector_model"), pdbl(params, "effect_strength"),
                    pdbl(params, "noise_amplification"),
                    pi32(params, "include_baseline_distortion"));
            },
            [&](const n4m_aug_detector_rolloff_handle_t* h, n4m_matrix_view_t x,
                n4m_matrix_view_t o) { return n4m_augmentation_detector_rolloff_apply(h, x, wl, o); },
            n4m_augmentation_detector_rolloff_destroy, X, O);
    } else if (current_ == "augmentation.edge_artifacts.stray_light") {
        direct_apply<n4m_aug_stray_light_handle_t>(
            params,
            [&](n4m_aug_stray_light_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_stray_light_create(
                    h, rng, pdbl(params, "stray_light_fraction"), pdbl(params, "edge_enhancement"),
                    pdbl(params, "edge_width"), pi32(params, "include_peak_truncation"));
            },
            n4m_augmentation_stray_light_apply, n4m_augmentation_stray_light_destroy, X, O);
    } else if (current_ == "augmentation.edge_artifacts.edge_artifacts") {
        const n4m_matrix_view_t wl = view(axis, 1, n_axis);
        const int32_t flags = (pint(params, "detector_roll_off") != 0 ? 0x1 : 0) |
                              (pint(params, "stray_light") != 0 ? 0x2 : 0) |
                              (pint(params, "edge_curvature") != 0 ? 0x4 : 0) |
                              (pint(params, "truncated_peaks") != 0 ? 0x8 : 0);
        direct_apply<n4m_aug_edge_artifacts_handle_t>(
            params,
            [&](n4m_aug_edge_artifacts_handle_t** h, n4m_rng_pcg64_state_t* rng) {
                return n4m_augmentation_edge_artifacts_create(
                    h, rng, flags, pdbl(params, "overall_strength"), pi32(params, "detector_model"));
            },
            [&](const n4m_aug_edge_artifacts_handle_t* h, n4m_matrix_view_t x, n4m_matrix_view_t o) {
                return n4m_augmentation_edge_artifacts_apply(h, x, wl, o);
            },
            n4m_augmentation_edge_artifacts_destroy, X, O);
    } else {
        throw std::runtime_error("no direct entry point for augmenter @" + current_);
    }
    return out;
}

// "X": the augmented rows, equal to the direct entry point's; "Y" exactly
// when the method mixes targets (y given), with the rows of "X".
void check_augmenter(const n4m_params_t* params, const n4m_fit_inputs_v1_t& in,
                     const n4m_method_result_t* result) {
    const double* data = nullptr;
    int64_t rows = 0, cols = 0;
    CHECK(n4m_method_result_get_double_matrix(result, "X", &data, &rows, &cols) == N4M_OK);
    CHECK(rows == kRows && cols == kCols);
    for (int64_t i = 0; i < rows * cols; ++i) CHECK(std::isfinite(data[i]));
    int32_t n_folds = 0;
    CHECK(n4m_method_result_get_n_folds(result, &n_folds) == N4M_ERR_INVALID_ARGUMENT);
    const std::vector<double> direct = direct_augmentation(params, in);
    CHECK(same_bits(direct.data(), data, direct.size()));
    int32_t entries = 0;
    CHECK(n4m_method_result_entry_count(result, &entries) == N4M_OK);
    const double* y = nullptr;
    const n4m_status_t has_y = n4m_method_result_get_double_matrix(result, "Y", &y, &rows, &cols);
    if (in.Y == nullptr) {
        CHECK(entries == 1 && has_y != N4M_OK);
        return;
    }
    CHECK(entries == 2 && has_y == N4M_OK && rows == kRows && cols == in.Y->cols);
    for (int64_t i = 0; i < rows * cols; ++i) CHECK(std::isfinite(y[i]));
}

// Target mixing uses the draw that mixes X: with y := X, "Y" equals "X"
// bitwise, row for row.
void check_target_mixing(n4m_context_t* ctx, int32_t index, const n4m_params_t* params,
                         const n4m_fit_inputs_v1_t& in) {
    n4m_fit_inputs_v1_t paired = in;
    paired.Y = in.X;
    ResultPtr r;
    CHECK(n4m_procedure_run(ctx, index, params, &paired, &r.p) == N4M_OK);
    const double *x = nullptr, *y = nullptr;
    int64_t xr = 0, xc = 0, yr = 0, yc = 0;
    CHECK(n4m_method_result_get_double_matrix(r.p, "X", &x, &xr, &xc) == N4M_OK);
    CHECK(n4m_method_result_get_double_matrix(r.p, "Y", &y, &yr, &yc) == N4M_OK);
    CHECK(xr == yr && xc == yc && same_bits(x, y, static_cast<size_t>(xr * xc)));
    // Mixed targets are convex combinations of two input targets.
    const n4m_matrix_view_t& Y = *in.Y;
    ResultPtr own;
    CHECK(n4m_procedure_run(ctx, index, params, &in, &own.p) == N4M_OK);
    CHECK(n4m_method_result_get_double_matrix(own.p, "Y", &y, &yr, &yc) == N4M_OK);
    const auto* targets = static_cast<const double*>(Y.data);
    const auto [lo, hi] = std::minmax_element(targets, targets + Y.rows * Y.cols);
    for (int64_t i = 0; i < yr * yc; ++i) CHECK(y[i] >= *lo - 1e-12 && y[i] <= *hi + 1e-12);
}

// Augmenters whose kernels hold nanometre constants: the axis must be a
// finite, strictly increasing wavelength grid.
const std::set<std::string> kNanometreAugmenters = {
    "augmentation.environmental.temperature", "augmentation.environmental.moisture",
    "augmentation.scattering.particle_size", "augmentation.edge_artifacts.detector_rolloff",
    "augmentation.edge_artifacts.edge_artifacts"};

void check_nanometre_axis(n4m_context_t* ctx, int32_t index, const n4m_params_t* params,
                          const n4m_fit_inputs_v1_t& in) {
    std::vector<double> decreasing(in.axis, in.axis + in.n_axis);
    std::reverse(decreasing.begin(), decreasing.end());
    std::vector<double> repeated(in.axis, in.axis + in.n_axis);
    repeated.at(5) = repeated.at(4);
    std::vector<double> not_finite(in.axis, in.axis + in.n_axis);
    not_finite.at(3) = std::nan("");
    for (const auto* axis : {&decreasing, &repeated, &not_finite}) {
        n4m_fit_inputs_v1_t bad = in;
        bad.axis = axis->data();
        ResultPtr r;
        CHECK(n4m_procedure_run(ctx, index, params, &bad, &r.p) == N4M_ERR_INVALID_ARGUMENT);
        CHECK(std::strstr(n4m_context_last_error(ctx), "nm") != nullptr);
        CHECK(r.p == nullptr);
    }
}

// ---- Generic procedures ----------------------------------------------------

using Direct = std::function<void(n4m_context_t*, const n4m_params_t*,
                                  const n4m_fit_inputs_v1_t&, const n4m_method_result_t*)>;

n4m_config_t* pls_config(const n4m_params_t* p) {
    n4m_config_t* cfg = nullptr;
    CHECK(n4m_config_create(&cfg) == N4M_OK);
    if (has_param(p, "n_components")) {
        CHECK(n4m_config_set_n_components(cfg, pi32(p, "n_components")) == N4M_OK);
    }
    CHECK(n4m_config_set_center_x(cfg, pi32(p, "center_x")) == N4M_OK);
    CHECK(n4m_config_set_scale_x(cfg, pi32(p, "scale_x")) == N4M_OK);
    CHECK(n4m_config_set_center_y(cfg, pi32(p, "center_y")) == N4M_OK);
    CHECK(n4m_config_set_scale_y(cfg, pi32(p, "scale_y")) == N4M_OK);
    return cfg;
}

// Runs a direct entry point returning a MethodResult and compares every
// named output with the procedure's.
void same_as(const n4m_method_result_t* procedure,
             const std::function<n4m_status_t(n4m_method_result_t**)>& direct) {
    ResultPtr expected;
    CHECK(direct(&expected.p) == N4M_OK);
    CHECK(same_result(expected.p, procedure));
}

void outlier_statistic(const n4m_params_t* p, const n4m_fit_inputs_v1_t& in,
                       const n4m_method_result_t* r, const char* name,
                       n4m_status_t (*fn)(n4m_matrix_view_t, int32_t, double, double*, int64_t,
                                          double*)) {
    std::vector<double> values(kRows);
    double ucl = 0.0;
    CHECK(fn(*in.X, pi32(p, "n_components"), pdbl(p, "alpha"), values.data(), kRows, &ucl) ==
          N4M_OK);
    const double* data = nullptr;
    int64_t rows = 0, cols = 0;
    CHECK(n4m_method_result_get_double_matrix(r, name, &data, &rows, &cols) == N4M_OK);
    CHECK(rows == 1 && cols == kRows && same_bits(values.data(), data, values.size()));
    CHECK(scalar_is(r, "ucl", ucl));
}

const std::map<std::string, Direct>& generic_checks() {
    static const std::map<std::string, Direct> checks = {
        {"diagnostics.approximate_press",
         [](n4m_context_t* ctx, const n4m_params_t* p, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             n4m_config_t* cfg = nullptr;
             CHECK(n4m_config_create(&cfg) == N4M_OK);
             same_as(r, [&](n4m_method_result_t** out) {
                 return n4m_metrics_approximate_press_compute(ctx, cfg, in.X, in.Y,
                                                              pi32(p, "max_components"), out);
             });
             n4m_config_destroy(cfg);
         }},
        {"diagnostics.model_selection",
         [](n4m_context_t* ctx, const n4m_params_t*, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             same_as(r, [&](n4m_method_result_t** out) {
                 return n4m_metrics_one_se_rule_compute(
                     ctx, static_cast<const double*>(in.X->data), static_cast<int32_t>(kComponents),
                     static_cast<int32_t>(kFolds), out);
             });
         }},
        {"diagnostics.pls_diagnostics",
         [](n4m_context_t* ctx, const n4m_params_t* p, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             n4m_config_t* cfg = pls_config(p);
             CHECK(n4m_config_set_store_scores(cfg, 1) == N4M_OK);
             n4m_model_t* model = nullptr;
             CHECK(n4m_model_fit(ctx, cfg, in.X, in.Y, &model) == N4M_OK);
             same_as(r, [&](n4m_method_result_t** out) {
                 return n4m_metrics_pls_diagnostics_compute(ctx, model, in.X, in.X, out);
             });
             n4m_model_destroy(model);
             n4m_config_destroy(cfg);
         }},
        {"diagnostics.pls_monitoring",
         [](n4m_context_t* ctx, const n4m_params_t* p, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             n4m_config_t* cfg = pls_config(p);
             CHECK(n4m_config_set_store_scores(cfg, 1) == N4M_OK);
             n4m_model_t* model = nullptr;
             CHECK(n4m_model_fit(ctx, cfg, in.X, in.Y, &model) == N4M_OK);
             same_as(r, [&](n4m_method_result_t** out) {
                 return n4m_metrics_pls_monitoring_run(ctx, model, in.X, in.X_target,
                                                       pdbl(p, "alpha"), out);
             });
             n4m_model_destroy(model);
             n4m_config_destroy(cfg);
         }},
        {"diagnostics.regression_metrics",
         [](n4m_context_t*, const n4m_params_t*, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             using Metric = n4m_status_t (*)(const double*, const double*, int64_t, double*);
             const std::pair<const char*, Metric> metrics[] = {
                 {"rmse", n4m_metrics_regression_metrics_rmse},
                 {"mae", n4m_metrics_regression_metrics_mae},
                 {"bias", n4m_metrics_regression_metrics_bias},
                 {"sep", n4m_metrics_regression_metrics_sep},
                 {"rpd", n4m_metrics_regression_metrics_rpd},
                 {"rpiq", n4m_metrics_regression_metrics_rpiq},
                 {"r2", n4m_metrics_regression_metrics_r2},
                 {"nrmse", n4m_metrics_regression_metrics_nrmse}};
             CHECK(r->scalars.size() == 8);
             for (const auto& [name, fn] : metrics) {
                 double v = 0.0;
                 CHECK(fn(static_cast<const double*>(in.Y->data),
                          static_cast<const double*>(in.X->data), kRows, &v) == N4M_OK);
                 CHECK(scalar_is(r, name, v));
             }
         }},
        {"aom_pop.linear_stack_compress",
         [](n4m_context_t*, const n4m_params_t* p, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             auto bias = doubles(p, "base_intercepts");
             auto weights = doubles(p, "meta_weights");
             auto meta = doubles(p, "meta_intercept");
             std::vector<double> coef(static_cast<size_t>(in.X->rows)), intercept(1);
             const n4m_matrix_view_t B = view(bias.data(), 1, kCols);
             const n4m_matrix_view_t W = view(weights.data(), kCols, 1);
             const n4m_matrix_view_t M = view(meta.data(), 1, 1);
             n4m_matrix_view_t C = view(coef.data(), in.X->rows, 1);
             n4m_matrix_view_t I = view(intercept.data(), 1, 1);
             CHECK(n4m_ensemble_linear_stack_compress(in.X, &B, &W, &M, &C, &I) == N4M_OK);
             const double* data = nullptr;
             int64_t rows = 0, cols = 0;
             CHECK(n4m_method_result_get_double_matrix(r, "coefficients", &data, &rows, &cols) ==
                   N4M_OK);
             CHECK(rows == in.X->rows && cols == 1 && same_bits(coef.data(), data, coef.size()));
             CHECK(n4m_method_result_get_double_matrix(r, "intercept", &data, &rows, &cols) ==
                   N4M_OK);
             CHECK(rows == 1 && cols == 1 && same_bits(intercept.data(), data, 1));
         }},
        {"utilities.hotelling_t2",
         [](n4m_context_t*, const n4m_params_t* p, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             outlier_statistic(p, in, r, "t2", n4m_outlier_detection_hotelling_t2);
         }},
        {"utilities.q_residuals",
         [](n4m_context_t*, const n4m_params_t* p, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             outlier_statistic(p, in, r, "q", n4m_outlier_detection_q_residuals);
         }},
        {"utilities.moments",
         [](n4m_context_t* ctx, const n4m_params_t*, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             same_as(r, [&](n4m_method_result_t** out) {
                 return n4m_lowlevel_moments_compute(ctx, in.X, in.Y, out);
             });
         }},
        {"utilities.signal_type_detector",
         [](n4m_context_t*, const n4m_params_t* p, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             n4m_signal_type_t type = N4M_SIGNAL_UNKNOWN;
             double confidence = 0.0;
             char reason[256] = {0};
             CHECK(n4m_transform_signal_type_detector(*in.X, in.axis, in.n_axis,
                                                      pdbl(p, "confidence_threshold"), &type,
                                                      &confidence, reason) == N4M_OK);
             CHECK(scalar_is(r, "signal_type", static_cast<double>(type)));
             CHECK(scalar_is(r, "confidence", confidence));
         }},
        {"utilities.sweep",
         [](n4m_context_t* ctx, const n4m_params_t* p, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             n4m_config_t* cfg = pls_config(p);
             const auto lambdas = doubles(p, "ridge_lambdas");
             std::vector<int32_t> components, folds;
             for (int64_t k : ints(p, "pls_components")) components.push_back(static_cast<int32_t>(k));
             for (int64_t i = 0; i < in.n_fold_ids; ++i) {
                 folds.push_back(static_cast<int32_t>(in.fold_ids[i]));
             }
             same_as(r, [&](n4m_method_result_t** out) {
                 return n4m_model_selection_sweep_run(
                     ctx, cfg, in.X, in.Y, folds.empty() ? pi32(p, "cv") : 0,
                     folds.empty() ? nullptr : folds.data(), in.n_fold_ids, lambdas.data(),
                     static_cast<int64_t>(lambdas.size()), components.data(),
                     static_cast<int64_t>(components.size()), pi32(p, "heads") + 1, out);
             });
             n4m_config_destroy(cfg);
         }},
        {"utilities.transfer_metrics",
         [](n4m_context_t*, const n4m_params_t* p, const n4m_fit_inputs_v1_t& in,
            const n4m_method_result_t* r) {
             n4m_transfer_metrics_t m{};
             CHECK(n4m_domain_adaptation_transfer_metrics_compute(
                       *in.X, *in.X_target, pi32(p, "n_components"), pi32(p, "k_neighbors"),
                       static_cast<uint64_t>(pint(p, "seed")), &m) == N4M_OK);
             CHECK(r->scalars.size() == 9);
             CHECK(scalar_is(r, "centroid_distance", m.centroid_distance));
             CHECK(scalar_is(r, "cka_similarity", m.cka_similarity));
             CHECK(scalar_is(r, "grassmann_distance", m.grassmann_distance));
             CHECK(scalar_is(r, "rv_coefficient", m.rv_coefficient));
             CHECK(scalar_is(r, "procrustes_disparity", m.procrustes_disparity));
             CHECK(scalar_is(r, "trustworthiness", m.trustworthiness));
             CHECK(scalar_is(r, "spread_distance", m.spread_distance));
             CHECK(scalar_is(r, "evr_source", m.evr_source));
             CHECK(scalar_is(r, "evr_target", m.evr_target));
         }},
    };
    return checks;
}

// ---- Conformance -----------------------------------------------------------

// Parameters without a default, sized for the test X; false when none.
bool fill_required(int32_t index, n4m_params_t* params) {
    n4m_method_info_v1_t info{};
    info.struct_size = sizeof(info);
    CHECK(n4m_method_info_v1(index, &info) == N4M_OK);
    bool any = false;
    for (int32_t k = 0; k < info.n_params; ++k) {
        n4m_param_info_v1_t pi{};
        pi.struct_size = sizeof(pi);
        CHECK(n4m_method_param_info_v1(index, k, &pi) == N4M_OK);
        if (pi.has_default) continue;
        any = true;
        // Linear stack compression of the X columns (base outputs) into one target.
        std::vector<double> v;
        if (std::strcmp(pi.name, "base_intercepts") == 0 ||
            std::strcmp(pi.name, "meta_weights") == 0) {
            for (int64_t j = 0; j < kCols; ++j) v.push_back(0.1 * static_cast<double>(j) - 1.0);
        } else if (std::strcmp(pi.name, "meta_intercept") == 0) {
            v.push_back(0.5);
        } else {
            throw std::runtime_error(std::string("no test value for required parameter ") +
                                     pi.name + " @" + current_);
        }
        CHECK(n4m_params_set_double_array(params, pi.name, v.data(),
                                          static_cast<int64_t>(v.size())) == N4M_OK);
    }
    return any;
}

n4m_status_t run(n4m_context_t* ctx, int32_t index, const n4m_params_t* params,
                 const n4m_fit_inputs_v1_t& in, ResultPtr& out) {
    return n4m_procedure_run(ctx, index, params, &in, &out.p);
}

struct Counts {
    int splitters = 0, augmenters = 0, generic = 0, target_mixing = 0, nanometre = 0;
};

void conformance(n4m_context_t* ctx, const Inputs& inputs, int32_t index, int32_t estimator_index,
                 Counts& counts) {
    n4m_method_info_v1_t info{};
    info.struct_size = sizeof(info);
    CHECK(n4m_method_info_v1(index, &info) == N4M_OK);
    current_ = info.method_id;
    if (info.kind != N4M_METHOD_PROCEDURE) return;

    const uint32_t role = info.roles;
    CHECK(role == N4M_ROLE_SPLITTER || role == N4M_ROLE_AUGMENTER || role == N4M_ROLE_GENERIC);
    CHECK(info.capabilities == 0 && std::strcmp(info.state_format, "") == 0);
    n4m_estimator_t* est = nullptr;
    CHECK(n4m_estimator_create(ctx, info.method_id, nullptr, &est) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(est == nullptr);

    n4m_params_t* params = nullptr;
    CHECK(n4m_params_create(ctx, index, &params) == N4M_OK);
    struct ParamsGuard {
        n4m_params_t* p;
        ~ParamsGuard() { n4m_params_destroy(p); }
    } guard{params};
    // Seeds are parameters: a non-default one must reach the kernel.
    if (role == N4M_ROLE_AUGMENTER) CHECK(has_param(params, "seed"));
    if (has_param(params, "seed")) CHECK(n4m_params_set_int(params, "seed", 7) == N4M_OK);
    // Drawn temperature / water-activity deltas instead of the no-op defaults.
    if (current_ == "augmentation.environmental.temperature") {
        CHECK(n4m_params_set_bool(params, "use_temp_range", 1) == N4M_OK);
    }
    if (current_ == "augmentation.environmental.moisture") {
        CHECK(n4m_params_set_bool(params, "use_aw_range", 1) == N4M_OK);
    }
    CHECK(n4m_params_set_int(params, "no_such_param", 1) == N4M_ERR_INVALID_ARGUMENT);
    const bool requires_params = fill_required(index, params);
    CHECK(n4m_params_validate(ctx, params) == N4M_OK);

    const n4m_fit_inputs_v1_t in = inputs.for_method(info);
    ResultPtr refused;
    // Missing required and unused inputs are refused with a named message.
    for (int k = 0; k < N4M_FIT_INPUT_COUNT; ++k) {
        if (info.inputs[k] != N4M_INPUT_REQUIRED) continue;
        n4m_fit_inputs_v1_t missing = in;
        if (k == N4M_FIT_INPUT_Y) missing.Y = nullptr;
        if (k == N4M_FIT_INPUT_GROUPS) missing.groups = nullptr;
        if (k == N4M_FIT_INPUT_TARGET_DOMAIN) missing.X_target = nullptr;
        if (k == N4M_FIT_INPUT_AXIS) missing.axis = nullptr;
        if (k == N4M_FIT_INPUT_FOLD_IDS) missing.fold_ids = nullptr;
        CHECK(run(ctx, index, params, missing, refused) == N4M_ERR_INVALID_ARGUMENT);
        CHECK(std::strstr(n4m_context_last_error(ctx), "missing required") != nullptr);
    }
    if (info.inputs[N4M_FIT_INPUT_GROUPS] == N4M_INPUT_NONE) {
        n4m_fit_inputs_v1_t extra = in;
        extra.groups = inputs.data.groups.data();
        extra.n_groups = kRows;
        CHECK(run(ctx, index, params, extra, refused) == N4M_ERR_INVALID_ARGUMENT);
        CHECK(std::strstr(n4m_context_last_error(ctx), "groups") != nullptr);
    }
    if (info.inputs[N4M_FIT_INPUT_Y] == N4M_INPUT_NONE) {
        n4m_fit_inputs_v1_t extra = in;
        extra.Y = &inputs.Y;
        CHECK(run(ctx, index, params, extra, refused) == N4M_ERR_INVALID_ARGUMENT);
    }
    // Parameters belong to one method; estimators are not procedures.
    n4m_params_t* foreign = nullptr;
    CHECK(n4m_params_create(ctx, estimator_index, &foreign) == N4M_OK);
    CHECK(run(ctx, index, foreign, in, refused) == N4M_ERR_INVALID_ARGUMENT);
    CHECK(run(ctx, estimator_index, foreign, in, refused) == N4M_ERR_INVALID_ARGUMENT);
    n4m_params_destroy(foreign);
    CHECK(refused.p == nullptr);

    // Defaults (NULL params) and explicit params both run, unless a
    // parameter has no default; same seed, same output; a column-major X
    // gives the same output.
    ResultPtr defaults, first, second, strided;
    if (requires_params) {
        CHECK(run(ctx, index, nullptr, in, defaults) == N4M_ERR_INVALID_ARGUMENT);
        CHECK(std::strstr(n4m_context_last_error(ctx), "missing required parameter") != nullptr);
    } else {
        CHECK(run(ctx, index, nullptr, in, defaults) == N4M_OK);
    }
    const std::vector<double> x_before(static_cast<const double*>(in.X->data),
                                       static_cast<const double*>(in.X->data) +
                                           in.X->rows * in.X->cols);
    CHECK(run(ctx, index, params, in, first) == N4M_OK);
    CHECK(run(ctx, index, params, in, second) == N4M_OK);
    CHECK(same_result(first.p, second.p));
    CHECK(same_bits(x_before.data(), static_cast<const double*>(in.X->data), x_before.size()));
    std::vector<double> x_cm(x_before.size());
    for (int64_t i = 0; i < in.X->rows; ++i) {
        for (int64_t j = 0; j < in.X->cols; ++j) {
            x_cm[static_cast<size_t>(j * in.X->rows + i)] =
                x_before[static_cast<size_t>(i * in.X->cols + j)];
        }
    }
    n4m_matrix_view_t Xc{};
    CHECK(n4m_matrix_view_init_colmajor(&Xc, x_cm.data(), in.X->rows, in.X->cols,
                                        N4M_DTYPE_F64) == N4M_OK);
    n4m_fit_inputs_v1_t column_major = in;
    column_major.X = &Xc;
    CHECK(run(ctx, index, params, column_major, strided) == N4M_OK);
    CHECK(same_result(first.p, strided.p));

    auto check_role = [&](const n4m_fit_inputs_v1_t& inputs_used, const n4m_method_result_t* r) {
        if (role == N4M_ROLE_SPLITTER) {
            check_splitter(params, inputs_used, r);
        } else if (role == N4M_ROLE_AUGMENTER) {
            check_augmenter(params, inputs_used, r);
        } else {
            const auto it = generic_checks().find(info.method_id);
            if (it == generic_checks().end()) {
                throw std::runtime_error("no direct entry point for generic procedure @" +
                                         current_);
            }
            it->second(ctx, params, inputs_used, r);
        }
    };
    check_role(in, first.p);
    if (role == N4M_ROLE_AUGMENTER && info.inputs[N4M_FIT_INPUT_Y] == N4M_INPUT_REQUIRED) {
        check_target_mixing(ctx, index, params, in);
        ++counts.target_mixing;
    }
    if (kNanometreAugmenters.count(current_) != 0) {
        check_nanometre_axis(ctx, index, params, in);
        ++counts.nanometre;
    }
    // Optional inputs may be left out.
    if (info.inputs[N4M_FIT_INPUT_AXIS] == N4M_INPUT_OPTIONAL) {
        n4m_fit_inputs_v1_t without = in;
        without.axis = nullptr;
        without.n_axis = 0;
        ResultPtr r;
        CHECK(run(ctx, index, params, without, r) == N4M_OK);
        check_role(without, r.p);
    }
    if (info.inputs[N4M_FIT_INPUT_FOLD_IDS] == N4M_INPUT_OPTIONAL) {
        n4m_fit_inputs_v1_t without = in;
        without.fold_ids = nullptr;
        without.n_fold_ids = 0;
        ResultPtr r;
        CHECK(run(ctx, index, params, without, r) == N4M_OK);
        check_role(without, r.p);
    }
    ++(role == N4M_ROLE_SPLITTER    ? counts.splitters
       : role == N4M_ROLE_AUGMENTER ? counts.augmenters
                                    : counts.generic);
}

void test_entry_points(n4m_context_t* ctx) {
    current_ = "entry points";
    ResultPtr out;
    n4m_fit_inputs_v1_t in{};
    in.struct_size = sizeof(in);
    CHECK(n4m_procedure_run(ctx, -1, nullptr, &in, &out.p) == N4M_ERR_INVALID_ARGUMENT);
    int32_t index = -1;
    CHECK(n4m_method_find("splitters.kennard_stone", &index) == N4M_OK);
    CHECK(n4m_procedure_run(ctx, index, nullptr, nullptr, &out.p) == N4M_ERR_NULL_POINTER);
    CHECK(n4m_procedure_run(ctx, index, nullptr, &in, &out.p) == N4M_ERR_NULL_POINTER);
    CHECK(std::strstr(n4m_context_last_error(ctx), "'X'") != nullptr);
    CHECK(n4m_procedure_run(ctx, index, nullptr, &in, nullptr) == N4M_ERR_NULL_POINTER);
    CHECK(out.p == nullptr);
}

}  // namespace

int main() {
    n4m_context_t* ctx = nullptr;
    if (n4m_context_create(&ctx) != N4M_OK) return 1;
    int failures = 0;
    auto guarded = [&](auto&& fn) {
        try {
            fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL %s (%s)\n", e.what(), n4m_context_last_error(ctx));
            ++failures;
        }
    };
    const Inputs inputs;
    guarded([&] { test_entry_points(ctx); });
    int32_t estimator_index = -1;
    n4m_method_find("models.pls.pls_regression", &estimator_index);
    int32_t count = 0;
    n4m_method_count(&count);
    Counts counts;
    for (int32_t i = 0; i < count; ++i) {
        guarded([&] { conformance(ctx, inputs, i, estimator_index, counts); });
    }
    n4m_context_destroy(ctx);
    if (counts.target_mixing != 2 ||
        counts.nanometre != static_cast<int>(kNanometreAugmenters.size())) {
        std::fprintf(stderr, "FAIL target-mixing or nanometre augmenters missing from the manifest\n");
        ++failures;
    }
    std::printf("n4m_procedure_tests: %d splitters, %d augmenters (%d target-mixing, %d in nm), "
                "%d generic, %d failures\n",
                counts.splitters, counts.augmenters, counts.target_mixing, counts.nanometre,
                counts.generic, failures);
    return failures == 0 ? 0 : 1;
}
