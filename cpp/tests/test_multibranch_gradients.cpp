// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Central numerical qualification of the production network, independent of its author.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/neural/multibranch_network.hpp"

namespace {
using n4m::core::neural::BranchShape;
using n4m::core::neural::Hyper;
using n4m::core::neural::MultibranchNetwork;
using n4m::core::neural::Training;

constexpr std::size_t kRows = 5;
constexpr std::size_t kWidth = 57;  // dense 2 + signal 5 + image 5x5x2
constexpr std::size_t kTargets = 2;

std::vector<double> features(std::size_t rows) {
    std::vector<double> values(rows * kWidth);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < kWidth; ++j) {
            values[i * kWidth + j] =
                std::sin(0.13 * static_cast<double>((i + 1) * (j + 1))) +
                0.07 * static_cast<double>(i) + 0.001 * static_cast<double>(j);
        }
    }
    return values;
}

std::vector<double> targets(const std::vector<double>& x, std::size_t rows) {
    std::vector<double> values(rows * kTargets);
    for (std::size_t i = 0; i < rows; ++i) {
        values[i * 2] = 0.4 * x[i * kWidth] - 0.2 * x[i * kWidth + 4] +
                        0.3 * x[i * kWidth + 20];
        values[i * 2 + 1] = x[i * kWidth + 1] * x[i * kWidth + 9] +
                            0.15 * x[i * kWidth + 45];
    }
    return values;
}

MultibranchNetwork network(std::int64_t head_hidden) {
    Hyper hyper;
    hyper.filters = 2;
    hyper.kernel_size = 2;
    hyper.pool_size = 2;
    hyper.embedding_dim = 2;
    hyper.head_hidden = head_hidden;
    const std::vector<BranchShape> shapes{{0, 2, 1, 1}, {1, 5, 1, 1}, {2, 5, 5, 2}};
    MultibranchNetwork result;
    result.configure(kWidth, kTargets, shapes, hyper);
    return result;
}

bool gradient_probe(std::int64_t head_hidden) {
    auto model = network(head_hidden);
    const auto x = features(kRows), y = targets(x, kRows);
    model.fit_scaling(x.data(), y.data(), kRows);
    std::vector<double> xs(x.size()), ys(y.size());
    model.standardize(x.data(), kRows, xs.data());
    model.standardize_targets(y.data(), kRows, ys.data());
    n4m_rng_pcg64 rng{};
    n4m_pcg64_engine_seed(&rng, 719);
    model.initialize(&rng);
    const std::size_t rows[]{0, 2, 4};  // exercises noncontiguous minibatch row selection
    std::vector<double> analytical;
    model.loss_gradient(xs.data(), ys.data(), rows, 3, &analytical);
    auto& parameters = model.parameters();
    if (analytical.size() != parameters.size()) return false;
    double worst = 0.0;
    for (std::size_t j = 0; j < parameters.size(); ++j) {
        const double original = parameters[j];
        const double step = 1e-6 * std::max(1.0, std::abs(original));
        parameters[j] = original + step;
        const double plus = model.loss_gradient(xs.data(), ys.data(), rows, 3, nullptr);
        parameters[j] = original - step;
        const double minus = model.loss_gradient(xs.data(), ys.data(), rows, 3, nullptr);
        parameters[j] = original;
        const double numerical = (plus - minus) / (2.0 * step);
        const double error = std::abs(numerical - analytical[j]) /
                             std::max({1.0, std::abs(numerical), std::abs(analytical[j])});
        worst = std::max(worst, error);
        if (!std::isfinite(error) || error > 3e-6) {
            std::printf("Gradient mismatch head=%lld parameter=%zu analytical=%.12g numerical=%.12g\n",
                        static_cast<long long>(head_hidden), j, analytical[j], numerical);
            return false;
        }
    }
    std::printf("head=%lld: %zu parameter derivatives checked, max normalized error %.3g\n",
                static_cast<long long>(head_hidden), parameters.size(), worst);
    return true;
}

bool learning_probe() {
    constexpr std::size_t rows = 24;
    auto model = network(3);
    const auto x = features(rows), y = targets(x, rows);
    Training training;
    training.seed = 719;
    training.epochs = 0;
    model.fit(x.data(), y.data(), rows, training);
    const auto before = model.parameters();
    training.epochs = 80;
    training.batch_size = 8;
    const auto loss = model.fit(x.data(), y.data(), rows, training);
    if (loss.size() != 81 || !(loss.back() < 0.5 * loss.front())) return false;
    for (const auto& branch : model.branches()) {
        if (!branch.conv) continue;
        double change = 0.0;
        for (std::size_t j = 0; j < branch.conv_k.size; ++j) {
            const auto index = branch.conv_k.offset + j;
            change = std::max(change, std::abs(model.parameters()[index] - before[index]));
        }
        if (change < 1e-8) return false;
    }
    std::printf("joint learning: MSE %.6g -> %.6g; both convolution branches updated\n",
                loss.front(), loss.back());
    return true;
}
}  // namespace

int main() {
    return gradient_probe(0) && gradient_probe(3) && learning_probe() ? 0 : 1;
}
