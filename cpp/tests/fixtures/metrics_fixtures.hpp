// SPDX-License-Identifier: CeCILL-2.1
// Generated mechanically from parity/fixtures/*metrics*.json.
#pragma once

#include "phase1_fixtures.hpp"

namespace pls4all::test::fixtures {

struct MetricsFixture {
    const char* id;
    MatrixRef observed;
    MatrixRef predicted;
    MatrixRef expected;
};

inline const double synthetic_metrics_regression_v1_observed[] = {
    0.20000000000000001, -1.1000000000000001, 1.3500000000000001, -0.45000000000000001,
    2.1000000000000001, 0.050000000000000003, 2.9500000000000002, 0.62,
    3.3999999999999999, 1.2, 4.1500000000000004, 1.95,
    5.0499999999999998, 2.3500000000000001, 5.7999999999999998, 3.1000000000000001,
};

inline const double synthetic_metrics_regression_v1_predicted[] = {
    0.25, -1.1300000000000001, 1.25, -0.37,
    2.2200000000000002, -0.020000000000000004, 2.9100000000000001, 0.71999999999999997,
    3.4899999999999998, 1.0899999999999999, 4.0200000000000005, 2.0099999999999998,
    5.1200000000000001, 2.3000000000000003, 5.7199999999999998, 3.1899999999999999,
};

inline const double synthetic_metrics_regression_v1_expected[] = {
    0.084150163398534164, 0.079375000000000001, 0.0031249999999999924, 0.99805199271001688,
    0.99805199271001688, 1.0021123951654836, -0.007451449348305772, 23.400173252929935,
    31.610158466386714,
};

inline const MetricsFixture kMetricsFixtures[] = {
    {
        "synthetic_metrics_regression_v1",
        MatrixRef{8, 2, synthetic_metrics_regression_v1_observed, sizeof(synthetic_metrics_regression_v1_observed) / sizeof(double), false},
        MatrixRef{8, 2, synthetic_metrics_regression_v1_predicted, sizeof(synthetic_metrics_regression_v1_predicted) / sizeof(double), false},
        MatrixRef{1, 9, synthetic_metrics_regression_v1_expected, sizeof(synthetic_metrics_regression_v1_expected) / sizeof(double), false}
    }
};

}  // namespace pls4all::test::fixtures
