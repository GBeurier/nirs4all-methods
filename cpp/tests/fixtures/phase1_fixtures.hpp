// SPDX-License-Identifier: CECILL-2.1
//
// Shared MatrixRef definition for the mechanically generated model-parity
// fixture headers (parity/python_generator/src/pls4all_parity/cpp_header.py).
//
// Every per-family header emitted by that generator opens with
// `#include "phase1_fixtures.hpp"` and references the MatrixRef aggregate
// below to describe an expected double matrix: shape, a pointer into the
// generated inline array, its element count, and whether the value is only
// determined up to a column sign (PLS latent matrices). The generator's
// `_matrix_ref` helper emits exactly `MatrixRef{rows, cols, name, size,
// sign_invariant}` so the field order here is load-bearing.
//
// Historically this file was itself generated (it carried the Phase-1 PLS1
// fixtures + a `Phase1Fixture` struct). After the Phase-A structural rename
// the generated fixture headers were dropped from the tree; this slim header
// re-supplies only the shared MatrixRef type the regenerated headers need.

#pragma once

#include <cstddef>
#include <cstdint>

namespace n4m::test::fixtures {

struct MatrixRef {
    std::int64_t rows;
    std::int64_t cols;
    const double* values;
    std::size_t size;
    bool sign_invariant;
};

}  // namespace n4m::test::fixtures
