// SPDX-License-Identifier: CECILL-2.1
//
// Portable, fail-closed optimizer checkpoint (N4MOPT v1). Every scalar is
// little-endian and every collection is length-prefixed. The format stores the
// ordered SearchSpace and normalized options together with the complete
// lifecycle/RNG/adaptive-sampler state; it never serializes C++ object bytes.

#include "core/optimization/optimizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace n4m::core::opt {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{{'N', '4', 'M', 'O', 'P', 'T', '\r', '\n'}};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kHeaderSize = 32;
constexpr std::size_t kChecksumSize = sizeof(std::uint64_t);
constexpr std::size_t kMaxCheckpointSize = 64U * 1024U * 1024U;
constexpr std::uint32_t kMaxItems = 100'000U;
constexpr std::uint32_t kMaxStringBytes = 1U * 1024U * 1024U;
// Wire-level cap. Decoding separately checks that the value is representable
// by the reader's actual steady_clock period/rep before reconstructing a
// time_point (the standard permits clocks finer than nanoseconds).
constexpr double kMaxElapsedSeconds = 1.0e9;

static_assert(sizeof(double) == sizeof(std::uint64_t));
static_assert(std::numeric_limits<double>::is_iec559,
              "N4MOPT requires IEEE-754 binary64 doubles");

constexpr std::int32_t signed_i32_from_bits(std::uint32_t raw) noexcept {
    if (raw <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        return static_cast<std::int32_t>(raw);
    }
    // The high bit of raw is set in this branch, so ~raw is guaranteed to fit
    // in int32_t. This avoids every unsigned-to-signed out-of-range conversion.
    return -1 - static_cast<std::int32_t>(~raw);
}

constexpr std::int64_t signed_i64_from_bits(std::uint64_t raw) noexcept {
    if (raw <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return static_cast<std::int64_t>(raw);
    }
    // As above, ~raw has its high bit clear and is representable in int64_t.
    return -1 - static_cast<std::int64_t>(~raw);
}

static_assert(signed_i32_from_bits(0U) == 0);
static_assert(signed_i32_from_bits(0x7fffffffU) == std::numeric_limits<std::int32_t>::max());
static_assert(signed_i32_from_bits(0x80000000U) == std::numeric_limits<std::int32_t>::min());
static_assert(signed_i32_from_bits(0xffffffffU) == -1);
static_assert(signed_i64_from_bits(0ULL) == 0);
static_assert(signed_i64_from_bits(0x7fffffffffffffffULL) ==
              std::numeric_limits<std::int64_t>::max());
static_assert(signed_i64_from_bits(0x8000000000000000ULL) ==
              std::numeric_limits<std::int64_t>::min());
static_assert(signed_i64_from_bits(0xffffffffffffffffULL) == -1);

std::uint64_t fnv1a64(const std::uint8_t* data, std::size_t size) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool valid_utf8(const std::string& value) noexcept {
    const auto* data = reinterpret_cast<const unsigned char*>(value.data());
    std::size_t i = 0;
    while (i < value.size()) {
        const unsigned char c = data[i++];
        if (c <= 0x7FU) continue;
        std::uint32_t code = 0;
        std::size_t continuation = 0;
        if ((c & 0xE0U) == 0xC0U) {
            code = c & 0x1FU;
            continuation = 1;
            if (code < 2U) return false;  // overlong
        } else if ((c & 0xF0U) == 0xE0U) {
            code = c & 0x0FU;
            continuation = 2;
        } else if ((c & 0xF8U) == 0xF0U) {
            code = c & 0x07U;
            continuation = 3;
        } else {
            return false;
        }
        if (continuation > value.size() - i) return false;
        for (std::size_t j = 0; j < continuation; ++j) {
            const unsigned char next = data[i++];
            if ((next & 0xC0U) != 0x80U) return false;
            code = (code << 6U) | (next & 0x3FU);
        }
        if ((continuation == 2 && code < 0x800U) ||
            (continuation == 3 && code < 0x10000U) || code > 0x10FFFFU ||
            (code >= 0xD800U && code <= 0xDFFFU)) {
            return false;
        }
    }
    return true;
}

struct Writer {
    std::vector<std::uint8_t> data;
    bool ok{true};

    void bytes(const void* source, std::size_t size) {
        if (!ok || size > kMaxCheckpointSize || data.size() > kMaxCheckpointSize - size) {
            ok = false;
            return;
        }
        const auto* first = static_cast<const std::uint8_t*>(source);
        data.insert(data.end(), first, first + size);
    }
    void u8(std::uint8_t value) { bytes(&value, sizeof(value)); }
    void u32(std::uint32_t value) {
        std::uint8_t raw[4] = {
            static_cast<std::uint8_t>(value & 0xffU),
            static_cast<std::uint8_t>((value >> 8U) & 0xffU),
            static_cast<std::uint8_t>((value >> 16U) & 0xffU),
            static_cast<std::uint8_t>((value >> 24U) & 0xffU),
        };
        bytes(raw, sizeof(raw));
    }
    void i32(std::int32_t value) { u32(static_cast<std::uint32_t>(value)); }
    void u64(std::uint64_t value) {
        std::uint8_t raw[8] = {};
        for (std::size_t i = 0; i < sizeof(raw); ++i) {
            raw[i] = static_cast<std::uint8_t>((value >> (i * 8U)) & 0xffU);
        }
        bytes(raw, sizeof(raw));
    }
    void i64(std::int64_t value) { u64(static_cast<std::uint64_t>(value)); }
    void f64(double value) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
    }
    void string(const std::string& value) {
        if (!valid_utf8(value) || value.find('\0') != std::string::npos ||
            value.size() > kMaxStringBytes ||
            value.size() > std::numeric_limits<std::uint32_t>::max()) {
            ok = false;
            return;
        }
        u32(static_cast<std::uint32_t>(value.size()));
        bytes(value.data(), value.size());
    }
};

struct Reader {
    const std::uint8_t* data{nullptr};
    std::size_t size{0};
    std::size_t pos{0};

    bool bytes(void* destination, std::size_t count) noexcept {
        if (count > size - pos) return false;
        std::memcpy(destination, data + pos, count);
        pos += count;
        return true;
    }
    bool u8(std::uint8_t& value) noexcept { return bytes(&value, sizeof(value)); }
    bool u32(std::uint32_t& value) noexcept {
        std::uint8_t raw[4] = {};
        if (!bytes(raw, sizeof(raw))) return false;
        value = static_cast<std::uint32_t>(raw[0]) |
                (static_cast<std::uint32_t>(raw[1]) << 8U) |
                (static_cast<std::uint32_t>(raw[2]) << 16U) |
                (static_cast<std::uint32_t>(raw[3]) << 24U);
        return true;
    }
    bool i32(std::int32_t& value) noexcept {
        std::uint32_t raw = 0;
        if (!u32(raw)) return false;
        value = signed_i32_from_bits(raw);
        return true;
    }
    bool u64(std::uint64_t& value) noexcept {
        std::uint8_t raw[8] = {};
        if (!bytes(raw, sizeof(raw))) return false;
        value = 0;
        for (std::size_t i = 0; i < sizeof(raw); ++i) {
            value |= static_cast<std::uint64_t>(raw[i]) << (i * 8U);
        }
        return true;
    }
    bool i64(std::int64_t& value) noexcept {
        std::uint64_t raw = 0;
        if (!u64(raw)) return false;
        value = signed_i64_from_bits(raw);
        return true;
    }
    bool f64(double& value) noexcept {
        std::uint64_t bits = 0;
        if (!u64(bits)) return false;
        std::memcpy(&value, &bits, sizeof(value));
        return true;
    }
    bool string(std::string& value) {
        std::uint32_t count = 0;
        if (!u32(count) || count > kMaxStringBytes || count > size - pos) return false;
        value.assign(reinterpret_cast<const char*>(data + pos), count);
        pos += count;
        return valid_utf8(value) && value.find('\0') == std::string::npos;
    }
    bool section(std::uint64_t count, Reader& out) noexcept {
        if (count > size - pos || count > kMaxCheckpointSize) return false;
        out = Reader{data + pos, static_cast<std::size_t>(count), 0};
        pos += static_cast<std::size_t>(count);
        return true;
    }
    bool consumed() const noexcept { return pos == size; }
    std::size_t remaining() const noexcept { return size - pos; }
};

void write_double_vector(Writer& writer, const std::vector<double>& values) {
    if (values.size() > kMaxItems) {
        writer.ok = false;
        return;
    }
    writer.u32(static_cast<std::uint32_t>(values.size()));
    for (const double value : values) writer.f64(value);
}

bool read_double_vector(Reader& reader, std::vector<double>& values) {
    std::uint32_t count = 0;
    if (!reader.u32(count) || count > kMaxItems ||
        count > reader.remaining() / sizeof(double)) {
        return false;
    }
    values.resize(count);
    for (double& value : values) {
        if (!reader.f64(value)) return false;
    }
    return true;
}

void write_u32_vector(Writer& writer, const std::vector<std::uint32_t>& values) {
    if (values.size() > kMaxItems) {
        writer.ok = false;
        return;
    }
    writer.u32(static_cast<std::uint32_t>(values.size()));
    for (const std::uint32_t value : values) writer.u32(value);
}

bool read_u32_vector(Reader& reader, std::vector<std::uint32_t>& values) {
    std::uint32_t count = 0;
    if (!reader.u32(count) || count > kMaxItems ||
        count > reader.remaining() / sizeof(std::uint32_t)) {
        return false;
    }
    values.resize(count);
    for (std::uint32_t& value : values) {
        if (!reader.u32(value)) return false;
    }
    return true;
}

void write_matrix(Writer& writer, const std::vector<std::vector<double>>& matrix) {
    if (matrix.size() > kMaxItems) {
        writer.ok = false;
        return;
    }
    writer.u32(static_cast<std::uint32_t>(matrix.size()));
    for (const auto& row : matrix) write_double_vector(writer, row);
}

bool read_matrix(Reader& reader, std::vector<std::vector<double>>& matrix) {
    std::uint32_t rows = 0;
    if (!reader.u32(rows) || rows > kMaxItems ||
        rows > reader.remaining() / sizeof(std::uint32_t)) {
        return false;
    }
    matrix.resize(rows);
    for (auto& row : matrix) {
        if (!read_double_vector(reader, row)) return false;
    }
    return true;
}

void encode_space(const SearchSpace& space, Writer& writer) {
    if (space.params.size() > kMaxItems || space.constraints.size() > kMaxItems) {
        writer.ok = false;
        return;
    }
    writer.u32(static_cast<std::uint32_t>(space.params.size()));
    for (const ParamSpec& param : space.params) {
        writer.string(param.name);
        writer.u32(static_cast<std::uint32_t>(param.kind));
        writer.f64(param.low);
        writer.f64(param.high);
        writer.f64(param.step);
        writer.u8(param.is_int ? 1U : 0U);
        writer.u8(param.is_log ? 1U : 0U);
        writer.u32(static_cast<std::uint32_t>(param.cat_type));
        if (param.labels.size() > kMaxItems) {
            writer.ok = false;
            return;
        }
        writer.u32(static_cast<std::uint32_t>(param.labels.size()));
        for (const std::string& label : param.labels) writer.string(label);
        write_double_vector(writer, param.num_values);
        writer.i32(param.tuple_length);
        writer.u8(param.tuple_element_is_int ? 1U : 0U);
    }
    writer.u32(static_cast<std::uint32_t>(space.constraints.size()));
    for (const Constraint& constraint : space.constraints) {
        if (constraint.param_refs.size() > kMaxItems ||
            constraint.label_refs.size() != constraint.param_refs.size()) {
            writer.ok = false;
            return;
        }
        writer.u32(static_cast<std::uint32_t>(constraint.kind));
        writer.u32(static_cast<std::uint32_t>(constraint.param_refs.size()));
        for (std::size_t i = 0; i < constraint.param_refs.size(); ++i) {
            writer.string(constraint.param_refs[i]);
            writer.string(constraint.label_refs[i]);
        }
    }
}

bool decode_space(Reader& reader, SearchSpace& space) {
    std::uint32_t n_params = 0;
    if (!reader.u32(n_params) || n_params == 0 || n_params > kMaxItems ||
        n_params > reader.remaining() / 40U) {
        return false;
    }
    space.params.resize(n_params);
    for (ParamSpec& param : space.params) {
        std::uint32_t kind = 0;
        std::uint8_t is_int = 0;
        std::uint8_t is_log = 0;
        std::uint32_t cat_type = 0;
        std::uint32_t n_labels = 0;
        std::uint8_t tuple_int = 0;
        if (!reader.string(param.name) || !reader.u32(kind) || !reader.f64(param.low) ||
            !reader.f64(param.high) || !reader.f64(param.step) || !reader.u8(is_int) ||
            !reader.u8(is_log) || !reader.u32(cat_type) || !reader.u32(n_labels) ||
            n_labels > kMaxItems || n_labels > reader.remaining() / sizeof(std::uint32_t) ||
            is_int > 1U || is_log > 1U) {
            return false;
        }
        param.kind = static_cast<n4m_param_kind_t>(kind);
        param.is_int = is_int != 0U;
        param.is_log = is_log != 0U;
        param.cat_type = static_cast<n4m_cat_type_t>(cat_type);
        param.labels.resize(n_labels);
        for (std::string& label : param.labels) {
            if (!reader.string(label)) return false;
        }
        if (!read_double_vector(reader, param.num_values) || !reader.i32(param.tuple_length) ||
            !reader.u8(tuple_int) || tuple_int > 1U) {
            return false;
        }
        param.tuple_element_is_int = tuple_int != 0U;
    }

    std::uint32_t n_constraints = 0;
    if (!reader.u32(n_constraints) || n_constraints > kMaxItems ||
        n_constraints > reader.remaining() / 8U) {
        return false;
    }
    space.constraints.resize(n_constraints);
    for (Constraint& constraint : space.constraints) {
        std::uint32_t kind = 0;
        std::uint32_t n_refs = 0;
        if (!reader.u32(kind) || !reader.u32(n_refs) || n_refs > kMaxItems) return false;
        constraint.kind = static_cast<n4m_constraint_kind_t>(kind);
        constraint.param_refs.resize(n_refs);
        constraint.label_refs.resize(n_refs);
        for (std::size_t i = 0; i < n_refs; ++i) {
            if (!reader.string(constraint.param_refs[i]) ||
                !reader.string(constraint.label_refs[i])) {
                return false;
            }
        }
    }

    // Rebuild the compiled condition metadata from the authoritative constraint
    // records. The same consistency rules are enforced by SearchSpace::validate.
    for (const Constraint& constraint : space.constraints) {
        if ((constraint.kind != N4M_CONSTRAINT_CONDITION_IN &&
             constraint.kind != N4M_CONSTRAINT_CONDITION_NOT_IN) ||
            constraint.param_refs.size() != 2 || constraint.label_refs.size() != 2) {
            continue;
        }
        ParamSpec* child = space.find(constraint.param_refs[0]);
        if (child == nullptr) return false;
        const bool is_in = constraint.kind == N4M_CONSTRAINT_CONDITION_IN;
        if (child->cond_parent.empty() ||
            (child->cond_parent == constraint.param_refs[1] && child->cond_is_in == is_in)) {
            child->cond_parent = constraint.param_refs[1];
            child->cond_is_in = is_in;
            if (!constraint.label_refs[1].empty()) {
                child->cond_labels.push_back(constraint.label_refs[1]);
            }
        }
    }
    return reader.consumed();
}

void encode_options(const n4m_optimizer_options_t& options, Writer& writer) {
    writer.u32(static_cast<std::uint32_t>(options.sampler));
    writer.u32(static_cast<std::uint32_t>(options.pruner));
    writer.u32(static_cast<std::uint32_t>(options.direction));
    writer.u32(static_cast<std::uint32_t>(options.eval_mode));
    writer.u32(static_cast<std::uint32_t>(options.metric));
    writer.u32(static_cast<std::uint32_t>(options.liar));
    writer.i32(options.n_startup_trials);
    writer.u64(options.seed);
    writer.f64(options.timeout_seconds);
    writer.i32(options.max_resource);
    writer.i32(options.reduction_factor);
}

bool decode_options(Reader& reader, n4m_optimizer_options_t& options) {
    std::uint32_t sampler = 0;
    std::uint32_t pruner = 0;
    std::uint32_t direction = 0;
    std::uint32_t eval_mode = 0;
    std::uint32_t metric = 0;
    std::uint32_t liar = 0;
    options = {};
    options.struct_size = sizeof(options);
    if (!reader.u32(sampler) || !reader.u32(pruner) || !reader.u32(direction) ||
        !reader.u32(eval_mode) || !reader.u32(metric) || !reader.u32(liar) ||
        !reader.i32(options.n_startup_trials) || !reader.u64(options.seed) ||
        !reader.f64(options.timeout_seconds) || !reader.i32(options.max_resource) ||
        !reader.i32(options.reduction_factor) || !reader.consumed()) {
        return false;
    }
    options.sampler = static_cast<n4m_sampler_kind_t>(sampler);
    options.pruner = static_cast<n4m_pruner_kind_t>(pruner);
    options.direction = static_cast<n4m_opt_direction_t>(direction);
    options.eval_mode = static_cast<n4m_eval_mode_t>(eval_mode);
    options.metric = static_cast<n4m_metric_t>(metric);
    options.liar = static_cast<n4m_liar_kind_t>(liar);
    return true;
}

bool valid_error_code(const std::string& code) noexcept {
    if (code.size() < 2 || code.size() > 64 || code[0] < 'A' || code[0] > 'Z') return false;
    return std::all_of(code.begin(), code.end(), [](char value) {
        return value == '_' || (value >= 'A' && value <= 'Z') ||
               (value >= '0' && value <= '9');
    });
}

bool finite_unit_matrix(const std::vector<std::vector<double>>& matrix,
                        std::size_t rows, std::size_t cols) noexcept {
    if (matrix.size() != rows) return false;
    for (const auto& row : matrix) {
        if (row.size() != cols) return false;
        for (const double value : row) {
            if (!std::isfinite(value) || value < 0.0 || value >= 1.0) return false;
        }
    }
    return true;
}

bool finite_vector(const std::vector<double>& values) noexcept {
    return std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
    });
}

bool pso_score_state_valid(const std::vector<std::vector<double>>& current_positions,
                           const std::vector<double>& pbest_scores,
                           const std::vector<std::vector<double>>& pbest_positions,
                           const std::vector<double>& gbest_position, double gbest_score,
                           bool have_gbest, bool materialized, std::int64_t population_base,
                           n4m_opt_direction_t direction) {
    if (direction != N4M_OPT_MINIMIZE && direction != N4M_OPT_MAXIMIZE) return false;
    const double worst = direction == N4M_OPT_MAXIMIZE
                             ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity();
    for (const double score : pbest_scores) {
        if (std::isnan(score) || (std::isinf(score) && score != worst)) return false;
    }

    if (!have_gbest) {
        if (gbest_score != 0.0 ||
            !std::all_of(pbest_scores.begin(), pbest_scores.end(),
                         [worst](double score) { return score == worst; })) {
            return false;
        }
        // Before the first completed swarm is folded in, personal-best
        // positions are exact copies of the initial particles and the neutral
        // global position is the exact midpoint of every axis. Once PSO moves
        // to a later swarm it always materializes a global best, even when all
        // trials failed and the score remains the directional worst sentinel.
        if (materialized &&
            (population_base != 0 || pbest_positions != current_positions)) {
            return false;
        }
        return !materialized ||
               std::all_of(gbest_position.begin(), gbest_position.end(),
                           [](double value) { return value == 0.5; });
    }

    if (!materialized || population_base <= 0 || pbest_scores.empty() ||
        pbest_positions.size() != pbest_scores.size() || std::isnan(gbest_score) ||
        (std::isinf(gbest_score) && gbest_score != worst)) {
        return false;
    }
    double best = pbest_scores.front();
    for (const double score : pbest_scores) {
        if ((direction == N4M_OPT_MAXIMIZE && score > best) ||
            (direction == N4M_OPT_MINIMIZE && score < best)) {
            best = score;
        }
    }
    if (gbest_score != best) return false;
    for (std::size_t i = 0; i < pbest_scores.size(); ++i) {
        if (pbest_scores[i] == best && pbest_positions[i] == gbest_position) return true;
    }
    return false;
}

bool population_base_valid(std::int64_t base, std::int32_t population_size,
                           std::int64_t next_id) noexcept {
    // Population state is initialized lazily while asking the member at
    // `base`; ask() increments next_id only after that member is committed.
    // Consequently a materialized population always owns at least one trial.
    if (base < 0 || population_size <= 0 || base >= next_id) return false;
    return base % population_size == 0 && next_id - base <= population_size;
}

bool subtract_seconds(std::chrono::steady_clock::time_point now, double seconds,
                      std::chrono::steady_clock::time_point& result) noexcept {
    using Duration = std::chrono::steady_clock::duration;
    using Rep = Duration::rep;

    if (!std::isfinite(seconds) || seconds < 0.0) return false;
    const long double ticks =
        static_cast<long double>(seconds) * static_cast<long double>(Duration::period::den) /
        static_cast<long double>(Duration::period::num);
    // Reject the upper boundary too: converting an integer rep's maximum to
    // long double may round upward on implementations where long double has no
    // more precision than double.
    if (!std::isfinite(ticks) || ticks < 0.0L ||
        ticks >= static_cast<long double>(std::numeric_limits<Rep>::max())) {
        return false;
    }
    const Rep delta = static_cast<Rep>(ticks);
    const Rep current = now.time_since_epoch().count();
    if constexpr (std::numeric_limits<Rep>::is_signed) {
        if (current < std::numeric_limits<Rep>::lowest() + delta) return false;
    } else if (delta > current) {
        return false;
    }
    result = std::chrono::steady_clock::time_point(Duration(current - delta));
    return true;
}

void set_failure(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

}  // namespace

struct OptimizerCheckpointAccess {
    static void write_trial(Writer& writer, const ::n4m_trial_s& trial,
                            std::chrono::steady_clock::time_point now) {
        writer.i64(trial.id);
        if (trial.params.size() > kMaxItems || trial.intermediates.size() > kMaxItems ||
            trial.intermediate_sequences.size() != trial.intermediates.size()) {
            writer.ok = false;
            return;
        }
        writer.u32(static_cast<std::uint32_t>(trial.params.size()));
        for (const auto& item : trial.params) {
            writer.string(item.first);
            writer.f64(item.second.value);
            writer.i32(item.second.cat_index);
            writer.string(item.second.cat_label);
            writer.u8(item.second.active ? 1U : 0U);
        }
        writer.u32(static_cast<std::uint32_t>(trial.status));
        writer.f64(trial.score);
        writer.u8(trial.has_score ? 1U : 0U);
        writer.f64(trial.duration_seconds);
        writer.i32(trial.rung);
        writer.u8(trial.has_error ? 1U : 0U);
        writer.u8(trial.pruned_by_policy ? 1U : 0U);
        writer.i64(trial.ask_sequence);
        writer.i64(trial.terminal_sequence);
        double age = std::chrono::duration<double>(now - trial.ask_time).count();
        if (!std::isfinite(age) || age < 0.0) age = 0.0;
        writer.f64(age);
        writer.u32(static_cast<std::uint32_t>(trial.intermediates.size()));
        for (std::size_t i = 0; i < trial.intermediates.size(); ++i) {
            writer.i32(trial.intermediates[i].first);
            writer.f64(trial.intermediates[i].second);
            writer.i64(trial.intermediate_sequences[i]);
        }
        writer.string(trial.error.code);
        writer.string(trial.error.message);
        writer.u8(trial.error.retryable ? 1U : 0U);
    }

    static bool read_trial(Reader& reader, ::n4m_trial_s& trial,
                           std::chrono::steady_clock::time_point now,
                           std::size_t expected_params) {
        std::uint32_t n_params = 0;
        std::uint32_t status = 0;
        std::uint8_t has_score = 0;
        std::uint8_t has_error = 0;
        std::uint8_t pruned = 0;
        double age = 0.0;
        if (!reader.i64(trial.id) || !reader.u32(n_params) || n_params > kMaxItems ||
            n_params != expected_params ||
            n_params > reader.remaining() / 21U) {
            return false;
        }
        trial.params.resize(n_params);
        for (auto& item : trial.params) {
            std::uint8_t active = 0;
            if (!reader.string(item.first) || !reader.f64(item.second.value) ||
                !reader.i32(item.second.cat_index) || !reader.string(item.second.cat_label) ||
                !reader.u8(active) || active > 1U) {
                return false;
            }
            item.second.active = active != 0U;
        }
        if (!reader.u32(status) || !reader.f64(trial.score) || !reader.u8(has_score) ||
            !reader.f64(trial.duration_seconds) || !reader.i32(trial.rung) ||
            !reader.u8(has_error) || !reader.u8(pruned) ||
            !reader.i64(trial.ask_sequence) || !reader.i64(trial.terminal_sequence) ||
            !reader.f64(age) || has_score > 1U || has_error > 1U || pruned > 1U ||
            !std::isfinite(age) || age < 0.0 || age > kMaxElapsedSeconds) {
            return false;
        }
        trial.status = static_cast<n4m_trial_status_t>(status);
        trial.has_score = has_score != 0U;
        trial.has_error = has_error != 0U;
        trial.pruned_by_policy = pruned != 0U;
        if (!subtract_seconds(now, age, trial.ask_time)) return false;
        std::uint32_t n_intermediates = 0;
        if (!reader.u32(n_intermediates) || n_intermediates > kMaxItems ||
            n_intermediates > reader.remaining() / 20U) {
            return false;
        }
        trial.intermediates.resize(n_intermediates);
        trial.intermediate_sequences.resize(n_intermediates);
        for (std::size_t i = 0; i < n_intermediates; ++i) {
            if (!reader.i32(trial.intermediates[i].first) ||
                !reader.f64(trial.intermediates[i].second) ||
                !reader.i64(trial.intermediate_sequences[i])) {
                return false;
            }
        }
        std::uint8_t retryable = 0;
        if (!reader.string(trial.error.code) || !reader.string(trial.error.message) ||
            !reader.u8(retryable) || retryable > 1U) {
            return false;
        }
        trial.error.retryable = retryable != 0U;
        return true;
    }

    static n4m_status_t encode(const Optimizer& optimizer, std::vector<std::uint8_t>& out,
                               std::string* error) {
        out.clear();
        Writer space_writer;
        encode_space(optimizer.space_, space_writer);
        Writer options_writer;
        encode_options(optimizer.opts_, options_writer);
        if (!space_writer.ok || !options_writer.ok) {
            set_failure(error, "optimizer space/options exceed checkpoint limits");
            return N4M_ERR_UNSUPPORTED;
        }

        Writer payload;
        payload.u64(fnv1a64(space_writer.data.data(), space_writer.data.size()));
        payload.u64(static_cast<std::uint64_t>(space_writer.data.size()));
        payload.bytes(space_writer.data.data(), space_writer.data.size());
        payload.u64(fnv1a64(options_writer.data.data(), options_writer.data.size()));
        payload.u64(static_cast<std::uint64_t>(options_writer.data.size()));
        payload.bytes(options_writer.data.data(), options_writer.data.size());

        const auto now = std::chrono::steady_clock::now();
        payload.u32(static_cast<std::uint32_t>(optimizer.rng_.kind));
        payload.u64(optimizer.rng_.sm_state);
        double elapsed = std::chrono::duration<double>(now - optimizer.start_time_).count();
        if (!std::isfinite(elapsed) || elapsed < 0.0 || elapsed > kMaxElapsedSeconds) {
            set_failure(error, "optimizer elapsed time is not checkpointable");
            return N4M_ERR_UNSUPPORTED;
        }
        payload.f64(elapsed);
        payload.i64(optimizer.next_id_);
        payload.i64(optimizer.next_event_sequence_);
        if (optimizer.trials_.size() > kMaxItems || optimizer.enqueued_.size() > kMaxItems) {
            set_failure(error, "optimizer history exceeds checkpoint limits");
            return N4M_ERR_UNSUPPORTED;
        }
        payload.u32(static_cast<std::uint32_t>(optimizer.trials_.size()));
        for (const auto& trial : optimizer.trials_) write_trial(payload, *trial, now);
        payload.u32(static_cast<std::uint32_t>(optimizer.enqueued_.size()));
        for (const auto& queued : optimizer.enqueued_) {
            if (queued.size() > kMaxItems) {
                payload.ok = false;
                break;
            }
            payload.u32(static_cast<std::uint32_t>(queued.size()));
            for (const auto& item : queued) {
                payload.string(item.first);
                payload.f64(item.second);
            }
        }

        payload.u32(static_cast<std::uint32_t>(optimizer.opts_.sampler));
        switch (optimizer.opts_.sampler) {
            case N4M_SAMPLER_RANDOM:
            case N4M_SAMPLER_LHS:
            case N4M_SAMPLER_TERNARY:
            case N4M_SAMPLER_TPE:
                break;  // derived state + base RNG/history are sufficient
            case N4M_SAMPLER_SOBOL: {
                const auto* sampler = dynamic_cast<const SobolSampler*>(&optimizer);
                if (sampler == nullptr) return N4M_ERR_INTERNAL;
                payload.i32(sampler->sobol_dims_);
                write_u32_vector(payload, sampler->x_);
                write_double_vector(payload, sampler->cached_point_);
                payload.i64(sampler->cached_ask_index_);
                payload.i64(sampler->next_index_);
                payload.u8(sampler->exhausted_ ? 1U : 0U);
                break;
            }
            case N4M_SAMPLER_GA: {
                const auto* sampler = dynamic_cast<const GaSampler*>(&optimizer);
                if (sampler == nullptr) return N4M_ERR_INTERNAL;
                write_matrix(payload, sampler->gen_);
                payload.i64(sampler->gen_base_id_);
                break;
            }
            case N4M_SAMPLER_PSO: {
                const auto* sampler = dynamic_cast<const PsoSampler*>(&optimizer);
                if (sampler == nullptr) return N4M_ERR_INTERNAL;
                write_matrix(payload, sampler->pos_);
                write_matrix(payload, sampler->vel_);
                write_matrix(payload, sampler->pbest_pos_);
                write_double_vector(payload, sampler->pbest_score_);
                write_double_vector(payload, sampler->gbest_pos_);
                payload.f64(sampler->gbest_score_);
                payload.u8(sampler->have_gbest_ ? 1U : 0U);
                payload.i64(sampler->iter_base_id_);
                break;
            }
            case N4M_SAMPLER_CMAES: {
                const auto* sampler = dynamic_cast<const CmaEsSampler*>(&optimizer);
                if (sampler == nullptr) return N4M_ERR_INTERNAL;
                write_double_vector(payload, sampler->m_);
                payload.f64(sampler->sigma_);
                write_double_vector(payload, sampler->C_);
                write_double_vector(payload, sampler->ps_);
                write_double_vector(payload, sampler->pc_);
                payload.i64(sampler->gen_count_);
                write_matrix(payload, sampler->pop_x_);
                payload.i64(sampler->gen_base_id_);
                break;
            }
            case N4M_SAMPLER_GP_EI: {
                const auto* sampler = dynamic_cast<const GpEiSampler*>(&optimizer);
                if (sampler == nullptr || sampler->proposals_.size() > kMaxItems) {
                    return sampler == nullptr ? N4M_ERR_INTERNAL : N4M_ERR_UNSUPPORTED;
                }
                payload.u32(static_cast<std::uint32_t>(sampler->proposals_.size()));
                for (const auto& item : sampler->proposals_) {
                    payload.i64(item.first);
                    write_double_vector(payload, item.second);
                }
                break;
            }
            default:
                set_failure(error, "optimizer sampler has no checkpoint codec");
                return N4M_ERR_NOT_IMPLEMENTED;
        }
        if (!payload.ok) {
            set_failure(error, "optimizer state exceeds checkpoint limits");
            return N4M_ERR_UNSUPPORTED;
        }

        Writer result;
        result.bytes(kMagic.data(), kMagic.size());
        result.u32(kFormatVersion);
        result.u32(kHeaderSize);
        const std::size_t unpadded = kHeaderSize + payload.data.size() + kChecksumSize;
        const std::size_t padding = (8U - (unpadded % 8U)) % 8U;
        const std::size_t total = unpadded + padding;
        if (total > kMaxCheckpointSize) {
            set_failure(error, "optimizer checkpoint exceeds the 64 MiB format limit");
            return N4M_ERR_UNSUPPORTED;
        }
        result.u64(static_cast<std::uint64_t>(total));
        result.u64(static_cast<std::uint64_t>(payload.data.size()));
        result.bytes(payload.data.data(), payload.data.size());
        for (std::size_t i = 0; i < padding; ++i) result.u8(0U);
        result.u64(fnv1a64(result.data.data(), result.data.size()));
        if (!result.ok || result.data.size() != total) return N4M_ERR_INTERNAL;
        out = std::move(result.data);
        return N4M_OK;
    }

    static bool trial_shape_valid(Optimizer& optimizer, const ::n4m_trial_s& trial,
                                  std::vector<std::int64_t>& events) {
        std::vector<std::pair<std::string, const ParamSpec*>> expected;
        for (const ParamSpec& param : optimizer.space_.params) {
            if (param.kind == N4M_PARAM_SORTED_TUPLE) {
                for (std::int32_t i = 0; i < param.tuple_length; ++i) {
                    expected.emplace_back(param.name + "#" + std::to_string(i), &param);
                }
            } else {
                expected.emplace_back(param.name, &param);
            }
        }
        if (trial.params.size() != expected.size() || trial.ask_sequence < 0 ||
            !std::isfinite(trial.duration_seconds) || trial.duration_seconds < 0.0 ||
            trial.rung < 0 || trial.intermediates.size() != trial.intermediate_sequences.size()) {
            return false;
        }
        ::n4m_trial_s activation_copy = trial;
        for (auto& item : activation_copy.params) item.second.active = true;
        optimizer.apply_conditions(activation_copy);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const auto& actual = trial.params[i];
            const ParamSpec& spec = *expected[i].second;
            if (actual.first != expected[i].first || !std::isfinite(actual.second.value) ||
                actual.second.active != activation_copy.params[i].second.active) {
                return false;
            }
            if (spec.kind == N4M_PARAM_CATEGORICAL || spec.kind == N4M_PARAM_ORDINAL) {
                const std::int32_t index = actual.second.cat_index;
                if (index < 0 || static_cast<std::size_t>(index) >= spec.num_values.size() ||
                    actual.second.value != spec.num_values[static_cast<std::size_t>(index)] ||
                    actual.second.cat_label != spec.labels[static_cast<std::size_t>(index)]) {
                    return false;
                }
            } else {
                if (actual.second.cat_index != -1 || !actual.second.cat_label.empty() ||
                    actual.second.value < spec.low || actual.second.value > spec.high) {
                    return false;
                }
                if ((spec.is_int || (spec.kind == N4M_PARAM_SORTED_TUPLE &&
                                     spec.tuple_element_is_int)) &&
                    std::trunc(actual.second.value) != actual.second.value) {
                    return false;
                }
            }
        }
        if (!optimizer.constraints_ok(trial)) return false;

        if (trial.status < N4M_TRIAL_RUNNING || trial.status > N4M_TRIAL_CANCELLED) return false;
        const bool completed = trial.status == N4M_TRIAL_COMPLETED;
        const bool error_status = trial.status == N4M_TRIAL_FAILED ||
                                  trial.status == N4M_TRIAL_CANCELLED;
        const bool terminal = trial.status != N4M_TRIAL_RUNNING;
        if (trial.has_score != completed || (completed && !std::isfinite(trial.score)) ||
            trial.has_error != error_status ||
            (error_status &&
             (!valid_error_code(trial.error.code) || trial.error.message.empty())) ||
            (!completed && trial.score != 0.0) ||
            (!error_status && (!trial.error.code.empty() || !trial.error.message.empty() ||
                               trial.error.retryable)) ||
            terminal != (trial.terminal_sequence >= 0) ||
            (terminal && trial.terminal_sequence <= trial.ask_sequence) ||
            (trial.pruned_by_policy &&
             (trial.status != N4M_TRIAL_PRUNED || trial.intermediates.empty()))) {
            return false;
        }
        events.push_back(trial.ask_sequence);
        std::int32_t last_step = -1;
        std::int64_t last_sequence = trial.ask_sequence;
        for (std::size_t i = 0; i < trial.intermediates.size(); ++i) {
            const auto& intermediate = trial.intermediates[i];
            const std::int64_t sequence = trial.intermediate_sequences[i];
            if (intermediate.first < 0 || intermediate.first <= last_step ||
                !std::isfinite(intermediate.second) || sequence <= last_sequence ||
                (trial.terminal_sequence >= 0 && sequence >= trial.terminal_sequence)) {
                return false;
            }
            last_step = intermediate.first;
            last_sequence = sequence;
            events.push_back(sequence);
        }
        if (terminal) events.push_back(trial.terminal_sequence);
        return true;
    }

    static bool validate_base_state(Optimizer& optimizer) {
        if (optimizer.rng_.kind != N4M_RNGK_SPLITMIX64 || optimizer.next_id_ < 0 ||
            optimizer.next_event_sequence_ < 0 ||
            static_cast<std::uint64_t>(optimizer.next_id_) != optimizer.trials_.size()) {
            return false;
        }
        std::vector<std::int64_t> events;
        for (std::size_t i = 0; i < optimizer.trials_.size(); ++i) {
            if (optimizer.trials_[i] == nullptr ||
                optimizer.trials_[i]->id != static_cast<std::int64_t>(i) ||
                !trial_shape_valid(optimizer, *optimizer.trials_[i], events)) {
                return false;
            }
        }
        if (events.size() != static_cast<std::uint64_t>(optimizer.next_event_sequence_)) {
            return false;
        }
        std::sort(events.begin(), events.end());
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (events[i] != static_cast<std::int64_t>(i)) return false;
        }
        return true;
    }

    static n4m_status_t decode(const std::uint8_t* data, std::size_t size,
                               std::unique_ptr<Optimizer>& out, std::string* error) {
        out.reset();
        if (data == nullptr) return N4M_ERR_NULL_POINTER;
        if (size < kHeaderSize + kChecksumSize || size > kMaxCheckpointSize) {
            set_failure(error, "optimizer checkpoint size is invalid");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        if (std::memcmp(data, kMagic.data(), kMagic.size()) != 0) {
            set_failure(error, "optimizer checkpoint magic is invalid");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        Reader header{data + kMagic.size(), kHeaderSize - kMagic.size(), 0};
        std::uint32_t version = 0;
        std::uint32_t header_size = 0;
        std::uint64_t total_size = 0;
        std::uint64_t payload_size = 0;
        if (!header.u32(version) || !header.u32(header_size) || !header.u64(total_size) ||
            !header.u64(payload_size) || !header.consumed()) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        if (version != kFormatVersion) {
            set_failure(error, "optimizer checkpoint format version is unsupported");
            return N4M_ERR_VERSION_INCOMPATIBLE;
        }
        if (header_size != kHeaderSize || total_size != size ||
            payload_size > size - kHeaderSize - kChecksumSize) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        const std::size_t trailer_offset = size - kChecksumSize;
        Reader trailer{data + trailer_offset, kChecksumSize, 0};
        std::uint64_t checksum = 0;
        if (!trailer.u64(checksum) || checksum != fnv1a64(data, trailer_offset)) {
            set_failure(error, "optimizer checkpoint checksum is invalid");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        const std::size_t payload_end = kHeaderSize + static_cast<std::size_t>(payload_size);
        const std::size_t canonical_padding =
            (8U - ((payload_end + kChecksumSize) % 8U)) % 8U;
        if (trailer_offset - payload_end != canonical_padding) {
            set_failure(error, "optimizer checkpoint padding is non-canonical");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        for (std::size_t i = payload_end; i < trailer_offset; ++i) {
            if (data[i] != 0U) return N4M_ERR_CORRUPT_BUFFER;
        }

        Reader payload{data + kHeaderSize, static_cast<std::size_t>(payload_size), 0};
        std::uint64_t space_hash = 0;
        std::uint64_t space_size = 0;
        Reader space_reader;
        if (!payload.u64(space_hash) || !payload.u64(space_size) ||
            !payload.section(space_size, space_reader) ||
            fnv1a64(space_reader.data, space_reader.size) != space_hash) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        SearchSpace space;
        if (!decode_space(space_reader, space)) return N4M_ERR_CORRUPT_BUFFER;

        std::uint64_t options_hash = 0;
        std::uint64_t options_size = 0;
        Reader options_reader;
        if (!payload.u64(options_hash) || !payload.u64(options_size) ||
            !payload.section(options_size, options_reader) ||
            fnv1a64(options_reader.data, options_reader.size) != options_hash) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_optimizer_options_t options{};
        if (!decode_options(options_reader, options)) return N4M_ERR_CORRUPT_BUFFER;
        std::string validation_error;
        if (validate_optimizer_options(options, &validation_error) != N4M_OK ||
            space.validate(options.sampler, &validation_error) != N4M_OK) {
            set_failure(error, "optimizer checkpoint option/search-space contract mismatch");
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_status_t make_status = N4M_OK;
        std::unique_ptr<Optimizer> optimizer =
            make_optimizer(space, options, &make_status, &validation_error);
        if (optimizer == nullptr) {
            set_failure(error, "optimizer checkpoint sampler/pruner is unavailable");
            return make_status == N4M_ERR_NOT_IMPLEMENTED ? make_status : N4M_ERR_CORRUPT_BUFFER;
        }

        std::uint32_t rng_kind = 0;
        double elapsed = 0.0;
        if (!payload.u32(rng_kind) || !payload.u64(optimizer->rng_.sm_state) ||
            !payload.f64(elapsed) || !payload.i64(optimizer->next_id_) ||
            !payload.i64(optimizer->next_event_sequence_) ||
            rng_kind != static_cast<std::uint32_t>(N4M_RNGK_SPLITMIX64) ||
            !std::isfinite(elapsed) || elapsed < 0.0 || elapsed > kMaxElapsedSeconds) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        optimizer->rng_.kind = N4M_RNGK_SPLITMIX64;
        const auto now = std::chrono::steady_clock::now();
        if (!subtract_seconds(now, elapsed, optimizer->start_time_)) {
            return N4M_ERR_CORRUPT_BUFFER;
        }

        std::uint32_t n_trials = 0;
        if (!payload.u32(n_trials) || n_trials > kMaxItems || optimizer->next_id_ < 0 ||
            static_cast<std::uint64_t>(optimizer->next_id_) != n_trials ||
            n_trials > payload.remaining() / 64U) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        std::size_t expected_params = 0;
        for (const ParamSpec& param : space.params) {
            const std::size_t add = param.kind == N4M_PARAM_SORTED_TUPLE
                                        ? static_cast<std::size_t>(param.tuple_length)
                                        : 1U;
            if (add > kMaxItems - expected_params) return N4M_ERR_CORRUPT_BUFFER;
            expected_params += add;
        }
        optimizer->trials_.clear();
        optimizer->trials_.reserve(n_trials);
        for (std::uint32_t i = 0; i < n_trials; ++i) {
            auto trial = std::make_unique<::n4m_trial_s>();
            if (!read_trial(payload, *trial, now, expected_params)) {
                return N4M_ERR_CORRUPT_BUFFER;
            }
            optimizer->trials_.push_back(std::move(trial));
        }

        std::uint32_t n_queued = 0;
        if (!payload.u32(n_queued) || n_queued > kMaxItems ||
            n_queued > payload.remaining() / sizeof(std::uint32_t)) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        std::vector<std::vector<std::pair<std::string, double>>> queued(n_queued);
        for (auto& entry : queued) {
            std::uint32_t count = 0;
            if (!payload.u32(count) || count > kMaxItems ||
                count > payload.remaining() / 12U) {
                return N4M_ERR_CORRUPT_BUFFER;
            }
            entry.resize(count);
            for (auto& item : entry) {
                if (!payload.string(item.first) || !payload.f64(item.second)) {
                    return N4M_ERR_CORRUPT_BUFFER;
                }
            }
        }

        std::uint32_t sampler_tag = 0;
        if (!payload.u32(sampler_tag) ||
            sampler_tag != static_cast<std::uint32_t>(options.sampler)) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        switch (options.sampler) {
            case N4M_SAMPLER_RANDOM:
            case N4M_SAMPLER_LHS:
            case N4M_SAMPLER_TERNARY:
            case N4M_SAMPLER_TPE:
                break;
            case N4M_SAMPLER_SOBOL: {
                auto* sampler = dynamic_cast<SobolSampler*>(optimizer.get());
                const std::int32_t expected_dims = sampler == nullptr ? -1 : sampler->sobol_dims_;
                std::uint8_t exhausted = 0;
                if (sampler == nullptr || !payload.i32(sampler->sobol_dims_) ||
                    !read_u32_vector(payload, sampler->x_) ||
                    !read_double_vector(payload, sampler->cached_point_) ||
                    !payload.i64(sampler->cached_ask_index_) ||
                    !payload.i64(sampler->next_index_) || !payload.u8(exhausted) ||
                    exhausted > 1U ||
                    sampler->sobol_dims_ != expected_dims || sampler->sobol_dims_ < 0 ||
                    sampler->x_.size() != static_cast<std::size_t>(sampler->sobol_dims_) ||
                    sampler->cached_point_.size() != sampler->x_.size() ||
                    !std::all_of(sampler->cached_point_.begin(),
                                 sampler->cached_point_.end(),
                                 [](double value) {
                                     return std::isfinite(value) && value >= 0.0 && value < 1.0;
                                 }) ||
                    sampler->next_index_ < 0 || sampler->next_index_ > optimizer->next_id_ ||
                    sampler->cached_ask_index_ < -1 ||
                    sampler->cached_ask_index_ > optimizer->next_id_ ||
                    sampler->cached_ask_index_ > sampler->next_index_) {
                    return N4M_ERR_CORRUPT_BUFFER;
                }
                constexpr std::uint32_t kSobolScale = std::uint32_t{1} << 30U;
                for (std::size_t i = 0; i < sampler->x_.size(); ++i) {
                    if (sampler->x_[i] >= kSobolScale ||
                        sampler->cached_point_[i] !=
                            static_cast<double>(sampler->x_[i]) /
                                static_cast<double>(kSobolScale)) {
                        return N4M_ERR_CORRUPT_BUFFER;
                    }
                }
                sampler->exhausted_ = exhausted != 0U;
                break;
            }
            case N4M_SAMPLER_GA: {
                auto* sampler = dynamic_cast<GaSampler*>(optimizer.get());
                if (sampler == nullptr || !read_matrix(payload, sampler->gen_) ||
                    !payload.i64(sampler->gen_base_id_) ||
                    !((sampler->gen_base_id_ == -1 && sampler->gen_.empty() &&
                       optimizer->next_id_ == 0) ||
                      (population_base_valid(sampler->gen_base_id_, sampler->pop_size_,
                                             optimizer->next_id_) &&
                       finite_unit_matrix(sampler->gen_,
                                          static_cast<std::size_t>(sampler->pop_size_),
                                          sampler->genome_length())))) {
                    return N4M_ERR_CORRUPT_BUFFER;
                }
                break;
            }
            case N4M_SAMPLER_PSO: {
                auto* sampler = dynamic_cast<PsoSampler*>(optimizer.get());
                std::uint8_t have_best = 0;
                const std::size_t count =
                    sampler == nullptr ? 0U
                                       : static_cast<std::size_t>(sampler->swarm_size_);
                const std::size_t width = space.params.size();
                if (sampler == nullptr || !read_matrix(payload, sampler->pos_) ||
                    !read_matrix(payload, sampler->vel_) ||
                    !read_matrix(payload, sampler->pbest_pos_) ||
                    !read_double_vector(payload, sampler->pbest_score_) ||
                    !read_double_vector(payload, sampler->gbest_pos_) ||
                    !payload.f64(sampler->gbest_score_) || !payload.u8(have_best) ||
                    !payload.i64(sampler->iter_base_id_) || have_best > 1U ||
                    !((sampler->iter_base_id_ == -1 && sampler->pos_.empty() &&
                       sampler->vel_.empty() && sampler->pbest_pos_.empty() &&
                       sampler->pbest_score_.empty() && sampler->gbest_pos_.empty() &&
                       have_best == 0U && optimizer->next_id_ == 0) ||
                      (population_base_valid(sampler->iter_base_id_, sampler->swarm_size_,
                                             optimizer->next_id_) &&
                       finite_unit_matrix(sampler->pos_, count, width) &&
                       sampler->vel_.size() == count && sampler->pbest_pos_.size() == count &&
                       finite_unit_matrix(sampler->pbest_pos_, count, width) &&
                       sampler->pbest_score_.size() == count &&
                       sampler->gbest_pos_.size() == width &&
                       finite_vector(sampler->gbest_pos_)))) {
                    return N4M_ERR_CORRUPT_BUFFER;
                }
                for (const auto& row : sampler->vel_) {
                    if (row.size() != width || !finite_vector(row)) return N4M_ERR_CORRUPT_BUFFER;
                }
                if (!pso_score_state_valid(
                        sampler->pos_, sampler->pbest_score_, sampler->pbest_pos_,
                        sampler->gbest_pos_, sampler->gbest_score_, have_best != 0U,
                        sampler->iter_base_id_ >= 0, sampler->iter_base_id_,
                        optimizer->direction())) {
                    return N4M_ERR_CORRUPT_BUFFER;
                }
                sampler->have_gbest_ = have_best != 0U;
                break;
            }
            case N4M_SAMPLER_CMAES: {
                auto* sampler = dynamic_cast<CmaEsSampler*>(optimizer.get());
                if (sampler == nullptr || !read_double_vector(payload, sampler->m_) ||
                    !payload.f64(sampler->sigma_) || !read_double_vector(payload, sampler->C_) ||
                    !read_double_vector(payload, sampler->ps_) ||
                    !read_double_vector(payload, sampler->pc_) ||
                    !payload.i64(sampler->gen_count_) || !read_matrix(payload, sampler->pop_x_) ||
                    !payload.i64(sampler->gen_base_id_) || sampler->m_.size() != sampler->P_ ||
                    sampler->C_.size() != sampler->P_ || sampler->ps_.size() != sampler->P_ ||
                    sampler->pc_.size() != sampler->P_ || !finite_vector(sampler->m_) ||
                    !std::isfinite(sampler->sigma_) || sampler->sigma_ <= 0.0 ||
                    !finite_vector(sampler->C_) || !finite_vector(sampler->ps_) ||
                    !finite_vector(sampler->pc_) || sampler->gen_count_ < 0 ||
                    !((sampler->gen_base_id_ == -1 && sampler->pop_x_.empty() &&
                       sampler->gen_count_ == 0 && optimizer->next_id_ == 0) ||
                      (population_base_valid(sampler->gen_base_id_, sampler->lambda_,
                                             optimizer->next_id_) &&
                       sampler->gen_count_ == sampler->gen_base_id_ / sampler->lambda_ &&
                       finite_unit_matrix(sampler->pop_x_,
                                          static_cast<std::size_t>(sampler->lambda_),
                                          sampler->P_)))) {
                    return N4M_ERR_CORRUPT_BUFFER;
                }
                for (const double value : sampler->C_) {
                    if (value <= 0.0) return N4M_ERR_CORRUPT_BUFFER;
                }
                break;
            }
            case N4M_SAMPLER_GP_EI: {
                auto* sampler = dynamic_cast<GpEiSampler*>(optimizer.get());
                std::uint32_t count = 0;
                if (sampler == nullptr || !payload.u32(count) || count > kMaxItems ||
                    count != n_trials ||
                    count > payload.remaining() / 12U) {
                    return N4M_ERR_CORRUPT_BUFFER;
                }
                sampler->proposals_.clear();
                for (std::uint32_t i = 0; i < count; ++i) {
                    std::int64_t id = 0;
                    std::vector<double> proposal;
                    if (!payload.i64(id) || !read_double_vector(payload, proposal) || id < 0 ||
                        static_cast<std::uint64_t>(id) >= n_trials ||
                        proposal.size() != sampler->P_ ||
                        !finite_vector(proposal) ||
                        !std::all_of(proposal.begin(), proposal.end(),
                                     [](double value) { return value >= 0.0 && value <= 1.0; }) ||
                        !sampler->proposals_.emplace(id, std::move(proposal)).second) {
                        return N4M_ERR_CORRUPT_BUFFER;
                    }
                }
                break;
            }
            default:
                return N4M_ERR_NOT_IMPLEMENTED;
        }
        if (!payload.consumed() || !validate_base_state(*optimizer)) {
            return N4M_ERR_CORRUPT_BUFFER;
        }
        for (auto& entry : queued) {
            if (optimizer->enqueue(std::move(entry)) != N4M_OK) {
                return N4M_ERR_CORRUPT_BUFFER;
            }
        }
        out = std::move(optimizer);
        return N4M_OK;
    }
};

n4m_status_t save_optimizer_checkpoint(const Optimizer& optimizer,
                                       std::vector<std::uint8_t>& out,
                                       std::string* error) {
    if (error != nullptr) error->clear();
    return OptimizerCheckpointAccess::encode(optimizer, out, error);
}

n4m_status_t load_optimizer_checkpoint(const std::uint8_t* data,
                                       std::size_t size,
                                       std::unique_ptr<Optimizer>& out,
                                       std::string* error) {
    if (error != nullptr) error->clear();
    return OptimizerCheckpointAccess::decode(data, size, out, error);
}

}  // namespace n4m::core::opt
