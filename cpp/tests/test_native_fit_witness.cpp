// SPDX-License-Identifier: CECILL-2.1
// Test-only CDLL probes linked to the real public FIT functions. No kernels.
#include "n4m/n4m.h"
#include "n4m/estimator.h"
#include "n4m/multimodal.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#if defined(_WIN32)
#  define N4M_TEST_PROBE_API __declspec(dllexport)
#else
#  define N4M_TEST_PROBE_API __attribute__((visibility("default")))
#endif

namespace {
constexpr std::size_t MAX_RECORDS = 65536;
struct Record {
    const char* kind = "";
    std::uintptr_t context = 0;
    std::uintptr_t handle = 0;
    std::uint64_t native_thread = 0;
    std::uint64_t entered_ns = 0;
    std::uint64_t exited_ns = 0;
};
std::array<Record, MAX_RECORDS> records{};
std::mutex state_mutex;
std::mutex serialized_fit_mutex;
std::atomic<bool> serialized{false};
std::atomic<std::uint64_t> inflight{0};
std::uint64_t active = 0, peak = 0, total = 0;
std::size_t capacity = MAX_RECORDS;
bool overflow = false;

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

class FitScope {
  public:
    FitScope(const char* kind, const void* context, const void* handle)
        : serial_lock_(serialized_fit_mutex, std::defer_lock) {
        inflight.fetch_add(1);
        try {
            // Negative control serializes both measurement and the real FIT.
            if (serialized.load()) serial_lock_.lock();
            std::lock_guard<std::mutex> lock(state_mutex);
            slot_ = total++;
            ++active;
            if (active > peak) peak = active;
            if (slot_ >= capacity) {
                overflow = true;
            } else {
                records[slot_] = {kind, reinterpret_cast<std::uintptr_t>(context),
                    reinterpret_cast<std::uintptr_t>(handle),
                    static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())),
                    now_ns(), 0};
            }
        } catch (...) {
            inflight.fetch_sub(1);
            throw;
        }
    }
    ~FitScope() {
        const auto ended = now_ns();
        std::lock_guard<std::mutex> lock(state_mutex);
        if (slot_ < capacity) records[slot_].exited_ns = ended;
        --active;
        inflight.fetch_sub(1);
    }
    FitScope(const FitScope&) = delete;
    FitScope& operator=(const FitScope&) = delete;

  private:
    std::unique_lock<std::mutex> serial_lock_;
    std::uint64_t slot_ = 0;
};
}  // namespace

extern "C" {
N4M_TEST_PROBE_API int n4m_test_native_fit_reset(int serialize, std::uint64_t limit) {
    try {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (active != 0 || inflight.load() != 0 || limit == 0 || limit > MAX_RECORDS ||
            (serialize != 0 && serialize != 1)) return 1;
        capacity = static_cast<std::size_t>(limit);
        active = peak = total = 0;
        overflow = false;
        serialized.store(serialize != 0);
        return 0;
    } catch (...) { return 2; }
}

N4M_TEST_PROBE_API const char* n4m_test_native_fit_snapshot() {
    try {
        std::lock_guard<std::mutex> lock(state_mutex);
        // Caller must join every worker before reading/resetting this store.
        if (active != 0 || inflight.load() != 0) return nullptr;
        std::ostringstream out;
        out << "{\"active\":" << active << ",\"inflight\":" << inflight.load()
            << ",\"peak_active\":" << peak << ",\"total_calls\":" << total
            << ",\"overflow\":" << (overflow ? "true" : "false") << ",\"records\":[";
        const auto count = total < capacity ? total : capacity;
        for (std::uint64_t i = 0; i < count; ++i) {
            const auto& record = records[i];
            if (i) out << ',';
            out << "{\"kind\":\"" << record.kind << "\",\"context\":" << record.context
                << ",\"handle\":" << record.handle << ",\"native_thread\":" << record.native_thread
                << ",\"entered_ns\":" << record.entered_ns << ",\"exited_ns\":" << record.exited_ns << '}';
        }
        out << "]}";
        static thread_local std::string snapshot;
        snapshot = out.str();
        return snapshot.c_str();
    } catch (...) { return nullptr; }
}

N4M_TEST_PROBE_API std::uint64_t n4m_test_native_fit_real_address(int kind) {
    if (kind == 1) return reinterpret_cast<std::uintptr_t>(&n4m_role_pipeline_fit);
    if (kind == 2) return reinterpret_cast<std::uintptr_t>(&n4m_multimodal_pipeline_fit);
    return 0;
}

N4M_TEST_PROBE_API n4m_status_t n4m_test_probe_role_fit(
    n4m_context_t* context, n4m_role_pipeline_t* handle, const n4m_fit_inputs_v1_t* inputs) {
    try {
        FitScope scope("role", context, handle);
        return n4m_role_pipeline_fit(context, handle, inputs);
    } catch (...) { return N4M_ERR_INTERNAL; }
}
N4M_TEST_PROBE_API n4m_status_t n4m_test_probe_multimodal_fit(
    n4m_context_t* context, n4m_multimodal_pipeline_t* handle, int32_t n_sources,
    const n4m_multimodal_source_view_v1_t* sources, const n4m_matrix_view_t* target) {
    try {
        FitScope scope("multimodal", context, handle);
        return n4m_multimodal_pipeline_fit(context, handle, n_sources, sources, target);
    } catch (...) { return N4M_ERR_INTERNAL; }
}
}  // extern "C"
