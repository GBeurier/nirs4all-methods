// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Private Native-EH qualification module only. Never link into n4m_wasm.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <malloc.h>
#include <new>

namespace {
// This fixture is single-threaded, like the private non-pthread Wasm target.
std::uint32_t allocation_calls = 0;
std::uint32_t injected_throws = 0;
std::uint32_t live_allocations = 0;
std::uint32_t allocations_to_skip = 0;
int armed_fault = 0;

class QualificationFailure final : public std::exception {
  public:
    const char* what() const noexcept override {
        return "n4m native EH qualification sentinel";
    }
};

void before_allocation() {
    ++allocation_calls;
    const int fault = armed_fault;
    if (fault != 0) {
        if (allocations_to_skip != 0) {
            --allocations_to_skip;
            return;
        }
        // Disarm BEFORE either exception is constructed/unwound.
        armed_fault = 0;
        ++injected_throws;
        if (fault == 1) {
            throw std::bad_alloc();
        }
        throw QualificationFailure();
    }
}

void* allocate(std::size_t size) {
    before_allocation();
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        ++live_allocations;
        return p;
    }
    throw std::bad_alloc();
}

void* allocate_aligned(std::size_t size, std::size_t alignment) {
    before_allocation();
    const std::size_t requested = size == 0 ? 1 : size;
    if (requested > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::bad_alloc();
    }
    const std::size_t rounded = (requested + alignment - 1) / alignment * alignment;
    if (void* p = std::aligned_alloc(alignment, rounded)) {
        ++live_allocations;
        return p;
    }
    throw std::bad_alloc();
}
void release(void* p) noexcept {
    if (p != nullptr) {
        --live_allocations;
        std::free(p);
    }
}
} // namespace

void* operator new(std::size_t size) { return allocate(size); }
void* operator new[](std::size_t size) { return allocate(size); }
void operator delete(void* p) noexcept { release(p); }
void operator delete[](void* p) noexcept { release(p); }
void operator delete(void* p, std::size_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t) noexcept { release(p); }
void* operator new(std::size_t size, std::align_val_t align) {
    return allocate_aligned(size, static_cast<std::size_t>(align));
}
void* operator new[](std::size_t size, std::align_val_t align) {
    return allocate_aligned(size, static_cast<std::size_t>(align));
}
void operator delete(void* p, std::align_val_t) noexcept { release(p); }
void operator delete[](void* p, std::align_val_t) noexcept { release(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { release(p); }

extern "C" {
int n4m_test_eh_arm(int fault, int skip) {
    if (armed_fault != 0 || skip < 0 || (fault != 1 && fault != 2)) {
        return 1;
    }
    allocations_to_skip = static_cast<std::uint32_t>(skip);
    armed_fault = fault;
    return 0;
}
std::uint32_t n4m_test_eh_allocations() { return allocation_calls; }
std::uint32_t n4m_test_eh_throws() { return injected_throws; }
int n4m_test_eh_armed() { return armed_fault; }
// One private snapshot export: caller allocates two uint32 slots BEFORE baselines.
// dlmalloc uordblks counts in-use chunks, including malloc-based C++ exceptions;
// it is not linear-memory capacity. No snapshot allocation or leak allowance.
void n4m_test_eh_memory_snapshot(std::uint32_t* out) {
    out[0] = live_allocations;
    out[1] = static_cast<std::uint32_t>(mallinfo().uordblks);
}
}
