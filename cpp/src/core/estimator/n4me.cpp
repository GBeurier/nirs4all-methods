// SPDX-License-Identifier: CECILL-2.1
//
// N4ME v1: portable fitted estimator state.
//
//   "N4ME" | u32 format | u32 writer ABI major, minor, patch
//   u32 len + method_id
//   u32 n_params, each: u32 len + name | u32 type | u64 count | count x (i64|f64)
//   u64 capabilities | u64 n_features_in | u64 n_outputs
//   u32 n_blocks, each: u32 tag | u64 len | bytes
//   u64 FNV-1a-64 of all preceding bytes
//
// All integers little-endian. Parameters are stored resolved (explicit or
// default), so a payload does not depend on the reader's defaults.

#include <cstring>
#include <limits>
#include <string>

#include "core/estimator/spec.hpp"

namespace n4m::estimator {

namespace {

constexpr unsigned char kMagic[4] = {'N', '4', 'M', 'E'};
constexpr std::uint64_t kMaxName = 256;
constexpr std::uint64_t kMaxBlocks = 64;

std::uint64_t fnv1a64(const unsigned char* data, std::size_t size) noexcept {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

class Writer {
  public:
    explicit Writer(std::vector<unsigned char>& out) : out_(out) {}
    void u32(std::uint32_t v) { le(v, 4); }
    void u64(std::uint64_t v) { le(v, 8); }
    void f64(double v) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, 8);
        le(bits, 8);
    }
    void bytes(const void* p, std::size_t n) {
        const auto* c = static_cast<const unsigned char*>(p);
        out_.insert(out_.end(), c, c + n);
    }
    void str(const char* s) {
        const std::size_t n = std::strlen(s);
        u32(static_cast<std::uint32_t>(n));
        bytes(s, n);
    }

  private:
    void le(std::uint64_t v, int n) {
        for (int i = 0; i < n; ++i) out_.push_back(static_cast<unsigned char>(v >> (8 * i)));
    }
    std::vector<unsigned char>& out_;
};

class Reader {
  public:
    Reader(const unsigned char* p, std::size_t n) : p_(p), n_(n) {}
    bool u32(std::uint32_t& v) {
        std::uint64_t w = 0;
        if (!le(w, 4)) return false;
        v = static_cast<std::uint32_t>(w);
        return true;
    }
    bool u64(std::uint64_t& v) { return le(v, 8); }
    bool f64(double& v) {
        std::uint64_t bits = 0;
        if (!le(bits, 8)) return false;
        std::memcpy(&v, &bits, 8);
        return true;
    }
    bool take(std::size_t n, const unsigned char*& out) {
        if (n > n_ - pos_) return false;
        out = p_ + pos_;
        pos_ += n;
        return true;
    }
    bool str(std::string& s, std::uint64_t max_len) {
        std::uint32_t n = 0;
        const unsigned char* data = nullptr;
        if (!u32(n) || n > max_len || !take(n, data)) return false;
        s.assign(reinterpret_cast<const char*>(data), n);
        return true;
    }
    std::size_t remaining() const noexcept { return n_ - pos_; }

  private:
    bool le(std::uint64_t& v, int n) {
        if (static_cast<std::size_t>(n) > n_ - pos_) return false;
        v = 0;
        for (int i = 0; i < n; ++i) v |= static_cast<std::uint64_t>(p_[pos_ + static_cast<std::size_t>(i)]) << (8 * i);
        pos_ += static_cast<std::size_t>(n);
        return true;
    }
    const unsigned char* p_;
    std::size_t n_;
    std::size_t pos_ = 0;
};

bool is_int_type(std::uint32_t t) noexcept {
    return t == N4M_METHOD_PARAM_INT || t == N4M_METHOD_PARAM_BOOL || t == N4M_METHOD_PARAM_ENUM ||
           t == N4M_METHOD_PARAM_INT_ARRAY;
}

n4m_status_t corrupt(n4m_context_t* ctx, const char* what) {
    set_error(ctx, what);
    return N4M_ERR_CORRUPT_BUFFER;
}

}  // namespace

n4m_status_t encode_state(n4m_context_t* ctx, const n4m_estimator_s& est,
                          std::vector<unsigned char>& out) {
    std::vector<StateBlock> blocks;
    n4m_status_t st = est.adapter->save_state(ctx, blocks);
    if (st != N4M_OK) return st;
    const MethodSpec& spec = est.params.spec();
    out.clear();
    Writer w(out);
    w.bytes(kMagic, 4);
    w.u32(N4M_ESTIMATOR_SERIALIZATION_FORMAT_VERSION);
    w.u32(N4M_ABI_VERSION_MAJOR);
    w.u32(N4M_ABI_VERSION_MINOR);
    w.u32(N4M_ABI_VERSION_PATCH);
    w.str(spec.method_id);
    w.u32(static_cast<std::uint32_t>(spec.n_params));
    std::vector<std::int64_t> ints;
    std::vector<double> doubles;
    for (std::int32_t i = 0; i < spec.n_params; ++i) {
        const ParamSpec& p = spec.params[i];
        if (!est.params.resolved(i, &ints, &doubles)) {
            set_error_named(ctx, "estimator has no value for required parameter", p.name);
            return N4M_ERR_INVALID_ARGUMENT;
        }
        w.str(p.name);
        w.u32(static_cast<std::uint32_t>(p.type));
        if (is_int_type(p.type)) {
            w.u64(ints.size());
            for (std::int64_t v : ints) w.u64(static_cast<std::uint64_t>(v));
        } else {
            w.u64(doubles.size());
            for (double v : doubles) w.f64(v);
        }
    }
    w.u64(est.adapter->capabilities());
    w.u64(static_cast<std::uint64_t>(est.adapter->n_features_in()));
    w.u64(static_cast<std::uint64_t>(est.adapter->n_outputs()));
    w.u32(static_cast<std::uint32_t>(blocks.size()));
    for (const StateBlock& b : blocks) {
        w.u32(b.tag);
        w.u64(b.bytes.size());
        w.bytes(b.bytes.data(), b.bytes.size());
    }
    w.u64(fnv1a64(out.data(), out.size()));
    return N4M_OK;
}

n4m_status_t decode_state(n4m_context_t* ctx, const unsigned char* bytes, std::size_t size,
                          std::uint64_t max_bytes, std::unique_ptr<n4m_estimator_s>& out) {
    if (size > max_bytes) {
        set_error(ctx, "N4ME payload exceeds the context state-size limit");
        return N4M_ERR_INVALID_ARGUMENT;
    }
    if (size < 4 + 16 + 8 || std::memcmp(bytes, kMagic, 4) != 0) {
        return corrupt(ctx, "not an N4ME payload");
    }
    Reader checksum(bytes + size - 8, 8);
    std::uint64_t stored = 0;
    if (!checksum.u64(stored) || stored != fnv1a64(bytes, size - 8)) {
        return corrupt(ctx, "N4ME checksum mismatch");
    }
    Reader r(bytes + 4, size - 12);
    std::uint32_t format = 0, major = 0, minor = 0, patch = 0;
    if (!r.u32(format) || !r.u32(major) || !r.u32(minor) || !r.u32(patch)) {
        return corrupt(ctx, "truncated N4ME header");
    }
    if (format != N4M_ESTIMATOR_SERIALIZATION_FORMAT_VERSION || major != N4M_ABI_VERSION_MAJOR) {
        set_error(ctx, "unsupported N4ME format or writer ABI major version");
        return N4M_ERR_VERSION_INCOMPATIBLE;
    }
    std::string method_id;
    if (!r.str(method_id, kMaxName)) return corrupt(ctx, "truncated N4ME method id");
    const std::int32_t index = method_index(method_id.c_str());
    if (index < 0) {
        set_error_named(ctx, "N4ME payload names an unknown method", method_id.c_str());
        return N4M_ERR_UNSUPPORTED;
    }
    const MethodSpec& spec = *method_at(index);
    Params params(spec);
    std::uint32_t n_params = 0;
    if (!r.u32(n_params) || n_params != static_cast<std::uint32_t>(spec.n_params)) {
        return corrupt(ctx, "N4ME parameter count does not match the method");
    }
    for (std::uint32_t k = 0; k < n_params; ++k) {
        std::string name;
        std::uint32_t type = 0;
        std::uint64_t count = 0;
        if (!r.str(name, kMaxName) || !r.u32(type) || !r.u64(count) ||
            count > r.remaining() / 8) {
            return corrupt(ctx, "truncated N4ME parameter");
        }
        const std::int32_t pi = param_index(spec, name.c_str());
        if (pi < 0 || static_cast<std::uint32_t>(spec.params[pi].type) != type ||
            params.raw(pi).set) {
            set_error_named(ctx, "N4ME parameter does not match the method", name.c_str());
            return N4M_ERR_CORRUPT_BUFFER;
        }
        n4m_status_t st = N4M_OK;
        if (is_int_type(type)) {
            std::vector<std::int64_t> v(static_cast<std::size_t>(count));
            for (auto& x : v) {
                std::uint64_t u = 0;
                r.u64(u);
                x = static_cast<std::int64_t>(u);
            }
            st = params.set_ints(name.c_str(), static_cast<n4m_method_param_type_t>(type), v.data(),
                                 static_cast<std::int64_t>(count));
        } else {
            std::vector<double> v(static_cast<std::size_t>(count));
            for (auto& x : v) r.f64(x);
            st = params.set_doubles(name.c_str(), static_cast<n4m_method_param_type_t>(type), v.data(),
                                    static_cast<std::int64_t>(count));
        }
        if (st != N4M_OK) {
            set_error_named(ctx, "N4ME parameter value is invalid", name.c_str());
            return N4M_ERR_CORRUPT_BUFFER;
        }
    }
    std::uint64_t caps = 0, n_features = 0, n_outputs = 0;
    std::uint32_t n_blocks = 0;
    if (!r.u64(caps) || !r.u64(n_features) || !r.u64(n_outputs) || !r.u32(n_blocks) ||
        n_blocks > kMaxBlocks) {
        return corrupt(ctx, "truncated N4ME state header");
    }
    std::vector<StateBlock> blocks(n_blocks);
    for (StateBlock& b : blocks) {
        std::uint64_t len = 0;
        const unsigned char* data = nullptr;
        if (!r.u32(b.tag) || !r.u64(len) || len > r.remaining() ||
            !r.take(static_cast<std::size_t>(len), data)) {
            return corrupt(ctx, "truncated N4ME state block");
        }
        b.bytes.assign(data, data + len);
    }
    if (r.remaining() != 0) return corrupt(ctx, "trailing bytes in N4ME payload");

    auto est = std::make_unique<n4m_estimator_s>(index, params, spec.factory(spec));
    n4m_status_t st = est->adapter->load_state(ctx, est->params, blocks);
    if (st != N4M_OK) return st;
    if (est->adapter->capabilities() != caps ||
        static_cast<std::uint64_t>(est->adapter->n_features_in()) != n_features ||
        static_cast<std::uint64_t>(est->adapter->n_outputs()) != n_outputs) {
        return corrupt(ctx, "N4ME header does not match the decoded state");
    }
    est->fitted = true;
    out = std::move(est);
    return N4M_OK;
}

}  // namespace n4m::estimator
