# SPDX-License-Identifier: CECILL-2.1
#
# Julia binding for the pls4all C ABI via ccall.
#
# Set the PLS4ALL_LIB environment variable to the libn4m.so path
# (or pls4all.dll / libn4m.dylib). Defaults to `libn4m` (resolved
# via the standard library search path).

module Pls4all

using Libdl

const _LIBP4A = Ref{String}("libn4m")

function __init__()
    _LIBP4A[] = get(ENV, "PLS4ALL_LIB", "libn4m")
    # Touch the library once so the dlopen happens at module load.
    Libdl.dlopen(_LIBP4A[], Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    return nothing
end

# Convenience macro: ccall a symbol on the cached libn4m path. Julia's
# ccall accepts the path string in the (symbol, libname) tuple and
# caches the resolved function pointer per (symbol, lib) pair.
macro p4a(name, rettype, argtypes, args...)
    quote
        ccall(($(esc(name)), Pls4all._LIBP4A[]),
              $(esc(rettype)),
              $(esc(argtypes)),
              $(map(esc, args)...))
    end
end

function version()
    return unsafe_string(@p4a(:n4m_get_version_string, Cstring, ()))
end

function abi_version()
    return (@p4a(:n4m_get_abi_version_major, UInt32, ()),
            @p4a(:n4m_get_abi_version_minor, UInt32, ()),
            @p4a(:n4m_get_abi_version_patch, UInt32, ()))
end

# Fit a SIMPLS PLS regression on (X, Y).
#
# Uses the raw-pointer `n4m_pls_fit_simple` C ABI entry (ABI 1.13+) to
# avoid the `Ref{matrix_view_t}` interop ambiguity that bites Julia's
# ccall layout when the struct contains a pointer field followed by
# int64s.
#
#   X — n × p Float64 matrix (Julia column-major; we materialise a
#        row-major copy for the C ABI).
#   Y — n × q Float64 matrix.
function pls_fit(X::AbstractMatrix{Float64},
                  Y::AbstractMatrix{Float64};
                  n_components::Integer = 3)
    n, p = size(X)
    nY, q = size(Y)
    n == nY || throw(ArgumentError(
        "X rows ($n) must equal Y rows ($nY)"))

    # transpose(X) on a Julia (n, p) matrix gives a (p, n) view whose
    # column-major linear storage equals row-major (n, p).
    X_rm = collect(transpose(X))
    Y_rm = collect(transpose(Y))

    coefs = Array{Float64}(undef, q, p)        # row-major (p, q) → JL (q, p)
    x_mean = Array{Float64}(undef, p)
    y_mean = Array{Float64}(undef, q)
    preds_rm = Array{Float64}(undef, q, n)     # row-major (n, q) → JL (q, n)

    GC.@preserve X_rm Y_rm coefs x_mean y_mean preds_rm begin
        s = @p4a(:n4m_pls_fit_simple, Int32,
                 (Ptr{Float64}, Ptr{Float64},
                  Int32, Int32, Int32, Int32,
                  Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
                  Ptr{Float64}),
                 pointer(X_rm), pointer(Y_rm),
                 Int32(n), Int32(p), Int32(q), Int32(n_components),
                 pointer(coefs), pointer(x_mean), pointer(y_mean),
                 pointer(preds_rm))
    end
    s == 0 || error("n4m_pls_fit_simple failed: $s")

    # `coefs` was filled row-major (p × q) into a column-major (q, p)
    # buffer; transpose materialises a column-major (p, q) array.
    coefficients = Array(transpose(coefs))
    predictions = Array(transpose(preds_rm))

    return (; coefficients=coefficients,
              x_mean=copy(x_mean),
              y_mean=copy(y_mean),
              predictions=predictions,
              n_components=Int(n_components))
end

include("Sklearn.jl")

end # module
