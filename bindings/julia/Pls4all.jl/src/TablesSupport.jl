# SPDX-License-Identifier: CECILL-2.1

import Tables

export table_matrix, response_matrix

function _selected_column_names(table; columns=nothing)
    cols = Tables.columntable(table)
    available = collect(Symbol.(propertynames(cols)))
    selected = columns === nothing ? available : collect(Symbol.(columns))
    missing = setdiff(selected, available)
    isempty(missing) || throw(ArgumentError(
        "table is missing requested column(s): $(join(string.(missing), ", "))"))
    return cols, selected
end

function _column_length(col, name::Symbol)
    try
        return length(col)
    catch err
        throw(ArgumentError("column $name does not expose length(): $err"))
    end
end

function _table_matrix_with_names(table; columns=nothing)
    Tables.istable(table) || throw(ArgumentError(
        "expected a Tables.jl-compatible table; got $(typeof(table))"))
    cols, names = _selected_column_names(table; columns=columns)
    isempty(names) && throw(ArgumentError("table has no selected columns"))

    nrows = _column_length(getproperty(cols, names[1]), names[1])
    out = Matrix{Float64}(undef, nrows, length(names))
    for (j, name) in enumerate(names)
        col = getproperty(cols, name)
        _column_length(col, name) == nrows || throw(ArgumentError(
            "column $name has a different row count"))
        i = 1
        for value in col
            ismissing(value) && throw(ArgumentError(
                "missing values are not supported in column $name"))
            try
                out[i, j] = Float64(value)
            catch err
                throw(ArgumentError(
                    "column $name contains a non-numeric value at row $i: $err"))
            end
            i += 1
        end
        i == nrows + 1 || throw(ArgumentError(
            "column $name iterated a different row count"))
    end
    return out, names
end

function _feature_matrix(X; columns=nothing)
    if X isa AbstractMatrix
        columns === nothing || throw(ArgumentError(
            "`columns` can only be used with Tables.jl-compatible inputs"))
        return Matrix{Float64}(X), nothing
    end
    return _table_matrix_with_names(X; columns=columns)
end

function _target_matrix(y)
    if y isa AbstractVector
        return reshape(Float64.(collect(y)), :, 1), nothing, true
    elseif y isa AbstractMatrix
        return Matrix{Float64}(y), nothing, false
    elseif Tables.istable(y)
        Y, names = _table_matrix_with_names(y)
        return Y, names, false
    else
        throw(ArgumentError(
            "expected target as vector, matrix, or Tables.jl-compatible table; got $(typeof(y))"))
    end
end

"""
    table_matrix(table; columns=nothing) -> Matrix{Float64}

Convert a Tables.jl-compatible column table, such as a named tuple of
vectors, `DataFrame`, `CSV.File`, or `Arrow.Table`, to the row-major
observation matrix expected by pls4all. If `columns` is provided, columns
are selected and ordered by that list.
"""
table_matrix(table; columns=nothing) =
    first(_table_matrix_with_names(table; columns=columns))

"""
    response_matrix(y) -> Matrix{Float64}

Convert a target vector, target matrix, or Tables.jl-compatible target table
to a numeric `(n, q)` response matrix.
"""
response_matrix(y) = first(_target_matrix(y))
