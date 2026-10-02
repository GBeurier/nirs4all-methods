// SPDX-License-Identifier: CECILL-2.1
// Thin MATLAB/Octave raw tensor translator for ABI 2.16.
#include "mex.h"
#include "n4m/multimodal.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(n4m_status_t status, n4m_context_t* ctx) {
    if (status != N4M_OK) throw std::runtime_error(std::string(n4m_status_to_string(status)) +
        ": " + (ctx ? n4m_context_last_error(ctx) : ""));
}
const mxArray* field(const mxArray* value, const char* name) {
    require(value && mxIsStruct(value) && mxGetNumberOfElements(value) == 1, "expected scalar struct");
    const mxArray* result = mxGetField(value, 0, name); require(result, "missing recipe/schema field"); return result;
}
void keys(const mxArray* value, const std::vector<std::string>& names) {
    require(value && mxIsStruct(value) && mxGetNumberOfElements(value) == 1 &&
        mxGetNumberOfFields(value) == static_cast<int>(names.size()), "invalid recipe/schema fields");
    for (int i = 0; i < mxGetNumberOfFields(value); ++i) {
        bool found = false; for (const auto& name : names) if (name == mxGetFieldNameByNumber(value, i)) found = true;
        require(found, "unknown recipe/schema field");
    }
}
std::string utf8_text(const mxArray* value, bool allow_nul) {
    require(value && mxIsChar(value) && mxGetM(value) <= 1, "expected UTF-8 character row string");
    if (!allow_nul) {
        const mxChar* data = mxGetChars(value);
        for (mwSize i = 0; i < mxGetNumberOfElements(value); ++i)
            require(data[i] != 0, "strings must not contain NUL");
    }
    // The host owns Unicode conversion. Copy the returned byte count rather
    // than using a C-string terminator: category cells may contain U+0000.
    using ArrayOwner = std::unique_ptr<mxArray, decltype(&mxDestroyArray)>;
    ArrayOwner encoding(mxCreateString("UTF-8"), mxDestroyArray);
    mxArray* args[] = {const_cast<mxArray*>(value), encoding.get()};
    mxArray* converted = nullptr;
    ArrayOwner failure(mexCallMATLABWithTrap(1, &converted, 2, args, "unicode2native"), mxDestroyArray);
    ArrayOwner bytes(converted, mxDestroyArray);
    require(!failure && bytes && mxIsUint8(bytes.get()), "UTF-8 string conversion failed");
    const auto count = mxGetNumberOfElements(bytes.get());
    return count == 0 ? std::string{} :
        std::string(static_cast<const char*>(mxGetData(bytes.get())), count);
}
std::string text(const mxArray* value) { return utf8_text(value, false); }
double number(const mxArray* value) {
    if (value && mxIsCell(value) && mxGetNumberOfElements(value) == 1) value = mxGetCell(value, 0);
    require(value && mxIsNumeric(value) && !mxIsComplex(value) && !mxIsSparse(value) &&
        mxGetNumberOfElements(value) == 1, "expected scalar number");
    const double result = mxGetScalar(value); require(std::isfinite(result), "number must be finite"); return result;
}
int64_t integer(const mxArray* value) {
    const double result = number(value);
    require(std::floor(result) == result && std::abs(result) <= 9007199254740991.0,
        "integer must be exactly representable"); return static_cast<int64_t>(result);
}
int boolean(const mxArray* value) {
    require(value && mxIsLogical(value) && mxGetNumberOfElements(value) == 1, "expected boolean");
    return mxIsLogicalScalarTrue(value) ? 1 : 0;
}
std::vector<int64_t> shape_vector(const mxArray* value) {
    require(value && (mxIsDouble(value) || mxIsCell(value)) && !mxIsComplex(value) && !mxIsSparse(value) &&
        mxGetNumberOfElements(value) <= 7, "shape needs at most seven double integer dimensions");
    std::vector<int64_t> result;
    for (mwSize i = 0; i < mxGetNumberOfElements(value); ++i) {
        const double d = mxIsCell(value) ? number(mxGetCell(value, i)) : mxGetPr(value)[i];
        require(std::isfinite(d) && d >= 1 && d <= 1048576 && std::floor(d) == d, "invalid shape dimension");
        result.push_back(static_cast<int64_t>(d));
    }
    return result;
}
const char* order[] = {"nir", "image", "series", "metadata"};
struct Schema {
    std::string name, representation, dtype, identity;
    std::vector<int64_t> shape;
};
Schema read_schema(const mxArray* schemas, int i) {
    const mxArray* s = field(schemas, order[i]); keys(s, {"representation_id", "input_shape", "dtype", "identity"});
    return {order[i], text(field(s, "representation_id")), text(field(s, "dtype")),
        text(field(s, "identity")), shape_vector(field(s, "input_shape"))};
}
struct Config {
    std::vector<Schema> schemas;
    std::vector<n4m_multimodal_source_spec_v1_t> sources;
    n4m_multimodal_recipe_v1_t recipe{};
    Config(const mxArray* raw, const mxArray* source_schemas) {
        keys(raw, {"schema_version", "fusion", "source_order", "encoders", "source_weights", "model"});
        keys(source_schemas, {"nir", "image", "series", "metadata"});
        require(integer(field(raw, "schema_version")) == 1 && text(field(raw, "fusion")) == "early", "expected recipe v1 early fusion");
        const mxArray* names = field(raw, "source_order");
        require(mxIsCell(names) && mxGetNumberOfElements(names) == 4, "expected ordered four sources");
        for (mwIndex i = 0; i < 4; ++i) require(text(mxGetCell(names, i)) == order[i], "source order differs");
        const mxArray* encoders = field(raw, "encoders"), *weights = field(raw, "source_weights");
        keys(encoders, {"nir", "image", "series", "metadata"}); keys(weights, {"nir", "image", "series", "metadata"});
        schemas.reserve(4); sources.resize(4);
        for (int i = 0; i < 4; ++i) schemas.push_back(read_schema(source_schemas, i));
        for (int i = 0; i < 4; ++i) {
            const auto& s = schemas[i]; auto& spec = sources[i]; const mxArray* encoder = field(encoders, order[i]);
            spec.struct_size = sizeof(spec); spec.name = s.name.c_str(); spec.representation_id = s.representation.c_str();
            spec.dtype = s.dtype.c_str(); spec.identity_utf8 = s.identity.data(); spec.identity_bytes = s.identity.size();
            spec.ndim = static_cast<int32_t>(s.shape.size()); spec.shape = s.shape.data();
            spec.weight = number(field(weights, order[i])); spec.numeric_column = spec.categorical_column = -1;
            const std::string kind = text(field(encoder, "kind"));
            if (kind == "standard_scaler") {
                keys(encoder, {"kind", "with_mean", "with_std"}); spec.encoder = N4M_MULTIMODAL_STANDARD_SCALER;
            } else if (kind == "tensor_pca") {
                keys(encoder, {"kind", "n_components", "random_state", "whiten"}); spec.encoder = N4M_MULTIMODAL_TENSOR_PCA;
                spec.n_components = integer(field(encoder, "n_components")); spec.random_state = integer(field(encoder, "random_state"));
                spec.whiten = boolean(field(encoder, "whiten"));
            } else if (kind == "column_transformer") {
                keys(encoder, {"kind", "numeric_columns", "categorical_columns", "with_mean", "with_std", "handle_unknown", "sparse_output", "drop"});
                spec.encoder = N4M_MULTIMODAL_COLUMN_TRANSFORMER;
                spec.numeric_column = integer(field(encoder, "numeric_columns")); spec.categorical_column = integer(field(encoder, "categorical_columns"));
                spec.ignore_unknown = text(field(encoder, "handle_unknown")) == "ignore";
                require(!boolean(field(encoder, "sparse_output")) && mxIsEmpty(field(encoder, "drop")), "unsupported mixed encoding options");
            } else throw std::runtime_error("unknown encoder");
            if (spec.encoder != N4M_MULTIMODAL_TENSOR_PCA) {
                spec.with_mean = boolean(field(encoder, "with_mean")); spec.with_std = boolean(field(encoder, "with_std"));
            }
        }
        const mxArray* model = field(raw, "model"); keys(model, {"method_id", "params"});
        require(text(field(model, "method_id")) == "models.regularized.ridge", "expected native Ridge");
        const mxArray* params = field(model, "params"); keys(params, {"alpha", "center_x", "center_y", "scale_x"});
        recipe.struct_size = sizeof(recipe); recipe.n_sources = 4; recipe.sources = sources.data();
        recipe.alpha = number(field(params, "alpha")); recipe.center_x = boolean(field(params, "center_x"));
        recipe.center_y = boolean(field(params, "center_y")); recipe.scale_x = boolean(field(params, "scale_x"));
    }
};
struct RawBlock {
    Schema schema;
    std::vector<int64_t> shape, strides;
    std::vector<double> numeric;
    std::vector<uint64_t> offsets;
    std::string categories;
    n4m_multimodal_source_view_v1_t view{};
};
struct Inputs {
    std::vector<RawBlock> blocks;
    std::vector<n4m_multimodal_source_view_v1_t> views;
    int64_t rows = -1;
    Inputs(const mxArray* raw, const mxArray* schemas) {
        keys(raw, {"nir", "image", "series", "metadata"}); keys(schemas, {"nir", "image", "series", "metadata"});
        blocks.resize(4); views.resize(4);
        for (int i = 0; i < 4; ++i) {
            auto& b = blocks[i]; b.schema = read_schema(schemas, i); const mxArray* data = field(raw, order[i]);
            const mwSize rank = mxGetNumberOfDimensions(data); const mwSize* dims = mxGetDimensions(data);
            require(rank <= b.schema.shape.size() + 1, "raw tensor rank differs from schema");
            b.shape.push_back(static_cast<int64_t>(dims[0]));
            for (size_t j = 0; j < b.schema.shape.size(); ++j) b.shape.push_back(j + 1 < rank ? static_cast<int64_t>(dims[j + 1]) : 1);
            for (size_t j = 0; j < b.schema.shape.size(); ++j)
                require(b.shape[j + 1] == b.schema.shape[j], "raw source shape differs from schema");
            b.strides.resize(b.shape.size()); b.strides[0] = 1;
            for (size_t j = 1; j < b.shape.size(); ++j) b.strides[j] = b.strides[j - 1] * b.shape[j - 1];
            require(rows == -1 || rows == b.shape[0], "source row counts differ"); rows = b.shape[0];
            auto& v = b.view; v.struct_size = sizeof(v); v.name = b.schema.name.c_str();
            v.representation_id = b.schema.representation.c_str(); v.dtype = b.schema.dtype.c_str();
            v.identity_utf8 = b.schema.identity.data(); v.identity_bytes = b.schema.identity.size();
            v.rank = static_cast<int32_t>(b.shape.size()); v.shape = b.shape.data(); v.strides = b.strides.data();
            if (i != 3) {
                require((mxIsDouble(data) || mxIsSingle(data)) && !mxIsComplex(data) && !mxIsSparse(data), "raw numeric tensors require real float32/64");
                v.numeric_data = mxGetData(data); v.numeric_dtype = mxIsDouble(data) ? N4M_DTYPE_F64 : N4M_DTYPE_F32;
            } else {
                require(mxIsCell(data) && rank == 2 && dims[1] == 2, "metadata must be an n-by-two raw cell matrix");
                b.numeric.resize(rows); b.offsets.push_back(0);
                for (int64_t j = 0; j < rows; ++j) {
                    const mxArray* value = mxGetCell(data, j);
                    require(value != nullptr, "metadata cells must not be missing");
                    if (mxIsChar(value)) {
                        const std::string raw_number = text(value); char* end = nullptr;
                        b.numeric[j] = std::strtod(raw_number.c_str(), &end);
                        require(end != raw_number.c_str(), "invalid numeric metadata string");
                        while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') ++end;
                        require(*end == 0 && std::isfinite(b.numeric[j]), "invalid numeric metadata string");
                    } else b.numeric[j] = number(value);
                    const std::string category = utf8_text(mxGetCell(data, rows + j), true);
                    require(category.size() <= 1048576 && b.categories.size() + category.size() <= 67108864, "UTF-8 cells exceed bounds");
                    b.categories += category; b.offsets.push_back(b.categories.size());
                }
                b.strides[0] = 1; v.numeric_data = b.numeric.data(); v.numeric_dtype = N4M_DTYPE_F64;
                v.categorical_utf8 = b.categories.data(); v.utf8_bytes = b.categories.size(); v.categorical_offsets = b.offsets.data();
            }
            views[i] = v;
        }
    }
};
struct Entry {
    n4m_context_t* context = nullptr;
    n4m_multimodal_pipeline_t* pipeline = nullptr;
    ~Entry() { n4m_multimodal_pipeline_destroy(pipeline); n4m_context_destroy(context); }
};
std::map<uint64_t, std::unique_ptr<Entry>> entries;
uint64_t next_id = 1;
bool registered = false;
void cleanup() { entries.clear(); }
uint64_t id(const mxArray* value) {
    require(value && mxIsUint64(value) && mxGetNumberOfElements(value) == 1, "expected native uint64 handle");
    return *static_cast<const uint64_t*>(mxGetData(value));
}
Entry& entry(const mxArray* value) {
    const auto it = entries.find(id(value)); require(it != entries.end(), "native multimodal pipeline is closed"); return *it->second;
}
mxArray* scalar_id(uint64_t value) {
    mxArray* result = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t*>(mxGetData(result)) = value; return result;
}
void dispatch(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs >= 1, "command required"); const std::string command = text(prhs[0]);
    if (command == "create" || command == "from_state") {
        require(nlhs == 1 && nrhs == (command == "create" ? 3 : 4), "create/from_state argument count");
        check(n4m_check_abi_compatibility(2, 16), nullptr); Config config(prhs[1], prhs[2]);
        auto value = std::make_unique<Entry>(); check(n4m_context_create(&value->context), value->context);
        if (command == "create") check(n4m_multimodal_pipeline_create(value->context, &config.recipe, &value->pipeline), value->context);
        else {
            require(mxIsUint8(prhs[3]) && !mxIsSparse(prhs[3]), "state must be uint8 bytes");
            require(mxGetNumberOfElements(prhs[3]) <= 67108864, "N4MF bytes exceed native limit");
            check(n4m_multimodal_pipeline_import_from_buffer(value->context, &config.recipe, mxGetData(prhs[3]), mxGetNumberOfElements(prhs[3]), &value->pipeline), value->context);
        }
        if (!registered) { mexAtExit(cleanup); registered = true; }
        const uint64_t token = next_id++; entries.emplace(token, std::move(value)); plhs[0] = scalar_id(token); mexLock(); return;
    }
    require(nrhs >= 2, "handle required");
    if (command == "close") {
        require(nrhs == 2 && nlhs == 0, "close takes only a handle");
        if (entries.erase(id(prhs[1]))) mexUnlock(); return;
    }
    auto& value = entry(prhs[1]);
    if (command == "fit") {
        require(nrhs == 5 && nlhs == 0, "fit(handle, blocks, schemas, y)"); Inputs inputs(prhs[2], prhs[3]);
        require(mxIsDouble(prhs[4]) && !mxIsSparse(prhs[4]) && !mxIsComplex(prhs[4]) &&
            mxGetNumberOfElements(prhs[4]) == static_cast<mwSize>(inputs.rows), "expected one double target per row");
        n4m_matrix_view_t target; check(n4m_matrix_view_init_rowmajor(&target, mxGetData(prhs[4]), inputs.rows, 1, N4M_DTYPE_F64), value.context);
        check(n4m_multimodal_pipeline_fit(value.context, value.pipeline, 4, inputs.views.data(), &target), value.context);
    } else if (command == "predict" || command == "transform") {
        require(nrhs == 4 && nlhs == 1, "predict/transform(handle, blocks, schemas)"); Inputs inputs(prhs[2], prhs[3]); int64_t width = 1;
        if (command == "transform") check(n4m_multimodal_pipeline_transform_cols(value.pipeline, &width), value.context);
        require(inputs.rows >= 0 && width > 0 && inputs.rows <= 16777216 / width, "output exceeds native bounds");
        mxArray* output = mxCreateDoubleMatrix(inputs.rows, width, mxREAL); n4m_matrix_view_t view;
        check(n4m_matrix_view_init_colmajor(&view, mxGetPr(output), inputs.rows, width, N4M_DTYPE_F64), value.context);
        const auto status = command == "transform" ? n4m_multimodal_pipeline_transform(value.context, value.pipeline, 4, inputs.views.data(), &view) :
            n4m_multimodal_pipeline_predict(value.context, value.pipeline, 4, inputs.views.data(), &view);
        if (status != N4M_OK) { mxDestroyArray(output); check(status, value.context); } plhs[0] = output;
    } else if (command == "export_state") {
        require(nrhs == 2 && nlhs == 1, "export_state takes a handle"); size_t size = 0;
        check(n4m_multimodal_pipeline_export_size(value.context, value.pipeline, &size), value.context);
        mxArray* output = mxCreateNumericMatrix(1, size, mxUINT8_CLASS, mxREAL);
        const auto status = n4m_multimodal_pipeline_export_to_buffer(value.context, value.pipeline, mxGetData(output), size, &size);
        if (status != N4M_OK) { mxDestroyArray(output); check(status, value.context); } plhs[0] = output;
    } else throw std::runtime_error("unknown multimodal command");
}
}
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try { dispatch(nlhs, plhs, nrhs, prhs); }
    catch (const std::exception& error) { mexErrMsgIdAndTxt("n4m:multimodal", "%s", error.what()); }
}
