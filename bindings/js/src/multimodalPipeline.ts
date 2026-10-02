// SPDX-License-Identifier: CECILL-2.1
// Raw tensor marshalling only. Learned encoders/fusion/Ridge live in libn4m.
import { checkStatus, getModule, makeMatrixView } from "./ffi.js";
import { withContext } from "./estimatorRoles.js";
import type { Matrix } from "./types.js";

export interface MultimodalSourceSchema {
    representation_id: string;
    input_shape: number[];
    dtype: string;
    identity: string;
}
export type MultimodalSourceSchemas = Record<string, MultimodalSourceSchema>;
export interface RawTensor {
    data: Float32Array | Float64Array;
    shape: number[];
    strides?: number[];
}
export type MultimodalBlocks = Record<string, RawTensor | Array<[number | string, string]>>;
export interface MultimodalRecipe {
    schema_version: number;
    fusion: string;
    source_order: string[];
    encoders: Record<string, Record<string, unknown>>;
    source_weights: Record<string, number>;
    model: { method_id: string; params: Record<string, unknown> };
}
const ORDER = ["nir", "image", "series", "metadata"];
function sourceOrder(value: unknown): string[] {
    if (!Array.isArray(value) || value.length < 1 || value.length > ORDER.length ||
        value.some((name) => typeof name !== "string" || !ORDER.includes(name)) ||
        new Set(value).size !== value.length)
        throw new TypeError("source_order must select 1..4 distinct U07 modalities");
    return [...value];
}
// wasm32 layouts of multimodal.h (double/int64 align to eight bytes).
const SPEC_SIZE = 96, RECIPE_SIZE = 40, VIEW_SIZE = 56;
function utf8(value: string): Uint8Array {
    if (typeof value !== "string") throw new TypeError("expected a Unicode string");
    const encoded = new TextEncoder().encode(value);
    if (new TextDecoder("utf-8", { fatal: true }).decode(encoded) !== value)
        throw new TypeError("string contains an unpaired surrogate");
    return encoded;
}
function keys(value: object, expected: string[], label: string): void {
    if (value === null || typeof value !== "object" ||
        Object.keys(value).sort().join("\0") !== [...expected].sort().join("\0")) {
        throw new TypeError(`${label} has invalid fields`);
    }
}
function integer(value: unknown): number {
    if (typeof value !== "number" || !Number.isSafeInteger(value)) throw new TypeError("expected an exact integer");
    return value;
}
function flag(value: unknown): number {
    if (typeof value !== "boolean") throw new TypeError("expected a boolean");
    return value ? 1 : 0;
}
class Arena {
    private allocations: number[] = [];
    alloc(size: number): number {
        const m = getModule(), p = m._malloc(Math.max(1, size));
        if (p === 0) throw new Error("native allocation failed");
        this.allocations.push(p); m.HEAPU8.fill(0, p, p + Math.max(1, size)); return p;
    }
    bytes(bytes: Uint8Array): number {
        const p = this.alloc(bytes.length); getModule().HEAPU8.set(bytes, p); return p;
    }
    text(value: string): number {
        if (typeof value !== "string" || value.includes("\0")) throw new TypeError("expected a string without NUL");
        return this.bytes(utf8(value + "\0"));
    }
    ints(values: number[]): number {
        const p = this.alloc(values.length * 8);
        values.forEach((v, i) => i64(p + i * 8, integer(v))); return p;
    }
    close(): void { this.allocations.reverse().forEach((p) => getModule()._free(p)); }
}
function i32(p: number, value: number): void { getModule().setValue(p, value, "i32"); }
function i64(p: number, value: number): void {
    new DataView(getModule().HEAPU8.buffer).setBigInt64(p, BigInt(value), true);
}
function f64(p: number, value: number): void {
    if (typeof value !== "number") throw new TypeError("expected a number");
    new DataView(getModule().HEAPU8.buffer).setFloat64(p, value, true);
}
function schema(p: number, value: MultimodalSourceSchema, arena: Arena): void {
    keys(value, ["representation_id", "input_shape", "dtype", "identity"], "source schema");
    i32(p + 8, arena.text(value.representation_id)); i32(p + 12, arena.text(value.dtype));
    if (typeof value.identity !== "string") throw new TypeError("identity must be the original schema string");
    const bytes = utf8(value.identity);
    i32(p + 16, arena.bytes(bytes)); i32(p + 20, bytes.length);
}
function configuration(recipe: MultimodalRecipe, schemas: MultimodalSourceSchemas, arena: Arena): number {
    keys(recipe, ["schema_version", "fusion", "source_order", "encoders", "source_weights", "model"], "recipe");
    if (recipe.schema_version !== 1 || recipe.fusion !== "early")
        throw new TypeError("expected v1 early-fusion recipe");
    const order = sourceOrder(recipe.source_order);
    keys(recipe.encoders, order, "encoders"); keys(recipe.source_weights, order, "weights"); keys(schemas, order, "schemas");
    keys(recipe.model, ["method_id", "params"], "model");
    keys(recipe.model.params, ["alpha", "center_x", "center_y", "scale_x"], "Ridge parameters");
    if (recipe.model.method_id !== "models.regularized.ridge") throw new TypeError("expected native Ridge");
    const sources = arena.alloc(order.length * SPEC_SIZE);
    order.forEach((name, index) => {
        const p = sources + index * SPEC_SIZE, encoder = recipe.encoders[name]!;
        i32(p, SPEC_SIZE); i32(p + 4, arena.text(name)); schema(p, schemas[name]!, arena);
        const shape = schemas[name]!.input_shape;
        i32(p + 24, shape.length); i32(p + 28, arena.ints(shape)); f64(p + 40, recipe.source_weights[name]!);
        i64(p + 80, -1); i64(p + 88, -1);
        if (encoder.kind === "standard_scaler") {
            keys(encoder, ["kind", "with_mean", "with_std"], "scaler"); i32(p + 32, 1);
            i32(p + 64, flag(encoder.with_mean)); i32(p + 68, flag(encoder.with_std));
        } else if (encoder.kind === "tensor_pca") {
            keys(encoder, ["kind", "n_components", "random_state", "whiten"], "PCA"); i32(p + 32, 2);
            i64(p + 48, integer(encoder.n_components)); i64(p + 56, integer(encoder.random_state)); i32(p + 72, flag(encoder.whiten));
        } else if (encoder.kind === "column_transformer") {
            keys(encoder, ["kind", "numeric_columns", "categorical_columns", "with_mean", "with_std", "handle_unknown", "sparse_output", "drop"], "mixed encoder");
            if (JSON.stringify(encoder.numeric_columns) !== "[0]" || JSON.stringify(encoder.categorical_columns) !== "[1]" ||
                encoder.handle_unknown !== "ignore" || encoder.sparse_output !== false || encoder.drop !== null)
                throw new TypeError("unsupported mixed-column recipe");
            i32(p + 32, 3); i32(p + 64, flag(encoder.with_mean)); i32(p + 68, flag(encoder.with_std));
            i32(p + 76, 1); i64(p + 80, 0); i64(p + 88, 1);
        } else throw new TypeError("unsupported encoder");
    });
    const p = arena.alloc(RECIPE_SIZE), params = recipe.model.params;
    i32(p, RECIPE_SIZE); i32(p + 4, order.length); i32(p + 8, sources); f64(p + 16, params.alpha as number);
    i32(p + 24, flag(params.center_x)); i32(p + 28, flag(params.center_y)); i32(p + 32, flag(params.scale_x)); return p;
}
function views(blocks: MultimodalBlocks, schemas: MultimodalSourceSchemas, arena: Arena, order: readonly string[]): { pointer: number; rows: number } {
    keys(blocks, [...order], "blocks"); keys(schemas, [...order], "schemas");
    const pointer = arena.alloc(order.length * VIEW_SIZE); let rows = -1;
    order.forEach((name, index) => {
        const p = pointer + index * VIEW_SIZE, raw = blocks[name]!;
        i32(p, VIEW_SIZE); i32(p + 4, arena.text(name)); schema(p, schemas[name]!, arena);
        let shape: number[], strides: number[], data: Float32Array | Float64Array;
        if (name === "metadata") {
            if (!Array.isArray(raw)) throw new TypeError("metadata must contain raw [numeric, category] rows");
            shape = [raw.length, 2]; strides = [1, 1]; data = new Float64Array(raw.length);
            const offsets = arena.alloc((raw.length + 1) * 8), chunks: Uint8Array[] = []; let bytes = 0;
            i64(offsets, 0);
            raw.forEach((row, i) => {
                if (!Array.isArray(row) || row.length !== 2 || typeof row[1] !== "string" ||
                    !(typeof row[0] === "number" || typeof row[0] === "string") ||
                    (typeof row[0] === "string" && !/^[\t\n\r ]*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[\t\n\r ]*$/.test(row[0]))) throw new TypeError("invalid declared metadata cells");
                data[i] = Number(row[0]);
                if (!Number.isFinite(data[i])) throw new TypeError("metadata numeric cell must be finite");
                const chunk = utf8(row[1]);
                if (chunk.length > 1024 * 1024 || bytes + chunk.length > 64 * 1024 * 1024) throw new TypeError("categorical UTF-8 input exceeds native bounds");
                chunks.push(chunk); bytes += chunk.length; i64(offsets + (i + 1) * 8, bytes);
            });
            const categories = arena.alloc(bytes); let position = categories;
            chunks.forEach((chunk) => { getModule().HEAPU8.set(chunk, position); position += chunk.length; });
            i32(p + 44, categories); i32(p + 48, bytes); i32(p + 52, offsets);
        } else {
            if (Array.isArray(raw) || !(raw.data instanceof Float32Array || raw.data instanceof Float64Array)) throw new TypeError("numeric source needs a typed raw tensor");
            shape = raw.shape.map(integer); data = raw.data;
            strides = raw.strides ? raw.strides.map(integer) : shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
            if (strides.length !== shape.length || strides.some((s) => s < 0)) throw new TypeError("invalid tensor strides");
            const last = shape.reduce((n, d, i) => n + Math.max(0, d - 1) * strides[i]!, 0);
            if (shape.every((d) => d > 0) && last >= data.length) throw new TypeError("raw tensor buffer is too short");
        }
        if (shape.slice(1).join("\0") !== schemas[name]!.input_shape.join("\0")) throw new TypeError("raw source shape differs from its schema");
        if (rows !== -1 && rows !== shape[0]) throw new TypeError("source row counts differ"); rows = shape[0]!;
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        i32(p + 24, shape.length); i32(p + 28, arena.ints(shape)); i32(p + 32, arena.ints(strides));
        i32(p + 36, arena.bytes(bytes)); i32(p + 40, data instanceof Float64Array ? 1 : 2);
    }); return { pointer, rows };
}

/** Complete native early-fusion pipeline, portable as a bounded N4MF state. */
export class MultimodalPipeline {
    private ptr = 0;
    private readonly sourceOrder: readonly string[];
    readonly recipe: MultimodalRecipe;
    readonly sourceSchemas: MultimodalSourceSchemas;
    constructor(recipe: MultimodalRecipe, sourceSchemas: MultimodalSourceSchemas) {
        this.recipe = structuredClone(recipe); this.sourceSchemas = structuredClone(sourceSchemas);
        this.sourceOrder = sourceOrder(this.recipe.source_order);
        const arena = new Arena();
        try {
            const config = configuration(this.recipe, this.sourceSchemas, arena), out = arena.alloc(4);
            withContext((ctx) => checkStatus(getModule().ccall("n4m_multimodal_pipeline_create", "number",
                ["number", "number", "number"], [ctx, config, out]) as number, ctx)); this.ptr = getModule().getValue(out, "i32");
        } finally { arena.close(); }
    }
    private handle(): number { if (!this.ptr) throw new Error("MultimodalPipeline is closed"); return this.ptr; }
    fit(blocks: MultimodalBlocks, y: Matrix | Float64Array): this {
        const arena = new Arena();
        try {
            const x = views(blocks, this.sourceSchemas, arena, this.sourceOrder), target = y instanceof Float64Array ? { data: y, rows: x.rows, cols: 1 } : y;
            if (target.rows !== x.rows || target.cols !== 1) throw new TypeError("expected one target per raw source row");
            const matrix = makeMatrixView(target.data, target.rows, 1);
            try { withContext((ctx) => checkStatus(getModule().ccall("n4m_multimodal_pipeline_fit", "number",
                ["number", "number", "number", "number", "number"], [ctx, this.handle(), this.sourceOrder.length, x.pointer, matrix.viewPtr]) as number, ctx)); }
            finally { matrix.free(); } return this;
        } finally { arena.close(); }
    }
    predict(blocks: MultimodalBlocks, schemas: MultimodalSourceSchemas = this.sourceSchemas): Matrix { return this.operation(blocks, schemas, false); }
    transform(blocks: MultimodalBlocks, schemas: MultimodalSourceSchemas = this.sourceSchemas): Matrix { return this.operation(blocks, schemas, true); }
    private operation(blocks: MultimodalBlocks, schemas: MultimodalSourceSchemas, transform: boolean): Matrix {
        const arena = new Arena();
        try {
            const x = views(blocks, schemas, arena, this.sourceOrder); let cols = 1;
            if (transform) { const out = arena.alloc(8); checkStatus(getModule().ccall("n4m_multimodal_pipeline_transform_cols", "number", ["number", "number"], [this.handle(), out]) as number); cols = Number(new DataView(getModule().HEAPU8.buffer).getBigInt64(out, true)); }
            const matrix = makeMatrixView(new Float64Array(x.rows * cols), x.rows, cols);
            try {
                withContext((ctx) => checkStatus(getModule().ccall("n4m_multimodal_pipeline_" + (transform ? "transform" : "predict"), "number",
                    ["number", "number", "number", "number", "number"], [ctx, this.handle(), this.sourceOrder.length, x.pointer, matrix.viewPtr]) as number, ctx));
                return { data: getModule().HEAPF64.slice(matrix.dataPtr / 8, matrix.dataPtr / 8 + x.rows * cols), rows: x.rows, cols };
            } finally { matrix.free(); }
        } finally { arena.close(); }
    }
    exportState(): Uint8Array {
        const arena = new Arena();
        try { return withContext((ctx) => {
            const size = arena.alloc(4), m = getModule();
            checkStatus(m.ccall("n4m_multimodal_pipeline_export_size", "number", ["number", "number", "number"], [ctx, this.handle(), size]) as number, ctx);
            const capacity = m.getValue(size, "i32"), buffer = arena.alloc(capacity);
            checkStatus(m.ccall("n4m_multimodal_pipeline_export_to_buffer", "number", ["number", "number", "number", "number", "number"], [ctx, this.handle(), buffer, capacity, size]) as number, ctx);
            return m.HEAPU8.slice(buffer, buffer + m.getValue(size, "i32"));
        }); } finally { arena.close(); }
    }
    static fromState(state: Uint8Array, recipe: MultimodalRecipe, sourceSchemas: MultimodalSourceSchemas): MultimodalPipeline {
        if (!(state instanceof Uint8Array) || state.length > 64 * 1024 * 1024) throw new TypeError("expected bounded N4MF bytes");
        const model = new MultimodalPipeline(recipe, sourceSchemas), arena = new Arena();
        try {
            const config = configuration(recipe, sourceSchemas, arena), buffer = arena.bytes(state), out = arena.alloc(4);
            withContext((ctx) => checkStatus(getModule().ccall("n4m_multimodal_pipeline_import_from_buffer", "number", ["number", "number", "number", "number", "number"], [ctx, config, buffer, state.length, out]) as number, ctx));
            model.dispose(); model.ptr = getModule().getValue(out, "i32"); return model;
        } catch (error) { model.dispose(); throw error; } finally { arena.close(); }
    }
    dispose(): void { if (this.ptr) { getModule().ccall("n4m_multimodal_pipeline_destroy", null, ["number"], [this.ptr]); this.ptr = 0; } }
}
