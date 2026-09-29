// SPDX-License-Identifier: CECILL-2.1
// Thin, owning WASM facade for the native n4m optimizer. Sampling, pruning,
// trace and checkpoint semantics stay exclusively in libn4m.

import { Context } from "./context.js";
import { checkStatus, getModule, type EmModule } from "./ffi.js";
import { MethodResult } from "./methodResult.js";

type ExactInteger = bigint | number;
export type SearchAxis =
    | { kind: "int" | "log_int"; low: ExactInteger; high: ExactInteger; step?: ExactInteger }
    | { kind: "float" | "log_float"; low: number; high: number; step?: number }
    | { kind: "categorical"; type: "string"; choices: readonly string[] }
    | { kind: "categorical"; type: "integer"; choices: readonly ExactInteger[] }
    | { kind: "categorical"; type: "float"; choices: readonly number[] }
    | { kind: "categorical"; type: "boolean"; choices: readonly boolean[] }
    | { kind: "ordinal"; choices: readonly number[] }
    | { kind: "sorted_tuple"; length: number; low: number; high: number; integer?: boolean };

export interface SearchConstraint {
    kind: "mutex_group" | "requires" | "exclude" | "condition_in" | "condition_not_in";
    refs: readonly string[];
    labels?: readonly (string | null)[];
}

export interface OptimizerOptions {
    sampler?: "random" | "sobol" | "lhs" | "ternary" | "ga" | "pso" | "cmaes" | "tpe" | "gp_ei";
    pruner?: "none" | "median" | "asha" | "hyperband" | "racing";
    direction?: "auto" | "minimize" | "maximize";
    evalMode?: "best" | "mean" | "robust_best";
    metric?: "rmse" | "mse" | "mae" | "r2" | "accuracy" | "balanced_accuracy" | "f1" | "logloss";
    liar?: "none" | "min" | "mean" | "max";
    startupTrials?: number;
    seed?: ExactInteger;
    timeoutSeconds?: number;
    maxResource?: number;
    reductionFactor?: number;
}

export type TrialValue = number | bigint | string | boolean | readonly (number | bigint)[];
export interface OptimizerTrial {
    id: bigint;
    parameters: Readonly<Record<string, TrialValue>>;
    rung: number;
    status: "running" | "completed" | "pruned" | "failed" | "cancelled";
}
export interface BatchResult {
    trials: readonly OptimizerTrial[];
    /** Nonzero only when native ask_batch committed a prefix before an error. */
    nativeStatus: number;
}
export interface OptimizerTrialRecord extends OptimizerTrial {
    readonly active: Readonly<Record<string, boolean>>;
    readonly score: number | null;
    readonly askSequence: bigint;
    readonly terminalSequence: bigint | null;
    readonly intermediates: readonly {
        sequence: bigint; step: number; score: number; shouldPrune: boolean;
    }[];
    readonly error: { code: string; message: string; retryable: boolean } | null;
}

const SAMPLERS = ["random", "sobol", "lhs", "ternary", "ga", "pso", "cmaes", "tpe", "gp_ei"];
const PRUNERS = ["none", "median", "asha", "hyperband", "racing"];
const DIRECTIONS = ["auto", "minimize", "maximize"];
const EVAL_MODES = ["best", "mean", "robust_best"];
const LIARS = ["none", "min", "mean", "max"];
const STATUSES = ["running", "completed", "pruned", "failed", "cancelled"] as const;
const CONSTRAINTS = ["mutex_group", "requires", "exclude", "condition_in", "condition_not_in"];
const METRICS: Readonly<Record<string, number>> = {
    rmse: 0, mse: 1, mae: 2, r2: 3, accuracy: 16, balanced_accuracy: 17,
    f1: 18, logloss: 19,
};
const MAX_EXACT_INT = 1n << 53n;
const MAX_BATCH = 10000;

function integer(value: ExactInteger, name: string): bigint {
    if (typeof value === "number") {
        if (!Number.isInteger(value) || Math.abs(value) > Number(MAX_EXACT_INT)) {
            throw new RangeError(`${name} must be an exactly representable integer`);
        }
        return BigInt(value);
    }
    if (typeof value !== "bigint") throw new TypeError(`${name} must be an integer`);
    return value;
}

function exactAxisInteger(value: ExactInteger, name: string): bigint {
    const result = integer(value, name);
    if (result < -MAX_EXACT_INT || result > MAX_EXACT_INT) {
        throw new RangeError(`${name} is outside the native exact-integer domain`);
    }
    return result;
}

function i32(value: number, name: string): number {
    if (!Number.isInteger(value) || value < -2147483648 || value > 2147483647) {
        throw new RangeError(`${name} must fit int32`);
    }
    return value;
}

function finite(value: number, name: string): number {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new RangeError(`${name} must be finite`);
    }
    return value;
}

function enumValue(value: string | undefined, names: readonly string[], fallback: string): number {
    const index = names.indexOf(value ?? fallback);
    if (index < 0) throw new RangeError(`Unknown native optimizer choice: ${value}`);
    return index;
}

class NativeArgs {
    private readonly pointers: number[] = [];
    constructor(readonly module: EmModule) {}

    alloc(bytes: number): number {
        const pointer = this.module._malloc(Math.max(1, bytes));
        if (pointer === 0) throw new Error("WASM allocation failed");
        this.pointers.push(pointer);
        return pointer;
    }

    string(value: string): number {
        const bytes = this.module.lengthBytesUTF8(value) + 1;
        const pointer = this.alloc(bytes);
        this.module.stringToUTF8(value, pointer, bytes);
        return pointer;
    }

    pointersArray(values: readonly number[]): number {
        const pointer = this.alloc(values.length * 4);
        const view = new DataView(this.module.HEAPU8.buffer);
        values.forEach((value, index) => view.setUint32(pointer + index * 4, value, true));
        return pointer;
    }

    doubles(values: readonly number[]): number {
        const pointer = this.alloc(values.length * 8);
        const view = new DataView(this.module.HEAPU8.buffer);
        values.forEach((value, index) => view.setFloat64(pointer + index * 8, value, true));
        return pointer;
    }

    int64(values: readonly bigint[]): number {
        const pointer = this.alloc(values.length * 8);
        const view = new DataView(this.module.HEAPU8.buffer);
        values.forEach((value, index) => view.setBigInt64(pointer + index * 8, value, true));
        return pointer;
    }

    int32(values: readonly number[]): number {
        const pointer = this.alloc(values.length * 4);
        const view = new DataView(this.module.HEAPU8.buffer);
        values.forEach((value, index) => view.setInt32(pointer + index * 4, value, true));
        return pointer;
    }

    free(): void {
        for (const pointer of this.pointers.reverse()) this.module._free(pointer);
    }
}

function addAxis(m: EmModule, args: NativeArgs, ptr: number, name: string, axis: SearchAxis,
                 context: number): void {
    const namePtr = args.string(name);
    let status: number;
    switch (axis.kind) {
        case "int":
        case "log_int":
            status = m.ccall("n4m_search_space_add_int", "number",
                ["number", "number", "i64", "i64", "i64", "number"],
                [ptr, namePtr, exactAxisInteger(axis.low, `${name}.low`),
                    exactAxisInteger(axis.high, `${name}.high`),
                    exactAxisInteger(axis.step ?? 1, `${name}.step`), Number(axis.kind === "log_int")]) as number;
            break;
        case "float":
        case "log_float":
            status = m.ccall("n4m_search_space_add_float", "number",
                ["number", "number", "number", "number", "number", "number"],
                [ptr, namePtr, finite(axis.low, `${name}.low`), finite(axis.high, `${name}.high`),
                    finite(axis.step ?? 0, `${name}.step`), Number(axis.kind === "log_float")]) as number;
            break;
        case "categorical": {
            if (axis.choices.length === 0) throw new RangeError(`${name} has no choices`);
            i32(axis.choices.length, `${name}.choices.length`);
            const type = ["string", "integer", "float", "boolean"].indexOf(axis.type);
            if (type < 0) throw new RangeError(`Unknown category type: ${axis.type}`);
            let valuesPtr: number;
            if (axis.type === "string") {
                valuesPtr = args.pointersArray(axis.choices.map(value => {
                    if (typeof value !== "string") throw new TypeError(`${name} choices must be strings`);
                    return args.string(value);
                }));
            } else if (axis.type === "integer") {
                valuesPtr = args.int64(axis.choices.map(value => exactAxisInteger(value, `${name}.choice`)));
            } else if (axis.type === "float") {
                valuesPtr = args.doubles(axis.choices.map(value => finite(value, `${name}.choice`)));
            } else {
                valuesPtr = args.int32(axis.choices.map(value => {
                    if (typeof value !== "boolean") throw new TypeError(`${name} choices must be booleans`);
                    return Number(value);
                }));
            }
            status = m.ccall("n4m_search_space_add_categorical", "number",
                ["number", "number", "number", "number", "number"],
                [ptr, namePtr, type, valuesPtr, axis.choices.length]) as number;
            break;
        }
        case "ordinal": {
            if (axis.choices.length === 0) throw new RangeError(`${name} has no choices`);
            i32(axis.choices.length, `${name}.choices.length`);
            const values = args.doubles(axis.choices.map(value => finite(value, `${name}.choice`)));
            status = m.ccall("n4m_search_space_add_ordinal", "number",
                ["number", "number", "number", "number"],
                [ptr, namePtr, values, axis.choices.length]) as number;
            break;
        }
        case "sorted_tuple":
            status = m.ccall("n4m_search_space_add_sorted_tuple", "number",
                ["number", "number", "number", "number", "number", "number"],
                [ptr, namePtr, i32(axis.length, `${name}.length`), finite(axis.low, `${name}.low`),
                    finite(axis.high, `${name}.high`), Number(axis.integer ?? false)]) as number;
            break;
    }
    checkStatus(status, context);
}

function addConstraint(m: EmModule, args: NativeArgs, ptr: number,
                       constraint: SearchConstraint, context: number): void {
    const kind = enumValue(constraint.kind, CONSTRAINTS, "mutex_group");
    if (constraint.refs.length === 0 ||
        (constraint.labels !== undefined && constraint.labels.length !== constraint.refs.length)) {
        throw new RangeError("Constraint refs and labels must have equal nonzero length");
    }
    const refs = args.pointersArray(constraint.refs.map(ref => args.string(ref)));
    const labels = args.pointersArray(constraint.refs.map((_, index) => {
        const value = constraint.labels?.[index];
        return value === null || value === undefined ? 0 : args.string(value);
    }));
    const status = m.ccall("n4m_search_space_add_constraint", "number",
        ["number", "number", "number", "number", "number"],
        [ptr, kind, refs, labels, constraint.refs.length]) as number;
    checkStatus(status, context);
}

function makeOptions(m: EmModule, args: NativeArgs, options: OptimizerOptions): number {
    const ptr = args.alloc(120); // sizeof(n4m_optimizer_options_t) on wasm32, ABI 2.15
    m.ccall("n4m_optimizer_options_init", null, ["number"], [ptr]);
    const view = new DataView(m.HEAPU8.buffer);
    if (view.getBigUint64(ptr, true) !== 120n) throw new Error("Optimizer options ABI mismatch");
    view.setInt32(ptr + 8, enumValue(options.sampler, SAMPLERS, "random"), true);
    view.setInt32(ptr + 12, enumValue(options.pruner, PRUNERS, "none"), true);
    view.setInt32(ptr + 16, enumValue(options.direction, DIRECTIONS, "auto"), true);
    view.setInt32(ptr + 20, enumValue(options.evalMode, EVAL_MODES, "mean"), true);
    const metric = options.metric ?? "rmse";
    if (!Object.hasOwn(METRICS, metric)) throw new RangeError(`Unknown native metric: ${metric}`);
    view.setInt32(ptr + 24, METRICS[metric]!, true);
    view.setInt32(ptr + 28, enumValue(options.liar, LIARS, "none"), true);
    view.setInt32(ptr + 32, i32(options.startupTrials ?? 10, "startupTrials"), true);
    const seed = integer(options.seed ?? 0, "seed");
    if (seed < 0n || seed > (1n << 64n) - 1n) throw new RangeError("seed must fit uint64");
    view.setBigUint64(ptr + 40, seed, true);
    view.setFloat64(ptr + 48, finite(options.timeoutSeconds ?? 0, "timeoutSeconds"), true);
    view.setInt32(ptr + 56, i32(options.maxResource ?? 0, "maxResource"), true);
    view.setInt32(ptr + 60, i32(options.reductionFactor ?? 0, "reductionFactor"), true);
    return ptr;
}

function copySpace(space: Readonly<Record<string, SearchAxis>>): Readonly<Record<string, SearchAxis>> {
    const entries = Object.entries(space);
    if (entries.length === 0) throw new RangeError("Search space must have at least one axis");
    return Object.fromEntries(entries.map(([name, axis]) => {
        if (!name || name.includes("#") || name === "__proto__") {
            throw new RangeError(`Invalid search axis name: ${name}`);
        }
        if (!axis || typeof axis !== "object") throw new TypeError(`Invalid search axis: ${name}`);
        const owned = ("choices" in axis
            ? { ...axis, choices: Object.freeze([...axis.choices]) }
            : { ...axis }) as SearchAxis;
        return [name, Object.freeze(owned)];
    }));
}

function poolStrings(result: MethodResult, prefix: string, count: number): string[] {
    const bytes = result.vectorInt(`${prefix}_utf8`);
    const offsets = result.vectorInt64(`${prefix}_offsets`);
    if (offsets.length !== count + 1) throw new Error(`Invalid native trace offsets: ${prefix}`);
    const decoder = new TextDecoder("utf-8", { fatal: true });
    const out: string[] = [];
    for (let index = 0; index < count; index += 1) {
        const start = Number(offsets[index]);
        const end = Number(offsets[index + 1]);
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) ||
            start < 0 || end < start || end > bytes.length) {
            throw new Error(`Invalid native trace string span: ${prefix}`);
        }
        out.push(decoder.decode(Uint8Array.from(bytes.subarray(start, end))));
    }
    return out;
}

function decodeRecords(result: MethodResult): OptimizerTrialRecord[] {
    if (result.scalar("trace_format_version") !== 1) {
        throw new Error("Unsupported native optimizer trace version");
    }
    const ids = result.vectorInt64("trial_ids_i64");
    const count = ids.length;
    const names = poolStrings(result, "trial_param_name", result.scalar("n_params"));
    const width = names.length;
    const values = result.matrix("trial_param_values");
    const kinds = result.vectorInt("trial_param_kind");
    const categoryTypes = result.vectorInt("trial_param_category_type");
    const integers = result.vectorInt("trial_param_integer");
    const categoryIndices = result.vectorInt("trial_param_category_index");
    const activeBits = result.vectorInt("trial_param_active");
    const labels = poolStrings(result, "trial_param_label", count * width);
    const scores = result.matrix("trial_scores").data;
    const statuses = result.matrix("trial_status").data;
    const rungs = result.matrix("trial_rung").data;
    const asks = result.vectorInt64("trial_ask_sequence");
    const terminals = result.vectorInt64("trial_terminal_sequence");
    const interOffsets = result.vectorInt64("trial_intermediate_offsets");
    const interSequences = result.vectorInt64("trial_intermediate_sequence");
    const interSteps = result.vectorInt("trial_intermediate_steps");
    const interScores = result.matrix("trial_intermediate_scores").data;
    const interPruned = result.vectorInt("trial_intermediate_should_prune");
    const errorCodes = poolStrings(result, "trial_error_code", count);
    const errorMessages = poolStrings(result, "trial_error_message", count);
    const retryable = result.vectorInt("trial_error_retryable");
    if (values.rows !== count || values.cols !== width || kinds.length !== width ||
        categoryTypes.length !== width || integers.length !== width ||
        categoryIndices.length !== count * width || activeBits.length !== count * width ||
        scores.length !== count || statuses.length !== count || rungs.length !== count ||
        asks.length !== count || terminals.length !== count || interOffsets.length !== count + 1 ||
        interSequences.length !== interSteps.length || interSteps.length !== interScores.length ||
        interScores.length !== interPruned.length || retryable.length !== count) {
        throw new Error("Invalid native optimizer trace shape");
    }
    const records: OptimizerTrialRecord[] = [];
    for (let row = 0; row < count; row += 1) {
        const parameters: Record<string, TrialValue> = Object.create(null);
        const active: Record<string, boolean> = Object.create(null);
        for (let col = 0; col < width; col += 1) {
            const offset = row * width + col;
            const name = names[col]!;
            const isActive = activeBits[offset] !== 0;
            active[name] = isActive;
            if (!isActive) continue;
            let value: TrialValue;
            if (kinds[col] === 4) {
                if (categoryIndices[offset]! < 0) throw new Error("Missing native category index");
                const label = labels[offset]!;
                const categoryType = categoryTypes[col];
                if (categoryType === 0) value = label;
                else if (categoryType === 1) value = BigInt(label);
                else if (categoryType === 2) value = Number(label);
                else if (categoryType === 3) value = label === "true";
                else throw new Error("Unknown native category type");
            } else if (integers[col] !== 0) {
                value = BigInt(values.data[offset]!);
            } else {
                value = values.data[offset]!;
            }
            parameters[name] = value;
        }
        const begin = Number(interOffsets[row]);
        const end = Number(interOffsets[row + 1]);
        if (!Number.isSafeInteger(begin) || !Number.isSafeInteger(end) ||
            begin < 0 || end < begin || end > interSteps.length) {
            throw new Error("Invalid native intermediate trace offsets");
        }
        const intermediates = Array.from({ length: end - begin }, (_, local) => {
            const index = begin + local;
            return { sequence: interSequences[index]!, step: interSteps[index]!,
                score: interScores[index]!, shouldPrune: interPruned[index] !== 0 };
        });
        const status = STATUSES[statuses[row]!];
        if (!status) throw new Error("Unknown native trace status");
        records.push({
            id: ids[row]!, parameters, active, status, rung: rungs[row]!,
            score: Number.isNaN(scores[row]) ? null : scores[row]!,
            askSequence: asks[row]!,
            terminalSequence: terminals[row] === -1n ? null : terminals[row]!,
            intermediates,
            error: errorCodes[row] || errorMessages[row]
                ? { code: errorCodes[row]!, message: errorMessages[row]!, retryable: retryable[row] !== 0 }
                : null,
        });
    }
    return records;
}

/** Owns a native optimizer and its context. Dispose it after use. */
export class Optimizer {
    private _ptr: number;
    private readonly context: Context;
    readonly space: Readonly<Record<string, SearchAxis>>;

    private constructor(context: Context, ptr: number, space: Readonly<Record<string, SearchAxis>>) {
        this.context = context;
        this._ptr = ptr;
        this.space = space;
    }

    static create(space: Readonly<Record<string, SearchAxis>>,
                  options: OptimizerOptions = {}, constraints: readonly SearchConstraint[] = []): Optimizer {
        const m = getModule();
        const ownedSpace = copySpace(space);
        const context = Context.create();
        const args = new NativeArgs(m);
        let spacePtr = 0;
        try {
            const spaceOut = args.alloc(4);
            checkStatus(m.ccall("n4m_search_space_create", "number", ["number"], [spaceOut]) as number,
                context.handle);
            spacePtr = m.getValue(spaceOut, "i32");
            for (const [name, axis] of Object.entries(ownedSpace)) {
                addAxis(m, args, spacePtr, name, axis, context.handle);
            }
            for (const constraint of constraints) addConstraint(m, args, spacePtr, constraint, context.handle);
            const optionsPtr = makeOptions(m, args, options);
            const optOut = args.alloc(4);
            const status = m.ccall("n4m_optimizer_create", "number",
                ["number", "number", "number", "number"],
                [context.handle, spacePtr, optionsPtr, optOut]) as number;
            checkStatus(status, context.handle);
            const pointer = m.getValue(optOut, "i32");
            if (!pointer) throw new Error("Native optimizer returned a null handle");
            return new Optimizer(context, pointer, ownedSpace);
        } catch (error) {
            context.destroy();
            throw error;
        } finally {
            if (spacePtr) m.ccall("n4m_search_space_destroy", null, ["number"], [spacePtr]);
            args.free();
        }
    }

    static load(blob: Uint8Array, space: Readonly<Record<string, SearchAxis>>): Optimizer {
        if (!(blob instanceof Uint8Array) || blob.length === 0) throw new TypeError("Expected N4MOPT bytes");
        const m = getModule();
        const ownedSpace = copySpace(space);
        const context = Context.create();
        const args = new NativeArgs(m);
        try {
            const bytes = args.alloc(blob.length);
            m.HEAPU8.set(blob, bytes);
            const optOut = args.alloc(4);
            checkStatus(m.ccall("n4m_optimizer_load", "number",
                ["number", "number", "i64", "number"],
                [context.handle, bytes, BigInt(blob.length), optOut]) as number, context.handle);
            const pointer = m.getValue(optOut, "i32");
            if (!pointer) throw new Error("Native optimizer returned a null handle");
            return new Optimizer(context, pointer, ownedSpace);
        } catch (error) {
            context.destroy();
            throw error;
        } finally {
            args.free();
        }
    }

    private get handle(): number {
        if (!this._ptr) throw new Error("Optimizer has been disposed");
        return this._ptr;
    }

    private decodeTrial(ptr: number): OptimizerTrial {
        const m = getModule();
        const args = new NativeArgs(m);
        try {
            const out8 = args.alloc(8);
            const out4 = args.alloc(4);
            checkStatus(m.ccall("n4m_trial_get_id", "number", ["number", "number"],
                [ptr, out8]) as number, this.context.handle);
            const id = BigInt(m.getValue(out8, "i64"));
            const parameters: Record<string, TrialValue> = Object.create(null);
            for (const [name, axis] of Object.entries(this.space)) {
                const nativeName = axis.kind === "sorted_tuple" ? `${name}#0` : name;
                const nativeNamePtr = args.string(nativeName);
                checkStatus(m.ccall("n4m_trial_is_active", "number",
                    ["number", "number", "number"], [ptr, nativeNamePtr, out4]) as number,
                    this.context.handle);
                if (!m.getValue(out4, "i32")) continue;
                const namePtr = axis.kind === "sorted_tuple" ? args.string(name) : nativeNamePtr;
                if (axis.kind === "int" || axis.kind === "log_int") {
                    checkStatus(m.ccall("n4m_trial_get_int", "number",
                        ["number", "number", "number"], [ptr, namePtr, out8]) as number,
                        this.context.handle);
                    parameters[name] = BigInt(m.getValue(out8, "i64"));
                } else if (axis.kind === "float" || axis.kind === "log_float" || axis.kind === "ordinal") {
                    checkStatus(m.ccall("n4m_trial_get_float", "number",
                        ["number", "number", "number"], [ptr, namePtr, out8]) as number,
                        this.context.handle);
                    parameters[name] = m.getValue(out8, "double");
                } else if (axis.kind === "categorical") {
                    const labelOut = args.alloc(4);
                    checkStatus(m.ccall("n4m_trial_get_category", "number",
                        ["number", "number", "number", "number"],
                        [ptr, namePtr, out4, labelOut]) as number, this.context.handle);
                    const index = m.getValue(out4, "i32");
                    if (index < 0 || index >= axis.choices.length) throw new Error("Native category index out of range");
                    parameters[name] = axis.choices[index]!;
                } else if (axis.kind === "sorted_tuple") {
                    const values: (number | bigint)[] = [];
                    for (let index = 0; index < axis.length; index += 1) {
                        const component = args.string(`${name}#${index}`);
                        const symbol = axis.integer ? "n4m_trial_get_int" : "n4m_trial_get_float";
                        checkStatus(m.ccall(symbol, "number", ["number", "number", "number"],
                            [ptr, component, out8]) as number, this.context.handle);
                        values.push(axis.integer ? BigInt(m.getValue(out8, "i64")) : m.getValue(out8, "double"));
                    }
                    parameters[name] = values;
                } else {
                    throw new Error(`Unsupported search axis kind: ${JSON.stringify(axis)}`);
                }
            }
            checkStatus(m.ccall("n4m_trial_get_rung", "number", ["number", "number"],
                [ptr, out4]) as number, this.context.handle);
            const rung = m.getValue(out4, "i32");
            checkStatus(m.ccall("n4m_trial_get_status", "number", ["number", "number"],
                [ptr, out4]) as number, this.context.handle);
            const status = STATUSES[m.getValue(out4, "i32")];
            if (!status) throw new Error("Unknown native trial status");
            return { id, parameters, rung, status };
        } finally {
            args.free();
        }
    }

    ask(): OptimizerTrial {
        const m = getModule();
        const args = new NativeArgs(m);
        try {
            const out = args.alloc(4);
            checkStatus(m.ccall("n4m_optimizer_ask", "number", ["number", "number"],
                [this.handle, out]) as number, this.context.handle);
            return this.decodeTrial(m.getValue(out, "i32"));
        } finally { args.free(); }
    }

    askBatch(n: number): BatchResult {
        i32(n, "batch size");
        if (n < 0 || n > MAX_BATCH) throw new RangeError(`batch size must be 0..${MAX_BATCH}`);
        const m = getModule();
        const args = new NativeArgs(m);
        try {
            const slots = args.alloc(n * 4);
            const count = args.alloc(4);
            const nativeStatus = m.ccall("n4m_optimizer_ask_batch", "number",
                ["number", "number", "number", "number"],
                [this.handle, n, slots, count]) as number;
            const committed = m.getValue(count, "i32");
            if (committed < 0 || committed > n) throw new Error("Native batch count out of range");
            const trials: OptimizerTrial[] = [];
            for (let index = 0; index < committed; index += 1) {
                trials.push(this.decodeTrial(m.getValue(slots + index * 4, "i32")));
            }
            if (nativeStatus !== 0 && committed === 0) checkStatus(nativeStatus, this.context.handle);
            return { trials, nativeStatus };
        } finally { args.free(); }
    }

    /** Queue numeric values; categorical axes use zero-based choice indices. */
    enqueue(parameters: Readonly<Record<string, ExactInteger>>): void {
        const entries = Object.entries(parameters);
        if (entries.length === 0) throw new RangeError("Warm-start candidate is empty");
        const m = getModule();
        const args = new NativeArgs(m);
        try {
            const names = args.pointersArray(entries.map(([name]) => args.string(name)));
            const values = args.doubles(entries.map(([name, value]) =>
                typeof value === "bigint" ? Number(exactAxisInteger(value, name)) : finite(value, name)));
            checkStatus(m.ccall("n4m_optimizer_enqueue", "number",
                ["number", "number", "number", "number"],
                [this.handle, names, values, entries.length]) as number, this.context.handle);
        } finally { args.free(); }
    }

    tell(trial: OptimizerTrial | ExactInteger, status: "completed" | "pruned" | "failed" | "cancelled",
         score?: number, error?: string): void {
        const id = typeof trial === "object" ? trial.id : integer(trial, "trial id");
        const code = STATUSES.indexOf(status);
        if (code < 1) throw new RangeError("Invalid terminal trial status");
        if (status === "completed") finite(score as number, "completed score");
        else if (score !== undefined) throw new RangeError("Only completed trials accept a score");
        if (error !== undefined && (status !== "failed" && status !== "cancelled")) {
            throw new RangeError("Only failed/cancelled trials accept an error");
        }
        const m = getModule();
        const args = new NativeArgs(m);
        try {
            const errorPtr = error === undefined ? 0 : args.string(error);
            checkStatus(m.ccall("n4m_optimizer_tell_result", "number",
                ["number", "i64", "number", "number", "number"],
                [this.handle, id, code, score ?? 0, errorPtr]) as number, this.context.handle);
        } finally { args.free(); }
    }

    intermediate(trial: OptimizerTrial | ExactInteger, step: number, score: number): boolean {
        const id = typeof trial === "object" ? trial.id : integer(trial, "trial id");
        i32(step, "step");
        finite(score, "intermediate score");
        const m = getModule();
        const args = new NativeArgs(m);
        try {
            const out = args.alloc(4);
            checkStatus(m.ccall("n4m_optimizer_tell_intermediate", "number",
                ["number", "i64", "number", "number", "number"],
                [this.handle, id, step, score, out]) as number, this.context.handle);
            return m.getValue(out, "i32") !== 0;
        } finally { args.free(); }
    }

    best(): { trial: OptimizerTrial; score: number } {
        const m = getModule();
        const args = new NativeArgs(m);
        try {
            const out = args.alloc(4);
            const scoreOut = args.alloc(8);
            checkStatus(m.ccall("n4m_optimizer_best", "number",
                ["number", "number", "number"], [this.handle, out, scoreOut]) as number,
                this.context.handle);
            return { trial: this.decodeTrial(m.getValue(out, "i32")), score: m.getValue(scoreOut, "double") };
        } finally { args.free(); }
    }

    /** Owning native rich trace v1 snapshot. Caller must destroy the result. */
    trials(sinceId: ExactInteger = 0): MethodResult {
        const id = integer(sinceId, "sinceId");
        if (id < 0n) throw new RangeError("sinceId must be nonnegative");
        const m = getModule();
        const args = new NativeArgs(m);
        try {
            const out = args.alloc(4);
            checkStatus(m.ccall("n4m_optimizer_get_trials", "number",
                ["number", "i64", "number"], [this.handle, id, out]) as number,
                this.context.handle);
            return new MethodResult(m.getValue(out, "i32"));
        } finally { args.free(); }
    }

    /** Decode an owning rich-trace snapshot to typed JS values. */
    trialRecords(sinceId: ExactInteger = 0): readonly OptimizerTrialRecord[] {
        const result = this.trials(sinceId);
        try { return decodeRecords(result); }
        finally { result.destroy(); }
    }

    /** Portable N4MOPT bytes, directly readable by Python/R/native bindings. */
    save(): Uint8Array {
        const m = getModule();
        const args = new NativeArgs(m);
        let arrayPtr = 0;
        try {
            const out = args.alloc(4);
            checkStatus(m.ccall("n4m_optimizer_save", "number", ["number", "number"],
                [this.handle, out]) as number, this.context.handle);
            arrayPtr = m.getValue(out, "i32");
            const viewPtr = args.alloc(48);
            checkStatus(m.ccall("n4m_array_view", "number", ["number", "number"],
                [arrayPtr, viewPtr]) as number, this.context.handle);
            const dataPtr = m.getValue(viewPtr, "i32") >>> 0;
            const rows = BigInt(m.getValue(viewPtr + 8, "i64"));
            const cols = BigInt(m.getValue(viewPtr + 16, "i64"));
            const dtype = m.getValue(viewPtr + 40, "i32");
            if (dtype !== 4 || rows !== 1n || cols <= 0n || dataPtr === 0) {
                throw new Error("Invalid native optimizer checkpoint array");
            }
            const byteLength = rows * cols * 8n;
            if (dataPtr > m.HEAPU8.byteLength ||
                byteLength > BigInt(m.HEAPU8.byteLength - dataPtr)) {
                throw new RangeError("Native optimizer checkpoint exceeds WASM memory");
            }
            return m.HEAPU8.slice(dataPtr, dataPtr + Number(byteLength));
        } finally {
            if (arrayPtr) m.ccall("n4m_array_free", null, ["number"], [arrayPtr]);
            args.free();
        }
    }

    dispose(): void {
        if (!this._ptr) return;
        getModule().ccall("n4m_optimizer_destroy", null, ["number"], [this._ptr]);
        this._ptr = 0;
        this.context.destroy();
    }
}
