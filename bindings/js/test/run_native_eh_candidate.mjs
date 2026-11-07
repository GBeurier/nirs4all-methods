// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Private candidate qualification. Authored tests are not executed evidence.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { registerHooks } from "node:module";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const protectedKeys = ["HOME", "home", "CODEX_HOME"];
const inherited = protectedKeys.map(k => [Object.hasOwn(process.env, k), process.env[k]]);
function envGuard(env = process.env) {
    assert(protectedKeys.every((k, j) => Object.hasOwn(env, k) === inherited[j][0]
        && env[k] === inherited[j][1]), "protected environment changed (values withheld)");
}
const hash = b => crypto.createHash("sha256").update(b).digest("hex");
const arg = name => {
    const i = process.argv.indexOf(name);
    assert(i >= 0 && process.argv[i + 1], `required ${name}`);
    return process.argv[i + 1];
};
function pinned(file, expected) {
    assert(/^[a-f0-9]{64}$/.test(expected), "explicit SHA256 required");
    const bytes = fs.readFileSync(file);
    assert.equal(hash(bytes), expected, `artifact identity: ${file}`);
    return bytes;
}
const vars = ["N4M_WASM_PATH", "N4M_WASM_GLUE_PATH",
    "N4M_EXPECTED_WASM_SHA256", "N4M_EXPECTED_GLUE_SHA256"];
assert(vars.every(k => process.env[k]), "private qualification needs all four artifact variables");
const normal = { wasm: path.resolve(process.env.N4M_WASM_PATH),
    glue: path.resolve(process.env.N4M_WASM_GLUE_PATH),
    wasmHash: process.env.N4M_EXPECTED_WASM_SHA256, glueHash: process.env.N4M_EXPECTED_GLUE_SHA256 };
// This explicit tests-first route admits ONLY the immutable old JS-EH pair.
// Its notes-only sidecar describes test admission, never artifact capability.
if (process.argv.includes("--tests-first-js-eh")) {
    assert.equal(normal.wasmHash, "7a8ecb9bcf555502252684806a829b3fbd4dd0421e4cf43c110082f4eabd5d5b");
    assert.equal(normal.glueHash, "8d0a66a55c745d19702eda31fc9502ef110bfcf0a03df8063fbcf2136528f36c");
    const bytes = pinned(normal.wasm, normal.wasmHash);
    pinned(normal.glue, normal.glueHash);
    const control = JSON.parse(pinned(arg("--red-sidecar"), arg("--red-sidecar-sha256")));
    assert.equal(control.purpose, "tests-first-absent-native-eh-contract");
    assert.equal(control.wasmSha256, normal.wasmHash);
    assert.equal(control.glueSha256, normal.glueHash);
    assert.equal(control.expectedAssertion, "NATIVE_EH_REQUIRED_CONTRACT_ABSENT");
    envGuard();
    const factory = (await import(pathToFileURL(normal.glue).href)).default;
    const m = await factory({ wasmBinary: bytes });
    const slot = m._malloc(4), id = m._malloc(64);
    assert(slot && id);
    let ctx = 0;
    try {
        assert.equal(m._n4m_context_create(slot), 0, "old pair actual context positive");
        ctx = m.HEAP32[slot / 4]; assert(ctx);
        m.stringToUTF8("preprocessing.scatter.snv", id, 64);
        assert.equal(m._n4m_method_find(id, slot), 0, "old pair actual metadata positive");
    } finally {
        if (ctx) m._n4m_context_destroy(ctx);
        m._free(id); m._free(slot);
    }
    console.log("NATIVE_EH_TESTS_FIRST_LOADED", JSON.stringify({ wasmSha256: normal.wasmHash,
        glueSha256: normal.glueHash, actualContextAndMetadata: true }));
    // This exact assertion must be the future RED, before any C++ tag witness.
    assert(m.n4mRuntimeRequirements, "NATIVE_EH_REQUIRED_CONTRACT_ABSENT");
    assert.fail("JS_EH_RED_INPUT_UNEXPECTEDLY_HAS_CONTRACT");
} else {
const boundary = { wasm: path.resolve(arg("--boundary-wasm")), glue: path.resolve(arg("--boundary-glue")),
    wasmHash: arg("--boundary-wasm-sha256"), glueHash: arg("--boundary-glue-sha256") };
const archive = path.resolve(arg("--owner-archive"));
const archiveHash = arg("--owner-archive-sha256");
pinned(archive, archiveHash);
const fixture = JSON.parse(pinned(arg("--fixture"), arg("--fixture-sha256")));
const indexPath = path.resolve(arg("--public-index"));
pinned(indexPath, arg("--public-index-sha256"));
// The public loader still imports its adjacent glue. No owner loader edit/copy here.
assert.equal(hash(fs.readFileSync(path.join(path.dirname(indexPath), "n4m.js"))), normal.glueHash);
assert.equal(hash(fs.readFileSync(path.join(path.dirname(indexPath), "n4m.wasm"))), normal.wasmHash);
const requiredOps = ["try", "catch", "throw", "catch_all", "rethrow", "delegate"];
function admission(pair, sidecarPath, sidecarHash, qualificationOnly) {
    pair.bytes = pinned(pair.wasm, pair.wasmHash);
    pinned(pair.glue, pair.glueHash);
    const c = JSON.parse(pinned(sidecarPath, sidecarHash));
    assert.equal(c.ehModel, "native-experimental");
    assert.equal(c.encoding, "wasm-legacy-eh");
    assert.equal(c.requiredFeature, "wasm-exception-handling");
    assert.equal(c.qualificationOnly, qualificationOnly);
    assert.equal(c.experimental, true);
    assert.equal(c.traceStack, true);
    assert.deepEqual(c.opcodes, requiredOps);
    assert.equal(path.resolve(c.ownerArchive), archive);
    pair.capabilities = c;
}
admission(normal, arg("--capabilities"), arg("--capabilities-sha256"), false);
admission(boundary, arg("--boundary-capabilities"), arg("--boundary-capabilities-sha256"), true);
envGuard();
const childAt = process.argv.indexOf("--child");
const modes = ["raw-positive", "public-positive", "raw-unavailable", "public-unavailable",
    "raw-unsupported", "public-unsupported", "raw-trace-unsupported", "public-trace-unsupported", "raw-corrupt-artifact", "public-corrupt-artifact", "boundary"];
if (childAt < 0) {
    for (const mode of modes) {
        envGuard();
        const env = { ...process.env };
        for (const k of ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"]) env[k] = "1";
        envGuard(env);
        const started = new Date().toISOString();
        const r = spawnSync(process.execPath, [fileURLToPath(import.meta.url), ...process.argv.slice(2), "--child", mode],
            { env, encoding: "utf8", timeout: 120000, maxBuffer: 1024 * 1024 });
        envGuard();
        process.stdout.write(r.stdout ?? "");
        process.stderr.write(r.stderr ?? "");
        console.log("NATIVE_EH_CHILD", JSON.stringify({ mode, pid: r.pid, started,
            finished: new Date().toISOString(), status: r.status, signal: r.signal,
            stdoutSha256: hash(r.stdout ?? ""), stderrSha256: hash(r.stderr ?? "") }));
        assert.ifError(r.error);
        assert.equal(r.signal, null, "abort/signal is not a typed refusal");
        assert.equal(r.status, 0, mode);
    }
} else {
    const mode = process.argv[childAt + 1];
    assert(modes.includes(mode));
    const pair = mode === "boundary" ? boundary : normal;
    const realWasm = globalThis.WebAssembly;
    const publicRoute = mode.startsWith("public-");
    let instantiations = 0, rawExports;
    const corruptArtifact = mode.endsWith("-corrupt-artifact");
    const observer = options => ({ ...options, wasmBinary: pair.bytes,
        instantiateWasm(imports, receive) {
            ++instantiations;
            const candidateBytes = Uint8Array.from(pair.bytes);
            if (corruptArtifact) candidateBytes[0] = 255; // labelled in-memory corrupt-magic control
            const compiled = new realWasm.Module(candidateBytes);
            const invokeImports = realWasm.Module.imports(compiled).filter(x => x.name.startsWith("invoke_"));
            assert.deepEqual(invokeImports, [], "native EH candidate still imports JS-EH invoke trampolines");
            const instance = new realWasm.Instance(compiled, imports);
            rawExports = instance.exports;
            receive(instance, compiled);
            return instance.exports;
        } });
    // Hook only factory options, not glue bytes, owner exports, or C++ exceptions.
    const hookKey = Symbol.for("n4m.private.nativeEH.factoryOptions");
    globalThis[hookKey] = observer;
    if (publicRoute) {
        const target = pathToFileURL(path.join(path.dirname(indexPath), "n4m.js")).href;
        const original = new URL(target); original.searchParams.set("nativeEHOriginal", "1");
        registerHooks({ load(url, context, nextLoad) {
            if (url !== target) return nextLoad(url, context);
            return { format: "module", shortCircuit: true, source:
                `import factory from ${JSON.stringify(original.href)};\n` +
                `export default options => factory(globalThis[Symbol.for("n4m.private.nativeEH.factoryOptions")](options));\n` };
        } });
    }
    const absent = mode.endsWith("-unavailable");
    const missingOpcodes = mode.endsWith("-unsupported") && !mode.includes("trace-");
    const missingTrace = mode.endsWith("trace-unsupported");
    const simulated = absent || missingOpcodes || missingTrace;
    if (absent) globalThis.WebAssembly = undefined;
    if (missingOpcodes) globalThis.WebAssembly = new Proxy(realWasm, {
        get(target, key) { return key === "validate" ? () => false : Reflect.get(target, key); } });
    if (missingTrace) {
        const noTrace = function(tag, values) { return new realWasm.Exception(tag, values); };
        noTrace.prototype = realWasm.Exception.prototype;
        globalThis.WebAssembly = new Proxy(realWasm, {
            get(target, key) { return key === "Exception" ? noTrace : Reflect.get(target, key); } });
    }
    let api;
    envGuard();
    if (publicRoute) api = await import(pathToFileURL(indexPath).href);
    async function load() {
        if (publicRoute) return api.loadModule();
        const factory = (await import(pathToFileURL(pair.glue).href)).default;
        return factory(observer({}));
    }
    if (simulated) {
        await assert.rejects(load, e => {
            assert.equal(e.name, "N4mRuntimeCapabilityError");
            assert.equal(e.code, absent ? "RuntimeUnavailable" : "Unsupported");
            assert.equal(e.requiredFeature, "wasm-exception-handling");
            return true;
        });
        assert.equal(instantiations, 0, "host gate must precede candidate instantiation (synthetic feature probe is separate)");
        if (publicRoute) assert.throws(() => api.getModule(), /not loaded|loadModule/i);
        console.log("NATIVE_EH_SIMULATION", JSON.stringify({ mode, candidateInstantiations: instantiations,
            scope: "isolated simulation, not browser support evidence" }));
    } else if (corruptArtifact) {
        await assert.rejects(load, e => {
            assert(e instanceof realWasm.CompileError, "corrupt artifact must retain its ordinary CompileError");
            assert.notEqual(e.name, "N4mRuntimeCapabilityError");
            assert.notEqual(e.code, "Unsupported");
            assert.notEqual(e.code, "RuntimeUnavailable");
            return true;
        });
        assert.equal(instantiations, 1, "host admitted; only candidate compilation was attempted");
        if (publicRoute) assert.throws(() => api.getModule(), /not loaded|loadModule/i);
        console.log("NATIVE_EH_CORRUPT_CONTROL", JSON.stringify({ mode, inMemoryMutation: true,
            pinnedPairUnchanged: true, candidateAttempts: instantiations }));
    } else {
        const m = await load();
        assert.equal(instantiations, 1);
        assert(m.n4mRuntimeRequirements, "NATIVE_EH_REQUIRED_CONTRACT_ABSENT");
        assert.equal(m.n4mRuntimeRequirements.encoding, "wasm-legacy-eh");
        assert.equal(m.n4mRuntimeRequirements.requiredFeature, "wasm-exception-handling");
        assert.deepEqual(m.n4mRuntimeRequirements.opcodes, requiredOps);
        assert.equal(m.n4mRuntimeRequirements.traceStack, true);
        // This is a post-instantiation witness, never a host-refusal classification.
        assert(rawExports.__cpp_exception instanceof realWasm.Tag, "missing emitted C++ exception tag witness");
        const witness = new realWasm.Exception(rawExports.__cpp_exception, [42], { traceStack: true });
        assert(witness.is(rawExports.__cpp_exception));
        assert.equal(witness.getArg(rawExports.__cpp_exception, 0), 42);
        assert.equal(typeof witness.stack, "string");
        assert(witness.stack.length > 0);
        if (mode !== "boundary") {
            assert(!Object.keys(m).some(k => k.startsWith("_n4m_test_eh_")), "normal module leaked private fixture");
        }
        const call = (name, ...args) => m[name](...args);
        const allocated = [];
        const malloc = n => {
            const p = m._malloc(n); assert(p); m.HEAPU8.fill(0, p, p + n); allocated.push(p); return p;
        };
        const text = t => { const p = malloc(m.lengthBytesUTF8(t) + 1);
            m.stringToUTF8(t, p, m.lengthBytesUTF8(t) + 1); return p; };
        const slot = malloc(4), id = text("preprocessing.scatter.snv");
        let ctx = 0, params = 0, est = 0;
        const ok = (s, label) => assert.equal(s, 0, label);
        const destroyEst = () => { if (est) { call("_n4m_estimator_destroy", est); est = 0; } };
        const create = () => {
            m.HEAP32[slot / 4] = 0;
            const s = call("_n4m_estimator_create", ctx, id, params, slot);
            est = m.HEAP32[slot / 4];
            ok(s, "valid create"); assert(est);
        };
        function matrix(values, rows, cols) {
            const data = malloc(values.length * 8), view = malloc(48);
            m.HEAPF64.set(values, data / 8);
            ok(call("_n4m_matrix_view_init_rowmajor", view, data, BigInt(rows), BigInt(cols), 1), "view");
            return { data, view };
        }
        try {
            ok(call("_n4m_context_create", slot), "context"); ctx = m.HEAP32[slot / 4];
            ok(call("_n4m_method_find", id, slot), "real metadata");
            ok(call("_n4m_params_create", ctx, m.HEAP32[slot / 4], slot), "params"); params = m.HEAP32[slot / 4];
            ok(call("_n4m_params_validate", ctx, params), "valid defaults");
            const x = matrix(fixture.matrix.flat(), 4, 4), out = matrix(Array(16).fill(-777.25), 4, 4);
            const input = malloc(120); m.HEAP32[input / 4] = 120; m.HEAP32[(input + 4) / 4] = x.view;
            const valid = () => {
                create(); ok(call("_n4m_estimator_fit", ctx, est, input), "recovery fit");
                ok(call("_n4m_estimator_transform", ctx, est, x.view, out.view), "recovery transform");
                assert.equal(Buffer.from(m.HEAPU8.subarray(out.data, out.data + 128)).toString("hex"), fixture.default_bits.wasm_8b0);
                destroyEst();
            };
            if (mode === "boundary") {
                // Snapshot storage is allocated before any baseline. mallinfo and
                // private snapshots allocate nothing. HEAP capacity is only an extra limit.
                const snapshotStorage = malloc(8);
                const snapshot = () => {
                    m._n4m_test_eh_memory_snapshot(snapshotStorage);
                    const view = new DataView(m.HEAPU8.buffer);
                    return { liveNew: view.getUint32(snapshotStorage, true),
                        heapInUse: view.getUint32(snapshotStorage + 4, true) };
                };
                const baseline = snapshot();
                assert(baseline.heapInUse > 0, "heap-in-use metric positive anchor missing");
                const balanced = label => assert.deepEqual(snapshot(), baseline, label);
                const start = m._n4m_test_eh_allocations();
                valid(); balanced("owner warmup must return both metrics to baseline");
                assert(m._n4m_test_eh_allocations() > start, "allocator positive control not reached");
                const createStart = m._n4m_test_eh_allocations();
                create();
                const createAllocations = m._n4m_test_eh_allocations() - createStart;
                assert(Number.isSafeInteger(createAllocations) && createAllocations > 0);
                const liveCreate = snapshot();
                assert(liveCreate.liveNew > baseline.liveNew, "live-new positive anchor missing");
                assert(liveCreate.heapInUse > baseline.heapInUse, "malloc heap-in-use positive anchor missing");
                destroyEst(); balanced("measured valid create/destroy must balance");
                const measuredAgain = m._n4m_test_eh_allocations();
                create(); assert.equal(m._n4m_test_eh_allocations() - measuredAgain, createAllocations);
                destroyEst(); balanced("same-run create allocation count must be stable");
                let badAllocCount = 0, sentinelCount = 0;
                function fault(kind, skip) {
                    balanced("before each fault");
                    assert.equal(m._n4m_test_eh_armed(), 0);
                    const a = m._n4m_test_eh_allocations(), t = m._n4m_test_eh_throws();
                    m.HEAP32[slot / 4] = 12345;
                    ok(m._n4m_test_eh_arm(kind, skip), "arm after owner warmup");
                    const status = call("_n4m_estimator_create", ctx, id, params, slot);
                    assert.equal(status, kind === 1 ? 9 : 255, "real owner guard returned wrong status");
                    assert.equal(m.HEAP32[slot / 4], 0, "failed create must null output");
                    assert.equal(m._n4m_test_eh_armed(), 0);
                    assert.equal(m._n4m_test_eh_allocations() - a, skip + 1);
                    assert.equal(m._n4m_test_eh_throws() - t, 1);
                    assert.equal(m.UTF8ToString(call("_n4m_context_last_error", ctx)), kind === 1
                        ? "out of memory" : "n4m native EH qualification sentinel");
                    balanced(`after fault ${kind} allocation ${skip}`);
                    if (kind === 1) ++badAllocCount; else ++sentinelCount;
                    valid(); balanced("after each same-module valid recovery");
                }
                // Every allocation index, including frames with resolved Params
                // and adapters already live. Any metric drift is a failure, not a warmup allowance.
                for (let skip = 0; skip < createAllocations; ++skip) { fault(1, skip); fault(2, skip); }
                const heap = m.HEAPU8.buffer.byteLength;
                for (let i = 0; i < 1000; ++i) {
                    const skip = i % createAllocations;
                    fault(1, skip); fault(2, skip);
                }
                assert.equal(m.HEAPU8.buffer.byteLength, heap, "additional bounded linear-memory capacity check");
                valid(); balanced("final valid recovery");
                console.log("NATIVE_EH_BOUNDARY", JSON.stringify({ createAllocations,
                    badAlloc: badAllocCount, sentinel: sentinelCount, everyAllocationIndex: true,
                    recovered: true, heapCapacityBytes: heap, memoryBaseline: baseline,
                    liveNewAndMallocInUseBalancedAfterEveryFault: true, exactOneInjectedThrow: true }));
            } else valid();
        } finally {
            destroyEst();
            if (params) call("_n4m_params_destroy", params);
            if (ctx) call("_n4m_context_destroy", ctx);
            for (const p of allocated.reverse()) m._free(p);
        }
        console.log("NATIVE_EH_POSITIVE", JSON.stringify({ mode, wasmSha256: pair.wasmHash,
            glueSha256: pair.glueHash, ownerArchiveSha256: archiveHash, instantiations }));
    }
    envGuard();
}
}
