// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
// Candidate-only --pre-js: this runs inside the factory before createWasm.
// No artifact CompileError is intercepted or relabelled by this host gate.
(function () {
    var feature = "wasm-exception-handling";
    function refuse(code, detail) {
        var error = new Error(code + ": private Native Wasm EH requires " + feature + ": " + detail);
        error.name = "N4mRuntimeCapabilityError";
        error.code = code;
        error.requiredFeature = feature;
        throw error;
    }
    var requirements = Object.freeze({
        ehModel: "native-experimental",
        encoding: "wasm-legacy-eh",
        experimental: true,
        requiredFeature: feature,
        opcodes: Object.freeze(["try", "catch", "throw", "catch_all", "rethrow", "delegate"]),
        traceStack: true
    });
    Module["n4mRuntimeRequirements"] = requirements;
    if (typeof WebAssembly === "undefined" || WebAssembly === null) {
        refuse("RuntimeUnavailable", "WebAssembly runtime is absent");
    }
    if (typeof WebAssembly.validate !== "function" || typeof WebAssembly.Tag !== "function"
        || typeof WebAssembly.Exception !== "function" || typeof WebAssembly.Module !== "function"
        || typeof WebAssembly.Instance !== "function") {
        refuse("Unsupported", "legacy EH opcodes and JS Tag/Exception APIs are required");
    }
    // Audited 114-byte synthetic feature module: tagged throw/catch, catch_all,
    // rethrow and delegate, including nested try. Only this tiny synthetic module
    // runs here; the candidate and its constructors have not run. The future emitted-opcode audit must
    // establish a subset of this complete set before qualification is credited.
    var section = function (id, body) { return [id, body.length].concat(body); };
    var body = function (code) { return [code.length].concat(code); };
    var types = section(1, [3, 96, 0, 0, 96, 1, 127, 0, 96, 0, 1, 127]);
    var functions = section(3, [4, 0, 2, 2, 2]);
    var tag = section(13, [1, 0, 1]);
    var exports = section(7, [5, 1, 122, 0, 0, 1, 97, 0, 1, 1, 98, 0, 2, 1, 99, 0, 3, 1, 116, 4, 0]);
    var code = section(10, [4].concat(
        body([0, 65, 42, 8, 0, 11]),
        body([0, 6, 127, 16, 0, 65, 0, 25, 65, 7, 11, 11]),
        body([0, 6, 127, 6, 127, 16, 0, 65, 0, 25, 9, 0, 11, 7, 0, 11, 11]),
        body([0, 6, 127, 6, 127, 16, 0, 65, 0, 24, 0, 7, 0, 11, 11])));
    var bytes = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0].concat(types, functions, tag, exports, code));
    if (!WebAssembly.validate(bytes)) {
        refuse("Unsupported", "the complete legacy EH instruction set is unavailable");
    }
    var supported = false;
    try {
        var probe = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports;
        var uncaughtPayload = false;
        try {
            probe.z();
        } catch (thrown) {
            uncaughtPayload = thrown instanceof WebAssembly.Exception
                && thrown.is(probe.t) && thrown.getArg(probe.t, 0) === 42;
        }
        var opcodeSupport = probe.a() === 7 && probe.b() === 42 && probe.c() === 42
            && probe.t instanceof WebAssembly.Tag && uncaughtPayload;
        var probeTag = new WebAssembly.Tag({ parameters: ["i32"] });
        var exception = new WebAssembly.Exception(probeTag, [42], { traceStack: true });
        supported = opcodeSupport && exception.is(probeTag) && exception.getArg(probeTag, 0) === 42
            && typeof exception.stack === "string" && exception.stack.length > 0;
    } catch (_) {
        // Only this synthetic capability probe is classified; never artifact
        // compilation/instantiation, owner execution, or unrelated exceptions.
        supported = false;
    }
    if (!supported) {
        refuse("Unsupported", "JS i32 Tag/Exception is/getArg and traceStack are required");
    }
}());
