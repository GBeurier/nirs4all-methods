// SPDX-License-Identifier: CECILL-2.1

import assert from "node:assert/strict";
import {
    Algorithm,
    SERIALIZED_MODEL_CAPABILITY_AFFINE,
    SERIALIZED_MODEL_CAPABILITY_PREDICT,
    inspectN4mm,
    loadModule,
} from "../dist/index.js";

class Writer {
    bytes = [];
    raw(values) { this.bytes.push(...values); }
    u32(value) {
        this.raw([value, value >>> 8, value >>> 16, value >>> 24].map((x) => x & 0xff));
    }
    u64(value) {
        let current = BigInt(value);
        for (let i = 0; i < 8; ++i) {
            this.bytes.push(Number(current & 0xffn));
            current >>= 8n;
        }
    }
    f64(value) {
        const raw = new Uint8Array(8);
        new DataView(raw.buffer).setFloat64(0, value, true);
        this.raw(raw);
    }
    vector(values) {
        this.u64(values.length);
        for (const value of values) this.f64(value);
    }
    finish() { return Uint8Array.from(this.bytes); }
}

function fnv1a64(bytes) {
    let hash = 14695981039346656037n;
    for (const byte of bytes) {
        hash ^= BigInt(byte);
        hash = BigInt.asUintN(64, hash * 1099511628211n);
    }
    return hash;
}

function importedLinearN4mm() {
    const writer = new Writer();
    writer.raw(new TextEncoder().encode("N4MM"));
    writer.u32(1);             // wire format
    writer.u32(2);             // writer ABI 2.4.0
    writer.u32(4);
    writer.u32(0);
    writer.u32(11);            // imported-linear algorithm
    writer.u32(0);             // NIPALS sentinel recipe
    writer.u32(0);             // regression deflation sentinel
    writer.u64(17);            // training samples
    writer.u64(2);             // features
    writer.u64(2);             // targets
    writer.u64(0);             // latent components
    for (let i = 0; i < 5; ++i) writer.u32(0);
    writer.f64(1e-6);
    writer.u32(1);
    writer.vector([0, 0]);
    writer.vector([1, 1]);
    writer.vector([1.5, -2]);
    writer.vector([1, 1]);
    writer.vector([2, 0.5, -1, 3]);
    for (let i = 0; i < 6; ++i) writer.vector([]);
    const withoutChecksum = writer.finish();
    writer.u64(fnv1a64(withoutChecksum));
    return writer.finish();
}

await loadModule();
const payload = importedLinearN4mm();
const info = inspectN4mm(payload);
assert.equal(info.schemaVersion, 1);
assert.equal(info.formatVersion, 1);
assert.deepEqual(info.writerAbi, [2, 4, 0]);
assert.equal(info.algorithm, Algorithm.IMPORTED_LINEAR_PREDICTOR);
assert.equal(info.solver, 0);
assert.equal(info.deflation, 0);
assert.equal(info.trainingSamples, 17n);
assert.equal(info.nFeatures, 2);
assert.equal(info.nTargets, 2);
assert.equal(info.nComponents, 0);
assert.equal(
    info.capabilities,
    SERIALIZED_MODEL_CAPABILITY_PREDICT | SERIALIZED_MODEL_CAPABILITY_AFFINE,
);

const corrupt = payload.slice();
corrupt[80] ^= 1;
assert.throws(
    () => inspectN4mm(corrupt),
    (error) => error !== null && typeof error === "object" && error.status === 14,
);

console.log("JS N4MM authoritative inspection smoke OK");
