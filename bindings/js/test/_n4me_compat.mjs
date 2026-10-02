// Test-only complete native state comparison across the known writer ABI bump.
import assert from "node:assert/strict";

const n4mmState = (api, packet, writerAbi) => {
    const info = api.inspectN4mm(packet); // Native complete decoder/checksum.
    assert.deepEqual(info.writerAbi, writerAbi);
    return [{ ...info, writerAbi: [0, 0, 0] }, packet.subarray(0, 8), packet.subarray(20, -8)];
};

export function decodeN4me(api, input) {
    const packet = Buffer.from(input);
    assert.ok(packet.length >= 28);
    assert.equal(packet.subarray(0, 4).toString("ascii"), "N4ME");
    assert.equal(packet.readUInt32LE(4), 1);
    const writerAbi = [8, 12, 16].map((offset) => packet.readUInt32LE(offset));
    let checksum = 0xcbf29ce484222325n;
    for (const byte of packet.subarray(0, -8)) {
        checksum = ((checksum ^ BigInt(byte)) * 0x100000001b3n) & 0xffffffffffffffffn;
    }
    assert.equal(packet.readBigUInt64LE(packet.length - 8), checksum);
    let offset = 20;
    const take = (size) => {
        assert.ok(Number.isSafeInteger(size) && size >= 0 && size <= packet.length - 8 - offset);
        const value = packet.subarray(offset, offset + size);
        offset += size;
        return value;
    };
    const u32 = () => take(4).readUInt32LE();
    const u64 = () => {
        const value = Number(take(8).readBigUInt64LE());
        assert.ok(Number.isSafeInteger(value));
        return value;
    };
    const text = () => take(u32());
    const methodId = text();
    const parameters = [];
    for (let count = u32(); count > 0; --count) {
        const name = text(), kind = u32(), length = u64();
        parameters.push([name, kind, length, take(length * 8)]);
    }
    const capabilitiesAndDimensions = take(24);
    const blocks = [];
    for (let count = u32(); count > 0; --count) {
        const tag = u32(), block = take(u64());
        if (tag === 0x4d4d344e) {
            blocks.push([tag, n4mmState(api, block, writerAbi)]);
        } else if (tag === 0x31534c43 && [
            "models.classification.pls_lda", "models.classification.pls_logistic",
            "models.sparse.sparse_pls_da",
        ].includes(methodId.toString("utf8"))) {
            assert.ok(block.length >= 16);
            const classes = Number(block.readBigUInt64LE(8));
            assert.ok(Number.isSafeInteger(classes) && classes >= 2 && classes <= 2 ** 20);
            const frame = 16 + classes * 8;
            assert.ok(frame + 16 <= block.length);
            const size = Number(block.readBigUInt64LE(frame));
            const packed = Number(block.readBigUInt64LE(frame + 8));
            assert.ok(Number.isSafeInteger(size) && size >= 28 && size <= 2 ** 30);
            assert.equal(packed, Math.ceil(size / 8));
            const start = frame + 16, end = start + size;
            assert.ok(start + packed * 8 <= block.length);
            blocks.push([tag, block.subarray(0, start),
                n4mmState(api, block.subarray(start, end), writerAbi), block.subarray(end)]);
        } else {
            blocks.push([tag, block]);
        }
    }
    assert.equal(offset, packet.length - 8);
    return { writerAbi, state: { methodId, parameters, capabilitiesAndDimensions, blocks } };
}

export function assertN4meReexportEquivalent(api, original, current, label) {
    const before = decodeN4me(api, original), after = decodeN4me(api, current);
    assert.deepEqual(before.writerAbi, [2, 15, 0], label);
    assert.deepEqual(after.writerAbi, api.abiVersion(), label);
    assert.deepEqual(Buffer.from(original).subarray(0, 8), Buffer.from(current).subarray(0, 8), label);
    // The only excluded bytes are the known writer ABI and its checksums.
    // All framing, parameters, class IDs, padding and learned bodies are exact.
    const oldNative = api.NativeEstimator.fromN4me(original);
    const newNative = api.NativeEstimator.fromN4me(current);
    try {
        assert.equal(oldNative.constructor, newNative.constructor, label);
        assert.equal(oldNative.methodId, newNative.methodId, label);
        assert.deepEqual(before.state, after.state, label);
    } finally {
        oldNative.dispose();
        newNative.dispose();
    }
}
