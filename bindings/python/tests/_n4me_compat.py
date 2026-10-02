"""Assert complete old/current native state equality across a writer ABI bump."""

from __future__ import annotations

import struct
from dataclasses import replace

import numpy as np

from n4m import abi_version, roles
from n4m.lowlevel.migration import inspect_n4mm


def _n4mm_state(block: bytes, writer_abi: tuple[int, int, int]) -> tuple:
    """Validate a complete nested model; retain all non-writer bytes."""
    info = inspect_n4mm(block)
    assert info.writer_abi == writer_abi
    return replace(info, writer_abi=(0, 0, 0)), block[:8], block[20:-8]


def _decode(payload: bytes) -> tuple[tuple[int, int, int], dict]:
    assert len(payload) >= 28 and payload[:4] == b"N4ME"
    format_version, major, minor, patch = struct.unpack_from("<4I", payload, 4)
    assert format_version == 1
    checksum = 0xCBF29CE484222325
    for value in payload[:-8]:
        checksum = ((checksum ^ value) * 0x100000001B3) & ((1 << 64) - 1)
    assert struct.unpack_from("<Q", payload, len(payload) - 8)[0] == checksum
    offset = 20

    def take(size: int) -> bytes:
        nonlocal offset
        assert 0 <= size <= len(payload) - 8 - offset
        value = payload[offset : offset + size]
        offset += size
        return value

    def integer(width: int) -> int:
        return int.from_bytes(take(width), "little")

    def text() -> str:
        return take(integer(4)).decode("utf-8")

    method_id = text()
    parameters = []
    for _ in range(integer(4)):
        name, kind, count = text(), integer(4), integer(8)
        # Retain every value's exact encoding, including floating-point bits.
        parameters.append((name, kind, count, take(count * 8)))
    capabilities, features, outputs = integer(8), integer(8), integer(8)
    blocks = []
    for _ in range(integer(4)):
        tag, size = integer(4), integer(8)
        block = take(size)
        if tag == 0x4D4D344E:  # ModelAdapter's complete nested N4MM packet.
            blocks.append((tag, _n4mm_state(block, (major, minor, patch))))
        elif tag == 0x31534C43 and method_id in {
            "models.classification.pls_lda",
            "models.classification.pls_logistic",
            "models.sparse.sparse_pls_da",
        }:
            # CLS1: input width, length-framed class IDs, then the model's
            # byte length and an f64 array packing eight raw bytes per value.
            # Preserve class IDs, framing, padding and every head-state byte.
            assert len(block) >= 16
            classes = struct.unpack_from("<Q", block, 8)[0]
            assert 2 <= classes <= 1 << 20
            model_offset = 16 + classes * 8
            assert model_offset + 16 <= len(block)
            model_size, packed_count = struct.unpack_from("<2Q", block, model_offset)
            assert 28 <= model_size <= 1 << 30
            assert packed_count == (model_size + 7) // 8
            model_start = model_offset + 16
            assert model_start + packed_count * 8 <= len(block)
            model_end = model_start + model_size
            blocks.append(
                (
                    tag,
                    block[:model_start],
                    _n4mm_state(block[model_start:model_end], (major, minor, patch)),
                    block[model_end:],
                )
            )
        else:
            blocks.append((tag, block))
    assert offset == len(payload) - 8
    return (major, minor, patch), {
        "method_id": method_id,
        "parameters": parameters,
        "capabilities": capabilities,
        "n_features_in": features,
        "n_outputs": outputs,
        "learned_blocks": blocks,
    }


def assert_n4me_reexport_equivalent(original: bytes, current: bytes) -> None:
    """Allow only the known writer ABI header and its dependent checksum."""
    old_abi, old_state = _decode(original)
    new_abi, new_state = _decode(current)
    assert old_abi == (2, 15, 0)
    assert new_abi == abi_version()
    assert old_state == new_state
    assert original[:8] == current[:8]
    # The decoded equality covers every outer body field and every learned
    # byte; recognized nested N4MM headers/checksums receive the same strictly
    # bounded writer-ABI exception. All other blocks remain wholly byte-equal.
    # Both complete packets must also pass the actual native decoder, which
    # validates parameters and every learned block against their method.
    before = roles.NativeEstimator.from_n4me(original)
    after = roles.NativeEstimator.from_n4me(current)
    assert type(before) is type(after)
    np.testing.assert_equal(before.get_params(), after.get_params())
