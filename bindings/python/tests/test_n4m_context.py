from __future__ import annotations

from n4m._context import Context


def test_context_num_threads_roundtrip() -> None:
    ctx = Context()
    try:
        ctx.num_threads = 2
        assert ctx.num_threads == 2
    finally:
        ctx.close()
