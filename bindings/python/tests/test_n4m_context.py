from __future__ import annotations

import copy

import pytest
from n4m._context import Context
from pls4all._context import Context as PlsContext


def test_context_num_threads_roundtrip() -> None:
    ctx = Context()
    try:
        ctx.num_threads = 2
        assert ctx.num_threads == 2
    finally:
        ctx.close()


@pytest.mark.parametrize("context_type", [Context, PlsContext])
def test_context_ownership_and_closed_operations(context_type) -> None:
    with context_type() as ctx:
        for copier in (copy.copy, copy.deepcopy):
            with pytest.raises(TypeError, match="Context is not copyable"):
                copier(ctx)
        ctx.num_threads = 2
        assert ctx.num_threads == 2
    ctx.close()  # Idempotent, including after context-manager cleanup.
    assert not ctx.handle.value
    with pytest.raises(RuntimeError, match="Context is closed"):
        _ = ctx.num_threads
    with pytest.raises(RuntimeError, match="Context is closed"):
        ctx.num_threads = 1
    with pytest.raises(RuntimeError, match="Context is closed"):
        ctx.__enter__()


def test_context_seed_and_error_access_after_close() -> None:
    ctx = Context()
    ctx.close()
    with pytest.raises(RuntimeError, match="Context is closed"):
        ctx.set_seed(1)
    ctx = PlsContext()
    ctx.close()
    for name in ("seed", "backend", "last_error"):
        with pytest.raises(RuntimeError, match="Context is closed"):
            getattr(ctx, name)
    with pytest.raises(RuntimeError, match="Context is closed"):
        ctx.clear_error()
