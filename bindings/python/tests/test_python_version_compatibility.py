"""Version-compatible native context annotations preserve the acquired subtype."""

from typing import get_type_hints

from n4m.roles._base import Self, _Context


def test_native_context_preserves_subtype_and_resolves_annotations():
    class ContextSubtype(_Context):
        pass

    owner = ContextSubtype()
    with owner as acquired:
        assert acquired is owner
        assert isinstance(acquired, ContextSubtype)
        assert acquired.handle.value
    hints = get_type_hints(_Context.__enter__)
    assert hints["return"] is Self
