# SPDX-License-Identifier: CECILL-2.1
"""Native build evidence is complete, closed and independent of GPU visibility."""

from __future__ import annotations

import ctypes
import itertools
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import n4m
from n4m import _build


def native_string(monkeypatch, text):
    buffer = ctypes.create_string_buffer(text)
    monkeypatch.setattr(_build, "lib", SimpleNamespace(
        n4m_get_build_info=lambda: ctypes.addressof(buffer)
    ))
    return buffer


@pytest.mark.parametrize("flags", list(itertools.product((False, True), repeat=3)))
def test_complete_compile_profiles_do_not_infer_runtime_device_availability(flags, monkeypatch):
    blas, openmp, cuda = flags
    text = ("descriptive compiler metadata\n"
            f"n4m-build-capabilities-v1;blas={int(blas)};openmp={int(openmp)};cuda={int(cuda)}")
    native_string(monkeypatch, text.encode())
    capabilities = n4m.build_capabilities()
    assert capabilities == n4m.BuildCapabilities(1, blas, openmp, cuda)
    assert capabilities.sequential_cpu is (not any(flags))
    assert n4m.build_info() == text
    with pytest.raises(FrozenInstanceError):
        capabilities.cuda = False


@pytest.mark.parametrize("text", [
    "", "old unstamped build", "blas=0;openmp=0;cuda=0",
    "n4m-build-capabilities-v2;blas=0;openmp=0;cuda=0",
    "n4m-build-capabilities-v1;blas=0;openmp=0",
    "n4m-build-capabilities-v1;blas=0;openmp=0;cuda=false",
    "n4m-build-capabilities-v1;blas=0;openmp=0;cuda=0;unknown=0",
    "n4m-build-capabilities-v1;blas=0;openmp=0;cuda=0;cuda=1",
    "n4m-build-capabilities-v1;blas=0;cuda=0;openmp=0",
    "n4m-build-capabilities-v1;blas=0;openmp=0;cuda=0 ",
    "n4m-build-capabilities-v1;blas=0;openmp=0;cuda=0\n"
    "n4m-build-capabilities-v1;blas=0;openmp=0;cuda=1",
    "n4m-build-capabilities-v1;blas=0;openmp=0;cuda=0\n"
    "n4m-build-capabilities-v2;blas=0;openmp=0;cuda=0",
])
def test_missing_ambiguous_or_unrecognized_profile_is_refused(text, monkeypatch):
    native_string(monkeypatch, text.encode())
    with pytest.raises(RuntimeError, match="complete supported build capability stamp"):
        n4m.build_capabilities()


def test_null_build_info_is_refused(monkeypatch):
    monkeypatch.setattr(_build, "lib", SimpleNamespace(n4m_get_build_info=lambda: None))
    with pytest.raises(RuntimeError, match="null build information"):
        n4m.build_capabilities()


def test_invalid_native_utf8_is_refused(monkeypatch):
    native_string(monkeypatch, b"\xff")
    with pytest.raises(RuntimeError, match="not UTF-8"):
        n4m.build_capabilities()


def test_actual_installed_native_profile_is_stamped():
    capabilities = n4m.build_capabilities()
    assert capabilities.schema_version == 1
    assert all(type(value) is bool for value in (
        capabilities.blas, capabilities.openmp, capabilities.cuda
    ))
    assert "n4m-build-capabilities-v1;" in n4m.build_info()
