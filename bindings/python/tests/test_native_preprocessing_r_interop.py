# SPDX-License-Identifier: CECILL-2.1
"""Optional bidirectional N4MP held-out replay with the installed n4m R binding."""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest
from n4m.compose import NativePreprocessingPipeline

_RSCRIPT = os.environ.get("N4M_RSCRIPT") or shutil.which("Rscript")
_R_LIBRARY = os.environ.get("N4M_R_TEST_LIB")
pytestmark = pytest.mark.skipif(
    not _RSCRIPT or not _R_LIBRARY,
    reason="set N4M_RSCRIPT and N4M_R_TEST_LIB for Python/R N4MP interop",
)

_TRAIN = np.array([
    [0.2, 1.2, 3.0, 4.5], [0.6, 1.4, 2.6, 4.8],
    [1.1, 2.0, 3.3, 4.2], [1.6, 2.4, 3.8, 4.0],
    [2.2, 2.8, 4.1, 5.2], [2.8, 3.1, 4.7, 5.7],
])
_HELDOUT = np.array([[0.9, 1.8, 2.8, 4.6], [2.5, 3.0, 4.5, 5.4]])


def _r_matrix(name: str, value: np.ndarray) -> str:
    numbers = ",".join(format(float(x), ".17g") for x in value.ravel())
    return f"{name} <- matrix(c({numbers}), nrow={value.shape[0]}, byrow=TRUE)\n"


def _r_run(body: str, *args: str) -> list[str]:
    script = (
        "suppressPackageStartupMessages(library(n4m))\n"
        + _r_matrix("train", _TRAIN)
        + _r_matrix("held", _HELDOUT)
        + body
    )
    env = os.environ.copy()
    env["R_LIBS_USER"] = _R_LIBRARY
    completed = subprocess.run(
        [_RSCRIPT, "--vanilla", "-e", script, *args],
        check=True, capture_output=True, text=True, env=env, timeout=60,
    )
    return completed.stdout.strip().splitlines()


def _prediction(line: str) -> np.ndarray:
    return np.fromstring(line, sep=",").reshape(_HELDOUT.shape)


def test_r_export_python_import_and_python_export_r_import():
    r_lines = _r_run(
        "fit <- n4m_preprocess_fit(train, list(n4m_preprocess_step('center'), "
        "n4m_preprocess_step('autoscale')))\n"
        "cat(paste(format(n4m_preprocess_export(fit)), collapse=''), '\\n', sep='')\n"
        "cat(paste(sprintf('%.17g', as.vector(t(n4m_preprocess_transform(fit, held)))), "
        "collapse=','), '\\n', sep='')\n"
    )
    assert len(r_lines) == 2
    with NativePreprocessingPipeline.from_bytes(
        bytes.fromhex(r_lines[0]), expected_operators=["center", "autoscale"]
    ) as imported:
        np.testing.assert_allclose(imported.transform(_HELDOUT),
                                   _prediction(r_lines[1]), rtol=0, atol=1e-12)
        assert [step.kind for step in imported.steps_] == ["center", "autoscale"]

    with NativePreprocessingPipeline(["center", "autoscale"]) as fitted:
        fitted.fit(_TRAIN)
        expected = fitted.transform(_HELDOUT)
        blob = fitted.to_bytes()
    py_lines = _r_run(
        "hex <- commandArgs(trailingOnly=TRUE)[1L]\n"
        "bytes <- as.raw(strtoi(substring(hex, seq(1, nchar(hex), 2L), "
        "seq(2, nchar(hex), 2L)), 16L))\n"
        "fit <- n4m_preprocess_import(bytes)\n"
        "cat(paste(sprintf('%.17g', as.vector(t(n4m_preprocess_transform(fit, held)))), "
        "collapse=','), '\\n', sep='')\n"
        "cat(paste(vapply(n4m_preprocess_plan(fit), function(step) step$kind, ''), "
        "collapse=','), '\\n', sep='')\n",
        blob.hex(),
    )
    assert len(py_lines) == 2
    np.testing.assert_allclose(_prediction(py_lines[0]), expected, rtol=0, atol=1e-12)
    assert py_lines[1] == "center,autoscale"
