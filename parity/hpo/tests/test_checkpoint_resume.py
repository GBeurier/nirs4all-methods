# SPDX-License-Identifier: CECILL-2.1
"""Golden parity for the portable optimizer checkpoint continuation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

if "N4M_LIB_PATH" not in os.environ:
    pytest.skip("no libn4m build (set N4M_LIB_PATH)", allow_module_level=True)

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "bindings" / "python" / "src"))

from parity.hpo.checkpoint_resume import generate_resume_pack  # noqa: E402


def test_checkpoint_resume_matches_frozen_random_and_tpe_tapes():
    golden_path = _ROOT / "parity" / "hpo" / "golden" / "checkpoint_resume.v1.json"
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    actual = generate_resume_pack()
    assert actual["format"] == expected["format"]
    assert len(actual["tapes"]) == len(expected["tapes"]) == 2
    for actual_tape, expected_tape in zip(
        actual["tapes"], expected["tapes"], strict=True
    ):
        assert actual_tape.keys() == expected_tape.keys()
        for key in ("sampler", "seed", "checkpoint_after"):
            assert actual_tape[key] == expected_tape[key]
        for actual_events, expected_events in (
            (actual_tape["prefix"], expected_tape["prefix"]),
            (actual_tape["continuation"], expected_tape["continuation"]),
        ):
            assert len(actual_events) == len(expected_events)
            for actual_event, expected_event in zip(
                actual_events, expected_events, strict=True
            ):
                assert actual_event["id"] == expected_event["id"]
                assert actual_event["k"] == expected_event["k"]
                assert actual_event["x"] == pytest.approx(
                    expected_event["x"], rel=0.0, abs=1e-15
                )
                assert actual_event["score"] == pytest.approx(
                    expected_event["score"], rel=0.0, abs=1e-14
                )
