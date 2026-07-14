"""Guard the documented nirs4all/n4m capability boundary.

The native bindings may expose kernels and single-estimator selection traces,
but conformal guarantees, robustness reports and workspace artifacts are owned
by the higher-level nirs4all lifecycle. This doc test keeps that boundary
explicit in the integration map and in the Python bindings README.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_MAP = REPO_ROOT / "docs" / "nirs4all_integration_map.md"
PYTHON_README = REPO_ROOT / "bindings" / "python" / "README.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains_normalized(text: str, phrase: str) -> bool:
    return " ".join(phrase.split()) in " ".join(text.split())


def test_integration_map_documents_native_python_boundary() -> None:
    text = _read(INTEGRATION_MAP)

    required_phrases = (
        "Native tuning / conformal / robustness capability boundary",
        "`n4m.model_selection.finetune_estimator(...)`",
        "`nirs4all.run(tuning=...)`",
        "`nirs4all.calibrate()`",
        "`predict_calibrated()`",
        "`CalibratedRunResult`",
        "`nirs4all.robustness()`",
        "`PredictResult.robustness()`",
        "`RobustnessReport`",
        "Thin translation layers over the C ABI",
        "must not reimplement tuning/conformal/robustness",
        "consume the `nirs4all` artifact/schema surfaces",
    )

    missing = [
        phrase
        for phrase in required_phrases
        if not _contains_normalized(text, phrase)
    ]
    assert not missing


def test_python_bindings_readme_documents_no_conformal_robustness_ownership() -> None:
    text = _read(PYTHON_README)

    required_phrases = (
        "Pure-native estimator selection in the full package",
        "does **not** fit a final estimator on all rows",
        "Relation to nirs4all conformal learning and robustness",
        "thin translation layers over the `libn4m` C ABI",
        "`nirs4all.run(tuning=...)`",
        "`nirs4all.calibrate()`",
        "`predict_calibrated()`",
        "`CalibratedRunResult`",
        "`nirs4all.robustness()`",
        "`RobustnessReport`",
        "must not reimplement tuning/conformal/robustness guarantees",
        "consume\n"
        "the public `nirs4all` artifact/schema surfaces",
        "`finetune_estimator(...)`",
    )

    missing = [
        phrase
        for phrase in required_phrases
        if not _contains_normalized(text, phrase)
    ]
    assert not missing
