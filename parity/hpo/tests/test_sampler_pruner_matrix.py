# SPDX-License-Identifier: CECILL-2.1
"""Exhaustive native conformance for all 9 sampler x 5 pruner cells."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

_HPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HPO))

from compatibility import (  # noqa: E402
    CONTRACT_PATH,
    PRUNERS,
    SAMPLERS,
    SCHEMA_PATH,
    CompatibilityContractError,
    execute_cell,
    execute_refusal_vector,
    load_contract,
    validate_contract,
)

CONTRACT = load_contract()


def _id(value: dict) -> str:
    return value.get("id", f"{value.get('sampler')}_{value.get('pruner')}")


def test_matrix_contract_is_local_json_schema_2020_12():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert document["$schema"] == SCHEMA_PATH.name
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)


def test_matrix_contract_is_exact_cartesian_product():
    pairs = [(cell["sampler"], cell["pruner"]) for cell in CONTRACT["cells"]]
    assert len(pairs) == len(set(pairs)) == 45
    assert set(pairs) == {
        (sampler, pruner) for sampler in SAMPLERS for pruner in PRUNERS
    }
    assert all(cell["creation_status"] == "N4M_OK" for cell in CONTRACT["cells"])
    assert all(cell["status"] == "supported" for cell in CONTRACT["cells"])


def test_matrix_contract_fails_closed_on_missing_or_duplicate_cell():
    missing = copy.deepcopy(CONTRACT)
    missing["cells"].pop()
    with pytest.raises(CompatibilityContractError, match="missing|45"):
        validate_contract(missing)

    duplicate = copy.deepcopy(CONTRACT)
    duplicate["cells"][-1] = copy.deepcopy(duplicate["cells"][0])
    with pytest.raises(CompatibilityContractError, match="duplicate"):
        validate_contract(duplicate)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.__setitem__("schema_version", True),
        lambda document: document["canonical_tape"]["space"][0]["args"].__setitem__(
            0, True
        ),
        lambda document: document["samplers"][4]["adaptive_transition"].__setitem__(
            "minimum_completed", True
        ),
        lambda document: document["pruners"][0].__setitem__("max_in_flight", True),
    ],
)
def test_matrix_semantics_reject_boolean_integer_aliases(mutation):
    malformed = copy.deepcopy(CONTRACT)
    mutation(malformed)
    with pytest.raises(CompatibilityContractError):
        validate_contract(malformed)


@pytest.mark.skipif(
    "N4M_LIB_PATH" not in os.environ,
    reason="no libn4m build (set N4M_LIB_PATH)",
)
@pytest.mark.parametrize("cell", CONTRACT["cells"], ids=_id)
def test_each_sampler_pruner_cell_replays_resumes_and_terminalizes(cell):
    report = execute_cell(CONTRACT, cell)
    assert report.sampler == cell["sampler"]
    assert report.pruner == cell["pruner"]
    assert report.n_trials == CONTRACT["canonical_tape"]["n_trials"]
    assert report.n_intermediates >= report.n_trials
    if cell["pruner"] == "none":
        assert report.n_pruned == 0
    else:
        assert report.n_pruned > 0


@pytest.mark.skipif(
    "N4M_LIB_PATH" not in os.environ,
    reason="no libn4m build (set N4M_LIB_PATH)",
)
@pytest.mark.parametrize("vector", CONTRACT["refusal_vectors"], ids=_id)
def test_each_transverse_refusal_returns_stable_status_not_message(vector):
    assert execute_refusal_vector(vector) == len(vector["applies_to"])
