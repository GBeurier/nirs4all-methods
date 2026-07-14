"""Contract and fingerprint tests for OrderedSearchSpaceSpec v1."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import jsonschema
import pytest

_HPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HPO))

from ordered_space_contract import (  # noqa: E402
    ContractError,
    _restricted_jcs_encode,
    c_abi_projection,
    space_fingerprint,
    validate_ordered_search_space,
)

_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE = _HPO / "contracts" / "ordered_search_space.v1.json"
_SCHEMA = _ROOT / "docs" / "contracts" / "ordered_search_space.v1.schema.json"


def _payload() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _golden() -> tuple[dict, str]:
    payload = _payload()
    return payload["contract"], payload["expected_fingerprint"]


def _apply_replacements(spec: dict, replacements: list[dict]) -> None:
    for replacement in replacements:
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in replacement["path"].removeprefix("/").split("/")
        ]
        parent = spec
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        leaf = parts[-1]
        value = copy.deepcopy(replacement["value"])
        if isinstance(parent, list):
            parent[int(leaf)] = value
        else:
            parent[leaf] = value


def test_schema_and_golden_contract_are_published_and_pinned():
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith("/ordered_search_space.v1.schema.json")

    contract, expected = _golden()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(contract, schema)
    validate_ordered_search_space(contract)
    assert space_fingerprint(contract) == expected


def test_axis_and_choice_order_are_fingerprint_semantic():
    contract, expected = _golden()
    reversed_axes = copy.deepcopy(contract)
    reversed_axes["axes"].reverse()
    assert space_fingerprint(reversed_axes) != expected

    reversed_choices = copy.deepcopy(contract)
    reversed_choices["axes"][0]["domain"]["choices"].reverse()
    assert space_fingerprint(reversed_choices) != expected


def test_current_c_abi_projection_is_pinned_without_dropping_host_fingerprint_fields():
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    contract = payload["contract"]

    assert c_abi_projection(contract) == payload["expected_c_abi_projection"]
    assert "target" not in c_abi_projection(contract)["params"][0]

    target_changed = copy.deepcopy(contract)
    target_changed["axes"][0]["target"]["path"] = ["alternate_family"]
    assert c_abi_projection(target_changed) == c_abi_projection(contract)
    assert space_fingerprint(target_changed) != space_fingerprint(contract)


def test_float_choice_condition_labels_match_native_max_digits10_projection():
    contract, _ = _golden()
    float_parent = copy.deepcopy(contract)
    float_parent["axes"][0]["domain"]["choices"] = [
        {"f64_bits": "3ff000001ad7f29b"},
        {"f64_bits": "3ff0000035afe535"},
    ]

    projected = c_abi_projection(float_parent)
    assert projected["constraints"][0]["label_refs"][1] == "1.0000001000000001"
    assert projected["constraints"][1]["label_refs"][1] == "1.0000001999999999"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda spec: spec["axes"][1].update(name="model_family"), "duplicate axis"),
        (lambda spec: spec["axes"][1].update(name="bad#axis"), "reserved '#'"),
        (
            lambda spec: spec["axes"][1]["activation"].update(parent="missing"),
            "unknown parent",
        ),
        (
            lambda spec: spec["axes"][1]["activation"].update(choice_index=9),
            "out of range",
        ),
        (
            lambda spec: spec["axes"][2]["domain"].update(
                low={"f64_bits": "7ff0000000000000"}
            ),
            "finite",
        ),
        (
            lambda spec: spec["axes"][2]["domain"].update(
                step={"f64_bits": "8000000000000000"}
            ),
            "negative zero",
        ),
        (
            lambda spec: spec["axes"][2]["target"].update(path=[]),
            "path must be non-empty",
        ),
        (
            lambda spec: spec["axes"][1]["domain"].update(step={"i64": "0"}),
            "integer step must be positive",
        ),
        (
            lambda spec: spec["axes"][2]["domain"].update(
                step={"f64_bits": "3fb999999999999a"}
            ),
            "log-float step must be zero",
        ),
        (
            lambda spec: (
                spec["axes"][0]["domain"]["choices"][1].clear()
                or spec["axes"][0]["domain"]["choices"][1].update(bool=True)
            ),
            "homogeneous scalar kind",
        ),
        (
            lambda spec: spec["axes"][1]["domain"].update(
                high={"i64": "9007199254740993"}
            ),
            "represented exactly by the current binary64 C ABI",
        ),
        (
            lambda spec: spec["axes"][0]["domain"]["choices"][0].update(
                string="\ud800"
            ),
            "valid Unicode scalar values",
        ),
    ],
)
def test_invalid_contracts_fail_closed(mutate, message):
    contract, _ = _golden()
    invalid = copy.deepcopy(contract)
    mutate(invalid)
    with pytest.raises(ContractError, match=message):
        validate_ordered_search_space(invalid)


@pytest.mark.parametrize(
    "vector",
    _payload()["invalid_contract_vectors"],
    ids=lambda vector: vector["id"],
)
def test_published_invalid_contract_vectors_fail_closed(vector):
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    contract, _ = _golden()
    invalid = copy.deepcopy(contract)
    _apply_replacements(invalid, vector["replacements"])

    with pytest.raises(ContractError, match=vector["expected_error"]):
        validate_ordered_search_space(invalid)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_activation_cycles_and_duplicate_patch_targets_are_rejected():
    contract, _ = _golden()
    cyclic = copy.deepcopy(contract)
    cyclic["axes"][1]["kind"] = "categorical"
    cyclic["axes"][1]["domain"] = {"choices": [{"i64": "1"}]}
    cyclic["axes"][0]["activation"] = {
        "mode": "when",
        "parent": "n_components",
        "choice_index": 0,
    }
    with pytest.raises(ContractError, match="activation cycle"):
        validate_ordered_search_space(cyclic)

    duplicate_target = copy.deepcopy(contract)
    duplicate_target["axes"][2]["target"] = copy.deepcopy(
        duplicate_target["axes"][1]["target"]
    )
    with pytest.raises(ContractError, match="duplicate ParameterPatch target"):
        validate_ordered_search_space(duplicate_target)


def test_constraint_refs_are_typed_ordered_and_known():
    contract, _ = _golden()
    constrained = copy.deepcopy(contract)
    constrained["constraints"] = [
        {
            "kind": "exclude",
            "refs": [
                {"axis": "model_family", "choice_index": 0},
                {"axis": "alpha"},
            ],
        }
    ]
    validate_ordered_search_space(constrained)

    invalid = copy.deepcopy(constrained)
    invalid["constraints"][0]["refs"][0]["choice_index"] = 7
    with pytest.raises(ContractError, match="choice_index is out of range"):
        validate_ordered_search_space(invalid)

    same_axis_distinct_choices = copy.deepcopy(contract)
    same_axis_distinct_choices["constraints"] = [
        {
            "kind": "exclude",
            "refs": [
                {"axis": "model_family", "choice_index": 0},
                {"axis": "model_family", "choice_index": 1},
            ],
        }
    ]
    validate_ordered_search_space(same_axis_distinct_choices)

    exact_duplicate = copy.deepcopy(same_axis_distinct_choices)
    exact_duplicate["constraints"][0]["refs"][1]["choice_index"] = 0
    with pytest.raises(ContractError, match="duplicate refs"):
        validate_ordered_search_space(exact_duplicate)


def test_canonical_bytes_do_not_depend_on_mapping_insertion_order():
    contract, expected = _golden()
    reordered = {
        "constraints": contract["constraints"],
        "axes": contract["axes"],
        "schema_version": contract["schema_version"],
        "schema": contract["schema"],
    }
    assert space_fingerprint(reordered) == expected


def test_jcs_key_order_is_utf16_not_utf8():
    # U+10000 sorts before U+E000 by UTF-16 code units (D800 < E000), while its
    # UTF-8 encoding sorts after it (F0 > EE). This vector catches use of the
    # TCV1/UTF-8 profile in the JCS-owned n4m contract.
    value = {"\ue000": 1, "\U00010000": 2}
    assert _restricted_jcs_encode(value) == '{"\U00010000":2,"\ue000":1}'
