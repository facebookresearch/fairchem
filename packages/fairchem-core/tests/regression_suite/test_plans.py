"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest

from fairchem.core.regression_suite.execution.runner import _smoke_cases
from fairchem.core.regression_suite.suites import SUITES, load_suite
from fairchem.core.regression_suite.suites.base import suite_contract_hash

# Baselines record these contract hashes, and Suite.plan() checks each suite's
# plan hash. Changing an observable contract (units, comparison, floor,
# report-only status, histogram) invalidates every saved run of the suite.
CONTRACT_HASHES = {
    "prediction-fidelity": "b5b09ab72c9eb3fddc6d69c869da1cd98668e3559811f26b5cbb2f8d7f260b7a",
    "edge-cases": "0e1166a66b2bf9975635895330bd477665c8d26998643d76493add5fd0a09e6e",
    "inference": "5738df055ea4397e8bec44a4741b573f2e0240024f41bf30777d333e108fc6d9",
    "physical-constraints": "cdb8253023144d62d1734929ef5f3c12bb37492784a297a8e27d7c86fd3e1c17",
    "elastic": "76e8b8cf93a4609e672c6f987262e4b6d99fe078c3b8c2a4ac555d799576fba7",
    "phonon-crystal": "9c970a59140a6d442c570db529568c4c7ddcb6dbef566d26350dd4deb0e5d450",
    "finite-difference-frequencies": "29cf4fbca890c7648dd9485a28c1a76f79aa494075b81987b36445697942812f",
    "autograd-frequencies": "4288a99843eca2d96c947e7ada586f8257ea42cf56122a58b788e925d9d9609e",
    "nve-md": "9235c1b401841f3d66972b7dd74cd9e447f0eab40ca8fb13cfeb61f822fa3ba3",
}


@pytest.mark.parametrize("suite_name", SUITES)
def test_suite_plans_contracts_and_structures_are_frozen(suite_name):
    suite = load_suite(suite_name)
    plan = suite.plan()

    assert suite_contract_hash(suite) == CONTRACT_HASHES[suite_name]
    assert suite.name == plan.suite == suite_name
    sources = {(case.structure, case.structure_index): case for case in plan.cases}
    for case in sources.values():
        assert suite.source_atoms(case).calc is None
    selected, _selection_id = _smoke_cases(suite, plan)
    assert [case.case_id for case in selected] == list(suite.smoke_case_ids)
    assert all(case.transform.kind == "identity" for case in selected)
