"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fairchem.core.regression_suite.artifacts.baselines import BaselineUnavailableError
from fairchem.core.regression_suite.spec.profiles import (
    build_inference_settings,
    json_settings,
)
from fairchem.core.regression_suite.spec.schema import (
    RegressionSchemaError,
    SuitePlan,
)
from fairchem.core.regression_suite.suites import load_suite
from fairchem.core.regression_suite.suites.base import suite_contract_hash

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from omegaconf import DictConfig

    from fairchem.core.regression_suite.suites.base import Suite
    from fairchem.core.units.mlip_unit import InferenceSettings


def validate_identity(
    identity: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """
    Reject a run whose recorded provenance differs from what its use requires.
    """
    mismatches = sorted(
        key for key, value in expected.items() if identity.get(key) != value
    )
    if mismatches:
        raise RegressionSchemaError(f"{label} identity differs at: {mismatches}")


def run_scope(identity: Mapping[str, Any]) -> str:
    """
    Return whether a run covers its suite's ``smoke`` cases or the ``full``
    plan.

    Every new run records its scope. Full runs recorded before that, such as
    those sealed in the published uma-s-1p2p1 baseline, have none, so a
    missing scope means ``full``.
    """
    return str(identity.get("scope", "full"))


def expected_identity(
    suite: Suite,
    plan: SuitePlan,
    settings: InferenceSettings,
    profile: DictConfig,
) -> dict[str, Any]:
    """
    Return the plan, contract, and settings fields a suite's run records.
    """
    return {
        "plan_hash": plan.plan_hash,
        "suite_contract_hash": suite_contract_hash(suite),
        "case_count": len(plan.cases),
        "inference_settings": json_settings(settings),
        "predictor_scope": str(profile.get("predictor_scope", "suite")),
    }


def suite_expectation(suite_name: str, profile: DictConfig) -> dict[str, Any]:
    """
    Describe the plan, contract, and settings that a suite's runs record.
    """
    suite = load_suite(suite_name)
    plan = suite.plan()
    settings = build_inference_settings(profile, suite.inference_settings_overrides)
    return expected_identity(suite, plan, settings, profile)


def check_baseline_plans(
    baseline: Mapping[str, Any], suites: Iterable[str], profile: DictConfig
) -> None:
    """
    Require every suite's plan and observable contract to match the baseline's.

    Raises:
        BaselineUnavailableError: Naming each mismatched suite and the commit
            its baseline entry was built at.
    """
    mismatches = []
    for suite in suites:
        expectation = suite_expectation(suite, profile)
        entry = baseline["suites"][suite]
        fields = [
            field
            for field in ("plan_hash", "suite_contract_hash")
            if expectation[field] != entry[field]
        ]
        if fields:
            mismatches.append(
                f"  {suite} (baseline commit {entry['git_commit']}): "
                f"{', '.join(fields)} changed"
            )
    if mismatches:
        raise BaselineUnavailableError(
            "this code's suite plans do not match the baseline:\n"
            + "\n".join(mismatches)
            + "\nPin a baseline revision built from matching code "
            "(--baseline hf:...@<revision>) or seal a new baseline at this commit."
        )
