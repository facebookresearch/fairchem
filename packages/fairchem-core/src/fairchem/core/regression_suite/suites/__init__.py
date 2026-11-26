"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib

from fairchem.core.regression_suite.suites.base import Suite

# Each suite and the class that implements it, in the order campaigns run and
# reports list them. A suite's module is imported when the suite is loaded.
_SUITE_CLASSES = {
    "prediction-fidelity": "prediction_fidelity.suite:PredictionFidelitySuite",
    "edge-cases": "edge_cases.suite:EdgeCasesSuite",
    "inference": "inference.suite:InferenceSuite",
    "physical-constraints": "physical_constraints.suite:PhysicalConstraintsSuite",
    "elastic": "elastic.suite:ElasticSuite",
    "phonon-crystal": "phonon_crystal.suite:PhononCrystalSuite",
    "finite-difference-frequencies": (
        "finite_difference_frequencies.suite:FiniteDifferenceFrequenciesSuite"
    ),
    "autograd-frequencies": "autograd_frequencies.suite:AutogradFrequenciesSuite",
    "nve-md": "nve_md.suite:NVEMDSuite",
}
SUITES = tuple(_SUITE_CLASSES)
STANDARD_SUITES = tuple(suite for suite in SUITES if suite != "nve-md")
CAMPAIGN_SUITES = {
    "standard": STANDARD_SUITES,
    "full": SUITES,
}


def load_suite(name: str) -> Suite:
    """
    Instantiate a suite by its name, written with hyphens or underscores.
    """
    key = name.replace("_", "-")
    if key not in _SUITE_CLASSES:
        raise ValueError(f"unknown suite: {name!r}")
    module, _, class_name = _SUITE_CLASSES[key].partition(":")
    return getattr(importlib.import_module(f"{__name__}.{module}"), class_name)()


__all__ = [
    "CAMPAIGN_SUITES",
    "STANDARD_SUITES",
    "SUITES",
    "Suite",
    "load_suite",
]
