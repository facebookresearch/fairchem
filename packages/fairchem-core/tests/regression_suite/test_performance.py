"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from fairchem.core.regression_suite.evaluation.performance import (
    build_performance_sidecar,
    compare_performance,
)
from fairchem.core.regression_suite.spec.schema import PerformanceMetricSpec


def _sidecar(run_id: str, accelerator: str) -> dict:
    return build_performance_sidecar(
        {"latency": PerformanceMetricSpec("ms/step", "minimize", ("atoms",))},
        [
            {
                "metric": "latency",
                "case_id": f"atoms-{atoms}",
                "transform_id": "identity",
                "cluster_id": f"atoms-{atoms}",
                "coordinates": {"atoms": atoms},
                "value": 0.4 * atoms,
            }
            for atoms in (50, 100)
        ],
        protocol={"name": "steady-state-v1"},
        identity={"run_id": run_id, "suite": "inference", "checkpoint": "uma"},
        environment={"device_type": "cuda", "accelerator_name": accelerator},
    )


def test_a_baseline_from_another_gpu_is_reported_not_compared():
    candidate = _sidecar("candidate", "H100")

    same = compare_performance(candidate, {"FP64": [_sidecar("a", "H100")]})
    result = compare_performance(candidate, {"FP64": [_sidecar("b", "GB300")]})
    assert same["status"] == "COMPARABLE"
    assert result["status"] == "NOT COMPARABLE"
    assert "hardware" in result["reason"]
