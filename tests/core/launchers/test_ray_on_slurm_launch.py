"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

from fairchem.core.launchers import ray_on_slurm_launch


def test_ray_environment_report_passes_config_metadata(monkeypatch) -> None:
    report_arguments = []
    monkeypatch.setattr(
        ray_on_slurm_launch,
        "write_environment_report",
        lambda **kwargs: report_arguments.append(kwargs),
    )
    job_config = SimpleNamespace(
        timestamp_id="timestamp",
        metadata=SimpleNamespace(log_dir="logs", commit="abc123"),
    )

    ray_on_slurm_launch.write_ray_environment_report(job_config)

    assert report_arguments == [
        {
            "log_dir": "logs",
            "run_type": "run",
            "timestamp_id": "timestamp",
            "submission_commit": "abc123",
        }
    ]
