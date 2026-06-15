"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
from tests.core.testing_utils import launch_main

pytestmark = [pytest.mark.pretrained("uma-s-1p2")]

COMMON_ARGS = [
    "job.device_type=CPU",
    "runner.timeiters=1",
    "runner.repeats=1",
    "runner.device=cpu",
    "runner.inference_settings.compile=False",
    "runner.inference_settings.execution_mode=general",
]


def _checkpoint_overrides(pretrained_checkpoint: str) -> list[str]:
    checkpoint_path = (
        pretrained_checkpoint
        if Path(pretrained_checkpoint).exists()
        else pretrained_checkpoint_path_from_name(pretrained_checkpoint)
    )
    return [
        "~uma_s_1p2",
        f"runner.model_checkpoints.uma_s_1p2={checkpoint_path}",
    ]


def test_uma_speed_benchmark_natoms_list(pretrained_checkpoint):
    """Run the UMA speed benchmark via CLI using natoms_list override."""
    with tempfile.TemporaryDirectory() as run_root:
        sys_args = [
            "-c",
            "configs/uma/speed/uma-speed.yaml",
            *COMMON_ARGS,
            *_checkpoint_overrides(pretrained_checkpoint),
            f"job.run_dir={run_root}",
            "runner.natoms_list=[20]",
        ]
        launch_main(sys_args)
        entries = list(Path(run_root).glob("*/"))
        assert entries, "Benchmark did not create a run directory"


def test_uma_speed_benchmark_input_system(water_xyz_file, pretrained_checkpoint):
    """Run the UMA speed benchmark using an explicit input_system (water.xyz)."""
    with tempfile.TemporaryDirectory() as run_root:
        input_system_override = f"+runner.input_system={{water: {water_xyz_file}}}"
        sys_args = [
            "-c",
            "configs/uma/speed/uma-speed.yaml",
            *COMMON_ARGS,
            *_checkpoint_overrides(pretrained_checkpoint),
            f"job.run_dir={run_root}",
            input_system_override,
            "runner.natoms_list=null",
        ]
        launch_main(sys_args)
        entries = list(Path(run_root).glob("*/"))
        assert entries, "Benchmark did not create a run directory"
