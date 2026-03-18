"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tests.core.testing_utils import launch_main

COMMON_ARGS = [
    "job.device_type=CPU",
    "runner.timeiters=1",
    "runner.repeats=1",
    "runner.device=cpu",
    "runner.inference_settings.compile=False",
    "runner.inference_settings.execution_mode=general",
]


def test_uma_speed_benchmark_natoms_list():
    """Run the UMA speed benchmark via CLI using natoms_list override."""
    with tempfile.TemporaryDirectory() as run_root:
        sys_args = [
            "-c",
            "configs/uma/speed/uma-speed.yaml",
            *COMMON_ARGS,
            f"job.run_dir={run_root}",
            "runner.natoms_list=[20]",
        ]
        launch_main(sys_args)
        entries = list(Path(run_root).glob("*/"))
        assert entries, "Benchmark did not create a run directory"


def test_uma_speed_benchmark_input_system(water_xyz_file):
    """Run the UMA speed benchmark using an explicit input_system (water.xyz)."""
    with tempfile.TemporaryDirectory() as run_root:
        input_system_override = f"+runner.input_system={{water: {water_xyz_file}}}"
        sys_args = [
            "-c",
            "configs/uma/speed/uma-speed.yaml",
            *COMMON_ARGS,
            f"job.run_dir={run_root}",
            input_system_override,
            "runner.natoms_list=null",
        ]
        launch_main(sys_args)
        entries = list(Path(run_root).glob("*/"))
        assert entries, "Benchmark did not create a run directory"
