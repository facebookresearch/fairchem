"""
GPU-only tests for the UMA inference speed benchmark runner.

These tests exercise the same code path as invoking:
    fairchem -c configs/uma/benchmark/uma-speed.yaml +run_dir_root=/tmp/

We reuse a tiny trained checkpoint produced by the existing UMA training
fixtures (`direct_mole_checkpoint`) to keep runtime minimal while still
loading a valid model. We then run the benchmark twice:
  1. Using natoms_list (synthetic FCC carbon crystal)
  2. Using an input_system (a small water cluster xyz file)

Assertions are intentionally light: we just ensure the CLI completes
without error and that a run directory is created.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tests.core.testing_utils import launch_main


@pytest.mark.gpu()
def test_uma_speed_benchmark_natoms_list(conserving_mole_checkpoint):
    """Run the UMA speed benchmark via CLI using natoms_list override.

    We point the model checkpoint mapping to the small test checkpoint
    produced by the direct_mole_checkpoint fixture and restrict to a
    very small natoms_list plus only 1 timing iteration to keep test fast. We also
    explicitly override runner.dataset_name to 'omol' to exercise the configurable
    dataset tag path.
    """

    checkpoint_pt, _ = conserving_mole_checkpoint
    # Override existing key (uma_sm_cons) instead of adding a new one to satisfy structured config.
    model_override = f"runner.model_checkpoints.uma_sm_cons={checkpoint_pt}"
    with tempfile.TemporaryDirectory() as run_root:
        sys_args = [
            "-c",
            "configs/uma/benchmark/uma-speed.yaml",
            f"+run_dir_root={run_root}",
            # shorten runtime dramatically
            "runner.timeiters=1",
            # use a tiny atom count
            "runner.natoms_list=[20]",
            model_override,
            "+runner.dataset_name=omol",
        ]
        launch_main(sys_args)
        # ensure a run directory was created and is non-empty
        entries = list(Path(run_root).glob("*/"))
        assert entries, "Benchmark did not create a run directory"


@pytest.mark.gpu()
def test_uma_speed_benchmark_input_system(conserving_mole_checkpoint, water_xyz_file):
    """Run the UMA speed benchmark using an explicit input_system (water.xyz)."""

    checkpoint_pt, _ = conserving_mole_checkpoint
    # Override existing key (uma_sm_cons) instead of adding a new one.
    model_override = f"runner.model_checkpoints.uma_sm_cons={checkpoint_pt}"

    with tempfile.TemporaryDirectory() as run_root:
        input_system_override = f"+runner.input_system={{water: {water_xyz_file}}}"
        sys_args = [
            "-c",
            "configs/uma/benchmark/uma-speed.yaml",
            f"+run_dir_root={run_root}",
            "runner.timeiters=1",
            input_system_override,
            model_override,
            "+runner.dataset_name=omol",
            "runner.natoms_list=[]",
        ]
        launch_main(sys_args)
        entries = list(Path(run_root).glob("*/"))
        assert entries, "Benchmark did not create a run directory"
