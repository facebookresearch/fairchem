"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests: CPU smoke tests for the public NVT and NPT Hydra examples.
Models: none; the UMA calculator factory is replaced with ASE EMT.
CI: test (core shard) — base CPU job.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytest
from ase.calculators.emt import EMT

from fairchem.core import FAIRChemCalculator
from fairchem.core._cli import get_hydra_config_from_yaml


@pytest.mark.parametrize("ensemble", ["nvt", "npt"])
def test_public_md_config_runs_with_emt(ensemble, monkeypatch, tmp_path):
    """
    Compose and run the checked-in MD examples without a model download.
    """
    monkeypatch.setattr(
        FAIRChemCalculator,
        "from_model_checkpoint",
        staticmethod(lambda **_: EMT()),
    )
    repo_root = Path(__file__).parents[3]
    config_path = repo_root / "configs" / "uma" / "md" / f"{ensemble}.yaml"
    run_dir = tmp_path / ensemble
    cfg = get_hydra_config_from_yaml(
        str(config_path),
        [
            f"job.run_dir={run_dir}",
            "job.device_type=CPU",
            "runner.steps=2",
            "runner.trajectory_interval=1",
            "runner.log_interval=1",
            "runner.checkpoint_interval=null",
            "runner.heartbeat_interval=null",
        ],
    )

    runner = hydra.utils.instantiate(cfg.runner)
    runner.job_config = cfg.job
    runner.run()

    results_dir = Path(cfg.job.metadata.results_dir)
    trajectory = pd.read_parquet(results_dir / "trajectory.parquet")
    assert list(trajectory["step"]) == [0, 1, 2]
    assert np.isfinite(trajectory["energy"]).all()
    assert np.isfinite(trajectory["temperature"]).all()
    assert (results_dir / "init_atoms.extxyz").is_file()
    assert (results_dir / "thermo.log").is_file()
    assert (results_dir / "metadata.json").is_file()

    if ensemble == "npt":
        assert trajectory["stress"].notna().all()
        assert np.isfinite(trajectory["pressure"]).all()
        volumes = [abs(np.linalg.det(np.vstack(cell))) for cell in trajectory["cell"]]
        assert all(volume > 0 for volume in volumes)
