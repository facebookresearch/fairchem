"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  Strict LAMMPS callback parity and the distributed predictor path.
Models: uma-s-1p1.
CI:     test_lammps_gpu; the public GPU runner has one T4, so the
        two-worker distributed case uses CPU/Gloo while exercising Ray and GP.
        The NCCL test runs conditionally on hosts exposing at least two GPUs.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("lammps")
pytest.importorskip("ray")

from fairchem.lammps.preflight import (  # noqa: E402
    BRIDGE_ATOL,
    BenchmarkCase,
    execute_case,
    generate_fcc_system,
    resolve_model_path,
)


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p1")
def test_gpu_preflight_has_strict_bridge_parity(pretrained_checkpoint):
    atoms = generate_fcc_system(8)
    atoms.set_atomic_numbers([6, 8, 6, 8, 6, 8, 6, 8])
    result = execute_case(
        atoms=atoms,
        task="omat",
        charge=0,
        spin=0,
        model_path=resolve_model_path(pretrained_checkpoint),
        case=BenchmarkCase("fp32_eager", graph_version=2, workers=1),
        timestep_fs=0.5,
        dynamics_steps=2,
    )

    assert result["status"] == "passed", result.get("error")
    assert result["bridge_energy_error_eV"] <= BRIDGE_ATOL
    assert result["bridge_force_max_error_eV_per_A"] <= BRIDGE_ATOL
    assert result["atom_mapping_passed"] is True
    assert result["direct_reference_passed"] is True


@pytest.mark.serial()
@pytest.mark.pretrained("uma-s-1p1")
def test_two_worker_predictor_runs_through_lammps(pretrained_checkpoint):
    result = execute_case(
        atoms=generate_fcc_system(4),
        task="omat",
        charge=0,
        spin=0,
        model_path=resolve_model_path(pretrained_checkpoint),
        case=BenchmarkCase("fp32_eager", graph_version=2, workers=2),
        device="cpu",
        timestep_fs=0.5,
        dynamics_steps=1,
    )

    assert result["status"] == "passed", result.get("error")
    assert result["bridge_passed"] is True
    assert result["direct_reference_passed"] is True


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p1")
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires two visible GPUs for graph-parallel NCCL coverage.",
)
def test_two_gpu_predictor_runs_through_lammps(pretrained_checkpoint):
    result = execute_case(
        atoms=generate_fcc_system(32),
        task="omat",
        charge=0,
        spin=0,
        model_path=resolve_model_path(pretrained_checkpoint),
        case=BenchmarkCase("fp32_eager", graph_version=2, workers=2),
        timestep_fs=0.5,
        dynamics_steps=1,
    )

    assert result["status"] == "passed", result.get("error")
    assert result["bridge_passed"] is True
    assert result["direct_reference_passed"] is True
