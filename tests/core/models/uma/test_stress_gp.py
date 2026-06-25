#!/usr/bin/env python
"""
Multi-GPU graph-parallel stress correctness test.

Validates that compute_forces_and_stress produces identical results
with and without graph parallelism. The test catches the double-reduction
bug where pos_virial was computed from already all-reduced gradients,
then re-reduced inside reduce_node_to_system.

Can be run in two modes:

1. CPU pytest (gloo backend, any machine):
    pytest tests/core/models/uma/test_stress_gp.py -v

2. Multi-GPU via torchrun (NCCL backend, requires GPUs):
    # Generate reference (1 GPU, no GP):
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
        tests/core/models/uma/test_stress_gp.py --save-ref /tmp/stress_ref.pt

    # Test with N GPUs:
    torchrun --nproc_per_node=N --master_port=29500 \
        tests/core/models/uma/test_stress_gp.py --ref-file /tmp/stress_ref.pt

    # Or use the runner script:
    bash tests/core/models/uma/run_stress_gp_test.sh
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from ase import Atoms

from fairchem.core.common import gp_utils
from fairchem.core.common.gp_utils import setup_gp
from fairchem.core.models.uma.outputs import (
    compute_energy,
    compute_forces_and_stress,
    reduce_node_to_system,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_water_box(num_molecules: int = 10, box_length: float = 12.0,
                   seed: int = 42) -> Atoms:
    """Create a periodic box of water molecules."""
    rng = np.random.RandomState(seed)
    positions, symbols = [], []
    for _ in range(num_molecules):
        center = rng.uniform(0, box_length, size=3)
        h1 = center + np.array([0.96, 0.0, 0.0])
        h2 = center + np.array([-0.24, 0.93, 0.0])
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        rot = np.array([
            [cos_t * cos_p, -sin_t, cos_t * sin_p],
            [sin_t * cos_p, cos_t, sin_t * sin_p],
            [-sin_p, 0, cos_p],
        ])
        for p in [center, h1, h2]:
            positions.append(center + rot @ (p - center))
            symbols.append("O" if len(symbols) % 3 == 0 else "H")
    return Atoms(symbols=symbols, positions=np.array(positions),
                 cell=[box_length] * 3, pbc=True)


class ToyEnergyModel(nn.Module):
    """Minimal model: per-atom MLP energy from positions → stress via autograd.

    This isolates the outputs.py stress computation from the full backbone,
    making the test focused on the GP reduction bug.
    """
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pos, cell, batch, num_systems):
        """Returns (energy, energy_part, forces, stress)."""
        node_energy = self.mlp(pos).squeeze(-1)  # [N]

        # reduce_node_to_system handles GP all-reduce internally
        energy, energy_part = reduce_node_to_system(node_energy, batch, num_systems)

        forces, stress = compute_forces_and_stress(
            energy_part, pos, cell, batch,
            training=self.training,
        )
        return energy, energy_part, forces, stress


# ---------------------------------------------------------------------------
# CPU pytest tests (gloo backend via spawn_multi_process)
# ---------------------------------------------------------------------------

def _stress_gp_worker(world_size: int, natoms: int, seed: int = 42):
    """Worker function run on each gloo rank."""
    rank = dist.get_rank()

    # Build identical model on every rank
    torch.manual_seed(seed)
    model = ToyEnergyModel(hidden=32)
    model.eval()

    # Build identical data
    atoms = make_water_box(num_molecules=max(natoms // 3, 1), seed=seed)
    natoms_actual = len(atoms)

    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32,
                       requires_grad=True)
    cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32).unsqueeze(0)
    cell.requires_grad_(True)
    batch = torch.zeros(natoms_actual, dtype=torch.long)

    with torch.no_grad():
        _, _, forces, stress = model(pos, cell, batch, num_systems=1)

    return {
        "forces": forces.detach(),
        "stress": stress.detach(),
        "rank": rank,
    }


def _run_gp_stress_test(world_size: int, natoms: int = 30):
    """Run GP stress test and compare against 1-GPU reference."""
    from fairchem.core.common.test_utils import (
        PGConfig,
        init_pg_and_rank_and_launch_test,
        spawn_multi_process,
    )

    # 1-GPU reference (no GP)
    ref_config = PGConfig(backend="gloo", world_size=1, gp_group_size=1, use_gp=False)
    ref_results = spawn_multi_process(
        ref_config, _stress_gp_worker,
        init_pg_and_rank_and_launch_test, 1, natoms,
    )
    ref = ref_results[0]

    # N-GPU with GP
    gp_config = PGConfig(backend="gloo", world_size=world_size,
                         gp_group_size=world_size, use_gp=True)
    gp_results = spawn_multi_process(
        gp_config, _stress_gp_worker,
        init_pg_and_rank_and_launch_test, world_size, natoms,
    )

    # All ranks should produce the same result; check rank 0
    gp = gp_results[0]

    forces_match = torch.allclose(ref["forces"], gp["forces"], atol=1e-4, rtol=1e-3)
    stress_match = torch.allclose(ref["stress"], gp["stress"], atol=1e-4, rtol=1e-3)

    forces_diff = (ref["forces"] - gp["forces"]).abs().max().item()
    stress_diff = (ref["stress"] - gp["stress"]).abs().max().item()

    return {
        "forces_match": forces_match,
        "stress_match": stress_match,
        "forces_max_diff": forces_diff,
        "stress_max_diff": stress_diff,
    }


class TestStressGP:
    """Pytest tests for GP stress correctness (CPU, gloo backend)."""

    @pytest.mark.parametrize("world_size", [2, 3, 4])
    def test_stress_gp_small(self, world_size):
        """Test stress matches 1-GPU reference on a small system."""
        result = _run_gp_stress_test(world_size, natoms=30)
        assert result["forces_match"], \
            f"Forces mismatch: max_diff={result['forces_max_diff']:.6e}"
        assert result["stress_match"], \
            f"Stress mismatch: max_diff={result['stress_max_diff']:.6e}"

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_stress_gp_medium(self, world_size):
        """Test stress on a medium system (90 atoms)."""
        result = _run_gp_stress_test(world_size, natoms=90)
        assert result["forces_match"], \
            f"Forces mismatch: max_diff={result['forces_max_diff']:.6e}"
        assert result["stress_match"], \
            f"Stress mismatch: max_diff={result['stress_max_diff']:.6e}"


# ---------------------------------------------------------------------------
# Multi-GPU torchrun mode (for SLURM / interactive GPU testing)
# ---------------------------------------------------------------------------

def _torchrun_main():
    """Entry point when run under torchrun for multi-GPU NCCL testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-file", type=str, default=None)
    parser.add_argument("--save-ref", type=str, default=None)
    parser.add_argument("--num-molecules", type=int, default=100)
    parser.add_argument("--gp-size", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    gp_size = args.gp_size if args.gp_size > 0 else world_size

    if rank == 0:
        print(f"Stress GP test: world_size={world_size}, gp_size={gp_size}, "
              f"molecules={args.num_molecules}")

    # Setup GP
    if gp_size > 1:
        setup_gp({"gp_gpus": gp_size, "distributed_backend": "nccl"})

    # Build model (identical on all ranks)
    torch.manual_seed(42)
    model = ToyEnergyModel(hidden=64).to(device)
    model.eval()

    # Build data
    atoms = make_water_box(num_molecules=args.num_molecules, seed=42)
    natoms = len(atoms)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32,
                       device=device, requires_grad=True)
    cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32,
                        device=device).unsqueeze(0)
    cell.requires_grad_(True)
    batch = torch.zeros(natoms, dtype=torch.long, device=device)

    if rank == 0:
        print(f"  Atoms: {natoms}")

    with torch.no_grad():
        energy, _, forces, stress = model(pos, cell, batch, num_systems=1)

    result = {
        "energy": energy.cpu(),
        "forces": forces.cpu(),
        "stress": stress.cpu(),
    }

    if rank == 0:
        for k, v in result.items():
            print(f"  {k}: shape={v.shape}, norm={v.float().norm():.6e}")

        if args.save_ref:
            torch.save(result, args.save_ref)
            print(f"  Reference saved to {args.save_ref}")
        elif args.ref_file:
            ref = torch.load(args.ref_file, weights_only=False, map_location="cpu")
            all_ok = True
            for key in ref:
                if key not in result:
                    print(f"  FAIL: Missing {key}")
                    all_ok = False
                    continue
                diff = (ref[key].float() - result[key].float()).abs()
                max_diff = diff.max().item()
                ok = torch.allclose(ref[key].float(), result[key].float(),
                                    atol=1e-4, rtol=1e-3)
                status = "PASS" if ok else "FAIL"
                print(f"  {status} {key}: max_diff={max_diff:.6e}")
                if not ok:
                    all_ok = False

            if all_ok:
                print(f"\n  All {world_size}-GPU stress tests PASSED!")
            else:
                print(f"\n  {world_size}-GPU stress tests FAILED!")
                dist.destroy_process_group()
                sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        # Running under torchrun
        _torchrun_main()
    else:
        # Running as pytest
        pytest.main([__file__, "-v"])
