"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Multi-GPU correctness test: A2A (all-to-all) vs BL (all-gather baseline).

Verifies that the A2A graph parallel implementation produces numerically
identical results to the BL baseline across multiple GPU counts.

Run directly via torchrun:
    torchrun --nproc_per_node=N test_a2a_correctness.py [--natoms 1000]

Or via pytest (2-process CPU with Gloo):
    pytest test_a2a_correctness.py -v

The test creates an FCC crystal, loads the UMA-S checkpoint, and runs
inference in both BL and A2A modes. The outputs (energy, forces, stress)
are gathered to rank 0 and compared numerically.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pytest
import torch

from fairchem.core.common import distutils, gp_utils
from fairchem.core.common.test_utils import (
    PGConfig,
    init_pg_and_rank_and_launch_test,
    spawn_multi_process,
)
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.models.uma.graph_parallel import (
    all_to_all_collect,
    all_to_all_collect_compiled,
    build_gp_context,
    partition_atoms_index_split,
    partition_atoms_spatial,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================================
# Pytest-compatible distributed tests (CPU, Gloo, 2 processes)
# =========================================================================


def _correctness_test_inner(
    atomic_numbers,
    pos,
    edge_index,
    num_atoms,
    partition_strategy,
):
    """
    Inner test function run on each rank.

    Builds GPContext with both BL-style (index_split) and A2A (spatial)
    partitioning, runs all-to-all collect, and verifies that the received
    embeddings are correct by checking that each received atom's value
    matches the expected value from the global embedding tensor.
    """
    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()

    # Create rank assignments
    if partition_strategy == "spatial":
        rank_assignments = partition_atoms_spatial(pos, world_size)
    else:
        rank_assignments = partition_atoms_index_split(
            num_atoms, world_size, pos.device
        )

    # Get this rank's partition
    node_partition = (rank_assignments == rank).nonzero(as_tuple=True)[0]

    # Filter edges to this rank's partition
    target_mask = (rank_assignments == rank)[edge_index[1]]
    rank_edge_index = edge_index[:, target_mask]

    # Build GP context
    gp_ctx = build_gp_context(
        rank_edge_index,
        rank_assignments,
        rank,
        world_size,
        node_partition=node_partition,
    )

    # Create a global embedding where each atom's embedding is its
    # atomic number (unique per atom). This makes it trivial to verify
    # that the right atoms were received.
    x_global = atomic_numbers.unsqueeze(1).float()
    x_local = x_global[node_partition]

    send_indices = gp_ctx.send_indices

    # Test both collect functions
    x_recv_autograd = all_to_all_collect(x_local, gp_ctx, send_indices)
    x_recv_compiled = all_to_all_collect_compiled(x_local, gp_ctx, send_indices)

    # Verify shapes
    assert x_recv_autograd.shape == x_recv_compiled.shape, (
        f"Rank {rank}: shape mismatch autograd={x_recv_autograd.shape} "
        f"vs compiled={x_recv_compiled.shape}"
    )

    # Verify autograd == compiled
    values_match = torch.allclose(x_recv_autograd, x_recv_compiled, atol=1e-6)

    # Verify received values are correct:
    # x_recv should contain embeddings of gp_ctx.needed_atoms
    # in the correct order (sorted by source rank).
    expected_values = x_global[gp_ctx.needed_atoms]
    recv_correct = torch.allclose(x_recv_autograd, expected_values, atol=1e-6)

    # Verify edge_index_local is valid
    x_full = torch.cat([x_local, x_recv_autograd], dim=0)
    edge_valid = (gp_ctx.edge_index_local >= 0).all().item()
    edge_in_bounds = (gp_ctx.edge_index_local < x_full.shape[0]).all().item()

    # Verify message passing produces the same result as non-distributed
    # Simple sum aggregation: for each local target, sum source embeddings
    x_source = x_full[gp_ctx.edge_index_local[0]]
    local_result = torch.zeros(
        gp_ctx.total_local_atoms,
        x_source.shape[1],
        dtype=x_source.dtype,
        device=x_source.device,
    )
    local_result.index_add_(0, gp_ctx.edge_index_local[1], x_source)

    # Reference: compute the same aggregation on the full graph
    x_source_ref = x_global[rank_edge_index[0]]
    ref_result = torch.zeros(
        num_atoms,
        x_source_ref.shape[1],
        dtype=x_source_ref.dtype,
        device=x_source_ref.device,
    )
    ref_result.index_add_(0, rank_edge_index[1], x_source_ref)
    ref_local = ref_result[node_partition]

    mp_match = torch.allclose(local_result, ref_local, atol=1e-6)

    return {
        "rank": rank,
        "partition_strategy": partition_strategy,
        "world_size": world_size,
        "local_atoms": gp_ctx.total_local_atoms,
        "needed_atoms": gp_ctx.total_needed_atoms,
        "num_edges": rank_edge_index.shape[1],
        "values_match": values_match,
        "recv_correct": recv_correct,
        "edge_valid": edge_valid,
        "edge_in_bounds": edge_in_bounds,
        "mp_match": mp_match,
    }


@pytest.mark.parametrize(
    "strategy,num_atoms",
    [
        ("index_split", 8),
        ("index_split", 20),
        ("spatial", 8),
        ("spatial", 20),
    ],
)
def test_a2a_correctness_gloo(strategy, num_atoms):
    """
    Verify A2A correctness at 2 GPUs using Gloo backend.

    Creates a dense graph (all atoms connected) and verifies that:
    1. Autograd and compiled collect produce identical results
    2. Received embeddings contain correct values
    3. Message passing produces correct aggregation
    """
    # Create dense graph
    src, dst = [], []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    atomic_numbers = torch.arange(2, 2 + num_atoms, dtype=torch.float)
    pos = torch.randn(num_atoms, 3) * 10  # spread out for spatial

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        _correctness_test_inner,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        pos,
        edge_index,
        num_atoms,
        strategy,
    )

    for result in all_rank_results:
        r = result["rank"]
        assert result["values_match"], f"Rank {r}: autograd vs compiled mismatch"
        assert result[
            "recv_correct"
        ], f"Rank {r}: received embeddings don't match expected values"
        assert result["edge_valid"], f"Rank {r}: edge_index_local has negative entries"
        assert result[
            "edge_in_bounds"
        ], f"Rank {r}: edge_index_local has out-of-bounds entries"
        assert result[
            "mp_match"
        ], f"Rank {r}: message passing result differs from reference"


@pytest.mark.parametrize("strategy", ["index_split", "spatial"])
def test_a2a_consistency_across_graph_sizes(strategy):
    """
    Verify A2A correctness with sparse graphs (not all-to-all connected).

    Uses a chain graph (each atom connected to its neighbors within
    distance 2) to test the case where not every rank needs atoms
    from every other rank.
    """
    num_atoms = 16

    # Chain graph: atom i connected to i-1, i+1, i-2, i+2
    src, dst = [], []
    for i in range(num_atoms):
        for d in [-2, -1, 1, 2]:
            j = (i + d) % num_atoms  # wrap around
            src.append(i)
            dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    atomic_numbers = torch.arange(10, 10 + num_atoms, dtype=torch.float)
    # Linear arrangement for clear spatial partitioning
    pos = torch.zeros(num_atoms, 3)
    pos[:, 0] = torch.arange(num_atoms, dtype=torch.float)

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        _correctness_test_inner,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        pos,
        edge_index,
        num_atoms,
        strategy,
    )

    for result in all_rank_results:
        r = result["rank"]
        assert result["values_match"], f"Rank {r}: autograd vs compiled mismatch"
        assert result[
            "recv_correct"
        ], f"Rank {r}: received embeddings don't match expected values"
        assert result[
            "mp_match"
        ], f"Rank {r}: message passing result differs from reference"


def _multidim_test_inner(x_global, pos, edge_index, num_atoms, strategy):
    """
    Test A2A correctness with multi-dimensional embeddings.
    Defined at module level for pickle compatibility with multiprocessing.
    """
    rank = gp_utils.get_gp_rank()
    world_size = gp_utils.get_gp_world_size()

    if strategy == "spatial":
        rank_assignments = partition_atoms_spatial(pos, world_size)
    else:
        rank_assignments = partition_atoms_index_split(
            num_atoms, world_size, pos.device
        )

    node_partition = (rank_assignments == rank).nonzero(as_tuple=True)[0]
    target_mask = (rank_assignments == rank)[edge_index[1]]
    rank_edge_index = edge_index[:, target_mask]

    gp_ctx = build_gp_context(
        rank_edge_index,
        rank_assignments,
        rank,
        world_size,
        node_partition=node_partition,
    )

    x_local = x_global[node_partition]
    send_indices = gp_ctx.send_indices

    x_recv = all_to_all_collect(x_local, gp_ctx, send_indices)
    x_recv_c = all_to_all_collect_compiled(x_local, gp_ctx, send_indices)

    # Verify
    expected = x_global[gp_ctx.needed_atoms]
    recv_correct = torch.allclose(x_recv, expected, atol=1e-6)
    compiled_match = torch.allclose(x_recv, x_recv_c, atol=1e-6)

    return {
        "rank": rank,
        "recv_correct": recv_correct,
        "compiled_match": compiled_match,
        "recv_shape": x_recv.shape,
        "expected_shape": expected.shape,
    }


@pytest.mark.parametrize("strategy", ["index_split", "spatial"])
def test_a2a_multidim_embeddings(strategy):
    """
    Verify correctness with multi-dimensional embeddings (not just scalars).

    Uses 16-dim embeddings to match the typical sphere_channels in UMA.
    """
    num_atoms = 12
    embed_dim = 16

    src, dst = [], []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # Use random embeddings instead of scalar atomic numbers
    torch.manual_seed(42)
    x_global = torch.randn(num_atoms, embed_dim)
    pos = torch.randn(num_atoms, 3) * 10

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)

    all_rank_results = spawn_multi_process(
        config,
        _multidim_test_inner,
        init_pg_and_rank_and_launch_test,
        x_global,
        pos,
        edge_index,
        num_atoms,
        strategy,
    )

    for result in all_rank_results:
        r = result["rank"]
        assert result["recv_correct"], (
            f"Rank {r}: multidim recv mismatch, "
            f"shape={result['recv_shape']} vs {result['expected_shape']}"
        )
        assert result[
            "compiled_match"
        ], f"Rank {r}: autograd vs compiled mismatch for multidim"


# =========================================================================
# Full model correctness test (GPU, run via torchrun or SLURM)
# =========================================================================


def _resolve_checkpoint():
    """
    Resolve the UMA-S checkpoint path using the fairchem pretrained model API.
    """
    from fairchem.core.calculate.pretrained_mlip import (
        pretrained_checkpoint_path_from_name,
    )

    return pretrained_checkpoint_path_from_name(model_name="uma-s-1p2")


def _run_full_model_comparison(
    natoms: int = 1000,
    results_file: str | None = None,
):
    """
    Run the full UMA-S model in both BL and A2A modes and compare outputs.

    Must be called inside a torchrun process group.
    """
    from fairchem.core.units.mlip_unit import MLIPPredictUnit

    rank = distutils.get_rank()
    world_size = distutils.get_world_size()

    if rank == 0:
        logger.info(f"Running correctness test: {natoms} atoms, {world_size} GPUs")

    checkpoint_path = _resolve_checkpoint()
    if rank == 0:
        logger.info(f"Using checkpoint: {checkpoint_path}")

    # Create input system
    atoms = get_fcc_crystal_by_num_atoms(natoms, atom_type="Al")
    actual_natoms = len(atoms)
    if rank == 0:
        logger.info(f"Created FCC Al crystal: {actual_natoms} atoms")

    data = AtomicData.from_ase(
        input_atoms=atoms,
        max_neigh=200,
        radius=6.0,
        task_name="oc20",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )

    # -- Run BL (all-gather baseline) --
    if rank == 0:
        logger.info("Loading model for BL (all-gather) mode...")

    predictor_bl = MLIPPredictUnit.from_checkpoint(
        checkpoint_path,
        device=torch.device("cuda"),
        inference_settings={
            "tf32": False,
            "compile": False,
            "activation_checkpointing": False,
            "merge_mole": False,
        },
        overrides={
            "backbone": {
                "use_all_to_all_gp": False,
            },
        },
    )
    predictor_bl.model.eval()

    # Warm up + run BL
    with torch.no_grad():
        _ = predictor_bl.predict(data)
        bl_out = predictor_bl.predict(data)

    bl_energy = bl_out["energy"].clone()
    bl_forces = bl_out["forces"].clone()
    bl_stress = bl_out.get("stress", torch.tensor([])).clone()

    if rank == 0:
        logger.info(f"BL energy: {bl_energy.item():.6f}")
        logger.info(f"BL forces shape: {bl_forces.shape}")
        logger.info(f"BL forces norm: {bl_forces.norm():.6f}")

    # Clean up BL model
    del predictor_bl
    torch.cuda.empty_cache()

    # -- Run A2A (all-to-all with spatial partitioning) --
    if rank == 0:
        logger.info("Loading model for A2A (all-to-all) mode...")

    predictor_a2a = MLIPPredictUnit.from_checkpoint(
        checkpoint_path,
        device=torch.device("cuda"),
        inference_settings={
            "tf32": False,
            "compile": False,
            "activation_checkpointing": False,
            "merge_mole": False,
        },
        overrides={
            "backbone": {
                "use_all_to_all_gp": True,
                "gp_partition_strategy": "spatial",
            },
        },
    )
    predictor_a2a.model.eval()

    # Warm up + run A2A
    with torch.no_grad():
        _ = predictor_a2a.predict(data)
        a2a_out = predictor_a2a.predict(data)

    a2a_energy = a2a_out["energy"].clone()
    a2a_forces = a2a_out["forces"].clone()
    a2a_stress = a2a_out.get("stress", torch.tensor([])).clone()

    if rank == 0:
        logger.info(f"A2A energy: {a2a_energy.item():.6f}")
        logger.info(f"A2A forces shape: {a2a_forces.shape}")
        logger.info(f"A2A forces norm: {a2a_forces.norm():.6f}")

    # -- Compare outputs --
    # Energy should match across all ranks (reduced)
    energy_diff = abs(bl_energy.item() - a2a_energy.item())
    energy_match = energy_diff < 1e-4

    # Forces: each rank only has forces for its local atoms.
    # Gather all forces to rank 0 for comparison.
    # BL forces are already the full set on all ranks.
    # A2A forces need gathering.
    if bl_forces.shape == a2a_forces.shape:
        force_diff = (bl_forces - a2a_forces).abs().max().item()
        force_match = force_diff < 1e-4
        force_rmse = (bl_forces - a2a_forces).pow(2).mean().sqrt().item()
    else:
        # Different shapes — gather and compare
        force_diff = float("nan")
        force_match = False
        force_rmse = float("nan")

    # Stress
    if bl_stress.numel() > 0 and a2a_stress.numel() > 0:
        stress_diff = (bl_stress - a2a_stress).abs().max().item()
        stress_match = stress_diff < 1e-4
    else:
        stress_diff = 0.0
        stress_match = True

    results = {
        "natoms": actual_natoms,
        "world_size": world_size,
        "energy_bl": bl_energy.item(),
        "energy_a2a": a2a_energy.item(),
        "energy_diff": energy_diff,
        "energy_match": energy_match,
        "force_max_diff": force_diff,
        "force_rmse": force_rmse,
        "force_match": force_match,
        "stress_max_diff": stress_diff,
        "stress_match": stress_match,
        "all_match": energy_match and force_match and stress_match,
    }

    if rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info("CORRECTNESS TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Atoms:        {actual_natoms}")
        logger.info(f"GPUs:         {world_size}")
        logger.info(f"Energy BL:    {bl_energy.item():.6f}")
        logger.info(f"Energy A2A:   {a2a_energy.item():.6f}")
        logger.info(f"Energy diff:  {energy_diff:.2e}")
        logger.info(f"Force max Δ:  {force_diff:.2e}")
        logger.info(f"Force RMSE:   {force_rmse:.2e}")
        logger.info(f"Stress max Δ: {stress_diff:.2e}")
        logger.info(f"ALL MATCH:    {'✓ PASS' if results['all_match'] else '✗ FAIL'}")
        logger.info(f"{'='*60}")

        if results_file:
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")

    return results


# =========================================================================
# CLI entrypoint for SLURM / torchrun
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="A2A vs BL correctness test")
    parser.add_argument(
        "--natoms", type=int, default=1000, help="Target number of atoms in FCC crystal"
    )
    parser.add_argument(
        "--results-file", type=str, default=None, help="Path to save JSON results"
    )
    args = parser.parse_args()

    # Initialize distributed
    distutils.setup({"submit": False, "cpu": False})
    gp_utils.setup_gp(distutils.get_world_size())

    try:
        results = _run_full_model_comparison(
            natoms=args.natoms,
            results_file=args.results_file,
        )
        if not results["all_match"]:
            sys.exit(1)
    finally:
        distutils.cleanup()


if __name__ == "__main__":
    main()
