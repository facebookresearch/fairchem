# example how to use checkpoint fixtures
from __future__ import annotations

import os
from functools import partial

import pytest
import torch
from e3nn.o3 import rand_matrix

from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.units.mlip_unit import MLIPPredictUnit

# Test equivariance in both fp32 and fp64
# If error in equivariance is due to numerical error in fp
# Then fp64 should have substainally lower variance than fp32
# Otherwise error might be from a bug


# variance in numerical error below num_tol
# variance in rotational error below rot_tol
@pytest.mark.parametrize(
    "dtype,num_tol,rot_tol",
    [
        (torch.float32, 1e-8, 1e-5),
        (torch.float64, 1e-25, 1e-22),
    ],
)
def test_direct_equivariance(
    dtype, num_tol, rot_tol, direct_checkpoint, fake_uma_dataset
):
    direct_inference_checkpoint_pt, _ = direct_checkpoint
    equivariance_on_pt(
        dtype, num_tol, rot_tol, direct_inference_checkpoint_pt, fake_uma_dataset
    )


@pytest.mark.parametrize(
    "dtype,num_tol,rot_tol",
    [
        (torch.float32, 1e-8, 1e-5),
        (torch.float64, 1e-25, 1e-22),
    ],
)
def test_direct_mole_equivariance(
    dtype, num_tol, rot_tol, direct_mole_checkpoint, fake_uma_dataset
):
    direct_mole_inference_checkpoint_pt, _ = direct_mole_checkpoint
    equivariance_on_pt(
        dtype, num_tol, rot_tol, direct_mole_inference_checkpoint_pt, fake_uma_dataset
    )


@pytest.mark.parametrize(
    "dtype,num_tol,rot_tol",
    [
        (torch.float32, 1e-8, 1e-5),
        (torch.float64, 1e-25, 1e-22),
    ],
)
def test_conserving_mole_equivariance(
    dtype, num_tol, rot_tol, conserving_mole_checkpoint, fake_uma_dataset
):
    conserving_mole_inference_checkpoint_pt, _ = conserving_mole_checkpoint
    equivariance_on_pt(
        dtype,
        num_tol,
        rot_tol,
        conserving_mole_inference_checkpoint_pt,
        fake_uma_dataset,
    )


def equivariance_on_pt(
    dtype, num_tol, rot_tol, inference_checkpoint_path, data_root_dir
):
    db = AseDBDataset(config={"src": os.path.join(data_root_dir, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
        target_dtype=dtype
    )

    n_repeats = 10
    for sample_idx in range(5):
        torch.manual_seed(42)
        rotations = [rand_matrix(dtype=dtype) for _ in range(n_repeats)]
        predictor = MLIPPredictUnit(inference_checkpoint_path, device="cpu")
        predictor.model = predictor.model.to(dtype)

        sample = a2g(db.get_atoms(sample_idx), task_name="oc20")
        sample.pos += 500
        sample.cell *= 2000
        batch = data_list_collater([sample], otf_graph=True)

        original_positions = batch.pos.clone()

        # numerical stability
        energies = []
        forces = []
        for _ in range(n_repeats):
            batch.pos = original_positions.clone()
            out = predictor.predict(batch)
            energies.append(out["energy"])
            forces.append(out.get("forces"))

        force_var = torch.stack(forces).var(dim=0).max()
        energy_var = torch.stack(energies).var()
        print(
            f"numerical test , {dtype} , energy_var: {energy_var}, force_var:{force_var}"
        )
        assert force_var < num_tol
        assert energy_var < num_tol

        # equivariance
        energies = []
        forces = []
        for rotation in rotations:
            batch.pos = original_positions.clone() @ rotation
            out = predictor.predict(batch)
            energies.append(out["energy"])
            forces.append(out.get("forces") @ rotation.T)

        force_var = torch.stack(forces).var(dim=0).max()
        energy_var = torch.stack(energies).var()
        assert torch.stack(energies).abs().sum() > 0.001
        print(
            f"equivariance test , {dtype} , energy_var: {energy_var}, force_var:{force_var}"
        )
        assert force_var < rot_tol
        assert energy_var < rot_tol
