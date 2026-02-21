# example how to use checkpoint fixtures
from __future__ import annotations

import os
from functools import partial

import numpy as np
import pytest
import torch

from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


# variance in numerical error below num_tol
@pytest.mark.parametrize(
    "dtype,num_tol,rot_tol",
    [
        (torch.float32, 1e-7, 1e-7),
        # (torch.float64, 1e-29, 1e-29),
    ],
)
def test_conserving_mole_aperiodic_on_pt(
    dtype, num_tol, rot_tol, conserving_mole_checkpoint, fake_uma_dataset
):
    inference_checkpoint_path, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
        target_dtype=dtype,
    )

    n_repeats = 10
    settings = InferenceSettings(base_precision_dtype=dtype)
    predictor_v1 = MLIPPredictUnit(
        inference_checkpoint_path,
        device="cpu",
        overrides={"backbone": {"radius_pbc_version": 1}},
        inference_settings=settings,
    )

    predictor_v2 = MLIPPredictUnit(
        inference_checkpoint_path,
        device="cpu",
        overrides={"backbone": {"radius_pbc_version": 2}},
        inference_settings=settings,
    )
    for sample_idx in range(5):
        torch.manual_seed(42)

        atoms = db.get_atoms(sample_idx)

        atoms1 = atoms.copy()
        atoms1.center(50000)
        sample1 = a2g(atoms1, task_name="oc20")
        batch1 = data_list_collater([sample1], otf_graph=True)

        atoms2 = atoms.copy()
        # atoms2.center(2000)
        atoms2.pbc = np.array([False, False, False])
        sample2 = a2g(atoms2, task_name="oc20")
        batch2 = data_list_collater([sample2], otf_graph=True)

        original_positions1 = batch1.pos.clone()
        original_positions2 = batch2.pos.clone()

        # numerical stability
        energies = []
        forces = []
        for _ in range(n_repeats):
            batch1.pos = original_positions1.clone()
            out1 = predictor_v1.predict(batch1)
            batch2.pos = original_positions2.clone()
            out2 = predictor_v2.predict(batch2)
            energies.append(out1["energy"] - out2["energy"])
            forces.append(out1.get("forces") - out2.get("forces"))

        force_var = torch.stack(forces).var(dim=0).max()
        energy_var = torch.stack(energies).var()
        print(
            f"numerical test , {dtype} , energy_var: {energy_var}, force_var:{force_var}"
        )
