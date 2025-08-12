# example how to use checkpoint fixtures
from __future__ import annotations

import os
from functools import partial

import numpy as np
import pytest
import torch

# Combine atoms and predict
from ase.build import make_supercell

from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.units.mlip_unit import MLIPPredictUnit


@pytest.mark.parametrize(
    "dtype,num_tol,charge",
    [
        (torch.float32, 1e-5, 0.0),
        (torch.float64, 1e-11, 0.0),
        pytest.param(
            torch.float32,
            1e-5,
            1.0,
            marks=pytest.mark.xfail(reason="Known issue with UMA approach"),
        ),
        pytest.param(
            torch.float64,
            1e-11,
            1.0,
            marks=pytest.mark.xfail(reason="Known issue with UMA approach"),
        ),
    ],
)
def test_extensivity_nonpbc(
    dtype, num_tol, charge, direct_checkpoint, fake_uma_dataset
):
    direct_inference_checkpoint_pt, _ = direct_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
        target_dtype=dtype,
    )

    atoms = db.get_atoms(0)
    atoms.info["charge"] = charge
    atoms.info["spin"] = 1

    # Get two samples and shift the second far away
    atoms1 = atoms.copy()
    atoms2 = atoms.copy()
    atoms1.pbc = False
    atoms2.pbc = False
    atoms2.positions += 1000.0  # Shift far to avoid interaction

    sample1 = a2g(atoms1, task_name="oc20")
    sample2 = a2g(atoms2, task_name="oc20")

    predictor = MLIPPredictUnit(direct_inference_checkpoint_pt, device="cpu")
    predictor.model = predictor.model.to(dtype)

    batch1 = data_list_collater([sample1], otf_graph=True)
    batch2 = data_list_collater([sample2], otf_graph=True)
    energy1 = predictor.predict(batch1)["energy"]
    energy2 = predictor.predict(batch2)["energy"]

    atoms_combined = atoms1 + atoms2
    atoms_combined.info["charge"] = charge * 2
    atoms_combined.info["spin"] = 1

    sample_combined = a2g(atoms_combined, task_name="oc20")
    batch_combined = data_list_collater([sample_combined], otf_graph=True)
    energy_combined = predictor.predict(batch_combined)["energy"]

    diff = torch.abs((energy1 + energy2) - energy_combined)
    print(f"Extensivity test, {dtype}, |E1+E2-E12|: {diff}")
    assert diff < num_tol


@pytest.mark.parametrize(
    "dtype,num_tol",
    [
        (torch.float32, 1e-5),
        (torch.float64, 1e-11),
    ],
)
def test_extensivity_pbc(dtype, num_tol, direct_checkpoint, fake_uma_dataset):
    direct_inference_checkpoint_pt, _ = direct_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=10,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
        target_dtype=dtype,
    )

    atoms_pbc = db.get_atoms(0)
    atoms_pbc.info["charge"] = 0
    atoms_pbc.info["spin"] = 1
    atoms_pbc.pbc = True

    P = np.eye(3, dtype=int)
    P[0, 0] = 2  # 2x1x1 supercell
    atoms_supercell = make_supercell(atoms_pbc, P)

    # this cant be charge'=2*charge and spin'=2*spin-1
    # because that gets different embeddings in UMA
    atoms_supercell.info["charge"] = atoms_pbc.info["charge"]
    atoms_supercell.info["spin"] = atoms_pbc.info["spin"]

    sample_pbc = a2g(atoms_pbc, task_name="oc20")
    sample_supercell = a2g(atoms_supercell, task_name="oc20")

    predictor = MLIPPredictUnit(direct_inference_checkpoint_pt, device="cpu")
    predictor.model = predictor.model.to(dtype)

    batch_pbc = data_list_collater([sample_pbc], otf_graph=True)
    batch_supercell = data_list_collater([sample_supercell], otf_graph=True)
    energy_pbc = predictor.predict(batch_pbc)["energy"]
    energy_supercell = predictor.predict(batch_supercell)["energy"]

    diff_pbc = torch.abs(2 * energy_pbc - energy_supercell)

    print(f"Extensivity test (PBC, tiled), {dtype}, |2*E-E_supercell|: {diff_pbc}")
    assert diff_pbc < num_tol
