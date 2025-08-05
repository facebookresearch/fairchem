from __future__ import annotations

import random
from functools import partial

import numpy as np
import pytest
import torch
from ase import Atoms

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.units.mlip_unit import MLIPPredictUnit

# molecule, transformation, pbc, should_energy_be_equal
test_cases = [
    ("chiral_methane", "full_mirror", False, True),
    ("chiral_methane", "full_mirror", True, True),
    ("chiral_ethane", "full_mirror", False, True),
    ("chiral_ethane", "full_mirror", True, True),
    ("chiral_ethane", "flip_one_carbon", False, False),
    ("chiral_ethane", "flip_one_carbon", True, False),
]


def seed_everywhere(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_atoms(symbols, coords, pbc, mode):
    atoms1 = Atoms(symbols=symbols, positions=coords)
    atoms2 = atoms1.copy()
    if mode == "full_mirror":
        atoms2.positions[:, 0] *= -1
    elif mode == "flip_one_carbon":
        atoms2.positions[3], atoms2.positions[4] = (
            atoms2.positions[4].copy(),
            atoms2.positions[3].copy(),
        )
    else:
        raise ValueError(f"Unknown transformation mode: {mode}")

    if pbc:
        cell = np.array([5.0, 5.0, 5.0])
        for atoms in [atoms1, atoms2]:
            atoms.set_cell(cell)
            atoms.pbc = True
            atoms.center(about=cell / 2)
    return atoms1, atoms2


def make_chiral_methane_atoms(pbc, mode):
    coords = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ]
    )
    symbols = ["C", "H", "Cl", "Br", "F"]
    return make_atoms(symbols, coords, pbc, mode)


def make_chiral_ethane_atoms(pbc, mode):
    coords = np.array(
        [
            [0.000, 0.000, 0.000],
            [1.54, 0.000, 0.000],
            [-0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [2.169, 0.629, 0.629],
            [2.169, -0.629, 0.629],
            [2.169, 0.629, -0.629],
        ]
    )
    symbols = ["C", "C", "H", "Cl", "Br", "H", "Cl", "F"]
    return make_atoms(symbols, coords, pbc, mode)


@pytest.mark.parametrize(
    "dtype,num_tol",
    [
        (torch.float32, 1e-6),
        (torch.float64, 1e-12),
    ],
)
@pytest.mark.parametrize("case_name,mode,pbc,should_be_equal", test_cases)
def test_uma_cases_all(
    case_name,
    mode,
    pbc,
    should_be_equal,
    direct_checkpoint,
    dtype,
    num_tol,
):
    seed_everywhere()
    direct_inference_checkpoint_pt, _ = direct_checkpoint
    predictor = MLIPPredictUnit(direct_inference_checkpoint_pt, device="cpu")
    predictor.model = predictor.model.to(dtype)

    a2g = partial(
        AtomicData.from_ase,
        max_neigh=100,
        radius=100,
        r_edges=False,
        r_data_keys=["spin", "charge"],
        target_dtype=dtype,
    )

    atom_builders = {
        "chiral_methane": make_chiral_methane_atoms,
        "chiral_ethane": make_chiral_ethane_atoms,
    }
    atoms1, atoms2 = atom_builders[case_name](pbc, mode)

    sample1 = a2g(atoms1, task_name="omol")
    sample2 = a2g(atoms2, task_name="omol")

    batch1 = data_list_collater([sample1], otf_graph=True)
    batch2 = data_list_collater([sample2], otf_graph=True)

    energy1 = predictor.predict(batch1)["energy"]
    energy2 = predictor.predict(batch2)["energy"]

    if should_be_equal:
        assert (
            torch.abs(energy2 - energy1) < num_tol
        ), f"UMA energy should be invariant for case={case_name}, mode={mode}, pbc={pbc}, dtype={dtype}"
    else:
        assert (
            torch.abs(energy2 - energy1) > num_tol
        ), f"UMA energy should differ for case={case_name}, mode={mode}, pbc={pbc}, dtype={dtype}"
