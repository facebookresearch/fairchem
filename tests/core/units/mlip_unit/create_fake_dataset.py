"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

fake_elements = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
)


@dataclass
class FakeDatasetConfig:
    name: str
    n_systems: int
    system_size_range: tuple[int, int]
    energy_std: float
    energy_mean: float
    forces_std: float
    src: str
    metadata_path: str | None = None
    split: str | None = None
    energy_field: str = "energy"
    forces_mean: float = 0.0
    seed: int = 0
    pbc: bool = True

    def get_split_config(self):
        if self.metadata_path is not None:
            return {"src": self.src, "metadata_path": self.metadata_path}
        return {"src": self.src}

    def get_normalization_constants_config(self):
        return {
            "energy": {"mean": self.energy_mean, "stdev": self.energy_std},
            "forces": {"mean": self.forces_mean, "stdev": self.forces_std},
        }


def calculate_forces_repulsive_force_field(atoms):
    positions = torch.tensor(atoms.positions)
    dists = torch.cdist(positions, positions)
    scaling = 1.0 / (dists**2).clamp(min=1e-6)
    pairwise_forces = (
        positions.unsqueeze(1) - positions.unsqueeze(0)
    ) * scaling.unsqueeze(2)
    return pairwise_forces.sum(axis=1)


# from metamate
def compute_energy(atoms, epsilon=1.0, sigma=1.0):
    positions = torch.tensor(atoms.positions)
    # Compute pairwise distances using torch.cdist
    distances = torch.cdist(positions, positions, p=2)

    # Avoid division by zero by setting diagonal to infinity
    distances.fill_diagonal_(float("inf"))

    # Calculate Lennard-Jones potential
    inv_distances = sigma / distances
    inv_distances6 = inv_distances**6
    inv_distances12 = inv_distances6**2
    lj_potential = 4 * epsilon * (inv_distances12 - inv_distances6)

    # Sum the potential energy for all unique pairs
    return torch.sum(lj_potential) / 2  # Divide by 2 to account for double counting


def generate_structures(fake_dataset_config: FakeDatasetConfig):
    systems = []

    np.random.seed(fake_dataset_config.seed)
    for _ in range(fake_dataset_config.n_systems):
        n_atoms = np.random.randint(
            fake_dataset_config.system_size_range[0],
            fake_dataset_config.system_size_range[1] + 1,
        )

        # 0.1 atoms per A^3
        sys_size = (n_atoms * 10) ** (1 / 3)
        atom_positions = np.random.uniform(0, sys_size, (n_atoms, 3))
        if fake_dataset_config.pbc:
            pbc = True
            cell = np.eye(3, dtype=np.float32) * sys_size
        else:
            pbc = False
            cell = None
        atom_symbols = np.random.choice(fake_elements, size=n_atoms)
        atoms = Atoms(
            symbols=atom_symbols, positions=atom_positions, cell=cell, pbc=pbc
        )

        forces = calculate_forces_repulsive_force_field(atoms)
        energy = compute_energy(atoms)
        systems.append({"atoms": atoms, "forces": forces, "energy": energy})

    forces = torch.vstack([system["forces"] for system in systems])
    energies = torch.vstack([system["energy"] for system in systems])

    energy_scaler = fake_dataset_config.energy_std / energies.std()
    energy_offset = (
        -energies.mean().item() + fake_dataset_config.energy_mean / energy_scaler
    )
    forces_scaler = fake_dataset_config.forces_std * forces.norm(dim=1, p=2).std()
    assert fake_dataset_config.forces_mean == 0.0

    structures = []
    for system in systems:
        atoms = system["atoms"]
        calc = SinglePointCalculator(
            atoms=atoms,
            forces=(system["forces"] * forces_scaler).numpy(),
            stress=np.random.random((6,)),
            **{
                fake_dataset_config.energy_field: (
                    (system["energy"] + energy_offset) * energy_scaler
                ).item(),
            },
        )
        atoms.calc = calc

        atoms.info["extensive_property"] = 3 * len(atoms)
        atoms.info["tensor_property"] = np.random.random((6, 6))
        atoms.info["charge"] = np.random.randint(-10, 10)
        atoms.info["spin"] = np.random.randint(0, 2)

        structures.append(atoms)

    return structures


def create_fake_dataset(fake_dataset_config: FakeDatasetConfig):
    # remove if they already exist
    if os.path.exists(fake_dataset_config.src):
        os.remove(fake_dataset_config.src)
    if fake_dataset_config.metadata_path is not None and os.path.exists(
        fake_dataset_config.metadata_path
    ):
        os.remove(fake_dataset_config.metadata_path)

    os.makedirs(os.path.dirname(fake_dataset_config.src), exist_ok=True)
    os.makedirs(os.path.dirname(fake_dataset_config.metadata_path), exist_ok=True)

    # generate the data
    structures = generate_structures(fake_dataset_config)

    # write data and metadata
    num_atoms = []
    # with db.connect(fake_dataset_config.src) as database:
    with connect(fake_dataset_config.src) as database:
        for _i, atoms in enumerate(structures):
            database.write(atoms, data=atoms.info)
            num_atoms.append(len(atoms))

    if fake_dataset_config.metadata_path is not None:
        np.savez(fake_dataset_config.metadata_path, natoms=num_atoms)
    return


def create_fake_uma_dataset(tmpdirname: str, train_size: int = 14, val_size: int = 10):
    systems_per_dataset = {"train": train_size, "val": val_size}
    dataset_configs = {
        "oc20": {
            train_or_val: FakeDatasetConfig(
                name="oc20",
                split=train_or_val,
                n_systems=systems_per_dataset[train_or_val],
                system_size_range=[5, 20],
                energy_std=24.901469505465872,
                forces_std=1.2,
                energy_mean=0.0,
                # energy_field="oc20_energy",
                src=f"{tmpdirname}/oc20/oc20_{train_or_val}.aselmdb",
                metadata_path=f"{tmpdirname}/oc20/oc20_{train_or_val}_metadata.npz",
                seed=0,
                pbc=True,
            )
            for train_or_val in ("train", "val")
        },
        "omol": {
            train_or_val: FakeDatasetConfig(
                name="omol",
                split=train_or_val,
                n_systems=systems_per_dataset[train_or_val],
                system_size_range=[2, 5],
                energy_std=1.8372538609816367,
                forces_std=1.0759386003767104,
                energy_mean=0.0,
                # energy_field="spice_energy",
                src=f"{tmpdirname}/omol/omol_{train_or_val}.aselmdb",
                metadata_path=f"{tmpdirname}/omol/omol_{train_or_val}_metadata.npz",
                seed=1,
                pbc=False,
            )
            for train_or_val in ("train", "val")
        },
    }

    # create all the datasets
    for train_and_val_fake_dataset_configs in dataset_configs.values():
        for fake_dataset_config in train_and_val_fake_dataset_configs.values():
            create_fake_dataset(fake_dataset_config)
    return tmpdirname
