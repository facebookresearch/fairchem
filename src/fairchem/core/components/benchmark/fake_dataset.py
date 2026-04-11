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

FAKE_ELEMENTS = (
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
    """
    Configuration for generating a single fake dataset split.
    """

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


def _calculate_forces_repulsive(atoms):
    positions = torch.tensor(atoms.positions)
    dists = torch.cdist(positions, positions)
    scaling = 1.0 / (dists**2).clamp(min=1e-6)
    pairwise_forces = (
        positions.unsqueeze(1) - positions.unsqueeze(0)
    ) * scaling.unsqueeze(2)
    return pairwise_forces.sum(axis=1)


def _compute_energy_lj(atoms, epsilon=1.0, sigma=1.0):
    positions = torch.tensor(atoms.positions)
    distances = torch.cdist(positions, positions, p=2)
    distances.fill_diagonal_(float("inf"))
    inv_distances = sigma / distances
    inv_distances6 = inv_distances**6
    inv_distances12 = inv_distances6**2
    lj_potential = 4 * epsilon * (inv_distances12 - inv_distances6)
    return torch.sum(lj_potential) / 2


def generate_structures(config: FakeDatasetConfig):
    """
    Generate fake atomic structures with LJ-like energies and repulsive forces.
    """
    systems = []

    np.random.seed(config.seed)
    for _ in range(config.n_systems):
        n_atoms = np.random.randint(
            config.system_size_range[0],
            config.system_size_range[1] + 1,
        )

        sys_size = (n_atoms * 10) ** (1 / 3)
        atom_positions = np.random.uniform(0, sys_size, (n_atoms, 3))
        if config.pbc:
            pbc = True
            cell = np.eye(3, dtype=np.float32) * sys_size
        else:
            pbc = False
            cell = None
        atom_symbols = np.random.choice(FAKE_ELEMENTS, size=n_atoms)
        atoms = Atoms(
            symbols=atom_symbols, positions=atom_positions, cell=cell, pbc=pbc
        )

        forces = _calculate_forces_repulsive(atoms)
        energy = _compute_energy_lj(atoms)
        systems.append({"atoms": atoms, "forces": forces, "energy": energy})

    forces = torch.vstack([s["forces"] for s in systems])
    energies = torch.vstack([s["energy"] for s in systems])

    energy_scaler = config.energy_std / energies.std()
    energy_offset = -energies.mean().item() + config.energy_mean / energy_scaler
    forces_scaler = config.forces_std * forces.norm(dim=1, p=2).std()
    assert config.forces_mean == 0.0

    structures = []
    for system in systems:
        atoms = system["atoms"]
        calc = SinglePointCalculator(
            atoms=atoms,
            forces=(system["forces"] * forces_scaler).numpy(),
            stress=np.random.random((6,)),
            **{
                config.energy_field: (
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


def create_fake_dataset(config: FakeDatasetConfig):
    """
    Write a single fake ASE database from a FakeDatasetConfig.
    """
    if os.path.exists(config.src):
        os.remove(config.src)
    if config.metadata_path is not None and os.path.exists(config.metadata_path):
        os.remove(config.metadata_path)

    os.makedirs(os.path.dirname(config.src), exist_ok=True)
    os.makedirs(os.path.dirname(config.metadata_path), exist_ok=True)

    structures = generate_structures(config)

    num_atoms = []
    with connect(config.src) as database:
        for atoms in structures:
            database.write(atoms, data=atoms.info)
            num_atoms.append(len(atoms))

    if config.metadata_path is not None:
        np.savez(config.metadata_path, natoms=num_atoms)


BENCHMARK_DATASET_SPECS = {
    "oc20": {
        "system_size_range": [4, 140],
        "n_train": 35,
        "pbc": True,
        "energy_std": 24.9,
        "forces_std": 1.2,
        "seed": 0,
    },
    "omol": {
        "system_size_range": [1, 110],
        "n_train": 27,
        "pbc": False,
        "energy_std": 1.84,
        "forces_std": 1.08,
        "seed": 1,
    },
    "omat": {
        "system_size_range": [1, 40],
        "n_train": 18,
        "pbc": True,
        "energy_std": 15.0,
        "forces_std": 1.5,
        "seed": 2,
    },
    "odac": {
        "system_size_range": [13, 340],
        "n_train": 10,
        "pbc": True,
        "energy_std": 20.0,
        "forces_std": 1.3,
        "seed": 3,
    },
    "omc": {
        "system_size_range": [12, 250],
        "n_train": 9,
        "pbc": True,
        "energy_std": 10.0,
        "forces_std": 1.0,
        "seed": 4,
    },
}


def create_fake_benchmark_dataset(tmpdirname: str, val_size: int = 5):
    """
    Generate all 5 UMA benchmark datasets with production-like system sizes.

    Per-dataset train sizes are set in BENCHMARK_DATASET_SPECS to match
    production sampling distributions (accounting for explicit sampling
    ratios in the training config).

    Skips generation if all expected files already exist (cached from a
    previous run).
    """
    all_exist = all(
        os.path.exists(f"{tmpdirname}/{name}/{name}_{split}.aselmdb")
        for name in BENCHMARK_DATASET_SPECS
        for split in ("train", "val")
    )
    if all_exist:
        return tmpdirname

    for name, spec in BENCHMARK_DATASET_SPECS.items():
        for split in ("train", "val"):
            n_systems = spec["n_train"] if split == "train" else val_size
            config = FakeDatasetConfig(
                name=name,
                split=split,
                n_systems=n_systems,
                system_size_range=spec["system_size_range"],
                energy_std=spec["energy_std"],
                forces_std=spec["forces_std"],
                energy_mean=0.0,
                src=f"{tmpdirname}/{name}/{name}_{split}.aselmdb",
                metadata_path=f"{tmpdirname}/{name}/{name}_{split}_metadata.npz",
                seed=spec["seed"],
                pbc=spec["pbc"],
            )
            create_fake_dataset(config)

    return tmpdirname
