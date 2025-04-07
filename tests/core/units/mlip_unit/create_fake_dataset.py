from __future__ import annotations

import collections
import os
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from fairchem.core.datasets import (
    AseDBDataset,
)

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

    def get_split_config(self):
        if self.metadata_path is not None:
            return {"src": self.src, "metadata_path": self.metadata_path}
        return {"src": self.src}

    def get_normalization_constants_config(self):
        return {
            "energy": {"mean": self.energy_mean, "stdev": self.energy_std},
            "forces": {"mean": self.forces_mean, "stdev": self.forces_std},
        }


def usable_config_from_val_and_train_fake_dataset_configs(
    train_fake_dataset_config: FakeDatasetConfig,
    val_fake_dataset_config: FakeDatasetConfig,
    transforms=None,
):
    assert train_fake_dataset_config.name == val_fake_dataset_config.name
    dataset_config = {
        "splits": {
            "train": train_fake_dataset_config.get_split_config(),
            "val": val_fake_dataset_config.get_split_config(),
        },
        "dataset_name": train_fake_dataset_config.name,
        "normalization_constants": train_fake_dataset_config.get_normalization_constants_config(),
        "format": "ase_db",
        "a2g_args": {
            f"r_{train_fake_dataset_config.energy_field}": True,
            "r_forces": True,
            "r_stress": True,
            "r_data_keys": ["extensive_property", "tensor_property"],
        },
    }
    if transforms is not None:
        dataset_config.update({"transforms": transforms})
    return dataset_config


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
        atom_positions = np.random.uniform(-20, 20, (n_atoms, 3))
        atom_symbols = np.random.choice(fake_elements, size=n_atoms)
        atoms = Atoms(symbols=atom_symbols, positions=atom_positions)
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


def create_fake_dev_set_datasets(
    tmpdirname: str, train_size: int = 500, val_size: int = 100
):
    systems_per_dataset = {"train": train_size, "val": val_size}
    dataset_configs = {
        "oc22": {
            train_or_val: FakeDatasetConfig(
                name="oc22",
                split=train_or_val,
                n_systems=systems_per_dataset[train_or_val],
                system_size_range=[2, 15],
                energy_std=25.229595396538468,
                forces_std=0.25678861141204834,
                energy_mean=0.0,
                # energy_field="oc22_energy",
                src=f"{tmpdirname}/oc22/oc22_{train_or_val}.aselmdb",
                metadata_path=f"{tmpdirname}/oc22/oc22_{train_or_val}_metadata.npz",
                seed=0,
            )
            for train_or_val in ("train", "val")
        },
        "oc20": {
            train_or_val: FakeDatasetConfig(
                name="oc20",
                split=train_or_val,
                n_systems=systems_per_dataset[train_or_val],
                system_size_range=[2, 17],
                energy_std=24.901469505465872,
                forces_std=0.5111534595489502,
                energy_mean=0.0,
                # energy_field="oc20_energy",
                src=f"{tmpdirname}/oc20/oc20_{train_or_val}.aselmdb",
                metadata_path=f"{tmpdirname}/oc20/oc20_{train_or_val}_metadata.npz",
                seed=1,
            )
            for train_or_val in ("train", "val")
        },
        "ani1x": {
            train_or_val: FakeDatasetConfig(
                name="ani1x",
                split=train_or_val,
                n_systems=systems_per_dataset[train_or_val],
                system_size_range=[2, 22],
                energy_std=2.8700712783472118,
                forces_std=2.131422996520996,
                energy_mean=0.0,
                # energy_field="ani1x_energy",
                src=f"{tmpdirname}/ani1x/ani1x_{train_or_val}.aselmdb",
                metadata_path=f"{tmpdirname}/ani1x/ani1x_{train_or_val}_metadata.npz",
                seed=2,
            )
            for train_or_val in ("train", "val")
        },
        "trans1x": {
            train_or_val: FakeDatasetConfig(
                name="trans1x",
                split=train_or_val,
                n_systems=systems_per_dataset[train_or_val],
                system_size_range=[3, 11],
                energy_std=1.787466168382901,
                forces_std=0.3591422140598297,
                energy_mean=0.0,
                # energy_field="trans1x_energy",
                src=f"{tmpdirname}/trans1x/trans1x_{train_or_val}.aselmdb",
                metadata_path=f"{tmpdirname}/trans1x/trans1x_{train_or_val}_metadata.npz",
                seed=3,
            )
            for train_or_val in ("train", "val")
        },
        "spice": {
            train_or_val: FakeDatasetConfig(
                name="spice",
                split=train_or_val,
                n_systems=systems_per_dataset[train_or_val],
                system_size_range=[4, 24],
                energy_std=1.8372538609816367,
                forces_std=1.0759386003767104,
                energy_mean=0.12719055238925644,
                # energy_field="spice_energy",
                src=f"{tmpdirname}/spice/spice_{train_or_val}.aselmdb",
                metadata_path=f"{tmpdirname}/spice/spice_{train_or_val}_metadata.npz",
                seed=4,
            )
            for train_or_val in ("train", "val")
        },
        "mptraj": {
            train_or_val: FakeDatasetConfig(
                name="mptraj",
                split=train_or_val,
                n_systems=systems_per_dataset[train_or_val],
                system_size_range=[5, 31],
                energy_std=22.621750653754294,
                forces_std=0.8124567446457275,
                energy_mean=1.4330214171684765,
                # energy_field="mptraj_energy",
                src=f"{tmpdirname}/mptraj/mptraj_{train_or_val}.aselmdb",
                metadata_path=f"{tmpdirname}/mptraj/mptraj_{train_or_val}_metadata.npz",
                seed=5,
            )
            for train_or_val in ("train", "val")
        },
    }

    # create all the datasets
    for train_and_val_fake_dataset_configs in dataset_configs.values():
        for fake_dataset_config in train_and_val_fake_dataset_configs.values():
            create_fake_dataset(fake_dataset_config)

    # generate the devset config
    datasets_config = {
        "datasets": {
            dataset_name: usable_config_from_val_and_train_fake_dataset_configs(
                dataset_configs[dataset_name]["train"],
                dataset_configs[dataset_name]["val"],
                transforms={
                    f"{dataset_name}_transform": {"dataset_name": dataset_name}
                },
            )
            for dataset_name in dataset_configs
        }
    }

    # for test case
    datasets_config["datasets"]["oc22"]["splits"]["test_string"] = datasets_config[
        "datasets"
    ]["oc22"]["splits"]["train"].copy()

    dataset_yaml_fn = f"{tmpdirname}/datasets.yaml"
    with open(dataset_yaml_fn, "w") as f:
        yaml.dump(datasets_config, f)

    return dataset_yaml_fn


def create_fake_puma_dataset(tmpdirname: str, train_size: int = 14, val_size: int = 10):
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
            )
            for train_or_val in ("train", "val")
        },
    }

    # create all the datasets
    for train_and_val_fake_dataset_configs in dataset_configs.values():
        for fake_dataset_config in train_and_val_fake_dataset_configs.values():
            create_fake_dataset(fake_dataset_config)
    return tmpdirname


def open_asedbdataset(db_path):
    a2g_args = {
        "r_energy": True,
        "r_forces": True,
        "r_stress": True,
        "r_data_keys": ["extensive_property", "tensor_property"],
    }
    return AseDBDataset(config={"src": str(db_path), "a2g_args": a2g_args})


# remote target ('this/nested/key') from config
def remove_from_config(config, target):
    if len(target) == 1 and target[0] in config:
        config.pop(target[0])
        return
    if target[0] in config:
        remove_from_config(
            config[target[0]],
            target[1:],
        )


# take a key as a list, 'this/deeper/key'.split('/') and replace it with the
# value replace_with, if the key does not exist optionally add it with
# add_if_not_exist
def replace_in_config(
    config: dict, target: str, replace_with, add_if_not_exist: bool = False
):
    if len(target) == 1 and (target[0] in config or add_if_not_exist):
        config[target[0]] = replace_with
        return
    if target[0] not in config and add_if_not_exist:
        config[target[0]] = {}
    if target[0] in config or add_if_not_exist:
        replace_in_config(
            config[target[0]],
            target[1:],
            replace_with=replace_with,
            add_if_not_exist=add_if_not_exist,
        )


# take in a regular config and output a config where the fake dataset
# has been plugged in. additionally try to make some changes for testing
# i.e. num_layers
def move_config_to_fake_dataset(
    yaml_fn: str,
    output_fn: str,
    fake_dataset_fn: str,
    num_layers: int = 2,
    dataset_size_replace: int | dict[str, int] = 100,
):
    with open(yaml_fn) as f:
        config = yaml.safe_load(f)
    replace_in_config(
        config=config,
        target=["includes"],
        replace_with=[fake_dataset_fn],
        add_if_not_exist=True,
    )

    for dataset_name, dataset_config in config.get("datasets", {}).items():
        if "key_mapping" in dataset_config:
            key_mapping = dataset_config["key_mapping"]
            if "y" in key_mapping:
                key_mapping["energy"] = key_mapping["y"]
                key_mapping.pop("y")
            if "force" in key_mapping:
                key_mapping["forces"] = key_mapping["force"]
                key_mapping.pop("force")
        if dataset_size_replace is not None:
            if "first_n" in dataset_config:
                dataset_config.pop("first_n")
            if isinstance(dataset_size_replace, dict):
                if dataset_name in dataset_size_replace:
                    target_size = dataset_size_replace[dataset_name]
                else:
                    continue
            else:
                target_size = dataset_size_replace
            remove_from_config(
                config=dataset_config,
                target=["sample_n"],
            )
            remove_from_config(
                config=dataset_config,
                target="splits/sample_n".split("/"),
            )
            replace_in_config(
                config=dataset_config,
                target="splits/train/sample_n".split("/"),
                replace_with=target_size,
                add_if_not_exist=True,
            )
            replace_in_config(
                config=dataset_config,
                target="splits/val/sample_n".split("/"),
                replace_with=target_size,
                add_if_not_exist=True,
            )

            replace_in_config(
                config=dataset_config,
                target="splits/test_string/sample_n".split("/"),
                replace_with=target_size,
                add_if_not_exist=True,
            )
    if num_layers is not None:
        replace_in_config(
            config=config, target="model/num_layers".split("/"), replace_with=num_layers
        )
        replace_in_config(
            config=config,
            target="model/backbone/num_layers".split("/"),
            replace_with=num_layers,
        )
    with open(output_fn, "w") as f:
        yaml.dump(config, f)
    return output_fn


def merge_dictionary(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_dictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d
