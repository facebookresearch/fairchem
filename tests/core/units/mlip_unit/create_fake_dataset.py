"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from fairchem.core.components.benchmark.fake_dataset import (
    BENCHMARK_DATASET_SPECS,
    FakeDatasetConfig,
    create_fake_benchmark_dataset,
    create_fake_dataset,
    generate_structures,
)

__all__ = [
    "FakeDatasetConfig",
    "create_fake_dataset",
    "create_fake_benchmark_dataset",
    "create_fake_uma_dataset",
    "generate_structures",
    "BENCHMARK_DATASET_SPECS",
]


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
                src=f"{tmpdirname}/omol/omol_{train_or_val}.aselmdb",
                metadata_path=f"{tmpdirname}/omol/omol_{train_or_val}_metadata.npz",
                seed=1,
                pbc=False,
            )
            for train_or_val in ("train", "val")
        },
    }

    for train_and_val_fake_dataset_configs in dataset_configs.values():
        for fake_dataset_config in train_and_val_fake_dataset_configs.values():
            create_fake_dataset(fake_dataset_config)
    return tmpdirname
