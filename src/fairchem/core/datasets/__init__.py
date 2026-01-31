# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from fairchem.core.common.utils import resolve_class

from .ase_datasets import AseDBDataset, AseReadDataset, AseReadMultiStructureDataset
from .base_dataset import create_dataset
from .collaters.simple_collater import (
    data_list_collater,
)

DATASET_REGISTRY: dict[str, type] = {
    "ase_read": AseReadDataset,
    "ase_read_multi": AseReadMultiStructureDataset,
    "ase_db": AseDBDataset,
}


def get_dataset_class(name: str) -> type:
    """Resolve a dataset class from its name or fully-qualified path.

    Args:
        name: Dataset name (e.g., "ase_read") or full path
            (e.g., "fairchem.core.datasets.ase_datasets.AseReadDataset")

    Returns:
        The dataset class
    """
    return resolve_class(name, DATASET_REGISTRY, "dataset")


__all__ = [
    "AseDBDataset",
    "AseReadDataset",
    "AseReadMultiStructureDataset",
    "create_dataset",
    "data_list_collater",
    "DATASET_REGISTRY",
    "get_dataset_class",
]
