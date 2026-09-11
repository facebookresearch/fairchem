"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from fairchem.core.datasets.mt_concat_dataset import ConcatDataset


class MetadataDataset(Dataset):
    def __init__(self, natoms: list[int]) -> None:
        self.natoms = np.asarray(natoms)

    def __len__(self) -> int:
        return len(self.natoms)

    def __getitem__(self, index: int) -> int:
        return index

    def get_metadata(self, attr: str, indices: int | list[int]) -> np.ndarray:
        assert attr == "natoms"
        return self.natoms[indices]


def test_get_metadata_preserves_vector_values_across_datasets():
    dataset = ConcatDataset(
        {
            "first": MetadataDataset([1, 2, 3]),
            "second": MetadataDataset([10, 20]),
        },
        sampling={
            "type": "explicit",
            "ratios": {"first": 1, "second": 2},
        },
    )

    metadata = dataset.get_metadata("natoms", [0, 3, 2, 4, 6])

    np.testing.assert_array_equal(metadata, [1, 10, 3, 20, 20])
