"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.datasets.ase_datasets import AseAtomsDataset


@runtime_checkable
class AtomsSequence(Protocol):
    def __getitem__(self, index) -> Atoms:
        pass

    def __len__(self) -> int:
        pass


class AtomsDatasetSequence(AtomsSequence):
    """
    Turn an AseAtomsDataset into an AtomsSequence that iterates over atoms objects.
    """

    def __init__(self, dataset: AseAtomsDataset):
        self.dataset = dataset

    def __getitem__(self, index) -> Atoms:
        return self.dataset.get_atoms(index)

    def __len__(self) -> int:
        return len(self.dataset)
