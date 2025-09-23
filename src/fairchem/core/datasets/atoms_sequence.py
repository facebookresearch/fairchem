"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, overload, runtime_checkable

from ase import Atoms

if TYPE_CHECKING:
    from fairchem.core.datasets.ase_datasets import AseAtomsDataset


@runtime_checkable
class AtomsSequence(Sequence[Atoms]):
    @overload
    def __getitem__(self, index: int) -> Atoms: ...

    @overload
    def __getitem__(self, index: slice) -> AtomsSequence: ...

    def __getitem__(self, index): ...

    def __len__(self) -> int: ...


class AtomsDatasetSequence:
    """
    Turn an AseAtomsDataset into an AtomsSequence that iterates over atoms objects.
    """

    def __init__(self, dataset: AseAtomsDataset):
        self.dataset = dataset

    def __getitem__(self, index) -> Atoms:
        if isinstance(index, int):
            return self.dataset.get_atoms(index)
        else:
            raise IndexError("Unsupported indexing")

    def __len__(self) -> int:
        return len(self.dataset)
