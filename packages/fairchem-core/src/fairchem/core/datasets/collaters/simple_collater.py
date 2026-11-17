"""
Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from typing import TypeVar

import torch

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

T_co = TypeVar("T_co", covariant=True)


def data_list_collater(
    data_list: list[AtomicData], otf_graph: bool = False, to_dict: bool = False
) -> AtomicData | dict[str, torch.Tensor]:
    # batch = Batch.from_data_list(data_list)
    batch = atomicdata_list_to_batch(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for _, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except (NotImplementedError, TypeError):
            logging.warning("No edge index information, set otf_graph=True")

    if to_dict:
        batch = dict(batch.items())

    return batch
