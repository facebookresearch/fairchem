"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

from torch.utils.data import DataLoader, Dataset

from fairchem.core.common import distutils, gp_utils


def get_dataloader(
    dataset: Dataset, batch_sampler_fn: callable, collate_fn: callable, num_workers, pin_memory=True,
) -> DataLoader:
    if gp_utils.initialized():
        num_replicas = gp_utils.get_dp_world_size()
        rank = gp_utils.get_dp_rank()
    else:
        num_replicas = distutils.get_world_size()
        rank = distutils.get_rank()

    logging.info(f"get_dataloader::Calling batch_sampler_fn={batch_sampler_fn}...")
    batch_sampler = batch_sampler_fn(
        dataset=dataset,
        num_replicas=num_replicas,
        rank=rank,
    )
    logging.info("get_dataloader::Calling Dataloader...")
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        batch_sampler=batch_sampler,
    )
    logging.info("get_dataloader::Done!")
    return dataloader
