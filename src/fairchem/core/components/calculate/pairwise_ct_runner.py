"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from abc import ABCMeta

import hydra
import numpy as np
import omegaconf
import torch
from tqdm import tqdm

from fairchem.core.components.runner import Runner


def count_pairs(tensor, max_value=100):
    # Ensure the tensor is of type long for indexing
    tensor = tensor.long()
    # Flatten the pairs into a single index
    indices = (tensor[:, 0] - 1) * max_value + (tensor[:, 1] - 1)
    # Count occurrences of each index
    counts = torch.bincount(indices, minlength=max_value * max_value)
    count_matrix = counts.view(max_value, max_value)

    return count_matrix


class PairwiseCountRunner(Runner, metaclass=ABCMeta):
    """Perform a single point calculation of several structures/molecules.

    This class handles the single point calculation of atomic structures using a specified calculator,
    processes the input data in chunks, and saves the results.
    """

    def __init__(
        self,
        dataset_cfg="/checkpoint/ocp/shared/pairwise_data/preview_config.yaml",
        ds_name="omat",
        radius=3.5,
        portion=0.01,
    ):
        """
        Args:
            dataset_cfg: Path to the dataset configuration file
            ds_name: Name of the dataset to process
            radius: Radius for the pairwise calculation
            portion: Portion of the dataset to process (default is 100)

        """
        self.dataset_cfg = dataset_cfg
        self.ds_name = ds_name
        self.radius = radius
        self.portion = portion

    def run(self):
        os.makedirs(
            f"/checkpoint/ocp/shared/pairwise_data/uma-preview-r{self.radius}_p{self.portion}",
            exist_ok=True,
        )

        job_num = self.job_config.metadata.array_job_num
        num_jobs = self.job_config.scheduler.num_array_jobs

        # canonical config of a training run. here I took uma_sm_direct (preview)
        cfg = omegaconf.OmegaConf.load(self.dataset_cfg)
        dataset_cfg = cfg["runner"]["train_dataloader"]["dataset"]
        dataset_names = sorted(dataset_cfg["dataset_configs"].keys())
        for ds in dataset_names:
            dataset_cfg["dataset_configs"][ds]["a2g_args"]["radius"] = self.radius
            dataset_cfg["dataset_configs"][ds]["a2g_args"]["max_neigh"] = 300

        concat_uma_dataset = hydra.utils.instantiate(dataset_cfg)

        ds_idx = dataset_names.index(self.ds_name)
        dataset = concat_uma_dataset.datasets[ds_idx]

        count_mtx = np.zeros([100, 100])
        downsample = int(1 / self.portion)
        chunk_indices = np.array_split(range(len(dataset)), num_jobs)[job_num][
            ::downsample
        ]
        for idx in tqdm(chunk_indices):
            data = dataset[idx]
            pairs = data.atomic_numbers[data.edge_index].T
            count_mtx += count_pairs(pairs).float().numpy()

        np.save(
            f"/checkpoint/ocp/shared/pairwise_data/uma-preview-r{self.radius}_p{self.portion}/{self.ds_name}_{num_jobs}_{job_num}.npy",
            count_mtx,
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        return
