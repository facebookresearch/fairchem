"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch


class MTCollater:
    def __init__(
        self,
        task_config,
        exclude_keys,
        otf_graph: bool = False,  # Deprecated, do not use!
    ) -> None:
        self.exclude_keys = exclude_keys
        self.task_config = task_config
        self.dataset_task_map = self._create_dataset_task_map(task_config)

    def __call__(self, data_list: list[AtomicData]) -> AtomicData:
        batch = self.data_list_collater(
            data_list,
            dataset_task_map=self.dataset_task_map,
            exclude_keys=self.exclude_keys,
        )
        return batch

    def data_list_collater(
        self,
        data_list: list[AtomicData],
        dataset_task_map: dict,
        exclude_keys: list,
    ) -> AtomicData:
        data_list = self._add_missing_attr(data_list, dataset_task_map)

        return atomicdata_list_to_batch(data_list, exclude_keys=exclude_keys)

    # takes in the task config
    def _create_dataset_task_map(self, config):
        datasets_task_map = {}
        for task, task_dict in config.items():
            for dataset in task_dict["datasets"]:
                if dataset not in datasets_task_map:
                    datasets_task_map[dataset] = {}
                datasets_task_map[dataset][task] = {"level": task_dict["level"]}
        return datasets_task_map

    def _add_missing_attr(self, data_list, dataset_task_map):
        """
        add missing data object attributes as inf
        """
        # find all tasks in a given batch
        # and collect all necessary info
        datasets_in_batch_to_task_configs = {}
        tasks_to_report_on = []

        for data in data_list:
            if data.dataset not in datasets_in_batch_to_task_configs:
                datasets_in_batch_to_task_configs[data.dataset] = dataset_task_map[
                    data.dataset
                ]
                for task, task_config in datasets_in_batch_to_task_configs[
                    data.dataset
                ].items():
                    # if this is the first time we have seen this kind of dataset
                    # record its out spec
                    if (
                        "out_spec"
                        not in datasets_in_batch_to_task_configs[data.dataset][task]
                    ):
                        if task_config["level"] == "system":
                            dim = list(getattr(data, task).shape)
                        elif task_config["level"] == "atom":
                            dim = list(getattr(data, task).shape[1:])
                        else:
                            raise ValueError(
                                f"task level must be either system or atom, found {task_config['level']}"
                            )
                        datasets_in_batch_to_task_configs[data.dataset][task][
                            "out_spec"
                        ] = {
                            "dim": dim,
                            "dtype": str(getattr(data, task).dtype),
                        }

                    # make sure all data in this batch has a valid output for this task
                    tasks_to_report_on.append(task)

        # tasks info
        set_all_tasks = True  # DDP breaks if we try and be clever here :(
        if set_all_tasks:
            task_config = self.task_config
            tasks_to_report_on = list(self.task_config)
        else:
            task_config = {
                task: val
                for _, td in datasets_in_batch_to_task_configs.items()
                for task, val in td.items()
            }

        # find missing tasks in the batch
        missing_tasks = {}
        for dataset, task_dict in datasets_in_batch_to_task_configs.items():
            missing_tasks[dataset] = list(
                set(tasks_to_report_on).difference(set(task_dict.keys()))
            )

        # create a copy before assigning attributes to inf
        data_list = [data.clone() for data in data_list]

        # set missing attributes to inf for all data objects in the batch
        # according to level, output dim, and dtype
        for data in data_list:
            for task in missing_tasks[data.dataset]:
                dim = task_config[task]["out_spec"]["dim"]
                dtype = getattr(torch, task_config[task]["out_spec"]["dtype"])
                if task_config[task]["level"] == "atom":
                    dim = [data.natoms] + dim
                setattr(data, task, torch.full(dim, torch.inf, dtype=dtype))
        return data_list
