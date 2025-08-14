"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from typing import TYPE_CHECKING, Sequence

import hydra
import numpy as np
import torch
from torchtnt.framework import PredictUnit, State

from fairchem.core.common.distutils import (
    CURRENT_DEVICE_TYPE_STR,
    get_device_for_local_rank,
)
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit.utils import (
    load_inference_model,
    tf32_context_manager,
)

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.mlip_unit import Task


def collate_predictions(predict_fn):
    @wraps(predict_fn)
    def collated_predict(
        predict_unit, data: AtomicData, undo_element_references: bool = True
    ):
        # Get the full prediction dictionary from the original predict method
        preds = predict_fn(predict_unit, data, undo_element_references)
        collated_preds = defaultdict(dict)
        
        # Create a mapping from model output keys to task information
        # Model outputs are in format like "dataset_property" (e.g., "oc20_energy")
        # We need to map these to tasks and identify which head they came from
        for i, dataset in enumerate(data.dataset):
            for task in predict_unit.dataset_to_tasks[dataset]:
                # Look for all model output keys that match this task
                matching_keys = []
                for output_key in preds.keys():
                    # Check if this output key corresponds to this task
                    # Format could be: "task_name", "head_dataset_property", etc.
                    if (output_key == task.name or 
                        output_key.endswith(f"_{dataset}_{task.property}") or
                        output_key.endswith(f"_{task.property}") and dataset in output_key):
                        matching_keys.append(output_key)
                
                # If no matching keys found, try the task name directly
                if not matching_keys and task.name in preds:
                    matching_keys = [task.name]
                
                for output_key in matching_keys:
                    if task.level == "system":
                        value = preds[output_key][i].unsqueeze(0)
                    elif task.level == "atom":
                        value = preds[output_key][data.batch == i]
                    else:
                        raise RuntimeError(
                            f"Unrecognized task level={task.level} found in data batch at position {i}"
                        )
                    # Use the full output key as the head identifier to maintain uniqueness
                    collated_preds[task.property][output_key] = value

        return dict(collated_preds)

    return collated_predict


class MLIPPredictUnit(PredictUnit[AtomicData]):
    def __init__(
        self,
        inference_model_path: str,
        device: str = "cpu",
        overrides: dict | None = None,
        inference_settings: InferenceSettings | None = None,
        seed: int = 41,
        atom_refs: dict | None = None,
    ):
        super().__init__()
        os.environ[CURRENT_DEVICE_TYPE_STR] = device

        self.seed(seed)
        # note these are different from the element references used for model training
        self.atom_refs = (
            {task.replace("_elem_refs", ""): refs for task, refs in atom_refs.items()}
            if atom_refs is not None
            else {}
        )

        if inference_settings is None:
            inference_settings = InferenceSettings()
        if overrides is None:
            overrides = {}
        if "backbone" not in overrides:
            overrides["backbone"] = {}
        if inference_settings.activation_checkpointing is not None:
            overrides["backbone"]["activation_checkpointing"] = (
                inference_settings.activation_checkpointing
            )
        if inference_settings.wigner_cuda is not None:
            overrides["backbone"]["use_cuda_graph_wigner"] = (
                inference_settings.wigner_cuda
            )
        if inference_settings.external_graph_gen is not None:
            overrides["backbone"][
                "otf_graph"
            ] = not inference_settings.external_graph_gen

        if inference_settings.internal_graph_gen_version is not None:
            overrides["backbone"]["radius_pbc_version"] = (
                inference_settings.internal_graph_gen_version
            )

        self.model, checkpoint = load_inference_model(
            inference_model_path, use_ema=True, overrides=overrides
        )

        all_tasks = [
            hydra.utils.instantiate(task_config)
            for task_config in checkpoint.tasks_config
        ]
        # Only keep tasks whose dataset matches one of self.datasets
        filtered_tasks = [
            t for t in all_tasks if any(ds in self.datasets for ds in getattr(t, 'datasets', []))
        ]
        self.tasks = {t.name: t for t in filtered_tasks}
        self.dataset_to_tasks = get_dataset_to_tasks_map(self.tasks.values())
        assert set(self.dataset_to_tasks.keys()).issubset(
            set(self.datasets)
        ), "Datasets in tasks is not a strict subset of datasets in backbone."
        assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'"

        self.device = get_device_for_local_rank() if device == "cuda" else "cpu"

        self.model.eval()

        self.lazy_model_intialized = False
        self.inference_mode = inference_settings

        # store composition embedding of system the model was merged on
        self.merged_on = None

    @property
    def direct_forces(self) -> bool:
        return self.model.module.backbone.direct_forces

    @property
    def datasets(self) -> list[str]:
        return self.model.module.backbone.dataset_list

    def get_available_heads(self) -> dict[str, list[str]]:
        """Get a mapping of properties to available head names.
        
        Returns:
            Dictionary mapping property names to lists of head names that predict that property
        """
        # This requires running a prediction to see what heads are available
        # For now, return an empty dict - this would need to be populated after first prediction
        return getattr(self, '_available_heads', {})
    
    def _update_available_heads(self, predictions: dict):
        """Update the internal mapping of available heads based on a prediction output."""
        if not hasattr(self, '_available_heads'):
            self._available_heads = defaultdict(list)
        
        for head_key in predictions.keys():
            for task in self.tasks.values():
                if (head_key == task.name or 
                    head_key.endswith(f"_{task.name}") or
                    any(dataset in head_key and task.property in head_key 
                        for dataset in task.datasets)):
                    if head_key not in self._available_heads[task.property]:
                        self._available_heads[task.property].append(head_key)

    def seed(self, seed: int):
        logging.info(f"Setting random seed to {seed}")
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def move_to_device(self):
        self.model.to(self.device)
        for task in self.tasks.values():
            task.normalizer.to(self.device)
            if task.element_references is not None:
                task.element_references.to(self.device)

    def predict_step(self, state: State, data: AtomicData) -> dict[str, torch.tensor]:
        return self.predict(data)

    def get_composition_charge_spin_dataset(self, data):
        composition_sum = data.atomic_numbers.new_zeros(
            self.model.module.backbone.max_num_elements,
            dtype=torch.int,
        ).index_add(
            0,
            data.atomic_numbers.to(torch.int),
            data.atomic_numbers.new_ones(data.atomic_numbers.shape[0], dtype=torch.int),
        )
        comp_charge_spin = (
            composition_sum,
            getattr(data, "charge", None),
            getattr(data, "spin", None),
        )
        return comp_charge_spin, getattr(data, "dataset", [None])

    @collate_predictions
    def predict(
        self, data: AtomicData, undo_element_references: bool = True
    ) -> dict[str, torch.tensor]:
        if not self.lazy_model_intialized:
            # merge everything on CPU
            if self.inference_mode.merge_mole:
                # replace backbone with non MOE version
                assert (
                    data.natoms.numel() == 1
                ), f"Cannot merge model with multiple systems in batch. Must be exactly 1 system, found {data.natoms.numel()}"
                self.model.module.backbone = (
                    self.model.module.backbone.merge_MOLE_model(data.clone())
                )
                self.model.eval()
            # move to device
            self.move_to_device()
            if self.inference_mode.compile:
                logging.warning(
                    "Model is being compiled this might take a while for the first time"
                )
                self.model = torch.compile(self.model, dynamic=True)
            self.lazy_model_intialized = True

        data_device = data.to(self.device)

        if self.inference_mode.merge_mole:
            if self.merged_on is None:
                # only get embeddings after moved to final device to get right types
                self.merged_on = self.get_composition_charge_spin_dataset(data_device)
            else:
                this_sys = self.get_composition_charge_spin_dataset(data_device)
                assert (
                    data_device.natoms.numel() == 1
                ), f"Cannot run merged model on batch with multiple systems. Must be exactly 1 system, found {data_device.natoms.numel()}"
                assert (
                    self.merged_on[0][0].isclose(this_sys[0][0], rtol=1e-5).all()
                ), "Cannot run on merged model on system. Embeddings seem different..."
                assert (
                    self.merged_on[0][1] == this_sys[0][1]
                ), f"Cannot run on merged model on system. Charge is diferrent {self.merged_on[0][1]} vs {this_sys[0][1]}"
                assert (
                    self.merged_on[0][2] == this_sys[0][2]
                ), f"Cannot run on merged model on system. Spin is diferrent {self.merged_on[0][2]} vs {this_sys[0][2]}"
                assert (
                    self.merged_on[1] == this_sys[1]
                ), f"Cannot run on merged model on system. Dataset is diferrent {self.merged_on[1]} vs {this_sys[1]}"

        inference_context = torch.no_grad() if self.direct_forces else nullcontext()
        tf32_context = (
            tf32_context_manager() if self.inference_mode.tf32 else nullcontext()
        )

        pred_output = {}
        with inference_context, tf32_context:
            output = self.model(data_device)
            # Only process tasks relevant to the current data.dataset
            relevant_datasets = set(data.dataset) if hasattr(data, 'dataset') else set()
            for task_name, task in self.tasks.items():
                # Only process if this task is for a relevant dataset
                if not relevant_datasets.intersection(set(getattr(task, 'datasets', []))):
                    continue
                
                # Look for matching output keys that correspond to this task
                # Keys might be like "omat_energy", "dataset_property", etc.
                matching_output_key = None
                for output_key in output.keys():
                    # Check if this output key matches this task
                    if (task_name in output_key or 
                        any(dataset in output_key and task.property in output_key 
                            for dataset in getattr(task, 'datasets', []))):
                        matching_output_key = output_key
                        break
                
                if matching_output_key is None:
                    continue
                    
                head_dict = output[matching_output_key]
                if isinstance(head_dict, dict):
                    # Multiple heads case - select appropriate head
                    if hasattr(task, 'head') and task.head in head_dict:
                        head_name = task.head
                        head_pred = head_dict[head_name]
                        # Extract tensor from prediction (could be nested dict)
                        if isinstance(head_pred, dict):
                            if task.property in head_pred:
                                value = head_pred[task.property]
                            else:
                                # Try to find any tensor value in the dict
                                value = next((v for v in head_pred.values() if isinstance(v, torch.Tensor)), None)
                                if value is None:
                                    continue
                        else:
                            value = head_pred
                    else:
                        # If only one head, use it; otherwise average or pick first
                        head_names = list(head_dict.keys())
                        if len(head_names) == 1:
                            head_pred = head_dict[head_names[0]]
                            # Extract tensor from prediction (could be nested dict)
                            if isinstance(head_pred, dict):
                                if task.property in head_pred:
                                    value = head_pred[task.property]
                                else:
                                    # Try to find any tensor value in the dict
                                    value = next((v for v in head_pred.values() if isinstance(v, torch.Tensor)), None)
                                    if value is None:
                                        continue
                            else:
                                value = head_pred
                        else:
                            # Average multiple heads for this property
                            head_values = []
                            for head_name in head_names:
                                head_pred = head_dict[head_name]
                                # Extract tensor from prediction (could be nested dict)
                                if isinstance(head_pred, dict):
                                    # Look for the property key in the nested dict
                                    if task.property in head_pred:
                                        head_tensor = head_pred[task.property]
                                    else:
                                        # Try to find any tensor value in the dict
                                        head_tensor = next((v for v in head_pred.values() if isinstance(v, torch.Tensor)), None)
                                        if head_tensor is None:
                                            continue
                                elif isinstance(head_pred, torch.Tensor):
                                    head_tensor = head_pred
                                else:
                                    continue
                                head_values.append(head_tensor)
                            
                            if head_values:
                                value = torch.stack(head_values).mean(dim=0)
                            else:
                                # Fallback if no valid tensors found
                                continue
                else:
                    # Single value case (backward compatibility)
                    value = head_dict
                    
                pred_output[task_name] = task.normalizer.denorm(value)
                if undo_element_references and task.element_references is not None:
                    pred_output[task_name] = task.element_references.undo_refs(
                        data_device, pred_output[task_name]
                    )

        # Update available heads mapping for future reference
        self._update_available_heads(pred_output)
        
        return pred_output


def get_dataset_to_tasks_map(tasks: Sequence[Task]) -> dict[str, list[Task]]:
    """Create a mapping from dataset names to their associated tasks.

    Args:
        tasks: A sequence of Task objects to be organized by dataset

    Returns:
        A dictionary mapping dataset names (str) to lists of Task objects
        that are associated with that dataset
    """
    dset_to_tasks_map = defaultdict(list)
    for task in tasks:
        for dataset_name in task.datasets:
            dset_to_tasks_map[dataset_name].append(task)
    return dict(dset_to_tasks_map)


def get_head_to_task_mapping(predictions: dict, tasks: Sequence[Task]) -> dict[str, list[Task]]:
    """Create a mapping from head names to their corresponding tasks.
    
    Args:
        predictions: Dictionary of predictions from the model
        tasks: Sequence of Task objects
        
    Returns:
        Dictionary mapping head names to lists of tasks they correspond to
    """
    head_to_tasks = defaultdict(list)
    
    for head_key in predictions.keys():
        for task in tasks:
            # Check if this head corresponds to this task
            if (head_key == task.name or 
                head_key.endswith(f"_{task.name}") or
                any(dataset in head_key and task.property in head_key 
                    for dataset in task.datasets)):
                head_to_tasks[head_key].append(task)
    
    return dict(head_to_tasks)
