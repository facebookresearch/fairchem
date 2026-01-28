"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from typing import TYPE_CHECKING, Protocol, Sequence

import hydra
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.utils.distributed import get_free_port
from torchtnt.framework import PredictUnit, State

from fairchem.core.common import gp_utils
from fairchem.core.common.distutils import (
    CURRENT_DEVICE_TYPE_STR,
    assign_device_for_local_rank,
    get_device_for_local_rank,
    setup_env_local_multi_gpu,
)
from fairchem.core.datasets.atomic_data import (
    AtomicData,
    atomicdata_list_to_batch,
)
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit.utils import (
    load_inference_model,
    tf32_context_manager,
)

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.mlip_unit import Task

import ray
from ray import remote
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def identify_single_atom_systems(data: AtomicData) -> tuple[torch.Tensor, torch.Tensor]:
    """Identify single isolated atoms (natoms==1 and pbc all False).

    Returns:
        tuple containing:
            - single_atom_mask: Boolean tensor indicating which systems are single atoms
            - regular_mask: Boolean tensor indicating which systems are regular (not single atoms)
    """
    is_single_atom = data.natoms == 1
    has_no_pbc = ~data.pbc.any(dim=1)
    single_atom_mask = is_single_atom & has_no_pbc
    return single_atom_mask, ~single_atom_mask


def extract_systems_by_mask(data: AtomicData, mask: torch.Tensor) -> AtomicData:
    """Extract systems from a batch based on a boolean mask.

    Args:
        data: The batched AtomicData
        mask: Boolean tensor of length num_graphs indicating which systems to extract

    Returns:
        A new AtomicData batch containing only the selected systems
    """
    indices = torch.where(mask)[0].tolist()
    if not indices:
        return None
    selected_data = [data.get_example(i) for i in indices]
    return atomicdata_list_to_batch(selected_data)


def compute_single_atom_outputs(
    data: AtomicData,
    single_atom_indices: list[int],
    atom_refs: dict[str, dict],
    tasks: dict[str, Task],
    dataset_names: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute outputs for single-atom systems using precomputed atom references.

    Args:
        data: The original batched AtomicData (used to get atomic numbers and charges)
        single_atom_indices: List of indices of single-atom systems in the original batch
        atom_refs: Dictionary mapping dataset names to their atom reference dictionaries
        tasks: Dictionary mapping task names to Task objects
        dataset_names: List of dataset names corresponding to each single-atom system
        device: Device to place output tensors on

    Returns:
        Dictionary mapping task names to output tensors
    """
    outputs = {}

    # Get atomic numbers and charges for single atoms
    atomic_numbers = []
    charges = []
    for idx in single_atom_indices:
        # For single atom systems, the atom index equals the batch index position
        atom_idx = data.natoms[:idx].sum().item() if idx > 0 else 0
        atomic_numbers.append(int(data.atomic_numbers[atom_idx].item()))
        charges.append(int(data.charge[idx].item()))

    num_single_atoms = len(single_atom_indices)

    for task_name, task in tasks.items():
        # Only handle tasks for datasets we're processing
        task_datasets = set(task.datasets)
        if not any(ds in task_datasets for ds in dataset_names):
            continue

        if task.property == "energy":
            energies = []
            for at_num, charge, ds_name in zip(
                atomic_numbers, charges, dataset_names, strict=False
            ):
                if ds_name not in atom_refs:
                    raise ValueError(
                        f"No atom references available for dataset '{ds_name}'. "
                        "Cannot compute single-atom energy."
                    )
                ds_refs = atom_refs[ds_name]
                # Handle both charge-dependent and charge-independent references
                # Same pattern as the original FAIRChemCalculator implementation
                try:
                    energy = ds_refs.get(at_num, {}).get(charge)
                except AttributeError:
                    energy = ds_refs[at_num]

                if energy is None:
                    raise ValueError(
                        f"No atom reference for element {at_num} with charge {charge} "
                        f"in dataset '{ds_name}'."
                    )
                energies.append(float(energy))

            outputs[task_name] = torch.tensor(
                energies, dtype=torch.float32, device=device
            )

        elif task.property == "forces":
            # Forces are zero for isolated single atoms
            outputs[task_name] = torch.zeros(
                (num_single_atoms, 3), dtype=torch.float32, device=device
            )

        elif task.property == "stress":
            # Stress is zero for isolated single atoms (flattened to 9)
            outputs[task_name] = torch.zeros(
                (num_single_atoms, 9), dtype=torch.float32, device=device
            )

    return outputs


def interleave_predictions(
    single_preds: dict[str, torch.Tensor] | None,
    regular_preds: dict[str, torch.Tensor] | None,
    single_atom_mask: torch.Tensor,
    tasks: dict[str, Task],
    data: AtomicData,
) -> dict[str, torch.Tensor]:
    """Merge single-atom and regular predictions maintaining original order.

    Args:
        single_preds: Predictions for single-atom systems (or None if no single atoms)
        regular_preds: Predictions for regular systems (or None if all single atoms)
        single_atom_mask: Boolean mask indicating which systems are single atoms
        tasks: Dictionary mapping task names to Task objects
        data: The original batched AtomicData

    Returns:
        Dictionary mapping task names to merged output tensors in original order
    """
    # Handle edge cases
    if single_preds is None:
        return regular_preds
    if regular_preds is None:
        return single_preds

    num_systems = single_atom_mask.shape[0]
    device = single_atom_mask.device

    # Get indices for interleaving
    single_indices = torch.where(single_atom_mask)[0]
    regular_indices = torch.where(~single_atom_mask)[0]

    merged = {}
    for task_name, task in tasks.items():
        if task_name not in single_preds and task_name not in regular_preds:
            continue

        single_tensor = single_preds.get(task_name)
        regular_tensor = regular_preds.get(task_name)

        # Skip if neither has this task
        if single_tensor is None and regular_tensor is None:
            continue

        if task.level == "system":
            # System-level outputs (energy, stress): one per system
            # Determine output shape and dtype from available tensors
            # Prefer regular_tensor dtype as it comes from the model
            if regular_tensor is not None:
                out_shape = (num_systems,) + regular_tensor.shape[1:]
                dtype = regular_tensor.dtype
            else:
                out_shape = (num_systems,) + single_tensor.shape[1:]
                dtype = single_tensor.dtype

            merged_tensor = torch.zeros(out_shape, dtype=dtype, device=device)

            if single_tensor is not None:
                merged_tensor[single_indices] = single_tensor.to(dtype)
            if regular_tensor is not None:
                merged_tensor[regular_indices] = regular_tensor

            merged[task_name] = merged_tensor

        elif task.level == "atom":
            # Atom-level outputs (forces): need to interleave by atom positions
            # For single atoms, there's one atom per system at the corresponding position
            total_atoms = data.natoms.sum().item()

            # Determine output shape and dtype - prefer model output dtype
            if regular_tensor is not None:
                out_shape = (total_atoms,) + regular_tensor.shape[1:]
                dtype = regular_tensor.dtype
            else:
                out_shape = (total_atoms,) + single_tensor.shape[1:]
                dtype = single_tensor.dtype

            merged_tensor = torch.zeros(out_shape, dtype=dtype, device=device)

            # Place single atom predictions
            if single_tensor is not None:
                single_atom_positions = []
                for idx in single_indices.tolist():
                    atom_pos = data.natoms[:idx].sum().item() if idx > 0 else 0
                    single_atom_positions.append(atom_pos)
                for pos, val in zip(single_atom_positions, single_tensor, strict=False):
                    merged_tensor[pos] = val.to(dtype)

            # Place regular predictions
            if regular_tensor is not None:
                regular_atom_positions = []
                for idx in regular_indices.tolist():
                    start_pos = data.natoms[:idx].sum().item() if idx > 0 else 0
                    num_atoms_in_sys = data.natoms[idx].item()
                    regular_atom_positions.extend(
                        range(start_pos, start_pos + num_atoms_in_sys)
                    )
                merged_tensor[regular_atom_positions] = regular_tensor

            merged[task_name] = merged_tensor

    return merged


def collate_predictions(predict_fn):
    @wraps(predict_fn)
    def collated_predict(
        predict_unit, data: AtomicData, undo_element_references: bool = True
    ):
        # Get the full prediction dictionary from the original predict method
        preds = predict_fn(predict_unit, data, undo_element_references)
        collated_preds = defaultdict(list)
        for i, dataset in enumerate(data.dataset):
            for task in predict_unit.dataset_to_tasks[dataset]:
                if task.level == "system":
                    collated_preds[task.property].append(
                        preds[task.name][i].unsqueeze(0)
                    )
                elif task.level == "atom":
                    collated_preds[task.property].append(
                        preds[task.name][data.batch == i]
                    )
                else:
                    raise RuntimeError(
                        f"Unrecognized task level={task.level} found in data batch at position {i}"
                    )

        return {prop: torch.cat(val) for prop, val in collated_preds.items()}

    return collated_predict


class MLIPPredictUnitProtocol(Protocol):
    def predict(self, data: AtomicData, undo_element_references: bool) -> dict: ...

    @property
    def dataset_to_tasks(self) -> dict[str, list]: ...


def merge_uma_model(model, data):
    # merge the backbone
    model.backbone = model.backbone.merge_MOLE_model(data)

    # merge any heads
    new_output_heads = torch.nn.ModuleDict()
    for head_name, head in model.output_heads.items():
        if hasattr(head, "merge_MOLE_model"):
            new_output_heads[head_name] = head.merge_MOLE_model(data)
        else:
            new_output_heads[head_name] = head
    model.output_heads = new_output_heads


class MLIPPredictUnit(PredictUnit[AtomicData], MLIPPredictUnitProtocol):
    def __init__(
        self,
        inference_model_path: str,
        device: str = "cpu",
        overrides: dict | None = None,
        inference_settings: InferenceSettings | None = None,
        seed: int = 41,
        atom_refs: dict | None = None,
        form_elem_refs: dict | None = None,
        assert_on_nans: bool = False,
    ):
        super().__init__()
        os.environ[CURRENT_DEVICE_TYPE_STR] = device

        self.set_seed(seed)
        # note these are different from the element references used for model training
        self.atom_refs = (
            {task.replace("_elem_refs", ""): refs for task, refs in atom_refs.items()}
            if atom_refs is not None
            else {}
        )
        self.form_elem_refs = form_elem_refs if form_elem_refs is not None else {}

        if inference_settings is None:
            inference_settings = InferenceSettings()
        if inference_settings.torch_num_threads is not None:
            torch.set_num_threads(inference_settings.torch_num_threads)
            torch.set_num_interop_threads(inference_settings.torch_num_threads)

        if overrides is None:
            overrides = {}
        if "backbone" not in overrides:
            overrides["backbone"] = {}
        # always disable always_use_pbc for inference
        overrides["backbone"]["always_use_pbc"] = False
        if inference_settings.activation_checkpointing is not None:
            overrides["backbone"]["activation_checkpointing"] = (
                inference_settings.activation_checkpointing
            )
        if inference_settings.edge_chunk_size is not None:
            overrides["backbone"]["edge_chunk_size"] = (
                inference_settings.edge_chunk_size
            )
        if inference_settings.external_graph_gen is not None:
            overrides["backbone"][
                "otf_graph"
            ] = not inference_settings.external_graph_gen

        if inference_settings.internal_graph_gen_version is not None:
            overrides["backbone"]["radius_pbc_version"] = (
                inference_settings.internal_graph_gen_version
            )

        if inference_settings.wigner_cuda:
            logging.warning(
                "The wigner_cuda flag is deprecated and will be removed in future versions."
            )

        self.model, checkpoint = load_inference_model(
            inference_model_path, use_ema=True, overrides=overrides
        )

        # Check if model natively supports single atom predictions
        self.supports_single_atoms = checkpoint.model_config.get(
            "supports_single_atoms", False
        )

        tasks = [
            hydra.utils.instantiate(task_config)
            for task_config in checkpoint.tasks_config
        ]
        self.tasks = {t.name: t for t in tasks}

        self._dataset_to_tasks = get_dataset_to_tasks_map(self.tasks.values())
        assert set(self._dataset_to_tasks.keys()).issubset(
            set(self.model.module.backbone.dataset_list)
        ), "Datasets in tasks is not a strict subset of datasets in backbone."
        assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'"

        self.device = get_device_for_local_rank() if device == "cuda" else "cpu"

        self.model.eval()

        self.lazy_model_intialized = False
        self.inference_settings = inference_settings

        # store composition embedding of system the model was merged on
        self.merged_on = None

        self.assert_on_nans = assert_on_nans

        if self.direct_forces:
            logging.warning(
                "This is a direct-force model. Direct force predictions may lead to discontinuities in the potential "
                "energy surface and energy conservation errors."
            )

    @property
    def direct_forces(self) -> bool:
        return self.model.module.backbone.direct_forces

    @property
    def dataset_to_tasks(self) -> dict[str, list]:
        return self._dataset_to_tasks

    def set_seed(self, seed: int):
        logging.debug(f"Setting random seed to {seed}")
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
        # Lazy initialization
        if not self.lazy_model_intialized:
            # merge everything on CPU
            if self.inference_settings.merge_mole:
                # replace backbone with non MOE version
                assert (
                    data.natoms.numel() == 1
                ), f"Cannot merge model with multiple systems in batch. Must be exactly 1 system, found {data.natoms.numel()}"

                merge_uma_model(self.model.module, data.clone())

                self.model.eval()
            # move to device
            self.move_to_device()
            if self.inference_settings.compile:
                logging.warning(
                    "Model is being compiled this might take a while for the first time"
                )
                self.model = torch.compile(self.model, dynamic=True)
            self.lazy_model_intialized = True

        # Identify single-atom systems (natoms==1 and pbc all False)
        if not self.supports_single_atoms:
            single_atom_mask, regular_mask = identify_single_atom_systems(data)

        # Check if we need single-atom handling
        if self.supports_single_atoms or not single_atom_mask.any():
            # Fast path: no single atoms or model supports them natively
            result = self._predict_with_model(data, undo_element_references)
        else:
            # Handle batch containing single-atom systems
            result = self._predict_with_single_atoms(
                data, single_atom_mask, regular_mask, undo_element_references
            )

        return result

    def _predict_with_single_atoms(
        self,
        data: AtomicData,
        single_atom_mask: torch.Tensor,
        regular_mask: torch.Tensor,
        undo_element_references: bool,
    ) -> dict[str, torch.Tensor]:
        """Handle prediction for batches containing single-atom systems.

        Single atoms (natoms==1, no PBC) cannot be processed by the model,
        so we use precomputed DFT reference energies instead.

        Args:
            data: The full batch of AtomicData
            single_atom_mask: Boolean mask for single-atom systems
            regular_mask: Boolean mask for regular (non-single-atom) systems
            undo_element_references: Whether to undo element references

        Returns:
            Dictionary mapping task names to prediction tensors
        """
        if not self.atom_refs:
            raise ValueError(
                "Single atom system encountered but no atom_refs available. "
                "Please call fairchem.core.pretrained_mlip.get_predict_unit() "
                "with an appropriate checkpoint name."
            )

        logging.warning(
            "Single atom system(s) detected; using precomputed DFT references "
            "instead of model predictions. Spin multiplicity is ignored for "
            "monoatomic systems."
        )

        # Process regular systems through the model
        regular_preds = None
        if regular_mask.any():
            regular_data = extract_systems_by_mask(data, regular_mask)
            if regular_data is not None:
                regular_preds = self._predict_with_model(
                    regular_data, undo_element_references
                )

        # Process single-atom systems using precomputed references
        single_atom_indices = torch.where(single_atom_mask)[0].tolist()
        single_dataset_names = [data.dataset[i] for i in single_atom_indices]

        single_preds = compute_single_atom_outputs(
            data=data,
            single_atom_indices=single_atom_indices,
            atom_refs=self.atom_refs,
            tasks=self.tasks,
            dataset_names=single_dataset_names,
            device=self.device,
        )

        # Interleave predictions back to original order
        return interleave_predictions(
            single_preds=single_preds,
            regular_preds=regular_preds,
            single_atom_mask=single_atom_mask.to(self.device),
            tasks=self.tasks,
            data=data.to(self.device),
        )

    def _predict_with_model(
        self, data: AtomicData, undo_element_references: bool = True
    ) -> dict[str, torch.tensor]:
        """Run actual ML model prediction on the data.

        This method contains the core model inference logic, separated from
        the single-atom handling in predict().

        Args:
            data: AtomicData batch to run inference on
            undo_element_references: Whether to undo element references in output

        Returns:
            Dictionary mapping task names to prediction tensors
        """
        # this needs to be .clone() to avoid issues with graph parallel modifying this data with MOLE
        data_device = data.to(self.device).clone()

        if self.inference_settings.merge_mole:
            if self.merged_on is None:
                # only get embeddings after moved to final device to get right types
                self.merged_on = self.get_composition_charge_spin_dataset(data_device)
            else:
                this_sys = self.get_composition_charge_spin_dataset(data_device)
                assert (
                    data_device.natoms.numel() == 1
                ), f"Cannot run merged model on batch with multiple systems. Must be exactly 1 system, found {data_device.natoms.numel()}"

                # Normalize compositions by total number of atoms to allow same reduced composition
                merged_comp = self.merged_on[0][0].float()
                this_comp = this_sys[0][0].float()
                merged_comp_norm = merged_comp / merged_comp.sum()
                this_comp_norm = this_comp / this_comp.sum()

                assert merged_comp_norm.isclose(
                    this_comp_norm, rtol=1e-5
                ).all(), "Cannot run on merged model on system. Relative compositions seem different..."
                assert (
                    self.merged_on[0][1] == this_sys[0][1]
                ), f"Cannot run on merged model on system. Charge is different {self.merged_on[0][1]} vs {this_sys[0][1]}"
                assert (
                    self.merged_on[0][2] == this_sys[0][2]
                ), f"Cannot run on merged model on system. Spin is different {self.merged_on[0][2]} vs {this_sys[0][2]}"
                assert (
                    self.merged_on[1] == this_sys[1]
                ), f"Cannot run on merged model on system. Dataset is different {self.merged_on[1]} vs {this_sys[1]}"

        inference_context = torch.no_grad() if self.direct_forces else nullcontext()
        tf32_context = (
            tf32_context_manager() if self.inference_settings.tf32 else nullcontext()
        )

        pred_output = {}
        with inference_context, tf32_context:
            output = self.model(data_device)
            for task_name, task in self.tasks.items():
                pred_output[task_name] = task.normalizer.denorm(
                    output[task_name][task.property]
                )
                if self.assert_on_nans:
                    assert torch.isfinite(
                        pred_output[task_name]
                    ).all(), f"NaNs/Infs found in prediction for task {task_name}.{task.property}"
                if undo_element_references and task.element_references is not None:
                    pred_output[task_name] = task.element_references.undo_refs(
                        data_device, pred_output[task_name]
                    )

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


def move_tensors_to_cpu(data):
    """
    Recursively move all PyTorch tensors in a nested data structure to CPU.

    Args:
        data: Input data structure (dict, list, tuple, tensor, or other)

    Returns:
        Data structure with all tensors moved to CPU
    """
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {key: move_tensors_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_cpu(item) for item in data)
    else:
        # Return as-is for non-tensor types (str, int, float, etc.)
        return data


class MLIPWorkerLocal:
    def __init__(
        self,
        worker_id: int,
        world_size: int,
        predictor_config: dict,
        master_port: int | None = None,
        master_address: str | None = None,
    ):
        self.worker_id = worker_id
        self.world_size = world_size
        self.predictor_config = predictor_config
        self.master_address = (
            ray.util.get_node_ip_address() if master_address is None else master_address
        )
        self.master_port = get_free_port() if master_port is None else master_port
        self.is_setup = False
        self.last_received_atomic_data = None

    def get_master_address_and_port(self):
        return (self.master_address, self.master_port)

    def get_device_for_local_rank(self):
        return get_device_for_local_rank()

    def _distributed_setup(
        self,
    ):
        # initialize distributed environment
        # TODO, this wont work for multi-node, need to fix master addr
        logging.info(f"Initializing worker {self.worker_id}...")
        setup_env_local_multi_gpu(self.worker_id, self.master_port, self.master_address)

        device = self.predictor_config.get("device", "cpu")
        assign_device_for_local_rank(device == "cpu", 0)
        backend = "gloo" if device == "cpu" else "nccl"
        dist.init_process_group(
            backend=backend,
            rank=self.worker_id,
            world_size=self.world_size,
        )
        gp_utils.setup_graph_parallel_groups(self.world_size, backend)
        self.predict_unit = hydra.utils.instantiate(self.predictor_config)
        self.device = get_device_for_local_rank()
        logging.info(
            f"Worker {self.worker_id}, gpu_id: {ray.get_gpu_ids()}, loaded predict unit: {self.predict_unit}, "
            f"on port {self.master_port}, with device: {self.device}, config: {self.predictor_config}"
        )
        self.is_setup = True

    def predict(
        self, data: AtomicData, use_nccl: bool = False
    ) -> dict[str, torch.tensor] | None:
        if not self.is_setup:
            self._distributed_setup()

        out = self.predict_unit.predict(data)
        if self.worker_id == 0:
            return move_tensors_to_cpu(out)

        if self.worker_id != 0 and use_nccl:
            self.last_received_atomic_data = data.to(self.device)
            while True:
                torch.distributed.broadcast(self.last_received_atomic_data.pos, src=0)
                self.predict_unit.predict(self.last_received_atomic_data)

        return None


@remote
class MLIPWorker(MLIPWorkerLocal):
    pass


class ParallelMLIPPredictUnit(MLIPPredictUnitProtocol):
    def __init__(
        self,
        inference_model_path: str,
        device: str = "cpu",
        overrides: dict | None = None,
        inference_settings: InferenceSettings | None = None,
        seed: int = 41,
        atom_refs: dict | None = None,
        form_elem_refs: dict | None = None,
        assert_on_nans: bool = False,
        num_workers: int = 1,
        num_workers_per_node: int = 8,
        log_level: int = logging.INFO,
    ):
        super().__init__()
        _mlip_pred_unit = MLIPPredictUnit(
            inference_model_path=inference_model_path,
            device="cpu",
            overrides=overrides,
            inference_settings=inference_settings,
            seed=seed,
            atom_refs=atom_refs,
            form_elem_refs=form_elem_refs,
        )
        self.inference_settings = inference_settings
        self._dataset_to_tasks = copy.deepcopy(_mlip_pred_unit.dataset_to_tasks)

        predict_unit_config = {
            "_target_": "fairchem.core.units.mlip_unit.predict.MLIPPredictUnit",
            "inference_model_path": inference_model_path,
            "device": device,
            "overrides": overrides,
            "inference_settings": inference_settings,
            "seed": seed,
            "atom_refs": atom_refs,
            "form_elem_refs": form_elem_refs,
            "assert_on_nans": assert_on_nans,
        }

        logging.basicConfig(
            level=log_level,
            force=True,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
        )
        # Optional: keep Ray/uvicorn chatty logs in check
        logging.getLogger("ray").setLevel(log_level)
        logging.getLogger("uvicorn").setLevel(log_level)
        if not ray.is_initialized():
            # in CI envrionment, we want to control the number of CPUs allocated to limit the pool of IDLE ray workers
            if os.environ.get("CI"):
                logging.info(
                    f"CI environment detected, initializing ray with limited CPUs: {num_workers_per_node}"
                )
                ray.init(
                    logging_level=log_level,
                    num_cpus=num_workers_per_node,
                    runtime_env={
                        "env_vars": {"RAY_DEBUG": "1"},
                    },
                )
            else:
                ray.init(logging_level=log_level)

        self.atomic_data_on_device = None

        num_nodes = math.ceil(num_workers / num_workers_per_node)
        num_workers_on_node_array = [num_workers_per_node] * num_nodes
        if num_workers % num_workers_per_node > 0:
            num_workers_on_node_array[-1] = num_workers % num_workers_per_node
        logging.info(
            f"Creating placement groups with {num_workers_on_node_array} workers on {device}"
        )

        # first create one placement group for each node
        num_gpu_per_worker = 1 if device == "cuda" else 0
        placement_groups = []
        for workers in num_workers_on_node_array:
            bundle = {"CPU": workers}
            if device == "cuda":
                bundle["GPU"] = workers
            pg = ray.util.placement_group([bundle], strategy="STRICT_PACK")
            placement_groups.append(pg)
        ray.get(pg.ready())  # Wait for each placement group to be scheduled

        # Need to still place worker to occupy space, otherwise ray double books this GPU
        rank0_worker = MLIPWorker.options(
            num_gpus=num_gpu_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_groups[0],
                placement_group_bundle_index=0,  # Use the first (and only) bundle in the PG
                placement_group_capture_child_tasks=True,  # Ensure child tasks also run in this PG
            ),
        ).remote(0, num_workers, predict_unit_config)

        local_gpu_or_cpu = ray.get(rank0_worker.get_device_for_local_rank.remote())
        os.environ[CURRENT_DEVICE_TYPE_STR] = local_gpu_or_cpu

        self.workers = []
        self.local_rank0 = MLIPWorkerLocal(
            worker_id=0,
            world_size=num_workers,
            predictor_config=predict_unit_config,
        )
        master_addr, master_port = self.local_rank0.get_master_address_and_port()
        logging.info(f"Started rank0 on {master_addr}:{master_port}")

        # next place all ranks in order and pack them on placement groups
        # ie: rank0-7 -> placement group 0, 8->15 -> placement group 1 etc.
        worker_id = 0
        for pg_idx, pg in enumerate(placement_groups):
            workers = num_workers_on_node_array[pg_idx]
            logging.info(
                f"Launching workers for placement group {pg_idx} (Node {pg_idx}), workers={workers}"
            )

            for i in range(workers):
                # skip the first one because it's already been initialized above
                if pg_idx == 0 and i == 0:
                    worker_id += 1
                    continue
                # Each actor requests 1 worker worth of resources and uses the specific placement group
                actor = MLIPWorker.options(
                    num_gpus=num_gpu_per_worker,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=0,  # Use the first (and only) bundle in the PG
                        placement_group_capture_child_tasks=True,  # Ensure child tasks also run in this PG
                    ),
                ).remote(
                    worker_id,
                    num_workers,
                    predict_unit_config,
                    master_port,
                    master_addr,
                )
                self.workers.append(actor)
                worker_id += 1

    def predict(self, data: AtomicData) -> dict[str, torch.tensor]:
        # put the reference in the object store only once
        if not self.inference_settings.merge_mole or self.atomic_data_on_device is None:
            data_ref = ray.put(data)
            # this will put the ray works into an infinite loop listening for broadcasts
            _futures = [
                w.predict.remote(data_ref, use_nccl=self.inference_settings.merge_mole)
                for w in self.workers
            ]
            self.atomic_data_on_device = data.clone()
        else:
            self.atomic_data_on_device.pos = data.pos.to(self.local_rank0.device)
            torch.distributed.broadcast(self.atomic_data_on_device.pos, src=0)

        return self.local_rank0.predict(self.atomic_data_on_device)

    @property
    def dataset_to_tasks(self) -> dict[str, list]:
        return self._dataset_to_tasks


class BatchServerPredictUnit(MLIPPredictUnitProtocol):
    """
    PredictUnit wrapper that uses Ray Serve for batched inference.

    This provides a clean interface compatible with MLIPPredictUnitProtocol
    while leveraging Ray Serve's batching capabilities under the hood.
    """

    def __init__(
        self,
        server_handle,
    ):
        """
        Args:
            server_handle: Ray Serve deployment handle for BatchPredictServer
            dataset_to_tasks: Mapping from dataset names to their associated tasks
            atom_refs: Optional atom references dictionary
        """
        self.server_handle = server_handle

    def predict(self, data: AtomicData, undo_element_references: bool = True) -> dict:
        """
        Args:
            data: AtomicData object (single system)
            undo_element_references: Whether to undo element references

        Returns:
            Prediction dictionary
        """
        result = self.server_handle.predict.remote(
            data, undo_element_references
        ).result()
        return result

    @property
    def dataset_to_tasks(self) -> dict:
        return self.server_handle.get_predict_unit_attribute.remote(
            "dataset_to_tasks"
        ).result()

    @property
    def atom_refs(self) -> dict | None:
        return self.server_handle.get_predict_unit_attribute.remote(
            "atom_refs"
        ).result()

    @property
    def inference_settings(self) -> InferenceSettings:
        return self.server_handle.get_predict_unit_attribute.remote(
            "inference_settings"
        ).result()
