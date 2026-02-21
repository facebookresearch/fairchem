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
from typing import TYPE_CHECKING, Protocol

import hydra
import numpy as np
import ray
import torch
import torch.distributed as dist
from ray import remote
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.elastic.utils.distributed import get_free_port
from torchtnt.framework import PredictUnit, State

from fairchem.core.common import gp_utils
from fairchem.core.common.distutils import (
    CURRENT_DEVICE_TYPE_STR,
    assign_device_for_local_rank,
    get_device_for_local_rank,
    setup_env_local_multi_gpu,
)
from fairchem.core.datasets.atomic_data import AtomicData, warn_if_upcasting
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit.single_atom_patch import (
    handle_single_atom_prediction,
)
from fairchem.core.units.mlip_unit.utils import (
    get_backbone_class_from_checkpoint,
    load_inference_model,
    tf32_context_manager,
)

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint


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

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None: ...

    @property
    def dataset_to_tasks(self) -> dict[str, list]: ...


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
        self._setup_refs(atom_refs, form_elem_refs)

        if inference_settings is None:
            inference_settings = InferenceSettings()
        self._setup_threads(inference_settings)

        if inference_settings.wigner_cuda:
            logging.warning(
                "The wigner_cuda flag is deprecated and will be removed in future versions."
            )

        # Load checkpoint first to get model type
        checkpoint = torch.load(
            inference_model_path, map_location="cpu", weights_only=False
        )

        # Check if model natively supports single atom predictions
        self.supports_single_atoms = checkpoint.model_config.get(
            "supports_single_atoms", False
        )

        # Build model-specific overrides
        final_overrides = self._build_overrides_from_settings(
            checkpoint, overrides, inference_settings
        )

        # Set default dtype during model construction so that non-persistent
        # buffers (SO3_Grid matrices, CoefficientMapping) are created at the
        # requested precision rather than being cast from float32 later.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(inference_settings.base_precision_dtype)

        try:
            # Load model with overrides, passing pre-loaded checkpoint
            self.model, checkpoint = load_inference_model(
                inference_model_path,
                use_ema=True,
                overrides=final_overrides,
                preloaded_checkpoint=checkpoint,
            )

            # Model sets up tasks
            self.model.module.setup_tasks(checkpoint.tasks_config)
        finally:
            torch.set_default_dtype(prev_dtype)

        self._setup_device(device)

        self.model.eval()
        self.lazy_model_intialized = False
        self.inference_settings = inference_settings
        self.assert_on_nans = assert_on_nans
        self._warned_upcast = False

        if self.model.module.direct_forces:
            logging.warning(
                "This is a direct-force model. Direct force predictions may lead to "
                "discontinuities in the potential energy surface and energy conservation errors."
            )

    @property
    def direct_forces(self) -> bool:
        return self.model.module.direct_forces

    @property
    def dataset_to_tasks(self) -> dict[str, list]:
        return self.model.module.dataset_to_tasks

    @property
    def tasks(self) -> dict:
        return self.model.module.tasks

    def set_seed(self, seed: int) -> None:
        """
        Initialize random seeds.
        """
        logging.debug(f"Setting random seed to {seed}")
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _setup_refs(self, atom_refs: dict | None, form_elem_refs: dict | None) -> None:
        """
        Setup element references.
        """
        self.atom_refs = (
            {task.replace("_elem_refs", ""): refs for task, refs in atom_refs.items()}
            if atom_refs
            else {}
        )
        self.form_elem_refs = form_elem_refs or {}

    def _setup_threads(self, settings: InferenceSettings) -> None:
        """
        Configure thread settings.
        """
        if settings.torch_num_threads is not None:
            torch.set_num_threads(settings.torch_num_threads)
            torch.set_num_interop_threads(settings.torch_num_threads)

    def _setup_device(self, device: str) -> None:
        """
        Setup inference device.
        """
        assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'"
        self.device = get_device_for_local_rank() if device == "cuda" else "cpu"

    def _build_overrides_from_settings(
        self,
        checkpoint: MLIPInferenceCheckpoint,
        user: dict | None,
        settings: InferenceSettings,
    ) -> dict:
        """Build backbone config overrides by delegating to model-specific logic."""
        overrides = {} if user is None else dict(user)
        if "backbone" not in overrides:
            overrides["backbone"] = {}

        # Delegate to model-specific classmethod
        backbone_cls = get_backbone_class_from_checkpoint(checkpoint)
        backbone_overrides = backbone_cls.build_inference_settings(settings)

        overrides["backbone"].update(backbone_overrides)

        # User overrides take precedence
        if user is not None and "backbone" in user:
            overrides["backbone"].update(user["backbone"])

        return overrides

    def move_to_device(self):
        self.model.to(self.device)
        for task in self.model.module.tasks.values():
            task.normalizer.to(self.device)
            if task.element_references is not None:
                task.element_references.to(self.device)

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        Validate and set defaults for calculator input data.

        Delegates to the model's backbone for model-specific validation.
        """
        self.model.module.validate_atoms_data(atoms, task_name)

    def predict_step(self, state: State, data: AtomicData) -> dict[str, torch.tensor]:
        return self.predict(data)

    @collate_predictions
    def predict(
        self, data: AtomicData, undo_element_references: bool = True
    ) -> dict[str, torch.tensor]:
        if not self.lazy_model_intialized:
            self._lazy_init(data)

        # Handle single-atom systems (natoms==1 and pbc all False)
        # Skip this check if the model natively supports single atoms
        if not self.supports_single_atoms:
            single_atom_result = handle_single_atom_prediction(
                data=data,
                atom_refs=self.atom_refs,
                tasks=self.tasks,
                device=self.device,
            )
            if single_atom_result is not None:
                return single_atom_result

        # Regular model prediction path
        # this needs to be .clone() to avoid issues with graph parallel modifying this data with MOLE
        data_device = data.to(self.device).clone()

        dtype = self.inference_settings.base_precision_dtype
        if not self._warned_upcast:
            self._warned_upcast = warn_if_upcasting(data_device.pos.dtype, dtype)
        for key, val in data_device:
            if torch.is_tensor(val) and val.is_floating_point():
                data_device[key] = val.to(dtype)

        # Model handles any per-prediction checks (e.g., MOLE consistency)
        self.model.module.on_predict_check(data_device)

        return self._run_inference(data_device, undo_element_references)

    def _lazy_init(self, data: AtomicData) -> None:
        """
        Lazy initialization on first predict call.
        """
        # Model handles its own preparation (MOLE merge, eval mode, etc.)
        self.model.module.prepare_for_inference(data, self.inference_settings)

        self.model.to(self.inference_settings.base_precision_dtype)

        self.move_to_device()

        if self.inference_settings.compile:
            logging.warning(
                "Model is being compiled this might take a while for the first time"
            )
            self.model = torch.compile(self.model, dynamic=True)

        self.lazy_model_intialized = True

    def _run_inference(self, data: AtomicData, undo_refs: bool) -> dict:
        """
        Execute model inference.
        """
        inference_context = torch.no_grad() if self.direct_forces else nullcontext()
        tf32_context = (
            tf32_context_manager() if self.inference_settings.tf32 else nullcontext()
        )

        with inference_context, tf32_context:
            output = self.model(data)
            return self._process_outputs(data, output, undo_refs)

    def _process_outputs(self, data: AtomicData, output: dict, undo_refs: bool) -> dict:
        """
        Denormalize and post-process model outputs.
        """
        pred_output = {}
        for task_name, task in self.model.module.tasks.items():
            pred_output[task_name] = task.normalizer.denorm(
                output[task_name][task.property]
            )
            if self.assert_on_nans:
                assert (
                    torch.isfinite(pred_output[task_name]).all()
                ), f"NaNs/Infs found in prediction for task {task_name}.{task.property}"
            if undo_refs and task.element_references is not None:
                pred_output[task_name] = task.element_references.undo_refs(
                    data, pred_output[task_name]
                )
        return pred_output


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
                torch.distributed.broadcast(self.last_received_atomic_data.cell, src=0)
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
        if inference_settings is None:
            inference_settings = InferenceSettings()
        self.inference_settings = inference_settings
        self._dataset_to_tasks = copy.deepcopy(_mlip_pred_unit.dataset_to_tasks)
        self._validate_atoms_data_fn = _mlip_pred_unit.model.module.validate_atoms_data

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
            self.atomic_data_on_device.cell = data.cell.to(self.local_rank0.device)
            torch.distributed.broadcast(self.atomic_data_on_device.pos, src=0)
            torch.distributed.broadcast(self.atomic_data_on_device.cell, src=0)

        return self.local_rank0.predict(self.atomic_data_on_device)

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        Validate and set defaults for calculator input data.

        Delegates to the model's validate_atoms_data captured at init time.
        """
        self._validate_atoms_data_fn(atoms, task_name)

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
        predict_unit: MLIPPredictUnit,
    ):
        """
        Args:
            server_handle: Ray Serve deployment handle for BatchPredictServer
            predict_unit: Local MLIPPredictUnit used for input validation.
                Validation must run locally because it mutates atoms.info.
        """
        self.server_handle = server_handle
        self._predict_unit = predict_unit

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

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        Validate and set defaults for calculator input data.

        Runs locally (not via Ray Serve) because validation mutates atoms.info.
        """
        self._predict_unit.validate_atoms_data(atoms, task_name)

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
