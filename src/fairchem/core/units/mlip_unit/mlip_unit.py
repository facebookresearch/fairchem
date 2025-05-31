"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.profiler import record_function
from torchtnt.framework import EvalUnit, State, TrainUnit
from torchtnt.utils.prepare_module import prepare_module

from fairchem.core.common import distutils, gp_utils
from fairchem.core.common.distutils import (
    get_device_for_local_rank,
)
from fairchem.core.common.logger import WandBSingletonLogger
from fairchem.core.common.registry import registry
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.mt_collater import MTCollater
from fairchem.core.modules.normalization.element_references import (  # noqa: TCH001
    ElementReferences,
)
from fairchem.core.modules.normalization.normalizer import Normalizer  # noqa: TCH001
from fairchem.core.modules.scheduler import CosineLRLambda
from fairchem.core.units.mlip_unit._metrics import Metrics, get_metrics_fn
from fairchem.core.units.mlip_unit.api.inference import (
    MLIPInferenceCheckpoint,
)
from fairchem.core.units.mlip_unit.utils import load_inference_model

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class OutputSpec:
    dim: list[int]
    dtype: str


class TrainStrategy(str, Enum):
    DDP = "ddp"
    FSDP = "fsdp"


@dataclass
class Task:
    name: str
    level: str
    property: str
    loss_fn: torch.nn.Module
    out_spec: OutputSpec
    normalizer: Normalizer
    datasets: list[str]
    element_references: Optional[ElementReferences] = None
    metrics: list[str] = field(default_factory=list)
    train_on_free_atoms: bool = True
    eval_on_free_atoms: bool = True


def convert_train_checkpoint_to_inference_checkpoint(
    dcp_checkpoint_loc: str, checkpoint_loc: str
) -> None:
    dcp_to_torch_save(dcp_checkpoint_loc, checkpoint_loc)

    inference_ckpt = torch.load(
        checkpoint_loc, map_location="cpu", weights_only=False
    )  # DCP model config
    train_eval_unit_state = inference_ckpt["config"]["runner"]["train_eval_unit"]
    unit_state = inference_ckpt["unit_state"]
    torch.save(
        MLIPInferenceCheckpoint(
            model_state_dict=unit_state["model"],
            ema_state_dict=unit_state["ema"],
            model_config=train_eval_unit_state["model"],
            tasks_config=train_eval_unit_state["tasks"],
        ),
        checkpoint_loc,
    )


def initialize_finetuning_model(
    checkpoint_location: str, overrides: dict | None = None, heads: dict | None = None
) -> torch.nn.Module:
    model, checkpoint = load_inference_model(checkpoint_location, overrides)

    logging.warning(
        f"initialize_finetuning_model starting from checkpoint_location: {checkpoint_location}"
    )

    checkpoint.model_config["heads"] = deepcopy(heads)
    model.finetune_model_full_config = checkpoint.model_config

    model.output_heads = None
    model.heads = heads
    del model.output_heads
    model.output_heads = {}
    head_names_sorted = sorted(heads.keys())
    assert len(set(head_names_sorted)) == len(
        head_names_sorted
    ), "Head names must be unique!"
    for head_name in head_names_sorted:
        head_config = heads[head_name]
        if "module" not in head_config:
            raise ValueError(
                f"{head_name} head does not specify module to use for the head"
            )
        module_name = head_config.pop("module")
        model.output_heads[head_name] = registry.get_model_class(module_name)(
            model.backbone,
            **head_config,
        )
    model.output_heads = torch.nn.ModuleDict(model.output_heads)
    return model


def get_output_mask(batch: AtomicData, task: Task) -> dict[str, torch.Tensor]:
    """Get a dictionary of boolean masks for each task and dataset in a batch.

    Comment(@abhshkdz): Structures in our `batch` are a mix from various
    sources, e.g. OC20, OC22, etc. That means for each loss computation,
    we need to pull out the attribute of interest from each structure.
    E.g. oc20_energy from OC20 structures, oc22_energy from OC22
    structures etc. Set up those mappings here. Supports two kinds for
    now: 1) for each structure-level output, mapping from output head
    to boolean indexing map for `out` and `batch`, s.t. we can index like
    batch.oc20_energy[oc20_map] for oc20_energy loss calculation. 2) for
    each atom-level output, a similar mapping from output head to boolean
    indexing map. s.t. we can index like batch.oc20_forces[oc20_map].
    """

    output_masks = {task.name: torch.isfinite(batch[task.name])}
    if "forces" in task.name:
        output_masks[task.name] = output_masks[task.name].all(dim=1)

    for dset in set(batch.dataset_name):
        dset_mask = torch.from_numpy(np.array(batch.dataset_name) == dset).to(
            batch.pos.device
        )
        if task.level == "atom":
            dset_mask = torch.repeat_interleave(dset_mask, batch.natoms)
            output_masks[f"{dset}.{task.name}"] = dset_mask & output_masks[task.name]
        elif "stress" in task.name:
            assert output_masks[task.name].shape[0] == dset_mask.shape[0]
            # we need to expand the target mask shape to match the output shape
            target_shape = output_masks[task.name].shape
            dset_expanded = dset_mask.view(len(dset_mask), -1).expand(target_shape)
            output_masks[f"{dset}.{task.name}"] = (
                dset_expanded & output_masks[task.name]
            )
        else:
            output_masks[f"{dset}.{task.name}"] = dset_mask & output_masks[task.name]

    return output_masks


def get_output_masks(
    batch: AtomicData, tasks: Sequence[Task]
) -> dict[str, torch.Tensor]:
    """Same as above but for a list of tasks."""
    output_masks = {}
    for task in tasks:
        output_masks.update(get_output_mask(batch, task))

    return output_masks


def compute_loss(
    tasks: Sequence[Task], predictions: dict[str, torch.Tensor], batch: AtomicData
) -> dict[str, float]:
    """Compute loss given a sequence of tasks

    Args:
        tasks: a sequence of Task
        predictions: dictionary of predictions
        batch: data batch

    Returns:
        dictionary of losses for each task
    """

    batch_size = batch.natoms.numel()
    num_atoms_in_batch = batch.natoms.sum()

    free_mask = batch.fixed == 0
    output_masks = get_output_masks(batch, tasks)

    loss_dict = {}
    for task in tasks:
        # TODO this might be a very expensive clone
        target = batch[task.name].clone()
        output_mask = output_masks[task.name]

        # element references are applied to the target before normalization
        # TODO the current implementation will not work for single tasks with
        # multiple element references or normalizers
        # apply element references to the target
        if task.element_references is not None:
            with record_function("element_refs"):
                target = task.element_references.apply_refs(batch, target)
        # Normalize the target
        target = task.normalizer.norm(target)

        # Setting up a mult_mask to multiply the loss by 1 for valid atoms
        # or structures, and by 0 for the others. This is better than
        # indexing the loss with the mask, because it ensures that the
        # computational graph is correct.

        # this is related to how Hydra outputs stuff in nested dicts:
        # ie: oc20_energy.energy
        pred_for_task = predictions[task.name][task.property]
        if task.level == "atom":
            pred_for_task = pred_for_task.view(num_atoms_in_batch, -1)
        else:
            pred_for_task = pred_for_task.view(batch_size, -1)

        if task.level == "atom" and task.train_on_free_atoms:
            mult_mask = free_mask & output_mask
        else:
            mult_mask = output_mask
        loss_dict[task.name] = task.loss_fn(
            pred_for_task,
            target,
            mult_mask=mult_mask,
            natoms=batch.natoms,
        )

    # Sanity check to make sure the compute graph is correct.
    for lc in loss_dict.values():
        assert hasattr(lc, "grad_fn")

    return loss_dict


def compute_metrics(
    task: Task,
    predictions: dict[str, torch.Tensor],
    batch: AtomicData,
    dataset_name: str | None = None,
) -> dict[str:Metrics]:
    """Compute metrics and update running metrics for a given task

    Args:
        task: a Task
        predictions: dictionary of predictions
        batch: data batch
        dataset_name: optional, if given compute metrics for given task using only labels from the given dataset
        running_metrics: optional dictionary of previous metrics to update.

    Returns:
        dictionary of (updated) metrics
    """
    # output masks include task level mask, and task.dataset level masks.
    mask_key = task.name if dataset_name is None else f"{dataset_name}.{task.name}"
    output_mask = get_output_mask(batch, task)[mask_key]

    natoms = torch.repeat_interleave(batch.natoms, batch.natoms)
    if task.level == "atom":
        if task.eval_on_free_atoms is True:
            output_mask = output_mask & (batch.fixed == 0)

        natoms_masked = natoms[output_mask]
        output_size = natoms_masked.numel()
    elif "stress" in task.name:
        natoms_masked = batch.natoms[output_mask.all(dim=1)]
        output_size = output_mask.sum()
    else:
        natoms_masked = batch.natoms[output_mask]
        output_size = output_mask.sum()

    # no metrics to report
    if output_size == 0:
        return {metric_name: Metrics() for metric_name in task.metrics}

    target_masked = batch[task.name][output_mask]
    pred = predictions[task.name][task.property].clone()
    # denormalize the prediction
    pred = task.normalizer.denorm(pred)
    # undo element references for energy tasks
    if task.element_references is not None:
        pred = task.element_references.undo_refs(
            batch,
            pred,
        )
    pred_masked = pred[output_mask]

    # reshape: (num_atoms_in_batch, -1) or (num_systems_in_batch, -1)
    # if task.level == "atom" or "stress" not in task.name:
    #     # TODO do not reshape based on task.name
    #     # tensor.view(..., -1) will add an extra dimension even if the input shape is the same as output shape
    #     # this will cause downstream broadcast operations to be wrong and is dangerous
    #     target_masked = target_masked.view(output_size, -1)

    assert (
        target_masked.shape == pred_masked.shape
    ), f"shape mismatch for {task} target: target: {target_masked.shape}, pred: {pred_masked.shape}"

    # TODO need a cleaner interface for this...
    # Lets package up the masked target and prediction into a dictionary,
    # so that it plays nicely with the metrics functions
    # this does not work for metrics that use more than a single prediction key, ie energy_forces_within_threshold
    target_dict = {task.property: target_masked, "natoms": natoms_masked}
    pred_dict = {task.property: pred_masked}

    # this is different from original mt trainer, it assumes a Task has a single normalizer\
    # (even if it is used across datasets)
    metrics = {}
    for metric_name in task.metrics:
        # now predict the metrics and update them
        metric_fn = get_metrics_fn(metric_name)
        metrics[metric_name] = metric_fn(
            pred_dict, target_dict, key=task.property
        )  # TODO change this to return Metrics dataclass

    return metrics


def mt_collater_adapter(tasks: list[Task], exclude_keys: list[str]):
    # this is required because the MTCollater needs the old json formated task config so we need to convert it here
    task_config_old = {}
    for task in tasks:
        task_config_old[task.name] = {
            "level": task.level,
            "property": task.property,
            "out_spec": {
                "dim": list(task.out_spec.dim),
                "dtype": str(task.out_spec.dtype),
            },
            "datasets": task.datasets,
            "train_on_free_atoms": task.train_on_free_atoms,
            "eval_on_free_atoms": task.eval_on_free_atoms,
        }
    return MTCollater(task_config_old, exclude_keys)


def _get_consine_lr_scheduler(
    warmup_factor: float,
    warmup_epochs: float,
    lr_min_factor: float,
    n_iters_per_epoch: int,
    optimizer: torch.optim.Optimizer,
    epochs: Optional[int] = None,
    steps: Optional[int] = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    assert (epochs is not None) ^ (
        steps is not None
    ), "Exactly one of epochs or steps must be None/Not-None (XOR)"
    scheduler_steps = int(epochs * n_iters_per_epoch) if steps is None else steps
    # fixed function for constructing a LambdaLR scheduler
    lambda_fn = CosineLRLambda(
        warmup_epochs=int(warmup_epochs * n_iters_per_epoch),
        warmup_factor=warmup_factor,
        epochs=scheduler_steps,
        lr_min_factor=lr_min_factor,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)


def _get_optimizer_wd(
    optimizer_fn: callable, model: torch.nn.Module
) -> torch.optim.Optimizer:
    weight_decay = optimizer_fn.keywords.get("weight_decay", 0)
    # split the params into the params with and without WD
    # some fairchem models implement a no_weight_decay and this
    # is used to return params such as embeddings that should have no wd.
    # TODO: use a protocol here instead of guessing that the attr exists
    if weight_decay > 0 and hasattr(model, "no_weight_decay"):
        model_params_no_wd = model.no_weight_decay()

        params_decay, params_no_decay, name_no_decay = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if any(name.endswith(skip_name) for skip_name in model_params_no_wd):
                params_no_decay.append(param)
                name_no_decay.append(name)
            else:
                params_decay.append(param)

        if distutils.is_master():
            logging.info("Parameters without weight decay:")
            logging.info(name_no_decay)

        optimizer = optimizer_fn(
            params=[
                {"params": params_no_decay, "weight_decay": 0},
                {"params": params_decay, "weight_decay": weight_decay},
            ]
        )
    else:
        optimizer = optimizer_fn(params=model.parameters())
    return optimizer


def _reshard_fsdp(model: torch.nn.Module) -> None:
    for m in FullyShardedDataParallel.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)


def set_sampler_state(state: State, epoch: int, step_start: int) -> None:
    logging.info(
        f"at beginning of epoch {epoch}, setting sampler start step to {step_start}"
    )
    if hasattr(state.train_state.dataloader, "batch_sampler"):
        # the batch sampler must have the set_epoch_and_start_iteration to be able to restore state
        assert hasattr(
            state.train_state.dataloader.batch_sampler,
            "set_epoch_and_start_iteration",
        )
        state.train_state.dataloader.batch_sampler.set_epoch_and_start_iteration(
            epoch, step_start
        )
    else:
        logging.warning(
            "AtomicData sampler not found in dataloader, no dataloader state restored!!"
        )


class MLIPTrainEvalUnit(TrainUnit[AtomicData], EvalUnit[AtomicData], Stateful):
    def __init__(
        self,
        job_config: DictConfig,
        model: torch.nn.Module,
        optimizer_fn: callable,
        cosine_lr_scheduler_fn: callable,
        tasks: list[Task],
        bf16: bool = False,
        print_every: int = 10,
        clip_grad_norm: float | None = None,
        ema_decay: float = 0.999,
        train_strategy: TrainStrategy = TrainStrategy.DDP,
        debug_checksums_save_path: str | None = None,
        profile_flops: bool = False,
    ):
        super().__init__()
        self.job_config = job_config
        self.tasks = tasks
        self.profile_flops = profile_flops

        for task in tasks:
            if task.element_references is not None:
                task.element_references.to(torch.device(get_device_for_local_rank()))

        # placeholder for autocast code, may need to move out to common
        self.bf16 = bf16
        self.autocast_enabled = self.bf16
        self.autocast_dtype = torch.bfloat16

        self.finetune_model_full_config = getattr(
            model, "finetune_model_full_config", None
        )

        # call optimizer function between wrapping in DDP
        # this is required for models that have a no_weight_decay function
        self.optimizer = _get_optimizer_wd(optimizer_fn, model)

        self.logger = (
            WandBSingletonLogger.get_instance()
            if distutils.is_master()
            and not self.job_config.debug
            and self.job_config.logger
            else None
        )
        self.debug_checksums_save_path = debug_checksums_save_path
        if self.debug_checksums_save_path:
            os.makedirs(debug_checksums_save_path, exist_ok=True)
        self.print_every = print_every
        self.clip_grad_norm = clip_grad_norm
        self.dp_world_size = (
            gp_utils.get_dp_world_size()
            if gp_utils.initialized()
            else distutils.get_world_size()
        )

        self.num_params = sum(p.numel() for p in model.parameters())
        if self.logger:
            self.logger.log_summary(
                {"num_params": self.num_params, "dp_world_size": self.dp_world_size}
            )

        model.to(torch.device(get_device_for_local_rank()))

        self.ema_decay = ema_decay
        self.ema_model = None
        self.train_strategy = train_strategy
        if train_strategy == TrainStrategy.DDP:
            self.model = prepare_module(
                model, device=torch.device(get_device_for_local_rank()), strategy="ddp"
            )
            if self.ema_decay is not None:
                self.ema_model = torch.optim.swa_utils.AveragedModel(
                    self.model,
                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                        self.ema_decay
                    ),
                )
        elif train_strategy == TrainStrategy.FSDP:
            # only wrap MOElinears for now, these are the layers that has large parameter size
            from fairchem.core.models.uma.escn_moe import MOELinear

            shard_group_size = job_config.scheduler.ranks_per_node
            mesh_2d = init_device_mesh(
                get_device_for_local_rank(),
                mesh_shape=(
                    int(distutils.get_world_size() // shard_group_size),
                    shard_group_size,
                ),
                mesh_dim_names=("replicate", "shard"),
            )
            fsdp_params = {
                "sharding_strategy": ShardingStrategy.HYBRID_SHARD,
                "device_mesh": mesh_2d,
                "auto_wrap_policy": ModuleWrapPolicy(module_classes=[MOELinear]),
                "use_orig_params": True,
            }
            if self.ema_decay is not None:
                ema_model = torch.optim.swa_utils.AveragedModel(
                    model,
                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                        self.ema_decay
                    ),
                )
                self.ema_model = FullyShardedDataParallel(ema_model, **fsdp_params)

                FullyShardedDataParallel.set_state_dict_type(
                    self.ema_model,
                    StateDictType.SHARDED_STATE_DICT,
                    state_dict_config=ShardedStateDictConfig(),
                )
            self.model = FullyShardedDataParallel(model, **fsdp_params)
            FullyShardedDataParallel.set_state_dict_type(
                self.model,
                StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

            logging.info(f"Create device mesh {mesh_2d} for FSDP")
        else:
            raise ValueError(f"Unknown Training Strategy {train_strategy}")

        # setup eval unit, make it share the same model as this unit and turn off the logger
        self.eval_unit = MLIPEvalUnit(job_config=job_config, model=None, tasks=tasks)
        eval_model = self.ema_model if self.ema_model is not None else self.model
        self.eval_unit.setup_train_eval_unit(eval_model)

        self.cosine_lr_scheduler_fn = cosine_lr_scheduler_fn
        self.scheduler = None

    def load_scheduler(self, train_dataloader_size: int) -> int:
        self.scheduler = self.cosine_lr_scheduler_fn(
            n_iters_per_epoch=train_dataloader_size,
            optimizer=self.optimizer,
        )

    def on_train_start(self, state: State) -> None:
        self.model.train()
        if self.profile_flops:
            # runtime import to make this feature optional
            from fairchem.core.components.common.flops_profile import get_flops_profile

            data = next(iter(state.train_state.dataloader)).to(
                get_device_for_local_rank()
            )
            flops = get_flops_profile(self.model, data, verbose=True)
            num_atoms_local = data.natoms.sum().item()
            flops_per_atom_param = flops / self.num_params / num_atoms_local
            if self.logger:
                self.logger.log_summary(
                    {
                        "train/fwd_flops": flops,
                        "train/fwd_flops_per_atom_param": flops_per_atom_param,
                    }
                )
            if "edge_index" in data:
                num_edges_local = data.edge_index.shape[1]
                flops_per_edge_param = flops / self.num_params / num_edges_local
                if self.logger:
                    self.logger.log_summary(
                        {
                            "train/fwd_flops_per_edge_param": flops_per_edge_param,
                            "train/num_edges_local": num_edges_local,
                        }
                    )

        if self.scheduler is None:
            self.load_scheduler(len(state.train_state.dataloader))
        self.previous_wall_time = time.time()
        # this should only be non-zero if we are resuming from a run
        epoch = self.train_progress.num_epochs_completed
        start_step = self.train_progress.num_steps_completed_in_epoch
        logging.info(f"on_train_start: setting sampler state to {epoch}, {start_step}")
        set_sampler_state(
            state,
            epoch,
            start_step,
        )

    def on_train_epoch_start(self, state: State) -> None:
        # we can safely set start steps to 0 here because this callback is NOT called when resuming from a run
        # https://github.com/pytorch/tnt/blob/master/torchtnt/framework/train.py#L187
        set_sampler_state(state, self.train_progress.num_epochs_completed, 0)

    def train_step(self, state: State, data: AtomicData) -> None:
        try:
            device = get_device_for_local_rank()
            batch_on_device = data.to(device)
            step = self.train_progress.num_steps_completed
            epoch = (
                self.train_progress.num_epochs_completed
                + self.train_progress.num_steps_completed_in_epoch
                / float(len(state.train_state.dataloader))
            )
            with torch.autocast(
                device_type=device,
                enabled=self.autocast_enabled,
                dtype=self.autocast_dtype,
            ):
                with record_function("forward"):
                    pred = self.model.forward(batch_on_device)
                with record_function("compute_loss"):
                    loss_dict = compute_loss(self.tasks, pred, batch_on_device)
            scalar_loss = sum(loss_dict.values())
            self.optimizer.zero_grad()
            with record_function("backward"):
                scalar_loss.backward()

            if self.debug_checksums_save_path:
                gp_size = 0
                gp_rank = 0
                if gp_utils.initialized():
                    gp_size = gp_utils.get_dp_world_size()
                    gp_rank = gp_utils.get_dp_rank()

                ddp_size = distutils.get_world_size()
                ddp_rank = distutils.get_rank()

                fn = os.path.join(
                    self.debug_checksums_save_path,
                    f"ddp{ddp_size}.{ddp_rank}_gp{gp_size}.{gp_rank}_step{step}.txt",
                )
                with open(fn, "w") as f:
                    f.write(f"Loss,{scalar_loss.item()}\n")
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            f.write(
                                f"Param,{step},{name},{param.abs().mean().item()}\n"
                            )
                            f.write(
                                f"Grad,{step},{name},{param.grad.abs().mean().item()}\n"
                            )

            if self.clip_grad_norm is not None:
                if self.train_strategy == TrainStrategy.FSDP:
                    grad_norm = self.model.clip_grad_norm_(
                        max_norm=self.clip_grad_norm,
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_grad_norm,
                    )

                if self.logger:
                    self.logger.log({"train/grad_norm": grad_norm}, step=step)
            self.optimizer.step()
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)

            self.optimizer.zero_grad(set_to_none=True)

            time_delta = time.time() - self.previous_wall_time
            self.previous_wall_time = time.time()
            num_atoms_local = data.natoms.sum().item()
            num_samples_local = data.natoms.numel()
            log_dict = {
                "train/loss": scalar_loss.item(),
                "train/lr": self.scheduler.get_lr()[0],
                "train/step": step,
                "train/epoch": epoch,
                "train/samples_per_second(approx)": num_samples_local
                * self.dp_world_size
                / float(time_delta),
                "train/atoms_per_second(approx)": num_atoms_local
                * self.dp_world_size
                / float(time_delta),
                "train/num_atoms_on_rank": num_atoms_local,
                "train/num_samples_on_rank": num_samples_local,
            }

            if self.logger:
                self.logger.log(log_dict, step=step, commit=True)

            if step % self.print_every == 0:
                logging.info(log_dict)

            self.scheduler.step()

            # TODO: compute metrics
            self.last_loss = scalar_loss.item()
        except Exception:
            logging.error(
                f"Exception during training! On step {self.train_progress.num_steps_completed}"
            )
            logging.error(
                f"Data info: {data}\ndata.dataset: {data.dataset}\n"
                + f"data.sid: {data.sid if 'sid' in data else None}\n"
                + f"data.natoms: {data.natoms}\n"
                + f"data.atomic_numbers: {data.atomic_numbers}"
            )
            raise

    def on_train_end(self, state: State) -> None:
        logging.info(
            f"Training Completed {self.train_progress.num_steps_completed} steps"
        )

    def state_dict(
        self,
    ) -> dict[str, Any]:
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        ema_state_dict = (
            get_model_state_dict(self.ema_model) if self.ema_model else None
        )
        state = {
            "progress": self.train_progress.state_dict(),
            "model": model_state_dict,
            "ema": ema_state_dict,
            "optim": optimizer_state_dict,
            "scheduler": self.scheduler.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        self.train_progress.load_state_dict(state_dict["progress"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.ema_model is not None:
            set_model_state_dict(
                self.ema_model,
                model_state_dict=state_dict["ema"],
            )
            self.ema_model.load_state_dict(state_dict["ema"])

    def eval_step(self, state: State, data: AtomicData) -> None:
        self.eval_unit.eval_step(state, data)

    def on_eval_epoch_start(self, state: State) -> None:
        self.eval_unit.on_eval_epoch_start(state)

    def on_eval_epoch_end(self, state: State) -> None:
        metrics = self.eval_unit.on_eval_epoch_end(state)
        if self.logger is not None:
            self.logger.log(metrics, commit=False)
        # Need to manually reshard the FSDP ema model: https://github.com/pytorch/pytorch/issues/117421#issuecomment-1890948734, otherwise we don't update the ema model weights correctly
        if self.ema_model and self.train_strategy == TrainStrategy.FSDP:
            _reshard_fsdp(self.ema_model)

    def get_finetune_model_config(self) -> DictConfig | None:
        return self.finetune_model_full_config


class MLIPEvalUnit(EvalUnit[AtomicData]):
    def __init__(
        self,
        job_config: DictConfig,
        model: torch.nn.Module,
        tasks: Sequence[Task],
        bf16: bool = False,
    ):
        """Evaluate your MLIPs and so forth.

        Args:
            job_config: a job config object specifying logger and job type
            model: model to evaluate
            evaluations: a list of evaluation objects
            bf16: whether to use autocast with bf16
        """
        super().__init__()
        self.job_config = job_config
        self.model = model
        self.tasks = tasks

        for task in tasks:
            if task.element_references is not None:
                task.element_references.to(torch.device(get_device_for_local_rank()))

        # dictionary of metrics for each dataset, split, task, and metric
        self.running_metrics: dict[str, dict[str, dict[str, Metrics]]] = {}
        self.total_loss_metrics: Metrics = Metrics()
        self.total_atoms: int = 0
        self.total_runtime: float = 0

        # allow the model to be set separately (this is used by the TrainEvalunit to initialize one model for both train and eval)
        if self.model is not None:
            self.model = prepare_module(
                model, device=torch.device(get_device_for_local_rank()), strategy="ddp"
            )
        self.logger = (
            WandBSingletonLogger.get_instance()
            if distutils.is_master()
            and not self.job_config.debug
            and self.job_config.logger
            else None
        )

        # TODO see placeholder comment in TrainEvalUnit as well
        self.autocast_enabled = bf16
        self.autocast_dtype = torch.bfloat16

    def setup_train_eval_unit(self, model: torch.nn.Module) -> None:
        self.model = model
        self.logger = None

    def on_eval_epoch_start(self, state: State) -> None:
        """Reset all metrics, and make sure model is in eval mode."""
        # TODO store ema here as well?
        self.model.eval()

        # create dictionary of running metrics with following schema:
        # task.name: {dataset.split: {metric: value}}}
        datasets_to_eval = state.eval_state.dataloader.dataset.dataset_names
        self.running_metrics = {
            task.name: {
                dataset: {metric: Metrics() for metric in task.metrics}
                for dataset in filter(
                    lambda x: any(dset in x for dset in task.datasets), datasets_to_eval
                )
            }
            for task in self.tasks
        }
        self.total_loss_metrics = Metrics()
        self.total_atoms = 0
        self.total_runtime = 0
        self.total_len = len(state.eval_state.dataloader)
        self.start_time = time.time()
        self.last_report = time.time()
        self.report_every = 180

    def eval_step(self, state: State, data: AtomicData) -> None:
        """Evaluates the model on a batch of data."""
        device = get_device_for_local_rank()
        data = data.to(device)
        self.total_atoms += data.natoms.sum().item()

        if (time.time() - self.last_report) > self.report_every:
            seconds_per_step = (time.time() - self.start_time) / max(
                1, self.eval_progress.num_steps_completed
            )
            eta_hours = self.total_len * seconds_per_step / (60.0 * 60.0)
            print(
                f"step: {self.eval_progress.num_steps_completed}, seconds_per_step: {seconds_per_step} eta_hours: {eta_hours}"
            )
            self.last_report = time.time()

        with torch.autocast(
            device_type=get_device_for_local_rank(),
            enabled=self.autocast_enabled,
            dtype=self.autocast_dtype,
        ):
            t0 = time.time()
            preds = self.model(data)
            self.total_runtime += time.time() - t0

        # compute the loss
        loss_dict = compute_loss(self.tasks, preds, data)
        total_loss = sum(loss_dict.values())
        self.total_loss_metrics += Metrics(metric=total_loss, total=total_loss, numel=1)

        # get the datasets with split names
        datasets_in_batch = set(data.dataset_name)

        # run each evaluation
        for task in self.tasks:
            # This filters out all the datasets.splits that are in the current batch and
            # and are also included in the task. Remove the dataset.splits not in the task, since we
            # wont compute metrics for those.
            # TODO overhaul the dataset names in task to avoid this filter?
            for dataset in filter(
                lambda x: any(dset_name in x for dset_name in task.datasets),
                datasets_in_batch,
            ):
                current_metrics = compute_metrics(task, preds, data, dataset)
                running_metrics = self.running_metrics[task.name][dataset]

                for metric_name in task.metrics:
                    running_metrics[metric_name] += current_metrics[metric_name]

                self.running_metrics[task.name][dataset].update(running_metrics)

                # update the loss metrics
                # loss_metrics = Metrics(
                #     metric=loss_dict[task.name], total=loss_dict[task.name], numel=1
                # )
                # self.running_metrics[task.name][dataset]["loss"] += loss_metrics

                # # total loss
                # loss = sum(loss_dict.values()).item()
                # self.total_loss_metrics += Metrics(metric=loss, total=loss, numel=1)

    def on_eval_epoch_end(self, state: State) -> dict:
        """Aggregate all metrics and log."""

        logging.info("Done eval epoch, aggregating metrics")
        device = get_device_for_local_rank()
        log_dict = {}
        for task, dataset_dict in self.running_metrics.items():
            for dataset, metrics_dict in dataset_dict.items():
                for metric_name, metrics in metrics_dict.items():
                    total = distutils.all_reduce(
                        metrics.total, average=False, device=device
                    )
                    numel = distutils.all_reduce(
                        metrics.numel, average=False, device=device
                    )
                    log_dict[f"val/{dataset},{task},{metric_name}"] = total / numel

        total_runtime = distutils.all_reduce(
            self.total_runtime, average=False, device=device
        )
        total_atoms = distutils.all_reduce(
            self.total_atoms, average=False, device=device
        )

        # we do not reduce across ranks here. DDP loss uses a _ddp_mean that
        # gives the an approximate loss -> loss_rank_i / average_num_samples_across_ranks
        log_dict["val/loss"] = self.total_loss_metrics.metric
        log_dict["val/atoms_per_second"] = total_atoms / total_runtime
        log_dict["val/epoch"] = self.eval_progress.num_epochs_completed

        if self.logger is not None:
            self.logger.log(log_dict, commit=True)

        log_str = "".join(
            f"  {k}: {log_dict[k]:.4f}\n" for k in sorted(log_dict.keys())
        )
        logging.info(f"Finished aggregating metrics: \n{log_str}")

        return log_dict
