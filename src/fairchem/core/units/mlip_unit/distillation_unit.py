"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Knowledge distillation training unit for training a smaller student model
using predictions from a larger teacher model.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Sequence

import hydra
import torch
from torch.profiler import record_function

from fairchem.core.common import distutils
from fairchem.core.common.distutils import get_device_for_local_rank
from fairchem.core.common.registry import registry
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.mlip_unit import (
    MLIPTrainEvalUnit,
    Task,
    compute_loss,
    get_output_masks,
)
from fairchem.core.units.mlip_unit.utils import load_inference_model

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torchtnt.framework import State


def initialize_student_model(
    backbone_config: dict,
    heads_config: dict,
    pass_through_head_outputs: bool = True,
) -> torch.nn.Module:
    """Initialize a fresh student model from scratch.

    Args:
        backbone_config: Configuration dict for the backbone model
        heads_config: Configuration dict for the output heads
        pass_through_head_outputs: Whether to pass through head outputs (default True for multi-task)

    Returns:
        Initialized HydraModel model
    """
    from fairchem.core.models.base import HydraModel

    # Instantiate backbone
    backbone = hydra.utils.instantiate(backbone_config)

    # Instantiate heads
    output_heads = {}
    head_names_sorted = sorted(heads_config.keys())
    for head_name in head_names_sorted:
        head_config = deepcopy(heads_config[head_name])
        if "module" not in head_config:
            raise ValueError(
                f"{head_name} head does not specify module to use for the head"
            )
        module_name = head_config.pop("module")
        output_heads[head_name] = registry.get_model_class(module_name)(
            backbone,
            **head_config,
        )

    model = HydraModel(
        backbone=backbone,
        heads=output_heads,
        pass_through_head_outputs=pass_through_head_outputs,
    )
    return model


def compute_distillation_loss(
    tasks: Sequence[Task],
    student_predictions: dict[str, torch.Tensor],
    teacher_predictions: dict[str, torch.Tensor],
    batch: AtomicData,
    distillation_loss_fn: torch.nn.Module,
    distillation_coefficient: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute distillation loss between student and teacher predictions.

    Args:
        tasks: Sequence of Task objects defining the outputs to distill
        student_predictions: Dictionary of student model predictions
        teacher_predictions: Dictionary of teacher model predictions
        batch: Data batch
        distillation_loss_fn: Loss function for distillation (e.g., MSE, L1)
        distillation_coefficient: Weight for distillation loss

    Returns:
        Dictionary of distillation losses for each task
    """
    batch_size = batch.natoms.numel()
    num_atoms_in_batch = batch.natoms.sum()

    output_masks = get_output_masks(batch, tasks)
    distill_loss_dict = {}

    for task in tasks:
        output_mask = output_masks[task.name]

        # Get student and teacher predictions for this task
        student_pred = student_predictions[task.name][task.property]
        teacher_pred = teacher_predictions[task.name][task.property]

        # Reshape based on task level
        if task.level == "atom":
            student_pred = student_pred.view(num_atoms_in_batch, -1)
            teacher_pred = teacher_pred.view(num_atoms_in_batch, -1)
        else:
            student_pred = student_pred.view(batch_size, -1)
            teacher_pred = teacher_pred.view(batch_size, -1)

        # Ensure shapes match
        assert student_pred.shape == teacher_pred.shape, (
            f"Shape mismatch for {task.name}: "
            f"student {student_pred.shape} vs teacher {teacher_pred.shape}"
        )

        # Compute distillation loss only on valid outputs
        if output_mask.any():
            loss = distillation_loss_fn(
                student_pred[output_mask],
                teacher_pred[output_mask].detach(),  # Detach teacher predictions
            )
            distill_loss_dict[f"{task.name}_distill"] = distillation_coefficient * loss
        else:
            distill_loss_dict[f"{task.name}_distill"] = torch.tensor(
                0.0, device=student_pred.device, requires_grad=True
            )

    return distill_loss_dict


class MLIPDistillationUnit(MLIPTrainEvalUnit):
    """Training unit for knowledge distillation.

    This unit extends MLIPTrainEvalUnit to support training a student model
    using both ground truth labels and soft labels from a teacher model.
    """

    def __init__(
        self,
        job_config: DictConfig,
        model: torch.nn.Module,
        teacher_checkpoint_location: str,
        optimizer_fn: callable,
        cosine_lr_scheduler_fn: callable,
        tasks: list[Task],
        distillation_coefficient: float = 1.0,
        ground_truth_coefficient: float = 1.0,
        distillation_loss_type: str = "mse",
        teacher_overrides: dict | None = None,
        distill_energy: bool = True,
        distill_forces: bool = True,
        distill_stress: bool = False,
        **kwargs,
    ):
        """Initialize the distillation training unit.

        Args:
            job_config: Job configuration
            model: Student model to train
            teacher_checkpoint_location: Path to teacher model checkpoint
            optimizer_fn: Optimizer factory function
            cosine_lr_scheduler_fn: Learning rate scheduler factory
            tasks: List of tasks to train on
            distillation_coefficient: Weight for distillation loss
            ground_truth_coefficient: Weight for ground truth loss
            distillation_loss_type: Type of loss for distillation ("mse" or "mae")
            teacher_overrides: Optional config overrides for teacher model
            distill_energy: Whether to distill energy predictions
            distill_forces: Whether to distill force predictions
            distill_stress: Whether to distill stress predictions
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            job_config=job_config,
            model=model,
            optimizer_fn=optimizer_fn,
            cosine_lr_scheduler_fn=cosine_lr_scheduler_fn,
            tasks=tasks,
            **kwargs,
        )

        self.distillation_coefficient = distillation_coefficient
        self.ground_truth_coefficient = ground_truth_coefficient
        self.distill_energy = distill_energy
        self.distill_forces = distill_forces
        self.distill_stress = distill_stress

        # Load teacher model
        logging.info(f"Loading teacher model from {teacher_checkpoint_location}")
        self.teacher_model, _ = load_inference_model(
            teacher_checkpoint_location,
            overrides=teacher_overrides,
            use_ema=True,  # Use EMA weights for teacher
        )

        # Move teacher to device and set to eval mode
        device = get_device_for_local_rank()
        self.teacher_model = self.teacher_model.to(device)
        self.teacher_model.eval()

        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Check if teacher uses conservative forces (needs grad for force computation)
        # Conservative models compute forces as F = -dE/dr, requiring gradients
        self._teacher_needs_grad = not getattr(
            self.teacher_model.backbone, "direct_forces", True
        )
        logging.info(
            f"Teacher model uses {'conservative' if self._teacher_needs_grad else 'direct'} forces"
        )

        # Setup distillation loss function
        if distillation_loss_type == "mse":
            self.distillation_loss_fn = torch.nn.MSELoss()
        elif distillation_loss_type == "mae":
            self.distillation_loss_fn = torch.nn.L1Loss()
        elif distillation_loss_type == "huber":
            self.distillation_loss_fn = torch.nn.HuberLoss()
        else:
            raise ValueError(
                f"Unknown distillation loss type: {distillation_loss_type}"
            )

        # Determine which tasks to distill
        self.distillation_tasks = []
        for task in self.tasks:
            if "energy" in task.name and self.distill_energy:
                self.distillation_tasks.append(task)
            elif "force" in task.name and self.distill_forces:
                self.distillation_tasks.append(task)
            elif "stress" in task.name and self.distill_stress:
                self.distillation_tasks.append(task)

        logging.info(
            f"Distillation tasks: {[t.name for t in self.distillation_tasks]}"
        )
        logging.info(
            f"Distillation coefficient: {self.distillation_coefficient}, "
            f"Ground truth coefficient: {self.ground_truth_coefficient}"
        )

    def train_step(self, state: State, data: AtomicData) -> None:
        """Execute a single training step with distillation.

        Args:
            state: Current training state
            data: Batch of atomic data
        """
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
                # Student forward pass
                with record_function("student_forward"):
                    student_pred = self.model.forward(batch_on_device)

                # Teacher forward pass
                # Note: For conservative models (direct_forces=False), we need gradients
                # enabled to compute forces as F = -dE/dr. However, we detach the outputs
                # so no gradients flow back through the teacher during backprop.
                with record_function("teacher_forward"):
                    if self._teacher_needs_grad:
                        # Conservative model - need grad for force computation
                        with torch.enable_grad():
                            teacher_pred = self.teacher_model.forward(batch_on_device)
                    else:
                        # Direct forces model - no grad needed
                        with torch.no_grad():
                            teacher_pred = self.teacher_model.forward(batch_on_device)

                # Compute ground truth loss
                with record_function("compute_gt_loss"):
                    gt_loss_dict = compute_loss(self.tasks, student_pred, batch_on_device)

                # Compute distillation loss
                with record_function("compute_distill_loss"):
                    distill_loss_dict = compute_distillation_loss(
                        tasks=self.distillation_tasks,
                        student_predictions=student_pred,
                        teacher_predictions=teacher_pred,
                        batch=batch_on_device,
                        distillation_loss_fn=self.distillation_loss_fn,
                        distillation_coefficient=self.distillation_coefficient,
                    )

            # Combine losses
            total_gt_loss = self.ground_truth_coefficient * sum(gt_loss_dict.values())
            total_distill_loss = sum(distill_loss_dict.values())
            scalar_loss = total_gt_loss + total_distill_loss

            # Backward pass
            self.optimizer.zero_grad()
            with record_function("backward"):
                scalar_loss.backward()

            # Gradient clipping
            if self.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )
            else:
                grad_norm = None

            # Optimizer step
            with record_function("optimizer_step"):
                self.optimizer.step()
                self.scheduler.step()

            # Update EMA model if enabled
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)

            # Logging
            if step % self.print_every == 0 and self.logger:
                log_dict = {
                    "train/step": step,
                    "train/epoch": epoch,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/total_loss": scalar_loss.item(),
                    "train/gt_loss": total_gt_loss.item(),
                    "train/distill_loss": total_distill_loss.item(),
                }

                # Log individual losses
                for key, value in gt_loss_dict.items():
                    log_dict[f"train/gt_{key}"] = value.item()
                for key, value in distill_loss_dict.items():
                    log_dict[f"train/{key}"] = value.item()

                if grad_norm is not None:
                    log_dict["train/grad_norm"] = grad_norm.item()

                self.logger.log(log_dict, step=step, commit=True)

        except Exception as e:
            logging.error(f"Error in train_step: {e}")
            raise
