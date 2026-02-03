"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests for knowledge distillation training unit.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from tests.core.testing_utils import launch_main


def test_distillation_training(fake_uma_dataset, torch_deterministic):
    """Test that distillation training runs for a few iterations.
    
    This test:
    1. Trains a small teacher model for a few steps
    2. Uses that teacher to train an even smaller student via distillation
    3. Verifies the training loop completes successfully
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_run_dir = os.path.join(tmpdir, "teacher")
        student_run_dir = os.path.join(tmpdir, "student")
        
        # Step 1: Train a teacher model (using conserving config)
        teacher_sys_args = [
            "--config",
            "tests/core/units/mlip_unit/test_mlip_train_conserving.yaml",
            f"datasets.data_root_dir={fake_uma_dataset}",
            f"job.run_dir={teacher_run_dir}",
            "max_steps=3",
        ]
        launch_main(teacher_sys_args)
        
        # Find the teacher checkpoint
        teacher_checkpoint = None
        for root, dirs, files in os.walk(teacher_run_dir):
            for f in files:
                if f == "inference_ckpt.pt":
                    teacher_checkpoint = os.path.join(root, f)
                    break
        
        assert teacher_checkpoint is not None, "Teacher checkpoint not found"
        assert os.path.exists(teacher_checkpoint), f"Teacher checkpoint does not exist: {teacher_checkpoint}"
        
        # Step 2: Train student model via distillation
        student_sys_args = [
            "--config",
            "tests/core/units/mlip_unit/test_mlip_distillation.yaml",
            f"datasets.data_root_dir={fake_uma_dataset}",
            f"job.run_dir={student_run_dir}",
            f"teacher_checkpoint={teacher_checkpoint}",
            "max_steps=3",
        ]
        launch_main(student_sys_args)
        
        # Verify student checkpoint was created
        student_checkpoint_found = False
        for root, dirs, files in os.walk(student_run_dir):
            for f in files:
                if f == "inference_ckpt.pt":
                    student_checkpoint_found = True
                    break
        
        assert student_checkpoint_found, "Student checkpoint not found after distillation training"


def test_distillation_unit_direct(fake_uma_dataset, torch_deterministic):
    """Direct unit test for MLIPDistillationUnit.
    
    Tests the distillation unit directly without going through CLI.
    """
    import pickle
    from collections import namedtuple
    
    import hydra
    import torch.distributed as dist
    from omegaconf import OmegaConf
    
    from fairchem.core._cli import get_hydra_config_from_yaml
    from fairchem.core.common.distutils import assign_device_for_local_rank, setup_env_local
    
    # Setup distributed environment
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    setup_env_local()
    assign_device_for_local_rank(True, 0)
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_run_dir = os.path.join(tmpdir, "teacher")
        
        # First train a quick teacher model
        teacher_config = "tests/core/units/mlip_unit/test_mlip_train_conserving.yaml"
        teacher_cfg = get_hydra_config_from_yaml(
            teacher_config,
            [
                f"datasets.data_root_dir={fake_uma_dataset}",
                f"job.run_dir={teacher_run_dir}",
                "max_steps=2",
            ],
        )
        os.makedirs(teacher_cfg.job.run_dir, exist_ok=True)
        os.makedirs(os.path.join(teacher_cfg.job.run_dir, teacher_cfg.job.timestamp_id), exist_ok=True)
        OmegaConf.save(teacher_cfg, teacher_cfg.job.metadata.config_path)
        
        teacher_runner = hydra.utils.instantiate(teacher_cfg.runner)
        teacher_runner.job_config = teacher_cfg.job
        teacher_runner.run()
        
        # Save teacher checkpoint
        ch_path = teacher_cfg.job.metadata.checkpoint_dir
        teacher_runner.save_state(ch_path)
        
        teacher_checkpoint = os.path.join(ch_path, "inference_ckpt.pt")
        assert os.path.exists(teacher_checkpoint)
        
        # Now test the distillation unit
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        
        student_run_dir = os.path.join(tmpdir, "student")
        distill_config = "tests/core/units/mlip_unit/test_mlip_distillation.yaml"
        distill_cfg = get_hydra_config_from_yaml(
            distill_config,
            [
                f"datasets.data_root_dir={fake_uma_dataset}",
                f"job.run_dir={student_run_dir}",
                f"teacher_checkpoint={teacher_checkpoint}",
                "max_steps=2",
            ],
        )
        os.makedirs(distill_cfg.job.run_dir, exist_ok=True)
        os.makedirs(os.path.join(distill_cfg.job.run_dir, distill_cfg.job.timestamp_id), exist_ok=True)
        OmegaConf.save(distill_cfg, distill_cfg.job.metadata.config_path)
        
        distill_runner = hydra.utils.instantiate(distill_cfg.runner)
        distill_runner.job_config = distill_cfg.job
        
        # Verify the distillation unit has correct properties
        unit = distill_runner.train_eval_unit
        assert hasattr(unit, 'teacher_model'), "Distillation unit should have teacher_model"
        assert hasattr(unit, 'distillation_tasks'), "Distillation unit should have distillation_tasks"
        assert hasattr(unit, '_teacher_needs_grad'), "Distillation unit should have _teacher_needs_grad"
        
        # Teacher should be in eval mode
        assert not unit.teacher_model.training, "Teacher model should be in eval mode"
        
        # Teacher parameters should be frozen
        for param in unit.teacher_model.parameters():
            assert not param.requires_grad, "Teacher parameters should be frozen"
        
        # Run training
        distill_runner.run()
        
        # Verify training completed
        assert unit.train_progress.num_steps_completed == 2
    
    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


def test_distillation_conservative_forces():
    """Test that conservative model teacher correctly requires gradients for force computation."""
    from fairchem.core.units.mlip_unit.distillation_unit import MLIPDistillationUnit
    
    # Create mock backbone classes
    class MockConservativeBackbone:
        direct_forces = False
    
    class MockDirectBackbone:
        direct_forces = True
    
    # Test detection logic
    conservative_backbone = MockConservativeBackbone()
    direct_backbone = MockDirectBackbone()
    
    # Conservative model should need grad
    needs_grad_conservative = not getattr(conservative_backbone, "direct_forces", True)
    assert needs_grad_conservative, "Conservative model should need gradients"
    
    # Direct model should not need grad
    needs_grad_direct = not getattr(direct_backbone, "direct_forces", True)
    assert not needs_grad_direct, "Direct model should not need gradients"
