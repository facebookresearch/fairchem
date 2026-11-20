"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.modules.scheduler import (
    CosineLRLambda,
    LRScheduler,
    warmup_lr_lambda,
)


def test_warmup_and_milestone_boundaries():
    config = {
        "warmup_steps": 10,
        "warmup_factor": 0.2,
        "lr_milestones": [100, 200],
        "lr_gamma": 0.1,
    }
    for step, expected in [
        (0, 0.2),
        (5, 0.6),
        (10, 1),
        (99, 1),
        (100, 0.1),
        (200, 0.01),
    ]:
        assert warmup_lr_lambda(step, config) == pytest.approx(expected)
    for invalid in ({**config, "warmup_epochs": 1}, {**config, "lr_milestones": [10]}):
        with pytest.raises(Exception, match="ConfigError"):
            warmup_lr_lambda(0, invalid)


def test_cosine_schedule_warmup_midpoint_and_floor():
    schedule = CosineLRLambda(10, 0.2, 100, 0.1)
    for step, expected in [(0, 0.2), (10, 1), (50, 0.55), (100, 0.1), (200, 0.1)]:
        assert schedule(step) == pytest.approx(expected)


def test_scheduler_updates_only_accepted_config_and_preserves_input():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    config = {"scheduler": "StepLR", "step_size": 1, "gamma": 0.5, "unused": 99}
    original = config.copy()
    schedule = LRScheduler(optimizer, config)
    optimizer.step()
    schedule.step()
    assert schedule.get_lr() == 0.5
    assert config == original
    null = LRScheduler(optimizer, {"scheduler": "Null"})
    null.step()
    assert null.get_lr() == 0.5


def test_plateau_requires_validation_and_reduces_learning_rate():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    schedule = LRScheduler(
        optimizer, {"scheduler": "ReduceLROnPlateau", "patience": 0, "factor": 0.5}
    )
    with pytest.raises(Exception, match="Validation set required"):
        schedule.step()
    schedule.step(metrics=1.0)
    schedule.step(metrics=2.0)
    assert schedule.get_lr() == 0.5


def test_default_scheduler_initializes_warmup():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    schedule = LRScheduler(
        optimizer,
        {
            "warmup_steps": 10,
            "warmup_factor": 0.2,
            "lr_milestones": [100],
            "lr_gamma": 0.1,
        },
    )
    assert schedule.get_lr() == 0.2
    optimizer.step()
    schedule.step()
    assert schedule.get_lr() == pytest.approx(0.28)
