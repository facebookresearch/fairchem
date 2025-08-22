"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import tempfile
import torch

import pytest

from fairchem.core.units.mlip_unit.mlip_unit import (
    UNIT_INFERENCE_CHECKPOINT,
    UNIT_RESUME_CONFIG,
)
from tests.core.testing_utils import launch_main
from tests.core.units.mlip_unit.create_fake_dataset import (
    create_fake_uma_dataset,
)


@pytest.fixture(scope="session")
def fake_uma_dataset():
    with tempfile.TemporaryDirectory() as tempdirname:
        datasets_yaml = create_fake_uma_dataset(tempdirname)
        yield datasets_yaml


@pytest.fixture(scope="session")
def direct_mole_checkpoint(fake_uma_dataset):
    # first train to completion
    temp_dir = tempfile.mkdtemp()
    timestamp_id = "12345"
    device = "CPU"

    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train.yaml",
        "num_experts=8",
        "checkpoint_every=10000",
        "datasets=aselmdb",
        f"+job.run_dir={temp_dir}",
        f"datasets.data_root_dir={fake_uma_dataset}",
        f"job.device_type={device}",
        f"+job.timestamp_id={timestamp_id}",
        "optimizer=savegrad",
        "max_steps=2",
        "max_epochs=null",
        "expected_loss=null",
        "act_type=gate",
        "ff_type=spectral",
    ]
    launch_main(sys_args)

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoint_dir = os.path.join(temp_dir, timestamp_id, "checkpoints", "step_0")
    checkpoint_state_yaml = os.path.join(checkpoint_dir, UNIT_RESUME_CONFIG)
    inference_checkpoint_pt = os.path.join(checkpoint_dir, UNIT_INFERENCE_CHECKPOINT)
    assert os.path.isdir(checkpoint_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    assert os.path.isfile(inference_checkpoint_pt)

    return inference_checkpoint_pt, checkpoint_state_yaml


@pytest.fixture(scope="session")
def direct_checkpoint(fake_uma_dataset):
    # first train to completion
    temp_dir = tempfile.mkdtemp()
    timestamp_id = "12345"
    device = "CPU"

    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train.yaml",
        "num_experts=0",
        "checkpoint_every=10000",
        "datasets=aselmdb",
        f"+job.run_dir={temp_dir}",
        f"datasets.data_root_dir={fake_uma_dataset}",
        f"job.device_type={device}",
        f"+job.timestamp_id={timestamp_id}",
        "optimizer=savegrad",
        "max_steps=2",
        "max_epochs=null",
        "expected_loss=null",
        "act_type=gate",
        "ff_type=spectral",
        # "max_neighbors=300"
    ]
    launch_main(sys_args)

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoint_dir = os.path.join(temp_dir, timestamp_id, "checkpoints", "step_0")
    checkpoint_state_yaml = os.path.join(checkpoint_dir, UNIT_RESUME_CONFIG)
    inference_checkpoint_pt = os.path.join(checkpoint_dir, UNIT_INFERENCE_CHECKPOINT)
    assert os.path.isdir(checkpoint_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    assert os.path.isfile(inference_checkpoint_pt)

    return inference_checkpoint_pt, checkpoint_state_yaml


@pytest.fixture(scope="session")
def conserving_mole_checkpoint(fake_uma_dataset):
    # first train to completion
    temp_dir = tempfile.mkdtemp()
    timestamp_id = "12345"
    device = "CPU"

    sys_args = [
        "--config",
        "tests/core/units/mlip_unit/test_mlip_train_conserving.yaml",
        "num_experts=8",
        "heads.energyandforcehead.module=fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
        "checkpoint_every=10000",
        "datasets=aselmdb_conserving",
        f"+job.run_dir={temp_dir}",
        f"datasets.data_root_dir={fake_uma_dataset}",
        f"job.device_type={device}",
        f"+job.timestamp_id={timestamp_id}",
        "optimizer=savegrad",
        "max_steps=2",
        "max_epochs=null",
        "expected_loss=null",
        "act_type=gate",
        "ff_type=spectral",
    ]
    launch_main(sys_args)

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoint_dir = os.path.join(temp_dir, timestamp_id, "checkpoints", "step_0")
    checkpoint_state_yaml = os.path.join(checkpoint_dir, UNIT_RESUME_CONFIG)
    inference_checkpoint_pt = os.path.join(checkpoint_dir, UNIT_INFERENCE_CHECKPOINT)
    assert os.path.isdir(checkpoint_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    assert os.path.isfile(inference_checkpoint_pt)

    return inference_checkpoint_pt, checkpoint_state_yaml
