"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from tests.core.models.uma.uma_nitro.nitro_test_utils import (
    direct_head_random_checkpoint,
    random_checkpoint,
)


@pytest.fixture(scope="session")
def nitro_random_checkpoint():
    """
    A randomly initialized UMA-S checkpoint object (not the pinned file).
    """
    return random_checkpoint()


@pytest.fixture(scope="session")
def nitro_direct_head_checkpoint():
    """
    A random single-dataset checkpoint with an unwrapped EFS head.
    """
    return direct_head_random_checkpoint()


@pytest.fixture(scope="session")
def nitro_random_checkpoint_path(nitro_random_checkpoint, tmp_path_factory):
    """
    The random checkpoint saved to disk, for loading through MLIPPredictUnit.
    """
    path = tmp_path_factory.mktemp("nitro") / "random_uma_s.pt"
    torch.save(nitro_random_checkpoint, path)
    return str(path)
