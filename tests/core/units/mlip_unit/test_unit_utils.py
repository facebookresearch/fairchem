"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.common import distutils, gp_utils
from fairchem.core.units.mlip_unit.utils import (
    float32_matmul_precision_context,
    get_model_float32_matmul_precision,
)


@pytest.mark.parametrize("initial_precision", ["highest", "high"])
def test_float32_matmul_precision_context_restores_caller(initial_precision):
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(initial_precision)
        with float32_matmul_precision_context("high"):
            assert torch.get_float32_matmul_precision() == "high"
        assert torch.get_float32_matmul_precision() == initial_precision
    finally:
        torch.set_float32_matmul_precision(original_precision)


def test_float32_matmul_precision_context_restores_after_error():
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with (
            pytest.raises(RuntimeError, match="failure"),
            float32_matmul_precision_context("high"),
        ):
            raise RuntimeError("failure")
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(original_precision)


def test_get_model_float32_matmul_precision_through_wrappers():
    model = SimpleNamespace(
        module=SimpleNamespace(
            module=SimpleNamespace(
                backbone=SimpleNamespace(float32_matmul_precision="high")
            )
        )
    )

    assert get_model_float32_matmul_precision(model) == "high"
    assert get_model_float32_matmul_precision(None) is None


class GradSaveOptimizer(torch.optim.AdamW):
    def __init__(
        self,
        params,
        save_path,
    ):
        super().__init__(params=params, lr=0.0, weight_decay=0.0)
        self.save_path = save_path
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
        self.save_step = 0
        # self.params = params

    def step(self, closure=None):
        if self.save_path:
            gp_size = 0
            gp_rank = 0
            if gp_utils.initialized():
                gp_size = gp_utils.get_gp_world_size()
                gp_rank = gp_utils.get_dp_rank()

            ddp_size = distutils.get_world_size()
            ddp_rank = distutils.get_rank()

            torch.save(
                {
                    "param": list(self.param_groups[0]["params"]),
                    "grad": [
                        param.grad
                        for param in self.param_groups[0]["params"]
                        if param.grad is not None
                    ],
                },
                f"{self.save_path}/ddp{ddp_size}.{ddp_rank}_gp{gp_size}.{gp_rank}_step{self.save_step}.pt",
            )
        self.save_step += 1
        super().step()
