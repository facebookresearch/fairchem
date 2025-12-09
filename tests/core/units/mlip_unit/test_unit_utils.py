"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import torch

from fairchem.core.common import distutils, gp_utils


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
                gp_size = gp_utils.get_dp_world_size()
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
