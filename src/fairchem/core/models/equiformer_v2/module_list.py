"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch


class ModuleListInfo(torch.nn.ModuleList):
    def __init__(self, info_str, modules=None) -> None:
        super().__init__(modules)
        self.info_str = str(info_str)

    def __repr__(self) -> str:
        return self.info_str
