"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the LICENSE
file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from .uma.escn_md import (
    MLP_EFS_Head,
    MLP_EFS_Head_LR,
    MLP_Energy_Head,
    MLP_Energy_Head_LR,
    MLP_Stress_Head,
    eSCNMDBackbone,
    eSCNMDBackboneLR,
)

# REMOVE LATER
from .allscaip import (
    AllScAIPBackbone,
    AllScAIPEnergyHead,
    AllScAIPDirectForceHead,
    AllScAIPGradientEnergyForceStressHead,
    AllScAIPEnergyHeadLR,
    AllScAIPGradientEnergyForceStressHeadLR
)


from .escaip import (
    EScAIPBackbone,
    EScAIPEnergyHead,
    EScAIPDirectForceHead,
    EScAIPGradientEFSHead
)


from .uma.escn_moe import eSCNMDMoeBackbone, eSCNMDMoeBackboneLR

from .uma.escn_md_les import eSCNMDBackboneLES, MLP_EFS_Head_LES, MLP_Energy_Head_LES
torch.set_float32_matmul_precision("high")

__all__ = [
    "MLP_EFS_Head_LR",
    "MLP_Energy_Head",
    "MLP_Energy_Head_LR",
    "MLP_Stress_Head",
    "MLP_Stress_Head_LR",
    "eSCNMDBackbone",
    "eSCNMDBackboneLR",
    "MLP_EFS_Head",
    "eSCNMDMoeBackbone",
    "eSCNMDMoeBackboneLR",
    "EScAIPBackbone",
    "EScAIPEnergyHead",
    "EScAIPDirectForceHead",
    "EScAIPGradientEFSHead",
    "eSCNMDBackboneLES",
    "MLP_EFS_Head_LES",
    "AllScAIPBackbone",
    "AllScAIPEnergyHead",
    "AllScAIPDirectForceHead",
    "AllScAIPGradientEnergyForceStressHead", 
    "AllScAIPEnergyHeadLR",
    "AllScAIPGradientEnergyForceStressHeadLR"
]
