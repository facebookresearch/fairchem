"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Public Triton operations for UMA model.

This module provides torch.autograd.Function classes for GPU-accelerated
operations in the UMA backbone. All operations have custom backward passes
optimized for force computation.

Main Operations:
    - FusedEdgeGatherWignerL2MTritonBwdEmitFunction: Forward gather + rotate
    - FusedMToLThenWignerLmax2Function: Backward rotate (M->L + Wigner)
"""

from __future__ import annotations

from ._kernels.gather_wigner_bwd import (
    FusedEdgeGatherWignerL2MTritonBwdEmitFunction,
    FusedEdgeGatherWignerL2MTritonV2BwdFunction,
)
from ._kernels.wigner_transform import (
    FusedMToLThenWignerLmax2Function,
    MToLThenWignerLmax2Function,
)

__all__ = [
    "FusedEdgeGatherWignerL2MTritonBwdEmitFunction",
    "FusedEdgeGatherWignerL2MTritonV2BwdFunction",
    "FusedMToLThenWignerLmax2Function",
    "MToLThenWignerLmax2Function",
]
