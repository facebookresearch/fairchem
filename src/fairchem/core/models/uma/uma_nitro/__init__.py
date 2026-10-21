"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

UMA Nitro: a standalone UMA-small model for energy, force, and stress inference.

Select it with ``inference_settings="nitro"``. The model is built directly
from the weights of uma-s-1p2p1, or of a fine-tune of it on Nitro's MD5
allowlist, and requires CUDA compute capability 8.x, 9.x, or 10.x. Capability
8.x uses portable IEEE matmul with Triton primitives; Hopper uses CuTe GEMM,
and Blackwell uses CuTe gather/scatter with cuBLAS GEMM.
"""

from __future__ import annotations

from fairchem.core.models.uma.uma_nitro._inference import (
    UMANitroModel,
    UMANitroNotSupportedError,
    kernel_backend_summary,
    verify_checkpoint_file,
)

__all__ = [
    "UMANitroModel",
    "UMANitroNotSupportedError",
    "kernel_backend_summary",
    "verify_checkpoint_file",
]
