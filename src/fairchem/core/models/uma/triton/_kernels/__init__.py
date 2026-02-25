"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Internal kernel implementations.
Triton JIT kernels and low-level Python wrappers.

NOTE: This is internal API. Import from fairchem.core.models.uma.triton instead.

Contains:
    - gather_wigner_fwd.py: Forward pass kernels (edge gather + Wigner + L->M)
    - gather_wigner_bwd.py: Backward pass kernels (M->L + W^T + scatter)
    - wigner_transform.py: Standalone Wigner multiply kernels
"""

from __future__ import annotations
