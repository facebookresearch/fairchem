"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Raw Triton kernels for the umas_flash execution backend. Launch them through
``fairchem.core.models.uma.flash.custom_ops``, which owns the autograd
definitions and the CUDA device guards.
"""

from __future__ import annotations
