"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Fused Triton implementation of the eSCN-MD backbone body, selected with
``InferenceSettings(execution_mode="umas_flash")``.

Cold start: the kernels are autotuned, and the first process to run a given
(kernel, channel count, GPU architecture) combination pays for it — minutes,
not seconds. The autotuners are declared ``cache_results=True`` so the winning
configurations are written to the Triton cache and every later process reads
them back. When launching many ranks at once, let one warm the cache first, or
give each rank its own ``TRITON_CACHE_DIR``, so they do not all tune in
parallel and time each other's kernels.
"""

from __future__ import annotations

# Registers the fairchem::flash_* operators with torch.ops on import.
import fairchem.core.models.uma.flash.custom_ops  # noqa: F401

from .features import FlashFeatures

__all__ = ["FlashFeatures"]
