"""
Copyright (c) Meta Platforms, Inc. and affiliates.
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# MOLEDGL runs its per-segment GEMM through a backend chosen at construction:
#   * "nvmath"       -> fairchem.core.common.segmentmm (cuBLAS via nvmath-python)
#   * "fairchem_cpp" -> the compiled fairchem_cpp.ops.segment_mm extension
# Both are optional at import time; the concrete op is bound once by
# ``replace_linear_with_MOLE`` and passed in as ``segment_mm_fn``.

# Canonical values accepted by ``moe_layer_type``. ``"dgl"`` is intentionally
# NOT accepted: it historically meant fairchem_cpp but was later repurposed to
# nvmath, so callers must now say which one explicitly.
MOLE_LAYER_TYPES = frozenset({"pytorch", "fairchem_cpp", "nvmath"})
# The subset backed by ``MOLEDGL`` (i.e. the per-segment cuBLAS/C++ ops). These
# are CUDA-only and mutually exclusive with activation_checkpointing/merge_mole.
_MOLEDGL_BACKENDS = frozenset({"fairchem_cpp", "nvmath"})


def normalize_mole_layer_type(name: str) -> str:
    """Validate a ``moe_layer_type`` string, rejecting the legacy ``"dgl"``."""
    if name == "dgl":
        raise ValueError(
            "moe_layer_type='dgl' is no longer supported; use 'fairchem_cpp' "
            "(compiled op) or 'nvmath' (cuBLAS via nvmath-python)."
        )
    if name not in MOLE_LAYER_TYPES:
        raise ValueError(
            f"moe_layer_type must be one of {sorted(MOLE_LAYER_TYPES)}; got {name!r}"
        )
    return name


def is_moledgl_backend(name: str) -> bool:
    """True if ``name`` selects a ``MOLEDGL`` (cuBLAS/C++) backend."""
    return name in _MOLEDGL_BACKENDS


def interval_intersection(interval1, interval2):
    """
    Compute intersection of two intervals [a, b] and [c, d]
    Returns None if no intersection, otherwise returns [start, end]
    """
    a, b = interval1
    c, d = interval2

    start = max(a, c)
    end = min(b, d)

    if start <= end:
        return [start, end]
    else:
        return None  # No intersection


def _softmax(x):
    return torch.softmax(x, dim=1) + 0.005


def _pnorm(x):
    return torch.nn.functional.normalize(x.abs() + 2 / x.shape[0], p=1.0, dim=1)


def norm_str_to_fn(act):
    if act == "softmax":
        return _softmax
    elif act == "pnorm":
        return _pnorm
    else:
        raise ValueError


@dataclass
class MOLEGlobals:
    # the linear coefficient for each expert
    expert_mixing_coefficients: torch.Tensor
    # if the input contains N separate systems, then the sizes represent the number of atoms in each system
    # this is used to for the MoLE to assign the correct parameters for each system
    mole_sizes: torch.Tensor
    # when using activation checkpointing, the inputs are chunked and given piecemeal so the start idx must be
    # updated each time the chunked operation happens. It's better to make this an input but in order for
    # the MolE interface to maintain functional equivalence to the Linear layer interface, this extra info
    # needs to be added here instead. (TODO: is there a cleaner way to do this?)
    ac_start_idx: int = 0


def init_linear(num_experts, use_bias, out_features, in_features):
    k = math.sqrt(1.0 / in_features)
    weights = nn.Parameter(
        k * 2 * (torch.rand(num_experts, out_features, in_features) - 0.5)
    )
    bias = nn.Parameter(k * 2 * (torch.rand(out_features) - 0.5)) if use_bias else None
    return weights, bias


class MOLEDGL(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        global_mole_tensors,
        bias: bool,
        *,
        segment_mm_fn,
    ):
        super().__init__()

        assert global_mole_tensors is not None
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weights, self.bias = init_linear(
            num_experts, bias, out_features, in_features
        )

        self.global_mole_tensors = global_mole_tensors
        # Per-segment GEMM op ``(A, B, seglen) -> C``, bound to a concrete
        # backend (nvmath/cuBLAS or fairchem_cpp) at construction time by
        # ``replace_linear_with_MOLE``. Keeping it a stored callable means
        # ``forward`` has no backend branch and no per-call closure.
        self.segment_mm_fn = segment_mm_fn

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->bio",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )
        x_shape = x.shape
        if x.ndim == 2:
            r = self.segment_mm_fn(x, weights, self.global_mole_tensors.mole_sizes)
        elif x.ndim == 3:
            r = self.segment_mm_fn(
                x.reshape(-1, x_shape[-1]),
                weights,
                self.global_mole_tensors.mole_sizes * x_shape[1],
            ).reshape(*x_shape[:-1], -1)
        else:
            raise ValueError("x.ndim not in (2,3) not allowed")
        if self.bias is not None:
            r += self.bias
        return r


class MOLE(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        global_mole_tensors: MOLEGlobals,
        bias: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weights, self.bias = init_linear(
            num_experts, bias, out_features, in_features
        )

        self.global_mole_tensors = global_mole_tensors

    def merged_linear_layer(self):
        linear = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
        ).to(self.weights.device)

        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )

        with torch.no_grad():
            linear.weight.copy_(weights[0])
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )

        out = []
        ac_start_idx = self.global_mole_tensors.ac_start_idx
        assert len(self.global_mole_tensors.mole_sizes) > 0
        # TODO: precompute these if needed but they should be small and on cpu
        start_idxs = [0] + torch.cumsum(
            self.global_mole_tensors.mole_sizes, dim=0
        ).tolist()
        mole_intervals = list(zip(start_idxs, start_idxs[1:]))

        # Because activation checkpointing can chunk the inputs, we need to only compute
        # the mole_size intervals that overlap with the current chunks
        # for example if mole_sizes = [10,10,15]
        # start_idxs -> [0,10,20,35]
        # mole_intervals -> [(0,10),(10,20),(20,35)]
        # if the input segment is (5,15) then we compute the following 2 segments
        # (5,10),(10,15)
        input_segment = (ac_start_idx, ac_start_idx + x.shape[0])

        for n, mole_segment in enumerate(mole_intervals):
            interval_overlap = interval_intersection(input_segment, mole_segment)
            if interval_overlap is not None:
                start = interval_overlap[0] - ac_start_idx
                end = interval_overlap[1] - ac_start_idx
                out.append(F.linear(x[start:end], weights[n], bias=self.bias))

        result = torch.concatenate(out, dim=0)
        assert (
            result.shape[0] == x.shape[0]
        ), f"result shape {result.shape}, does not match input shape {x.shape} at dim 0"
        return result
