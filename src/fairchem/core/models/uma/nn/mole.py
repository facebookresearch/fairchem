"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn as nn
from torch.nn import functional as F

fairchem_cpp_found = False
with suppress(ModuleNotFoundError):
    import fairchem_cpp  # try to use DGL if available

    fairchem_cpp_found = True


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

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->bio",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )
        x_shape = x.shape
        if x.ndim == 2:
            r = fairchem_cpp.ops.segment_mm(
                x, weights, self.global_mole_tensors.mole_sizes
            )
        elif x.ndim == 3:
            r = fairchem_cpp.ops.segment_mm(
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

    def ragged_dot(self, lhs: torch.Tensor, rhs: torch.Tensor, group_sizes: torch.Tensor):
        start_idxs = [0] + torch.cumsum(group_sizes, dim=0).tolist()
        out = []
        for n, (start, end) in enumerate(pairwise(start_idxs)):
            if end > start:
                out.append(F.linear(lhs[start:end], rhs[n]))
        return torch.concatenate(out, dim=0)

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi,be->boi",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )

        assert self.global_mole_tensors.mole_sizes.shape[0] > 0
        ac_start_idx = self.global_mole_tensors.ac_start_idx
        cs = self.global_mole_tensors.mole_sizes.cumsum(0)
        cs = cs.clamp(min=ac_start_idx, max=ac_start_idx + x.shape[0]) - ac_start_idx
        group_sizes = torch.diff(cs, prepend=cs.new_zeros(1))
        result = self.ragged_dot(x, weights, group_sizes)
        if self.bias is not None:
            result += self.bias
        assert (
            result.shape[0] == x.shape[0]
        ), f"result shape {result.shape}, does not match input shape {x.shape} at dim 0"
        return result
