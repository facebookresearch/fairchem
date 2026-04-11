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

from fairchem.core.models.uma.nn.segment_mm import (
    segment_mm_double_backward,
)


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
    # if the input contains N separate systems, then the sizes represent
    # the number of atoms in each system; this is used for the MoLE to
    # assign the correct parameters for each system
    mole_sizes: torch.Tensor
    # when using activation checkpointing, the inputs are chunked and
    # given piecemeal so the start idx must be updated each time the
    # chunked operation happens.
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
            r = segment_mm_double_backward(
                x,
                weights,
                self.global_mole_tensors.mole_sizes,
            )
        elif x.ndim == 3:
            r = segment_mm_double_backward(
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
        start_idxs = [0] + torch.cumsum(
            self.global_mole_tensors.mole_sizes, dim=0
        ).tolist()
        mole_intervals = list(zip(start_idxs, start_idxs[1:]))

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
