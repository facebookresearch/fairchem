"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn as nn

from .matmul import linear_with_folded_batch

# MOLE.forward groups systems of similar size, pads each system to the largest
# one in its group and runs one bmm per group. A group grows while its padded
# rows stay within this multiple of its real rows, which bounds the extra memory.
MAX_PADDING_RATIO = 2.0

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
    # Padded layouts for MOLE.forward, valid for one mole_sizes tensor
    _padded_plans: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def padded_plan(self, num_rows: int, device: torch.device) -> PaddedPlan | None:
        """
        Return the padded layout of the systems in rows
        [ac_start_idx, ac_start_idx + num_rows), or None to use the loop.

        The layout is computed once per mole_sizes tensor and chunk, so the MOLE
        layers of one forward pass share it. Replacing mole_sizes or modifying
        it in place invalidates the cache.

        Args:
            num_rows: Number of rows in the layer input.
            device: Device of the layer input.

        Returns:
            The padded layout, or None to use the per-system loop.
        """
        sizes = self.mole_sizes
        cache = self._padded_plans
        if cache.get("sizes") is not sizes or cache.get("version") != sizes._version:
            cache.clear()
            cache.update(sizes=sizes, version=sizes._version)
        key = (self.ac_start_idx, num_rows, device)
        if key not in cache:
            cache[key] = _build_padded_plan(
                sizes.tolist(), self.ac_start_idx, num_rows, device
            )
        return cache[key]


class PaddedGroup(NamedTuple):
    """
    Systems of similar size that share one bmm.
    """

    # System index of each block in the group, used to select its mixed weights
    systems: torch.Tensor
    # Rows per block (the largest system in the group)
    block_rows: int
    # First padded row of the group
    offset: int


class PaddedPlan(NamedTuple):
    """
    Layout for running one MOLE layer on several systems with a few bmm calls.
    """

    groups: tuple[PaddedGroup, ...]
    # Total padded rows over all groups
    padded_rows: int
    # Padded position of each input row, in input order
    index: torch.Tensor


def _build_padded_plan(
    sizes: list[int], start: int, num_rows: int, device: torch.device
) -> PaddedPlan | None:
    """
    Build the padded layout for input rows [start, start + num_rows).

    Systems are sorted by size and grouped greedily so that each group's padded
    rows stay within MAX_PADDING_RATIO times its real rows.

    Args:
        sizes: Rows per system for the whole batch.
        start: Global index of the first input row (activation checkpointing).
        num_rows: Number of input rows.
        device: Device for the index tensors.

    Returns:
        The layout, or None when the per-system loop is as good (at most one
        system has rows, or every system needs its own group) or the sizes do
        not cover the input.
    """
    end = start + num_rows
    segments = []  # (system, first input row, rows), in input order
    offset = 0
    for system, size in enumerate(sizes):
        first, last = max(offset, start), min(offset + size, end)
        if last > first:
            segments.append((system, first - start, last - first))
        offset += size
    if len(segments) < 2 or sum(seg[2] for seg in segments) != num_rows:
        return None

    grouped = []  # each group sorted by rows, largest first
    for segment in sorted(segments, key=lambda seg: -seg[2]):
        if grouped:
            group = grouped[-1]
            real = sum(seg[2] for seg in group) + segment[2]
            if (len(group) + 1) * group[0][2] <= MAX_PADDING_RATIO * real:
                group.append(segment)
                continue
        grouped.append([segment])
    if len(grouped) == len(segments):
        return None

    groups = []
    index = torch.empty(num_rows, dtype=torch.long)
    padded_rows = 0
    for group in grouped:
        block_rows = group[0][2]
        for block, (_, first, rows) in enumerate(group):
            begin = padded_rows + block * block_rows
            index[first : first + rows] = torch.arange(begin, begin + rows)
        systems = torch.tensor([seg[0] for seg in group])
        groups.append(PaddedGroup(systems.to(device), block_rows, padded_rows))
        padded_rows += len(group) * block_rows
    return PaddedPlan(tuple(groups), padded_rows, index.to(device))


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

    def forward(self, x):
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )

        assert len(self.global_mole_tensors.mole_sizes) > 0
        plan = self.global_mole_tensors.padded_plan(x.shape[0], x.device)
        if plan is not None:
            return self._padded_forward(x, weights, plan)

        out = []
        ac_start_idx = self.global_mole_tensors.ac_start_idx
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
                out.append(
                    linear_with_folded_batch(x[start:end], weights[n], bias=self.bias)
                )

        result = torch.concatenate(out, dim=0)
        assert (
            result.shape[0] == x.shape[0]
        ), f"result shape {result.shape}, does not match input shape {x.shape} at dim 0"
        return result

    def _padded_forward(self, x, weights, plan: PaddedPlan):
        """
        Apply each system's mixed weights to its rows with one bmm per group.

        Rows are scattered into zero-padded per-system blocks, each group of
        blocks is multiplied by its weights in one batched matmul, and rows are
        gathered back in input order. This replaces one matmul per system per
        layer, which made batched inference bound by CPU-side kernel launches.

        Args:
            x: Input of shape (rows, ..., in_features).
            weights: Mixed weights per system, (systems, out_features, in_features).
            plan: Padded layout from MOLEGlobals.padded_plan.

        Returns:
            Output of shape (rows, ..., out_features).
        """
        rows = x.reshape(x.shape[0], -1, x.shape[-1])
        width, in_features = rows.shape[1], rows.shape[2]
        padded = rows.new_zeros(plan.padded_rows, width, in_features)
        padded = padded.index_copy(0, plan.index, rows)
        outputs = []
        for group in plan.groups:
            blocks = group.systems.shape[0]
            chunk = padded[group.offset : group.offset + blocks * group.block_rows]
            out = torch.bmm(
                chunk.reshape(blocks, group.block_rows * width, in_features),
                weights[group.systems].transpose(1, 2),
            )
            outputs.append(out.reshape(blocks * group.block_rows, width, -1))
        out = torch.cat(outputs).index_select(0, plan.index)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], out.shape[-1])
