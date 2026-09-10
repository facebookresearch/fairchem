"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass

import torch
import torch.nn as nn

from .matmul import linear_with_folded_batch

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
    # Index maps for the batched MOLE path, built once per batch by
    # set_padded_segments. None means the per-system loop is used.
    pad_index: torch.Tensor | None = None
    unpad_index: torch.Tensor | None = None
    pad_shape: tuple[int, int] | None = None


def set_padded_segments(
    globals_obj: MOLEGlobals,
    sizes: list[int],
    device: torch.device,
    max_pad_ratio: float = 1.25,
) -> None:
    """
    Build the index maps that turn the per-system loop into one bmm.

    Rows of the MOLE input are contiguous per system, system b owning
    sizes[b] rows. pad_index gathers them into a [B, Smax] padded layout
    (padding slots point at row 0 and are never read back), and unpad_index
    maps every real row back to its padded slot. The padded layout copies
    the input and multiplies the GEMM work by B * Smax / sum(sizes), so it is
    only built when that ratio is at most max_pad_ratio; otherwise the loop
    over split views is used.
    """
    num_systems = len(sizes)
    total = sum(sizes)
    if (
        num_systems < 2
        or total == 0
        or num_systems * max(sizes) > max_pad_ratio * total
    ):
        globals_obj.pad_index = None
        globals_obj.unpad_index = None
        globals_obj.pad_shape = None
        return
    max_size = max(sizes)
    sizes_t = torch.tensor(sizes, device=device)
    starts = torch.cumsum(sizes_t, 0) - sizes_t
    system_of_row = torch.repeat_interleave(
        torch.arange(num_systems, device=device), sizes_t, output_size=sum(sizes)
    )
    rows = torch.arange(sum(sizes), device=device)
    unpad_index = system_of_row * max_size + rows - starts[system_of_row]
    pad_index = torch.zeros(num_systems * max_size, dtype=torch.long, device=device)
    pad_index[unpad_index] = rows
    globals_obj.pad_index = pad_index
    globals_obj.unpad_index = unpad_index
    globals_obj.pad_shape = (num_systems, max_size)


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
            # coefficients [B, E] @ weights [E, O*I] -> [B, O, I]. Written as a
            # plain matmul so that the weight gradient comes back as the
            # contiguous [E, O*I] product and matches the parameter layout,
            # which lets DDP use it in place instead of copying it into the
            # bucket view every step.
            coefficients = self.global_mole_tensors.expert_mixing_coefficients
            flat_weights = self.weights.flatten(1)
            if flat_weights.shape[0] == 1 and coefficients.shape[1] != 1:
                # a single expert bank against wider coefficients: the einsum
                # this replaces broadcast the expert dim, i.e. summed them
                coefficients = coefficients.sum(dim=1, keepdim=True)
            weights = torch.mm(coefficients, flat_weights).view(
                -1, self.out_features, self.in_features
            )

        ac_start_idx = self.global_mole_tensors.ac_start_idx
        assert len(self.global_mole_tensors.mole_sizes) > 0
        pad_index = self.global_mole_tensors.pad_index
        if (
            pad_index is not None
            and ac_start_idx == 0
            and x.shape[0] == self.global_mole_tensors.unpad_index.shape[0]
        ):
            return self._forward_batched(x, weights, pad_index)

        mole_sizes = self.global_mole_tensors.mole_sizes
        if (
            ac_start_idx == 0
            and mole_sizes.device.type == "cpu"
            and x.shape[0] == int(mole_sizes.sum())
        ):
            # Whole input: split into per-system views. One op forward and one
            # cat backward, instead of B slices whose backward each zero-fills
            # a full-size tensor.
            out = [
                linear_with_folded_batch(segment, weights[n], bias=self.bias)
                for n, segment in enumerate(x.split(mole_sizes.tolist(), dim=0))
            ]
            return torch.concatenate(out, dim=0)

        out = []
        # TODO: precompute these if needed but they should be small and on cpu
        start_idxs = [0] + torch.cumsum(mole_sizes, dim=0).tolist()
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

    def _forward_batched(self, x, weights, pad_index):
        """
        All systems in one bmm over a padded [B, Smax, ...] layout.

        Padded slots hold a copy of row 0; their outputs are never gathered
        back, so their gradient is zero and they contribute nothing to the
        weight gradient.
        """
        num_systems, max_size = self.global_mole_tensors.pad_shape
        x_pad = x.index_select(0, pad_index)
        x_pad = x_pad.reshape(num_systems, -1, self.in_features)
        out = torch.bmm(x_pad, weights.transpose(1, 2))
        out = out.reshape(num_systems * max_size, *x.shape[1:-1], self.out_features)
        out = out.index_select(0, self.global_mole_tensors.unpad_index)
        if self.bias is not None:
            out = out + self.bias
        return out
