"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import logging
import threading

import torch
from torch import distributed as dist
from torch.distributed.nn.functional import all_reduce, reduce_scatter

"""
Functions to support graph parallel training.
This is based on the Megatron-LM implementation:
https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/model_parallel/initialize.py
"""

########## INITIALIZATION ##########

_GRAPH_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None

_tls = threading.local()


def pad_input(input: torch.Tensor, padded_size: int):
    # pad using functional
    # if input.shape[0]!=padded_size:
    #    input=torch.nn.functional.pad(input,(0,0,0,0,0,1)).contiguous()

    # pad using manual tensor cat
    if input.shape[0] != padded_size:
        input = torch.cat(
            [
                input,
                torch.zeros(
                    (padded_size - input.shape[0], *input.shape[1:]),
                    device=input.device,
                    dtype=input.dtype,
                ),
            ],
            dim=0,
        )

        assert input.shape[0] == padded_size

    return input


def ensure_div(a: int, b: int) -> None:
    assert a % b == 0


def divide_and_check_no_remainder(a: int, b: int) -> int:
    ensure_div(a, b)
    return a // b


def setup_graph_parallel_groups(
    graph_parallel_group_size: int, distributed_backend: str
) -> None:
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    assert (
        graph_parallel_group_size <= world_size
    ), "graph parallel group size must be at most world size"

    ensure_div(world_size, graph_parallel_group_size)
    dp_size = world_size // graph_parallel_group_size
    rank = dist.get_rank()

    if rank == 0:
        logging.info(
            f"> initializing graph parallel with size {graph_parallel_group_size}"
        )
        logging.info(f"> initializing ddp with size {dp_size}")

    groups = torch.arange(world_size).reshape(dp_size, graph_parallel_group_size)
    found = [x.item() for x in torch.where(groups == rank)]

    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    for j in range(graph_parallel_group_size):
        group = dist.new_group(groups[:, j].tolist(), backend=distributed_backend)
        if j == found[1]:
            _DATA_PARALLEL_GROUP = group
    global _GRAPH_PARALLEL_GROUP
    assert _GRAPH_PARALLEL_GROUP is None, "graph parallel group is already initialized"
    for i in range(dp_size):
        group = dist.new_group(groups[i, :].tolist(), backend=distributed_backend)
        if i == found[0]:
            _GRAPH_PARALLEL_GROUP = group


def setup_gp(config) -> None:
    gp_size = config["gp_gpus"]
    backend = config["distributed_backend"]
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()

    gp_size = min(gp_size, world_size)
    ensure_div(world_size, gp_size)
    dp_size = world_size // gp_size
    rank = dist.get_rank()

    if rank == 0:
        logging.info(f"> initializing graph parallel with size {gp_size}")
        logging.info(f"> initializing ddp with size {dp_size}")

    groups = torch.arange(world_size).reshape(dp_size, gp_size)
    found = [x.item() for x in torch.where(groups == rank)]

    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    for j in range(gp_size):
        group = dist.new_group(groups[:, j].tolist(), backend=backend)
        if j == found[1]:
            _DATA_PARALLEL_GROUP = group
    global _GRAPH_PARALLEL_GROUP
    assert _GRAPH_PARALLEL_GROUP is None, "graph parallel group is already initialized"
    for i in range(dp_size):
        group = dist.new_group(groups[i, :].tolist(), backend=backend)
        if i == found[0]:
            _GRAPH_PARALLEL_GROUP = group


def cleanup_gp() -> None:
    global _DATA_PARALLEL_GROUP
    global _GRAPH_PARALLEL_GROUP
    assert _GRAPH_PARALLEL_GROUP is not None
    assert _DATA_PARALLEL_GROUP is not None
    with contextlib.suppress(ValueError):
        dist.destroy_process_group(_DATA_PARALLEL_GROUP)
    with contextlib.suppress(ValueError):
        dist.destroy_process_group(_GRAPH_PARALLEL_GROUP)
    _DATA_PARALLEL_GROUP = None
    _GRAPH_PARALLEL_GROUP = None


def initialized() -> bool:
    return _GRAPH_PARALLEL_GROUP is not None


def get_dp_group():
    return _DATA_PARALLEL_GROUP


def get_gp_group():
    return _GRAPH_PARALLEL_GROUP


def get_dp_rank() -> int:
    return dist.get_rank(group=get_dp_group())


def get_gp_rank() -> int:
    return dist.get_rank(group=get_gp_group())


def get_dp_world_size() -> int:
    return dist.get_world_size(group=get_dp_group())


def get_gp_world_size() -> int:
    return 1 if not initialized() else dist.get_world_size(group=get_gp_group())


########## DIST METHODS ##########


def size_list_fn(size: int, parts: int) -> list[int]:
    return [size // parts + (1 if idx < size % parts else 0) for idx in range(parts)]


def reduce_from_model_parallel_region(input: torch.Tensor) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ReduceFromModelParallelRegion.apply(input)


class ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        # return _reduce(ctx, input) # this operates in place
        return all_reduce(input, group=get_gp_group())  # this operats out of place

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def scatter_to_model_parallel_region(input: torch.Tensor) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ScatterToModelParallelRegion.apply(input)


# this returns the values in place
class ScatterToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx.split_sizes = size_list_fn(input.shape[0], get_gp_world_size())
        return input.split(ctx.split_sizes)[get_gp_rank()]

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: torch.Tensor):
        return gather_from_model_parallel_region_sum_grad(
            grad_output, sum(ctx.split_sizes)
        )


def gather_from_model_parallel_region(
    input: torch.Tensor,
    natoms: int,
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    world_size = get_gp_world_size()
    size_list = size_list_fn(natoms, world_size)

    input = pad_input(
        input, natoms // world_size + (1 if natoms % world_size != 0 else 0)
    )

    tensor_list_w_padding = GatherFromModelParallelRegionGradPadded.apply(input)

    return torch.cat(
        [
            t.narrow(0, 0, s) if t.shape[0] != s else t
            for t, s in zip(tensor_list_w_padding, size_list)
        ],
        dim=0,
    )


def gather_from_model_parallel_region_sum_grad(
    input: torch.Tensor,
    natoms: int,
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    world_size = get_gp_world_size()
    size_list = size_list_fn(natoms, world_size)

    input = pad_input(
        input, natoms // world_size + (1 if natoms % world_size != 0 else 0)
    )

    tensor_list_w_padding = GatherFromModelParallelRegionSumGradPadded.apply(input)

    return torch.cat(
        [
            t.narrow(0, 0, s) if t.shape[0] != s else t
            for t, s in zip(tensor_list_w_padding, size_list)
        ],
        dim=0,
    )


class GatherFromModelParallelRegionGradPadded(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.rank = get_gp_rank()
        ctx.group = get_gp_group()
        tensor_list = [torch.empty_like(input) for _ in range(get_gp_world_size())]
        dist.all_gather(tensor_list, input, group=ctx.group)
        return tuple(tensor_list)

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, *grad_outputs):
        return grad_outputs[ctx.rank]


class GatherFromModelParallelRegionSumGradPadded(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.rank = get_gp_rank()
        ctx.group = get_gp_group()
        if dist.get_backend() == "gloo":
            ctx.shape = input.shape
        tensor_list = [torch.empty_like(input) for _ in range(get_gp_world_size())]
        dist.all_gather(tensor_list, input, group=ctx.group)
        return tuple(tensor_list)

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, *grad_outputs):
        if dist.get_backend() == "gloo":
            grad_output = all_reduce(torch.cat(grad_outputs, dim=0), group=ctx.group)
            ctx.padded_size = grad_outputs[0].shape[0]
            result = grad_output[
                ctx.padded_size * ctx.rank : ctx.padded_size * ctx.rank + ctx.shape[0]
            ]
            return result
        local_grad_output = grad_outputs[ctx.rank]
        output_tensor = torch.empty_like(local_grad_output)
        return reduce_scatter(output_tensor, grad_outputs, group=ctx.group)


def scale_backward_grad(input: torch.Tensor) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ScaleBackwardGrad.apply(input)


# Leave forward untouched but upscale the gradient by a factor of gp_group_size
# DDP reduces a mean across the loss, if we have gp_group_size=2 and 6 ranks
# that means we do (a_1+a_2+a_3+b_1+b_2+b_3)/6 in ddp mean. This gets us the
# correct loss but the grad is wrong by a factor of gp_group_size
# dL/d_a1 = 1/6 but it should be dL/da = 1/2 (for the equivalanet non GP run
# with 2 ranks)
# we coud perform an extra round of all_reduce, but this would increase
# communication overhead, instead we can just upscsale the gradient only and
# avoid over head communication
class ScaleBackwardGrad(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: torch.Tensor):
        return dist.get_world_size(get_gp_group()) * grad_output
