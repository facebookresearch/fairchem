"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import logging

import torch
from torch import distributed as dist
from torch.distributed._functional_collectives import all_gather_tensor_autograd
from torch.distributed.nn.functional import all_reduce
from torch.distributed._functional_collectives import (
            all_reduce as functional_all_reduce,
        )
"""
Functions to support graph parallel training.
This is based on the Megatron-LM implementation:
https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/model_parallel/initialize.py
"""

########## INITIALIZATION ##########

_GRAPH_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None
_CACHED_BACKEND = None
_CACHED_GP_WORLD_SIZE = None
_CACHED_GP_RANK = None


def _populate_cache() -> None:
    global _CACHED_BACKEND, _CACHED_GP_WORLD_SIZE, _CACHED_GP_RANK
    _CACHED_BACKEND = dist.get_backend()
    _CACHED_GP_WORLD_SIZE = dist.get_world_size(group=_GRAPH_PARALLEL_GROUP)
    _CACHED_GP_RANK = dist.get_rank(group=_GRAPH_PARALLEL_GROUP)


def ensure_div(a: int, b: int) -> None:
    assert a % b == 0


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
    _populate_cache()


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
    _populate_cache()


def cleanup_gp() -> None:
    global _DATA_PARALLEL_GROUP
    global _GRAPH_PARALLEL_GROUP
    global _CACHED_BACKEND, _CACHED_GP_WORLD_SIZE, _CACHED_GP_RANK
    assert _GRAPH_PARALLEL_GROUP is not None
    assert _DATA_PARALLEL_GROUP is not None
    with contextlib.suppress(ValueError):
        dist.destroy_process_group(_DATA_PARALLEL_GROUP)
    with contextlib.suppress(ValueError):
        dist.destroy_process_group(_GRAPH_PARALLEL_GROUP)
    _DATA_PARALLEL_GROUP = None
    _GRAPH_PARALLEL_GROUP = None
    _CACHED_BACKEND = None
    _CACHED_GP_WORLD_SIZE = None
    _CACHED_GP_RANK = None


def initialized() -> bool:
    return _GRAPH_PARALLEL_GROUP is not None


def get_dp_group():
    return _DATA_PARALLEL_GROUP


def get_gp_group():
    return _GRAPH_PARALLEL_GROUP


def get_dp_rank() -> int:
    return dist.get_rank(group=get_dp_group())


def get_gp_rank() -> int:
    if _CACHED_GP_RANK is not None:
        return _CACHED_GP_RANK
    return dist.get_rank(group=get_gp_group())


def get_dp_world_size() -> int:
    return dist.get_world_size(group=get_dp_group())


def get_gp_world_size() -> int:
    if _CACHED_GP_WORLD_SIZE is not None:
        return _CACHED_GP_WORLD_SIZE
    return 1 if not initialized() else dist.get_world_size(group=get_gp_group())


########## DIST METHODS ##########


def size_list_fn(size: int, parts: int) -> list[int]:
    return [size // parts + (1 if idx < size % parts else 0) for idx in range(parts)]


def reduce_from_model_parallel_region(
    input: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    if training or _CACHED_BACKEND == "gloo":
        return ReduceFromModelParallelRegion.apply(input)
    # Inference on nccl: use compile-traceable functional collective
    return functional_all_reduce(input, "sum", get_gp_group())


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
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input.chunk(get_gp_world_size())[get_gp_rank()]

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: torch.Tensor):
        return gather_from_model_parallel_region_sum_grad(grad_output)


# Only used in Linear_Force_Head
def gather_from_model_parallel_region(
    input: torch.Tensor,
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"

    tensor_list_w_padding = GatherFromModelParallelRegionGradPadded.apply(input)

    return torch.cat(
        tensor_list_w_padding,
        dim=0,
    )


def gather_from_model_parallel_region_sum_grad(
    input: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"

    if training or _CACHED_BACKEND == "gloo":
        return torch.cat(
            GatherFromModelParallelRegionSumGradPadded.apply(input),
            dim=0,
        )
    # Inference on nccl: use compile-traceable functional collective.
    # all_gather_tensor_autograd already returns a single tensor
    # concatenated along gather_dim, so no chunk+cat needed.
    return all_gather_tensor_autograd(input, gather_dim=0, group=get_gp_group())


class GatherFromModelParallelRegionGradPadded(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        tensor_list = [torch.empty_like(input) for _ in range(get_gp_world_size())]
        dist.all_gather(tensor_list, input, group=get_gp_group())
        return tuple(tensor_list)

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, *grad_outputs):
        return grad_outputs[get_gp_rank()]


class GatherFromModelParallelRegionSumGradPadded(torch.autograd.Function):
    """Gloo-only fallback: all_gather forward, all_reduce+slice backward.

    Used instead of all_gather_tensor_autograd on gloo because the
    _c10d_functional reduce_scatter backward lacks second-order
    autograd support.
    """

    @staticmethod
    @torch.compiler.disable
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        tensor_list = [torch.empty_like(input) for _ in range(get_gp_world_size())]
        dist.all_gather(tensor_list, input, group=get_gp_group())
        return tuple(tensor_list)

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, *grad_outputs):
        grad_output = all_reduce(torch.cat(grad_outputs, dim=0), group=get_gp_group())
        size = grad_outputs[0].shape[0]
        return grad_output[size * get_gp_rank() : size * (get_gp_rank() + 1)]


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
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return dist.get_world_size(get_gp_group()) * grad_output
