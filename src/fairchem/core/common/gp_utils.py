"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch
from torch import distributed as dist
from torch.distributed.nn.functional import all_gather, all_reduce, reduce_scatter

"""
Functions to support graph parallel training.
This is based on the Megatron-LM implementation:
https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/model_parallel/initialize.py
"""

########## INITIALIZATION ##########

_GRAPH_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None


def edge_partition_by_node_idxs(min_node, max_node, edge_index):
    return torch.where(
        torch.logical_and(
            edge_index[1] >= min_node,
            edge_index[1] <= max_node,  # TODO: 0 or 1?
        )
    )[0]


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


@torch.enable_grad()
def pad_tensor(
    tensor: torch.Tensor, dim: int = -1, target_size: int | None = None
) -> torch.Tensor:
    size = tensor.size(dim)
    if target_size is None:
        world_size = get_gp_world_size()
        pad_size = 0 if size % world_size == 0 else world_size - size % world_size
    else:
        pad_size = target_size - size
    if pad_size == 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    padding = torch.empty(pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=dim)


def trim_tensor(tensor: torch.Tensor, sizes: torch.Tensor | None = None, dim: int = 0):
    size = tensor.size(dim)
    world_size = get_gp_world_size()
    if size % world_size == 0:
        return tensor, sizes
    trim_size = size - size % world_size
    if dim == 0:
        tensor = tensor[:trim_size]
    elif dim == 1:
        tensor = tensor[:, :trim_size]
    else:
        raise ValueError
    if sizes is not None:
        sizes[-1] = sizes[-1] - size % world_size
    return tensor, sizes


def _reduce(ctx: Any, input: torch.Tensor) -> torch.Tensor:
    group = get_gp_group()
    if ctx:
        ctx.mark_dirty(input)
    dist.all_reduce(input, group=group)
    return input


def _split(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    rank = get_gp_rank()
    group = get_gp_group()
    world_size = dist.get_world_size(group=group)
    sizes = [
        input.shape[dim] // world_size
        + (1 if idx < input.shape[dim] % world_size else 0)
        for idx in range(world_size)
    ]
    return torch.split(input, sizes, dim=dim)[rank]


def size_list_fn(size, parts):
    return [size // parts + (1 if idx < size % parts else 0) for idx in range(parts)]


def _gather_assume_padded(
    input: torch.Tensor, size_list, async_op=False
) -> torch.Tensor:
    group = get_gp_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input

    assert max(size_list) == min(size_list)

    # gloo does not support all_gather with different sized tensors
    slice_size = max(size_list)
    all_atoms = torch.zeros(
        (slice_size * world_size,) + input.shape[1:],
        device=input.device,
        dtype=input.dtype,
    )
    tensor_list = list(all_atoms.split(slice_size, dim=0))

    handle = dist.all_gather(tensor_list, input, group=group, async_op=async_op)
    if async_op:
        return all_atoms, handle

    return all_atoms, None


def _gather_with_padding_gloo(
    input: torch.Tensor, size_list, async_op=False
) -> torch.Tensor:
    group = get_gp_group()
    rank = get_gp_rank()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input

    # gloo does not support all_gather with different sized tensors
    slice_size = max(size_list)
    all_atoms = torch.zeros(
        (slice_size * world_size,) + input.shape[1:],
        device=input.device,
        dtype=input.dtype,
    )
    tensor_list = list(all_atoms.split(slice_size, dim=0))
    # assert(input.requires_grad==True)
    if input.shape[0] < slice_size:
        rg = input.requires_grad
        input = pad_tensor(input, 0, slice_size)
        assert input.requires_grad == rg

    handle = dist.all_gather(tensor_list, input, group=group, async_op=async_op)
    # assert(input.requires_grad==True)
    tensor_list[rank] = input  # pop back in our local copy (requires grad)
    if async_op:
        # assert(tensor_list[rank].requires_grad==True)
        return tensor_list, handle

    tensor_list = [
        tensor.narrow(0, 0, size) for tensor, size in zip(tensor_list, size_list)
    ]
    return torch.cat(
        tensor_list,
        dim=0,
    )


def _gather_with_padding(input: torch.Tensor, size_list) -> torch.Tensor:
    group = get_gp_group()
    rank = get_gp_rank()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input

    all_atoms = torch.zeros(
        (sum(size_list),) + input.shape[1:], device=input.device, dtype=input.dtype
    )
    tensor_list = list(all_atoms.split(size_list, dim=0))

    dist.all_gather(tensor_list, input, group=group)
    tensor_list[rank] = input  # pop back in our local copy (requires grad)

    node_offset = sum(size_list[:rank])
    all_atoms[node_offset : node_offset + input.shape[0]] = input
    return all_atoms


class CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return _reduce(None, grad_output)


class ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        # return _reduce(ctx, input) # this operates in place
        return all_reduce(input, group=get_gp_group())  # this operats out of place

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


# this returns the values in place
class ScatterToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        result = _split(input, dim)
        ctx.save_for_backward(torch.tensor(dim))
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (dim,) = ctx.saved_tensors
        return _gather_with_padding(grad_output.clone(), dim.item()), None


class GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, size_list) -> torch.Tensor:
        if dist.get_backend() == "gloo":
            return _gather_with_padding_gloo(input, size_list)
        return _gather_with_padding(input, size_list)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        result = _split(grad_output, 0)
        return result, None, None


class GatherFromModelParallelRegionSumGradAsyncGLOO(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input: torch.Tensor, size_list: list[int], async_op
    ) -> torch.Tensor:
        ctx.rank = get_gp_rank()
        ctx.group = get_gp_group()
        ctx.offset = sum(size_list[: ctx.rank])
        ctx.shape = input.shape
        ctx.world_size = get_gp_world_size()
        ctx.size_list = size_list

        ctx.padded_size = max(size_list)
        all_atoms = torch.zeros(
            (ctx.padded_size * ctx.world_size,) + input.shape[1:],
            device=input.device,
            dtype=input.dtype,
        )
        # TODO CAN OPTIMIZE< dont need all zeros!
        if input.shape[0] != ctx.padded_size:
            _input = torch.zeros(
                (ctx.padded_size, *input.shape[1:]),
                device=input.device,
                dtype=input.dtype,
            )
            _input[: ctx.shape[0]] = input
            input = _input
        tensor_list = list(all_atoms.split(ctx.padded_size, dim=0))
        for x in tensor_list:
            assert x.shape[0] == ctx.padded_size

        ctx.all_atoms_shape = all_atoms.shape
        handle = dist.all_gather(tensor_list, input, group=ctx.group, async_op=async_op)
        # print("NO WAY THIS WORKS")
        return all_atoms, handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        output_tensor = torch.empty(
            (ctx.padded_size * ctx.world_size,) + ctx.shape[1:],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # print("WTF IS GOIN ON???")

        dist.all_to_all_single(
            output=output_tensor,
            input=grad_output,
            output_split_sizes=[ctx.padded_size] * ctx.world_size,
            input_split_sizes=[ctx.padded_size] * ctx.world_size,
            group=ctx.group,
            async_op=False,
        )
        result = output_tensor.view(
            ctx.world_size, ctx.padded_size, *ctx.shape[1:]
        ).sum(dim=0)[: ctx.shape[0]]

        # #previous
        # grad_output = all_reduce(grad_output, group=ctx.group)

        # #result = _split(grad_output, 0)
        # #result=torch.split(grad_output,ctx.size_list)[ctx.rank]
        # if ctx.gloo_backend:
        #     result=grad_output[ctx.padded_size*ctx.rank:ctx.padded_size*ctx.rank+ctx.shape[0]]
        # else:
        #     result=grad_output[ctx.offset:ctx.offset+ctx.shape[0]]
        return result, None, None, None


class GatherFromModelParallelRegionSumGradAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, natoms: int) -> torch.Tensor:
        ctx.rank = get_gp_rank()
        ctx.group = get_gp_group()
        ctx.shape = input.shape
        ctx.world_size = get_gp_world_size()
        ctx.size_list = size_list_fn(natoms, ctx.world_size)
        # ctx.offset = sum(size_list[: ctx.rank])

        all_atoms = torch.empty(
            (natoms,) + input.shape[1:],
            device=input.device,
            dtype=input.dtype,
        )

        tensor_list = []
        offset = 0
        for size in ctx.size_list:
            # tensor_list.append(all_atoms[offset:offset+size])
            tensor_list.append(all_atoms.narrow(0, offset, size))
            offset += size
        # tensor_list = list(all_atoms.split(ctx.size_list, dim=0))

        ctx.all_atoms_shape = all_atoms.shape
        handle = dist.all_gather(tensor_list, input, group=ctx.group, async_op=True)
        return all_atoms, handle

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output_tensor = torch.empty(
            (ctx.shape[0] * ctx.world_size,) + ctx.shape[1:],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )

        dist.all_to_all_single(
            output=output_tensor,
            input=grad_output,
            output_split_sizes=[ctx.shape[0]] * ctx.world_size,
            input_split_sizes=ctx.size_list,
            group=ctx.group,
            async_op=False,
        )
        # handle.wait()
        result = output_tensor.view(ctx.world_size, *ctx.shape).sum(dim=0)

        # #previous
        # grad_output = all_reduce(grad_output, group=ctx.group)

        # #result = _split(grad_output, 0)
        # #result=torch.split(grad_output,ctx.size_list)[ctx.rank]
        # if ctx.gloo_backend:
        #     result=grad_output[ctx.padded_size*ctx.rank:ctx.padded_size*ctx.rank+ctx.shape[0]]
        # else:
        #     result=grad_output[ctx.offset:ctx.offset+ctx.shape[0]]
        return result, None


class GatherFromModelParallelRegionSumGradNoAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, natoms: int) -> torch.Tensor:
        ctx.rank = get_gp_rank()
        ctx.group = get_gp_group()
        ctx.shape = input.shape
        ctx.world_size = get_gp_world_size()
        ctx.size_list = size_list_fn(natoms, ctx.world_size)
        # ctx.offset = sum(size_list[: ctx.rank])

        all_atoms = torch.empty(
            (natoms,) + input.shape[1:],
            device=input.device,
            dtype=input.dtype,
            requires_grad=False,
        )

        # tensor_list=[]
        # offset=0
        # with torch.no_grad():
        #     for size in ctx.size_list:
        #         #tensor_list.append(all_atoms[offset:offset+size])
        #         tensor_list.append(all_atoms.narrow(0,offset,size))
        #         offset+=size
        # with torch.no_grad():
        # tensor_list = list(all_atoms.split(ctx.size_list, dim=0))
        tensor_list = [
            torch.empty(
                (size,) + input.shape[1:],
                device=input.device,
                dtype=input.dtype,
                requires_grad=False,
            )
            for size in ctx.size_list
        ]
        # print("ALL ATOMS",all_atoms.requires_grad,[ t.requires_grad for t in tensor_list])
        ctx.all_atoms_shape = all_atoms.shape
        dist.all_gather(tensor_list, input, group=ctx.group, async_op=False)
        return tensor_list

    @staticmethod
    def backward(ctx, grad_outputs):
        local_grad_output = grad_outputs[ctx.rank]
        output_tensor = torch.empty_like(local_grad_output)

        dist.reduce_scatter(output_tensor, grad_outputs, group=ctx.group)
        result = output_tensor
        # dist.all_to_all_single(
        #     output=output_tensor,
        #     input=grad_output,
        #     output_split_sizes=[ctx.shape[0]] * ctx.world_size,
        #     input_split_sizes=ctx.size_list,
        #     group=ctx.group,
        #     async_op=False,
        # )
        # # handle.wait()
        # result = output_tensor.view(ctx.world_size, *ctx.shape).sum(dim=0)

        # #previous
        # grad_output = all_reduce(grad_output, group=ctx.group)

        # #result = _split(grad_output, 0)
        # #result=torch.split(grad_output,ctx.size_list)[ctx.rank]
        # if ctx.gloo_backend:
        #     result=grad_output[ctx.padded_size*ctx.rank:ctx.padded_size*ctx.rank+ctx.shape[0]]
        # else:
        #     result=grad_output[ctx.offset:ctx.offset+ctx.shape[0]]
        return result, None


class GatherFromModelParallelRegionSumGradPaddedNoAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.rank = get_gp_rank()
        ctx.group = get_gp_group()
        tensor_list = [torch.empty_like(input) for _ in range(get_gp_world_size())]
        dist.all_gather(tensor_list, input, group=ctx.group, async_op=False)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        local_grad_output = grad_outputs[ctx.rank]
        output_tensor = torch.empty_like(local_grad_output)
        return reduce_scatter(output_tensor, grad_outputs, group=ctx.group)


class GatherFromModelParallelRegionSumGradPaddedNoAsyncGLOO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.rank = get_gp_rank()
        ctx.group = get_gp_group()
        ctx.shape = input.shape
        tensor_list = [torch.empty_like(input) for _ in range(get_gp_world_size())]
        dist.all_gather(tensor_list, input, group=ctx.group, async_op=False)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = all_reduce(torch.cat(grad_outputs, dim=0), group=ctx.group)
        ctx.padded_size = grad_outputs[0].shape[0]
        result = grad_output[
            ctx.padded_size * ctx.rank : ctx.padded_size * ctx.rank + ctx.shape[0]
        ]
        return result, None


class GatherFromModelParallelRegionSumGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, size_list: int) -> torch.Tensor:
        if dist.get_backend() == "gloo":
            return _gather_with_padding_gloo(input, size_list)
        return _gather_with_padding(input, size_list)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = get_gp_group()
        # use dist internal # does not work
        # reduced_grad_output = grad_output.clone()
        # dist.all_reduce(
        #    reduced_grad_output, group=group
        # )  # This is an inplace operation
        # grad_output = reduced_grad_output

        # use functional version instead
        grad_output = all_reduce(grad_output, group=group)

        result = _split(grad_output, 0)
        return result, None, None


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


def copy_to_model_parallel_region(input: torch.Tensor) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return CopyToModelParallelRegion.apply(input)


def reduce_from_model_parallel_region(input: torch.Tensor) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ReduceFromModelParallelRegion.apply(input)


def scatter_to_model_parallel_region(
    input: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ScatterToModelParallelRegion.apply(input, dim)


def gather_from_model_parallel_region(input: torch.Tensor, size_list) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return GatherFromModelParallelRegion.apply(input, size_list)


def gather_from_model_parallel_region_sum_grad(
    input: torch.Tensor, size_list
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return GatherFromModelParallelRegionSumGrad.apply(input, size_list)


def gather_from_model_parallel_region_sum_grad_noasync(
    input: torch.Tensor, natoms: int, gloo_backend: bool
) -> torch.Tensor:
    # TODO REMOVE ASSDERT?
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    # input = input.contiguous()
    world_size = get_gp_world_size()
    size_list = size_list_fn(natoms, world_size)
    padded_size = natoms // world_size + (1 if natoms % world_size != 0 else 0)

    # pad using functional
    # if input.shape[0]!=padded_size:
    #    input=torch.nn.functional.pad(input,(0,0,0,0,0,1)).contiguous()

    # pad using manual tensor cat
    if input.shape[0] != padded_size:
        _input = torch.empty(
            (padded_size, *input.shape[1:]),
            device=input.device,
            dtype=input.dtype,
        )
        _input[: input.shape[0]] = input
        _input[input.shape[0] :] = 0
        input = _input

    input = input.contiguous()
    assert input.shape[0] == padded_size

    if gloo_backend:
        tensor_list_w_padding = (
            GatherFromModelParallelRegionSumGradPaddedNoAsyncGLOO.apply(input)
        )
    else:
        tensor_list_w_padding = all_gather(input, group=get_gp_group())
        # tensor_list_w_padding= GatherFromModelParallelRegionSumGradPaddedNoAsync.apply(input)

    # return torch.cat([t[:s] for t, s in zip(tensor_list_w_padding, size_list)], dim=0)
    return torch.cat(
        [
            t.narrow(0, 0, s) if t.shape[0] != s else t
            for t, s in zip(tensor_list_w_padding, size_list)
        ],
        dim=0,
    )


def gather_from_model_parallel_region_sum_grad_async(
    input: torch.Tensor, async_op: bool, natoms: int
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return GatherFromModelParallelRegionSumGradAsync.apply(input, async_op, natoms)


def gather_from_model_parallel_region_sum_grad_async_gloo(
    input: torch.Tensor, size_list: list[int], async_op: bool
) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return GatherFromModelParallelRegionSumGradAsyncGLOO.apply(
        input, size_list, async_op
    )


def scale_backward_grad(input: torch.Tensor) -> torch.Tensor:
    assert initialized(), "Cannot use graph parallel with initializing gp group, must call setup_gp from gp_utils.py!"
    return ScaleBackwardGrad.apply(input)
