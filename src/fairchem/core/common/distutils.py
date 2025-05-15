"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import timedelta
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.elastic.utils.distributed import get_free_port
from torchtnt.utils.distributed import get_file_init_method, get_tcp_init_method

from fairchem.core.common.typing import none_throws

T = TypeVar("T")
DISTRIBUTED_PORT = 13356
CURRENT_DEVICE_TYPE_STR = "CURRRENT_DEVICE_TYPE"


def os_environ_get_or_throw(x: str) -> str:
    if x not in os.environ:
        raise RuntimeError(f"Could not find {x} in ENV variables")
    return none_throws(os.environ.get(x))


def get_init_method(
    init_method,
    world_size: int | None,
    rank: int | None = None,
    node_list: str | None = None,
    filename: str | None = None,
):
    """
    Get the initialization method for a distributed job based on the specified method type.

    Args:
        init_method: The initialization method type, either "tcp" or "file".
        world_size: The total number of processes in the distributed job.
        rank: The rank of the current process (optional).
        node_list: The list of nodes for SLURM-based distributed job (optional, used with "tcp").
        filename: The shared file path for file-based initialization (optional, used with "file").

    Returns:
        The initialization method string to be used by PyTorch's distributed module.

    Raises:
        ValueError: If an invalid init_method is provided.
    """
    if init_method == "tcp":
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", node_list]
        )
        return get_tcp_init_method(
            world_size=world_size,
            hostname=hostnames.split()[0].decode("utf-8"),
            rank=rank,
            port=get_free_port() if world_size == 1 else DISTRIBUTED_PORT,
        )
    elif init_method == "file":
        return get_file_init_method(world_size=world_size, rank=rank, filename=filename)
    else:
        raise ValueError(f"Invalid init_method: {init_method}")


def setup(config) -> None:
    timeout = timedelta(minutes=config.get("timeout", 30))
    if config["submit"]:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            try:
                nnodes = int(os_environ_get_or_throw("SLURM_NNODES"))
                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os_environ_get_or_throw("SLURM_NTASKS"))
                    nnodes = int(os_environ_get_or_throw("SLURM_NNODES"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    assert config["world_size"] % nnodes == 0
                    gpus_per_node = config["world_size"] // nnodes
                    node_id = int(os_environ_get_or_throw("SLURM_NODEID"))
                    rank = node_id * gpus_per_node
                    local_rank = 0
                else:
                    assert ntasks_per_node == config["world_size"] // nnodes
                    rank = int(os_environ_get_or_throw("SLURM_PROCID"))
                    local_rank = int(os_environ_get_or_throw("SLURM_LOCALID"))

                init_method = get_init_method(
                    config["init_method"],
                    world_size=config["world_size"],
                    rank=rank,
                    node_list=node_list,
                    filename=os.path.join(
                        config["shared_file_dir"],
                        f".distributed-shared-file-{config['array_job_num']}",
                    ),
                )
                logging.info(f"Torch distributed initialized with: {init_method}")

                # ensures GPU0 does not have extra context/higher peak memory
                logging.info(
                    f"local rank: {local_rank}, visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}"
                )

                assign_device_for_local_rank(config["cpu"], local_rank)

                dist.init_process_group(
                    backend="nccl",
                    init_method=init_method,
                    timeout=timeout,
                )
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass
    else:  # local mode
        if "MASTER_ADDR" not in os.environ:
            assert (
                config["world_size"] == 1
            ), "Can only setup master address and port at this point for a single rank, otherwise we assume the processes and the comm addr/port have already been setup"
            setup_env_local()
        local_rank = int(os.environ["LOCAL_RANK"])
        assign_device_for_local_rank(config["cpu"], local_rank)

        dist.init_process_group(
            backend=config["distributed_backend"],
            rank=int(os.environ["RANK"]),
            world_size=config["world_size"],
            timeout=timeout,
        )


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
    if CURRENT_DEVICE_TYPE_STR in os.environ:
        os.environ.pop(CURRENT_DEVICE_TYPE_STR)


def initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if initialized() else 1


def is_master() -> bool:
    return get_rank() == 0


def synchronize() -> None:
    if get_world_size() == 1:
        return
    dist.barrier()


def broadcast(
    tensor: torch.Tensor, src, group=dist.group.WORLD, async_op: bool = False
) -> None:
    if get_world_size() == 1:
        return
    dist.broadcast(tensor, src, group, async_op)


def broadcast_object_list(
    object_list: list[Any], src: int, group=dist.group.WORLD, device: str | None = None
) -> None:
    if get_world_size() == 1:
        return
    dist.broadcast_object_list(object_list, src, group, device)


def all_reduce(
    data, group=dist.group.WORLD, average: bool = False, device=None
) -> torch.Tensor:
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.to(device)
    dist.all_reduce(tensor, group=group)
    if average:
        tensor /= get_world_size()
    if not isinstance(data, torch.Tensor):
        result = tensor.cpu().numpy() if tensor.numel() > 1 else tensor.item()
    else:
        result = tensor
    return result


def all_gather(data, group=dist.group.WORLD, device=None) -> list[torch.Tensor]:
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.to(device)
    tensor_list = [tensor.new_zeros(tensor.shape) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor, group=group)
    if not isinstance(data, torch.Tensor):
        result = [tensor.cpu().numpy() for tensor in tensor_list]
    else:
        result = tensor_list
    return result


def gather_objects(data: T, group: dist.ProcessGroup = dist.group.WORLD) -> list[T]:
    """Gather a list of pickleable objects into rank 0"""
    if get_world_size() == 1:
        return [data]

    output = [None for _ in range(get_world_size())] if is_master() else None
    dist.gather_object(data, output, group=group, dst=0)
    return output


def assign_device_for_local_rank(cpu: bool, local_rank: int) -> None:
    if cpu:
        os.environ[CURRENT_DEVICE_TYPE_STR] = "cpu"
    else:
        assert torch.cuda.is_available(), "cannot set cpu=false and no cuda available!"
        os.environ[CURRENT_DEVICE_TYPE_STR] = "cuda"
        torch.cuda.set_device(local_rank)


def get_device_for_local_rank() -> str:
    if os.environ.get(CURRENT_DEVICE_TYPE_STR) is None:
        os.environ[CURRENT_DEVICE_TYPE_STR] = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        logging.warning(
            f"WARNING: assign_device_for_local_rank was never called, automatically defaulting to using {os.environ[CURRENT_DEVICE_TYPE_STR]}"
        )
        return os.environ[CURRENT_DEVICE_TYPE_STR]

    if "cuda" in os.environ[CURRENT_DEVICE_TYPE_STR]:
        assert torch.cuda.is_available(), "cannot set cpu=false and no cuda available!"
        return f"cuda:{torch.cuda.current_device()}"
    elif os.environ[CURRENT_DEVICE_TYPE_STR] == "cpu":
        return "cpu"
    else:
        raise ValueError(f"unsupported device type: {CURRENT_DEVICE_TYPE_STR}")


def setup_env_local():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_PORT"] = str(get_free_port())
