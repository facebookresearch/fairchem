"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from functools import partial

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)
from torch.nn.parallel import DistributedDataParallel

from fairchem.core.common.test_utils import (
    PGConfig,
    init_pg_and_rank_and_launch_test,
    spawn_multi_process,
)
from fairchem.core.units.mlip_unit.zero_redundancy import (
    ShardedZeroOptimizerState,
    build_zero_redundancy_optimizer,
)


def _make_model_and_optimizer(optimizer_name: str):
    torch.manual_seed(7)
    model = DistributedDataParallel(
        torch.nn.Sequential(
            torch.nn.Linear(4, 6),
            torch.nn.Tanh(),
            torch.nn.Linear(6, 2),
        )
    )
    if optimizer_name == "soap":
        from pytorch_optimizer import SOAP

        optimizer_class = SOAP
        optimizer_kwargs = {"precondition_frequency": 2}
    else:
        optimizer_class = torch.optim.AdamW
        optimizer_kwargs = {}
    parameters = list(model.parameters())
    optimizer = build_zero_redundancy_optimizer(
        partial(
            optimizer_class,
            lr=1e-2,
            weight_decay=1e-3,
            **optimizer_kwargs,
        ),
        [
            {"params": parameters[:2], "weight_decay": 0.0},
            {"params": parameters[2:], "weight_decay": 1e-3},
        ],
    )
    return model, optimizer


def _take_step(model, optimizer, step: int) -> None:
    inputs = torch.arange(20, dtype=torch.float32).reshape(5, 4) / (step + 2)
    targets = torch.arange(10, dtype=torch.float32).reshape(5, 2) / (step + 3)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()


def _flat_parameters(model) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


def _save_and_continue(checkpoint_path: str, optimizer_name: str):
    model, optimizer = _make_model_and_optimizer(optimizer_name)
    for step in range(3):
        _take_step(model, optimizer, step)

    sharded_state = ShardedZeroOptimizerState(model, optimizer)
    local_state = sharded_state.state_dict()
    dcp.save(
        {
            "model": get_model_state_dict(model),
            "optimizer": local_state,
        },
        checkpoint_id=checkpoint_path,
    )

    state_before_step = copy.deepcopy(local_state)
    _take_step(model, optimizer, 3)
    return sorted(local_state["state"]), _flat_parameters(model), state_before_step


def _load_and_continue(checkpoint_path: str, optimizer_name: str):
    model, optimizer = _make_model_and_optimizer(optimizer_name)
    sharded_state = ShardedZeroOptimizerState(model, optimizer)
    sharded_state.initialize_local_state()
    state = {
        "model": get_model_state_dict(model),
        "optimizer": sharded_state.state_dict(),
    }
    dcp.load(state, checkpoint_id=checkpoint_path)
    set_model_state_dict(model, model_state_dict=state["model"])
    sharded_state.load_state_dict(state["optimizer"])

    state_before_step = copy.deepcopy(sharded_state.state_dict())
    _take_step(model, optimizer, 3)
    return (
        sorted(state["optimizer"]["state"]),
        _flat_parameters(model),
        state_before_step,
    )


def _spawn(world_size: int, method, checkpoint_path: str, optimizer_name: str):
    return spawn_multi_process(
        PGConfig(backend="gloo", world_size=world_size, use_gp=False),
        method,
        init_pg_and_rank_and_launch_test,
        checkpoint_path,
        optimizer_name,
    )


def _compare_adamw_and_zero_updates(_checkpoint_path: str, _optimizer_name: str):
    torch.manual_seed(11)
    plain_model = DistributedDataParallel(torch.nn.Linear(4, 3))
    zero_model = DistributedDataParallel(copy.deepcopy(plain_model.module))
    plain_optimizer = torch.optim.AdamW(
        plain_model.parameters(), lr=0.1, weight_decay=1e-3
    )
    zero_optimizer = build_zero_redundancy_optimizer(
        partial(torch.optim.AdamW, lr=0.1, weight_decay=1e-3),
        zero_model.parameters(),
    )
    plain_scheduler = torch.optim.lr_scheduler.LambdaLR(
        plain_optimizer, lambda step: (step + 1) / 5
    )
    zero_scheduler = torch.optim.lr_scheduler.LambdaLR(
        zero_optimizer, lambda step: (step + 1) / 5
    )

    for step in range(4):
        inputs = torch.arange(20, dtype=torch.float32).reshape(5, 4) / (step + 1)
        for model, optimizer, scheduler in (
            (plain_model, plain_optimizer, plain_scheduler),
            (zero_model, zero_optimizer, zero_scheduler),
        ):
            optimizer.zero_grad(set_to_none=True)
            model(inputs).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        torch.testing.assert_close(
            _flat_parameters(zero_model),
            _flat_parameters(plain_model),
            rtol=0,
            atol=0,
        )


def test_zero_matches_adamw_updates():
    _spawn(2, _compare_adamw_and_zero_updates, "", "")


def test_sharded_zero_checkpoint_repartitions_across_world_sizes(tmp_path):
    checkpoint_path = str(tmp_path / "zero_checkpoint")

    saved = _spawn(2, _save_and_continue, checkpoint_path, "adamw")
    resumed_one = _spawn(1, _load_and_continue, checkpoint_path, "adamw")
    resumed_three = _spawn(3, _load_and_continue, checkpoint_path, "adamw")

    saved_state_names = [set(names) for names, _, _ in saved]
    assert saved_state_names[0].isdisjoint(saved_state_names[1])
    assert len(saved_state_names[0] | saved_state_names[1]) == 4

    expected_parameters = saved[0][1]
    for _, parameters, _ in saved[1:] + resumed_one + resumed_three:
        torch.testing.assert_close(parameters, expected_parameters)

    assert len(resumed_one[0][0]) == 4
    assert sum(len(names) for names, _, _ in resumed_three) == 4


def test_sharded_zero_checkpoint_supports_soap(tmp_path):
    pytest.importorskip("pytorch_optimizer")
    checkpoint_path = str(tmp_path / "zero_soap_checkpoint")

    saved = _spawn(2, _save_and_continue, checkpoint_path, "soap")
    resumed = _spawn(1, _load_and_continue, checkpoint_path, "soap")

    saved_optimizer_state = {}
    for _, _, rank_state in saved:
        saved_optimizer_state.update(rank_state["state"])
    torch.testing.assert_close(
        resumed[0][2]["state"],
        saved_optimizer_state,
    )
    assert resumed[0][2]["param_groups"] == saved[0][2]["param_groups"]
    torch.testing.assert_close(resumed[0][1], saved[0][1])
