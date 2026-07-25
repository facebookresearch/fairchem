"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer


def build_zero_redundancy_optimizer(
    optimizer_fn: partial,
    params: Any,
) -> ZeroRedundancyOptimizer:
    """
    Build a ZeRO optimizer from FAIRChem's partial optimizer factory.

    Args:
        optimizer_fn: Partial optimizer constructor from the Hydra config.
        params: Parameters or parameter groups to optimize.

    Returns:
        A ZeRO stage-one optimizer wrapping the configured optimizer.
    """
    if not isinstance(optimizer_fn, partial):
        raise TypeError("ZeRO requires optimizer_fn to be a functools.partial")
    if optimizer_fn.args:
        raise ValueError("ZeRO does not support positional optimizer arguments")
    return ZeroRedundancyOptimizer(
        params,
        optimizer_class=optimizer_fn.func,
        **optimizer_fn.keywords,
    )


class ShardedZeroOptimizerState:
    """
    Translate rank-local ZeRO state to a world-size-independent DCP layout.

    Optimizer tensors are keyed by model parameter name. Each rank exposes only
    the states owned by its local optimizer, allowing DCP to save and restore
    them without consolidating the complete optimizer on any rank.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: ZeroRedundancyOptimizer,
    ) -> None:
        self.model = model
        self.optimizer = optimizer

    def _parameter_names(self) -> dict[torch.Tensor, str]:
        names = {}
        for name, parameter in self.model.named_parameters():
            names[parameter] = name.removeprefix("module.")
        return names

    def _global_param_groups(self) -> list[dict[str, Any]]:
        names = self._parameter_names()
        groups = []
        for param_group in self.optimizer.param_groups:
            group = {
                key: value for key, value in param_group.items() if key != "params"
            }
            try:
                group["params"] = [names[param] for param in param_group["params"]]
            except KeyError as error:
                raise ValueError(
                    "ZeRO optimizer contains a parameter not found in the model"
                ) from error
            groups.append(group)
        return groups

    def state_dict(self) -> dict[str, Any]:
        """
        Return only this rank's optimizer state using global parameter names.
        """
        names = self._parameter_names()
        local_state = {}
        for parameter, state in self.optimizer.optim.state.items():
            try:
                local_state[names[parameter]] = state
            except KeyError as error:
                raise ValueError(
                    "Local ZeRO optimizer contains a parameter not found in the model"
                ) from error
        return {
            "state": local_state,
            "param_groups": self._global_param_groups(),
        }

    @torch.no_grad()
    def initialize_local_state(self) -> None:
        """
        Materialize local optimizer tensors used as DCP load placeholders.
        """
        local_optimizer = self.optimizer.optim
        if local_optimizer.state:
            return
        if any(
            parameter.grad is not None
            for group in local_optimizer.param_groups
            for parameter in group["params"]
        ):
            raise RuntimeError("Cannot initialize ZeRO state with existing gradients")

        learning_rates = []
        for group in local_optimizer.param_groups:
            learning_rates.append(group.get("lr"))
            if "lr" in group:
                group["lr"] = 0.0
            for parameter in group["params"]:
                if parameter.requires_grad:
                    parameter.grad = torch.zeros_like(parameter)

        local_optimizer.step()
        local_optimizer.zero_grad(set_to_none=True)
        for group, learning_rate in zip(local_optimizer.param_groups, learning_rates):
            if learning_rate is not None:
                group["lr"] = learning_rate
        for outer_group, local_group in zip(
            self.optimizer.param_groups, local_optimizer.param_groups
        ):
            for key, value in local_group.items():
                if key != "params":
                    outer_group[key] = value

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load this rank's optimizer shard from the name-keyed DCP state.

        Args:
            state_dict: Rank-local optimizer state populated by DCP.
        """
        saved_groups = state_dict["param_groups"]
        current_groups = self._global_param_groups()
        if len(saved_groups) != len(current_groups):
            raise ValueError("ZeRO checkpoint parameter-group count changed")
        for saved_group, current_group in zip(saved_groups, current_groups):
            if saved_group["params"] != current_group["params"]:
                raise ValueError("ZeRO checkpoint parameter-group membership changed")

        names = self._parameter_names()
        local_optimizer = self.optimizer.optim
        local_state_dict = local_optimizer.state_dict()
        name_to_local_id = {}
        for state_group, parameter_group in zip(
            local_state_dict["param_groups"], local_optimizer.param_groups
        ):
            for parameter_id, parameter in zip(
                state_group["params"], parameter_group["params"]
            ):
                name_to_local_id[names[parameter]] = parameter_id

        local_state_dict["state"] = {
            name_to_local_id[name]: state for name, state in state_dict["state"].items()
        }
        for local_group, saved_group in zip(
            local_state_dict["param_groups"], saved_groups
        ):
            for key, value in saved_group.items():
                if key != "params":
                    local_group[key] = value

        local_optimizer.load_state_dict(local_state_dict)
        for outer_group, saved_group in zip(self.optimizer.param_groups, saved_groups):
            for key, value in saved_group.items():
                if key != "params":
                    outer_group[key] = value
