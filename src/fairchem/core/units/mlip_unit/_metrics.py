"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Hashable

NONE_SLICE = slice(None)


@dataclass
class Metrics:
    metric: float = 0.0
    total: float = 0.0
    numel: int = 0

    def __iadd__(self, other):
        self.total += other.total
        self.numel += other.numel
        self.metric = self.total / self.numel
        return self


def metrics_dict(metric_fun: Callable) -> Callable:
    """Wrap up the return of a metrics function"""

    @wraps(metric_fun)
    def wrapped_metrics(
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        key: Hashable = None,
        **kwargs,
    ) -> Metrics:
        error = metric_fun(prediction, target, key, **kwargs)
        return Metrics(
            metric=torch.mean(error).item(),
            total=torch.sum(error).item(),
            numel=error.numel(),
        )

    return wrapped_metrics


@metrics_dict
def cosine_similarity(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
):
    # cast to float 32 to avoid 0/nan issues in fp16
    # https://github.com/pytorch/pytorch/issues/69512
    return torch.cosine_similarity(prediction[key].float(), target[key].float())


@metrics_dict
def mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:
    return torch.abs(target[key] - prediction[key])


@metrics_dict
def mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:
    return (target[key] - prediction[key]) ** 2


@metrics_dict
def rmse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> torch.Tensor:
    return torch.sqrt(((target[key] - prediction[key]) ** 2).sum(dim=-1))


@metrics_dict
def per_atom_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:
    return torch.abs(target[key] - prediction[key]) / target["natoms"].unsqueeze(1)


@metrics_dict
def per_atom_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:
    return ((target[key] - prediction[key]) / target["natoms"].unsqueeze(1)) ** 2


@metrics_dict
def magnitude_error(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
    p: int = 2,
) -> torch.Tensor:
    assert prediction[key].shape[1] > 1
    return torch.abs(
        torch.norm(prediction[key], p=p, dim=-1) - torch.norm(target[key], p=p, dim=-1)
    )


def forcesx_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> Metrics:
    return mae(prediction["forces"][:, 0], target["forces"][:, 0])


def forcesx_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> Metrics:
    return mse(prediction["forces"][:, 0], target["forces"][:, 0])


def forcesy_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> Metrics:
    return mae(prediction["forces"][:, 1], target["forces"][:, 1])


def forcesy_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> Metrics:
    return mse(prediction["forces"][:, 1], target["forces"][:, 1])


def forcesz_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> Metrics:
    return mae(prediction["forces"][:, 2], target["forces"][:, 2])


def forcesz_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> Metrics:
    return mse(prediction["forces"][:, 2], target["forces"][:, 2])


def energy_forces_within_threshold(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> Metrics:
    # Note that this natoms should be the count of free atoms we evaluate over.
    assert target["natoms"].sum() == prediction["forces"].size(0)
    assert target["natoms"].size(0) == prediction["energy"].size(0)

    # compute absolute error on per-atom forces and energy per system.
    # then count the no. of systems where max force error is < 0.03 and max
    # energy error is < 0.02.
    f_thresh = 0.03
    e_thresh = 0.02

    success = 0
    total = int(target["natoms"].size(0))

    error_forces = torch.abs(target["forces"] - prediction["forces"])
    error_energy = torch.abs(target["energy"] - prediction["energy"])

    start_idx = 0
    for i, n in enumerate(target["natoms"]):
        if (
            error_energy[i] < e_thresh
            and error_forces[start_idx : start_idx + n].max() < f_thresh
        ):
            success += 1
        start_idx += n

    return Metrics(metric=success / total, total=success, numel=total)


def energy_within_threshold(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> Metrics:
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    # threshold is set based on OC20 leaderboard
    e_thresh = 0.02
    error_energy = torch.abs(target["energy"] - prediction["energy"])

    success = (error_energy < e_thresh).sum().item()
    total = target["energy"].size(0)

    return Metrics(metric=success / total, total=success, numel=total)


def average_distance_within_threshold(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = None,
) -> Metrics:
    pred_pos = torch.split(prediction["positions"], prediction["natoms"].tolist())
    target_pos = torch.split(target["positions"], target["natoms"].tolist())

    mean_distance = []
    for idx, ml_pos in enumerate(pred_pos):
        mean_distance.append(
            np.mean(
                np.linalg.norm(
                    min_diff(
                        ml_pos.detach().cpu().numpy(),
                        target_pos[idx].detach().cpu().numpy(),
                        target["cell"][idx].detach().cpu().numpy(),
                        target["pbc"].tolist(),
                    ),
                    axis=1,
                )
            )
        )

    success = 0
    intv = np.arange(0.01, 0.5, 0.001)
    for i in intv:
        success += sum(np.array(mean_distance) < i)

    total = len(mean_distance) * len(intv)

    return Metrics(metric=success / total, total=success, numel=total)


def min_diff(
    pred_pos: torch.Tensor,
    dft_pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
):
    """
    Calculate the minimum difference between predicted and target positions considering periodic boundary conditions.
    """
    pos_diff = pred_pos - dft_pos
    fractional = np.linalg.solve(cell.T, pos_diff.T).T

    for i, periodic in enumerate(pbc):
        # Yes, we need to do it twice
        if periodic:
            fractional[:, i] %= 1.0
            fractional[:, i] %= 1.0

    fractional[fractional > 0.5] -= 1

    return np.matmul(fractional, cell)


def get_metrics_fn(function_name: str) -> Callable:
    contents = globals()
    if function_name.startswith("_") or function_name not in contents:
        raise ValueError(f"Unknown metric function name {function_name}")
    return contents[function_name]
