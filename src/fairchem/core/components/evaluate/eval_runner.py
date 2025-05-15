"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtnt.framework.evaluate import evaluate

from fairchem.core.components.runner import Runner

if TYPE_CHECKING:
    import torch
    from torchtnt.framework import EvalUnit
    from torchtnt.framework.callback import Callback


class EvalRunner(Runner):
    def __init__(
        self,
        dataloader: torch.utils.data.dataloader,
        eval_unit: EvalUnit,
        callbacks: list[Callback] | None = None,
        max_steps_per_epoch: int | None = None,
    ):
        self.dataloader = dataloader
        self.eval_unit = eval_unit
        self.callbacks = callbacks if callbacks is not None else []
        self.max_steps_per_epoch = max_steps_per_epoch

    def run(self) -> None:
        evaluate(
            self.eval_unit,
            eval_dataloader=self.dataloader,
            max_steps_per_epoch=self.max_steps_per_epoch,
            callbacks=self.callbacks,
        )

    # during preemptions
    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        # need the unit to have a save_state protocol
        pass

    def load_state(self, checkpoint_location: str | None) -> None:
        # need the unit to have a load_state protocol
        pass
