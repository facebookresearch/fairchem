"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from typing import TYPE_CHECKING

from torchtnt.framework.callback import Callback

from fairchem.core.common import distutils
from fairchem.core.common.utils import get_subdirectories_sorted_by_time

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchtnt.framework.state import State
    from torchtnt.framework.unit import TTrainUnit


class TrainCheckpointCallback(Callback):
    def __init__(
        self,
        checkpoint_every_n_steps: int | None,
        max_saved_checkpoints: int = 2,
    ):
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.max_saved_checkpoints = max_saved_checkpoints
        self.save_callback = None
        self.load_callback = None
        self.checkpoint_dir = None

    def set_runner_callbacks(
        self,
        save_callback: Callable[[str], None],
        load_callback: Callable[[str | None], None],
        checkpoint_dir: str,
    ) -> None:
        self.save_callback = save_callback
        self.load_callback = load_callback
        self.checkpoint_dir = checkpoint_dir

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        # Step and epoch counts are both current at train-step start.
        save_callback, checkpoint_dir = self._checkpoint_hooks()
        step = unit.train_progress.num_steps_completed
        if (
            self.checkpoint_every_n_steps is not None
            and step > 0
            and step % self.checkpoint_every_n_steps == 0
        ):
            save_callback(os.path.join(checkpoint_dir, f"step_{step}"))
            if distutils.is_master():
                checkpoint_dirs_by_time = get_subdirectories_sorted_by_time(
                    checkpoint_dir
                )
                for dir, _ in checkpoint_dirs_by_time[: -self.max_saved_checkpoints]:
                    if not os.path.islink(dir):
                        shutil.rmtree(dir)

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        if self.checkpoint_every_n_steps is not None:
            save_callback, checkpoint_dir = self._checkpoint_hooks()
            save_callback(os.path.join(checkpoint_dir, "final"))

    def _checkpoint_hooks(self) -> tuple[Callable[[str], None], str]:
        if self.save_callback is None or self.checkpoint_dir is None:
            raise RuntimeError("TrainCheckpointCallback was not initialized.")
        return self.save_callback, self.checkpoint_dir


class StopBeforeTimeoutCallback(Callback):
    def __init__(
        self,
        timeout_hr: float,
        stop_before_timeout_min: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if timeout_hr <= 0:
            raise ValueError("timeout_hr must be positive")
        if stop_before_timeout_min <= 0:
            raise ValueError("stop_before_timeout_min must be positive")
        if stop_before_timeout_min >= timeout_hr * 60:
            raise ValueError("stop_before_timeout_min must be smaller than timeout_hr")

        self.timeout_hr = timeout_hr
        self.stop_before_timeout_min = stop_before_timeout_min
        self.clock = clock
        self.deadline = clock() + timeout_hr * 3600 - stop_before_timeout_min * 60
        self._has_stopped = False

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        self._stop_if_deadline_passed(state, unit)

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        self._stop_if_deadline_passed(state, unit)

    def _stop_if_deadline_passed(self, state: State, unit: TTrainUnit) -> None:
        if self._has_stopped or self.clock() < self.deadline:
            return
        self._has_stopped = True
        if state.eval_state is not None:
            # TorchTNT may still launch scheduled in-loop eval after state.stop().
            state.eval_state._max_steps_per_epoch = 0
        logging.warning(
            "Stopping training before Slurm timeout: timeout_hr=%s, "
            "stop_before_timeout_min=%s, step=%s",
            self.timeout_hr,
            self.stop_before_timeout_min,
            unit.train_progress.num_steps_completed,
        )
        state.stop()
