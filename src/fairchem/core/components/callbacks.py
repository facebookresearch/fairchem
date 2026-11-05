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
from fairchem.core.common.utils import get_subdirectories_sorted_by_step

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
            and step % self.checkpoint_every_n_steps == 0
        ):
            checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}")
            save_callback(checkpoint_path)
            if distutils.is_master():
                self._rotate(checkpoint_dir, checkpoint_path)

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        if self.checkpoint_every_n_steps is not None:
            save_callback, checkpoint_dir = self._checkpoint_hooks()
            save_callback(os.path.join(checkpoint_dir, "final"))

    def _rotate(self, checkpoint_dir: str, checkpoint_path: str) -> None:
        """
        Delete all but the newest ``max_saved_checkpoints`` step checkpoints.

        Checkpoints are ranked by step, not modification time. A ``final``
        left over from before this save is deleted too: the run was extended
        past it, and ``get_most_recent_viable_checkpoint_path`` would otherwise
        still pick it. Symlinks and other directories are never deleted.

        Args:
            checkpoint_dir: The directory holding the checkpoint directories.
            checkpoint_path: The checkpoint just saved, which is always kept.
        """
        steps = get_subdirectories_sorted_by_step(checkpoint_dir)
        stale = [path for path, _ in steps[: -self.max_saved_checkpoints]]
        stale.append(os.path.join(checkpoint_dir, "final"))
        for path in stale:
            is_stale_dir = os.path.isdir(path) and not os.path.islink(path)
            if path != checkpoint_path and is_stale_dir:
                shutil.rmtree(path)

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
