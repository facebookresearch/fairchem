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

    from torchtnt.framework import EvalUnit
    from torchtnt.framework.state import State
    from torchtnt.framework.unit import TTrainUnit


class TrainCheckpointCallback(Callback):
    def __init__(
        self,
        checkpoint_every_n_steps: int | None,
        max_saved_checkpoints: int = 2,
        monitor: str | None = None,
        mode: str = "min",
        save_top_k: int = 1,
    ):
        """
        Callback to save checkpoints.
        Can save periodically based on train steps AND/OR save the best K models based on a monitored metric.

        Args:
            checkpoint_every_n_steps: Save a checkpoint every N train steps. Set to None to disable periodic saving.
            max_saved_checkpoints: Max number of periodic checkpoints to keep.
            monitor: The name of the validation metric to monitor (e.g., "val/loss"). Set to None to disable best-model saving.
            mode: "min" or "max". Whether to save the minimum or maximum value of the monitored metric.
            save_top_k: The number of best models to save.
        """
        if checkpoint_every_n_steps is None and monitor is None:
            logging.warning(
                "TrainCheckpointCallback: Neither 'checkpoint_every_n_steps' nor 'monitor' was specified. "
                "No checkpoints will be saved during training."
            )
        if monitor is not None and mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'.")

        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.max_saved_checkpoints = max_saved_checkpoints
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_metrics: list[tuple[float, str]] = []
        self.save_callback = None
        self.load_callback = None
        self.checkpoint_dir = None
        self._current_best_val = float("inf") if self.mode == "min" else -float("inf")

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
        if self.checkpoint_every_n_steps is None:
            return

        # Step and epoch counts are both current at train-step start.
        save_callback, checkpoint_dir = self._checkpoint_hooks()
        step = unit.train_progress.num_steps_completed
        if step > 0 and step % self.checkpoint_every_n_steps == 0:
            save_callback(os.path.join(checkpoint_dir, f"step_{step}"))
            if distutils.is_master():
                checkpoint_dirs_by_time = get_subdirectories_sorted_by_time(
                    checkpoint_dir
                )
                for dir, _ in checkpoint_dirs_by_time[: -self.max_saved_checkpoints]:
                    if not os.path.islink(dir):
                        # Skip best_model checkpoints - they are managed separately
                        if not os.path.basename(dir).startswith("best_model_"):
                            shutil.rmtree(dir)

    def on_eval_end(self, state: State, unit: EvalUnit) -> None:
        if self.monitor is None:
            return

        save_callback, checkpoint_dir = self._checkpoint_hooks()

        # Get metrics from unit (stored during on_eval_epoch_end)
        metrics = getattr(unit, "last_eval_metrics", None)
        if metrics is None:
            logging.debug("No metrics found on unit. Skipping best model check.")
            return

        current_metric = metrics.get(self.monitor)
        if current_metric is None:
            logging.debug(
                "Metric '%s' not found in unit metrics. Skipping best model check.",
                self.monitor,
            )
            return

        try:
            current_metric = float(current_metric)
        except (ValueError, TypeError):
            logging.warning(
                "Metric '%s' has non-numeric value '%s'. Skipping best model check.",
                self.monitor,
                current_metric,
            )
            return

        is_better = (
            len(self.best_metrics) < self.save_top_k
            or (self.mode == "min" and current_metric < self._current_best_val)
            or (self.mode == "max" and current_metric > self._current_best_val)
        )

        if is_better:
            self._current_best_val = current_metric
            step = unit.train_progress.num_steps_completed
            new_best_path = os.path.join(
                checkpoint_dir,
                f"best_model_step_{step}_metric_{current_metric:.4f}",
            )

            logging.info(
                "New best model found! Metric %s: %.4f. Saving to %s",
                self.monitor,
                current_metric,
                new_best_path,
            )
            save_callback(new_best_path)

            self.best_metrics.append((current_metric, new_best_path))
            sort_reverse = self.mode == "max"
            self.best_metrics.sort(key=lambda x: x[0], reverse=sort_reverse)

            if len(self.best_metrics) > self.save_top_k:
                _, path_to_remove = self.best_metrics.pop()
                if distutils.is_master() and os.path.exists(path_to_remove):
                    logging.info("Removing old best checkpoint: %s", path_to_remove)
                    try:
                        shutil.rmtree(path_to_remove)
                    except OSError as e:
                        logging.warning(
                            "Failed to remove old checkpoint %s: %s",
                            path_to_remove,
                            e,
                        )

            self._current_best_val = self.best_metrics[-1][0]

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
