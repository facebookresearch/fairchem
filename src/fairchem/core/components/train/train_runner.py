"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Protocol, Union, runtime_checkable

from torchtnt.framework.fit import fit

from fairchem.core.common.utils import get_subdirectories_sorted_by_time
from fairchem.core.components.runner import Runner
from fairchem.core.components.callbacks import (
    StopBeforeTimeoutCallback,
    TrainCheckpointCallback,
)

if TYPE_CHECKING:
    import torch
    from torchtnt.framework import EvalUnit, TrainUnit
    from torchtnt.framework.callback import Callback


@runtime_checkable
class Checkpointable(Protocol):
    """
    Protocol that Units used by this trainer should implement if they want save and resume functionality
    This is in addition to Pytorch's Stateful protocol because it allows units implement custom logic
    that's required for checkpointing
    """

    def save_state(self, checkpoint_location: str) -> None:
        """
        Save the unit state to a checkpoint path

        Args:
            checkpoint_location: The checkpoint path to save to
        """

        ...

    def load_state(self, checkpoint_location: str | None) -> None:
        """
        Loads the state given a checkpoint path

        Args:
            checkpoint_location: The checkpoint path to restore from
        """

        ...


def get_most_recent_viable_checkpoint_path(checkpoint_dir: str | None) -> str | None:
    if not checkpoint_dir:
        return None

    ckpt_dirs_time = get_subdirectories_sorted_by_time(checkpoint_dir)
    most_recent_viable_checkpoint = None
    for sub_dir_path, _ in ckpt_dirs_time[::-1]:
        items = os.listdir(sub_dir_path)
        if items and ".metadata" in items:
            most_recent_viable_checkpoint = sub_dir_path
            break
    return most_recent_viable_checkpoint


class TrainEvalRunner(Runner):
    def __init__(
        self,
        train_dataloader: torch.utils.data.dataloader,
        eval_dataloader: torch.utils.data.dataloader,
        train_eval_unit: Union[TrainUnit, EvalUnit, Checkpointable],
        callbacks: list[Callback] | None = None,
        max_epochs: int | None = 1,
        evaluate_every_n_steps: Optional[int] = None,
        max_steps: int | None = None,
        max_eval_steps_per_epoch: int | None = None,
        stop_before_timeout_min: float | None = None,
    ):
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_eval_unit = train_eval_unit
        self.callbacks = callbacks if callbacks is not None else []
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.max_eval_steps_per_epoch = max_eval_steps_per_epoch
        self.stop_before_timeout_min = stop_before_timeout_min

        checkpoint_callbacks = [
            c for c in self.callbacks if isinstance(c, TrainCheckpointCallback)
        ]
        assert len(checkpoint_callbacks) <= 1
        self.checkpoint_callback = (
            checkpoint_callbacks[0] if len(checkpoint_callbacks) == 1 else None
        )
        logging.info(f"Train Dataloader size {len(self.train_dataloader)}")
        logging.info(f"Eval Dataloader size {len(self.eval_dataloader)}")

    def run(self) -> None:
        self._set_runner_callbacks(self.callbacks)

        callbacks = self.callbacks
        if self.stop_before_timeout_min is not None:
            callbacks = [
                *callbacks,
                StopBeforeTimeoutCallback(
                    timeout_hr=self._get_slurm_timeout_hr(),
                    stop_before_timeout_min=self.stop_before_timeout_min,
                ),
            ]

        fit(
            self.train_eval_unit,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            max_eval_steps_per_epoch=self.max_eval_steps_per_epoch,
            callbacks=callbacks,
            evaluate_every_n_steps=self.evaluate_every_n_steps,
        )

    def _set_runner_callbacks(self, callbacks: list[Callback]) -> None:
        for callback in callbacks:
            set_callbacks = getattr(callback, "set_runner_callbacks", None)
            if callable(set_callbacks):
                set_callbacks(
                    self.save_state,
                    self.load_state,
                    self.job_config.metadata.checkpoint_dir,
                )

    def _get_slurm_timeout_hr(self) -> float:
        try:
            return float(self.job_config.scheduler.slurm.timeout_hr)
        except AttributeError as e:
            raise ValueError(
                "runner.stop_before_timeout_min requires "
                "job.scheduler.slurm.timeout_hr"
            ) from e

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        # in the case of preemption, don't attempt to save a new checkpoint but try to move an existing to the checkpoint_location
        # this is because submitit's preemption routine only calls checkpoint on master and dcp will deadlock if its not called on all ranks
        if is_preemption:
            most_recent_checkpoint_path = get_most_recent_viable_checkpoint_path(
                self.job_config.metadata.checkpoint_dir
            )
            if most_recent_checkpoint_path:
                if os.path.lexists(checkpoint_location):
                    logging.warning(
                        f"Checkpoint location {checkpoint_location} already exists, removing it"
                    )
                    os.remove(checkpoint_location)
                os.symlink(most_recent_checkpoint_path, checkpoint_location)
                logging.info(
                    f"When the job resumes from preemption, it will be using the state found at {most_recent_checkpoint_path}, which has been symlinked to {checkpoint_location}"
                )
                return True
            else:
                logging.info(
                    "Did not find a viable checkpoint, no preemption checkpoint is available"
                )
                return False

        self.train_eval_unit.save_state(checkpoint_location)
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        # if checkpoint_location is given, load that, otherwise attempt to load from latest checkpoint
        if checkpoint_location:
            checkpoint_to_load = checkpoint_location
        else:
            # we could be here because of a node failure where a checkpoint exists but the preemption code was never triggered
            # in this case we attempt to find the last known checkpoint
            # NOTE we must do this otherwise an automatically requeue by the cluster could restart this job from step 0
            most_recent_checkpoint_path = get_most_recent_viable_checkpoint_path(
                self.job_config.metadata.checkpoint_dir
            )
            if most_recent_checkpoint_path:
                logging.info(
                    f"Last existing checkpoints found at {most_recent_checkpoint_path}, starting from here"
                )
                checkpoint_to_load = most_recent_checkpoint_path
            else:
                logging.info("No existing checkpoints found, starting from scratch")
                return

        self.train_eval_unit.load_state(checkpoint_to_load)
