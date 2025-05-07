"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from torchtnt.framework.callback import Callback
from torchtnt.framework.fit import fit

from fairchem.core.common import distutils
from fairchem.core.common.utils import get_subdirectories_sorted_by_time
from fairchem.core.components.runner import Runner
from fairchem.core.units.mlip_unit.mlip_unit import (
    convert_train_checkpoint_to_inference_checkpoint,
)

if TYPE_CHECKING:
    from torch.distributed.checkpoint.stateful import Stateful
    from torchtnt.framework import EvalUnit, TrainUnit
    from torchtnt.framework.state import State
    from torchtnt.framework.unit import TTrainUnit


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


class TrainCheckpointCallback(Callback):
    def __init__(
        self,
        checkpoint_every_n_steps: int,
        max_saved_checkpoints: int = 2,
    ):
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.max_saved_checkpoints = max_saved_checkpoints
        self.save_callback = None
        self.load_callback = None
        self.checkpoint_dir = None

    def set_runner_callbacks(
        self, save_callback: callable, load_callback: callable, checkpoint_dir: str
    ) -> None:
        self.save_callback = save_callback
        self.load_callback = load_callback
        self.checkpoint_dir = checkpoint_dir

    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        # We try to save the checkpoint on_train_step_start instead of at the on_train_step_end because both the step and epoch counts are consistently updated before it gets here
        # if we did this at on_train_step_end, the step would be correct but the epoch would have not been incremented and break the edge case on the last step of an epoch
        assert (
            self.save_callback
        ), "Must initialize set_checkpoint_call_backs from Runner!"
        step = unit.train_progress.num_steps_completed
        if (
            self.checkpoint_every_n_steps is not None
            and step % self.checkpoint_every_n_steps == 0
        ):
            self.save_callback(os.path.join(self.checkpoint_dir, f"step_{step}"))
            # on main rank only
            # if there are too many checkpoints, delete the oldest one
            if distutils.is_master():
                checkpoint_dirs_by_time = get_subdirectories_sorted_by_time(
                    self.checkpoint_dir
                )
                for dir, _ in checkpoint_dirs_by_time[: -self.max_saved_checkpoints]:
                    shutil.rmtree(dir)

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        if self.checkpoint_every_n_steps is not None:
            # also always checkpoint on train end
            assert (
                self.save_callback
            ), "Must initialize set_checkpoint_call_backs from Runner!"
            self.save_callback(os.path.join(self.checkpoint_dir, "final"))


class TrainEvalRunner(Runner):
    def __init__(
        self,
        train_dataloader: torch.utils.data.dataloader,
        eval_dataloader: torch.utils.data.dataloader,
        train_eval_unit: Union[TrainUnit, EvalUnit, Stateful],
        callbacks: list[Callback] | None = None,
        max_epochs: int | None = 1,
        evaluate_every_n_steps: Optional[int] = None,
        max_steps: int | None = None,
        save_inference_ckpt: bool = True,
    ):
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_eval_unit = train_eval_unit
        self.callbacks = callbacks if callbacks is not None else []
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.save_inference_ckpt = save_inference_ckpt

        checkpoint_callbacks = [
            c for c in callbacks if isinstance(c, TrainCheckpointCallback)
        ]
        assert len(checkpoint_callbacks) <= 1
        self.checkpoint_callback = (
            checkpoint_callbacks[0] if len(checkpoint_callbacks) == 1 else None
        )
        logging.info(f"Train Dataloader size {len(self.train_dataloader)}")
        logging.info(f"Eval Dataloader size {len(self.eval_dataloader)}")

    def run(self) -> None:
        if self.checkpoint_callback is not None:
            self.checkpoint_callback.set_runner_callbacks(
                self.save_state,
                self.load_state,
                self.job_config.metadata.checkpoint_dir,
            )

        fit(
            self.train_eval_unit,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            callbacks=self.callbacks,
            evaluate_every_n_steps=self.evaluate_every_n_steps,
        )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        # in the case of preemption, don't attempt to save a new checkpoint but try to move an existing to the checkpoint_location
        # this is because submitit's preemption routine only calls checkpoint on master and dcp will deadlock if its not called on all ranks
        if is_preemption:
            most_recent_checkpoint_path = get_most_recent_viable_checkpoint_path(
                self.job_config.metadata.checkpoint_dir
            )
            if most_recent_checkpoint_path:
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

        # save a "train_state.yaml" that can be easily used for resuming runs
        os.makedirs(checkpoint_location, exist_ok=True)
        config = OmegaConf.load(self.job_config.metadata.config_path)
        config.job.runner_state_path = checkpoint_location
        # TODO: TrainEvalRunner shouldn't know about this and we need to potentially create
        # a protocol that includes this kind of information. the reason these patches exist
        # is because the model backbone config of the finetuning model is stored in the
        # starting checkpoint and we need to decouple it from the finetuning checkpoints,
        # such that the finetuning checkpoint can be used on its own without relying on knowing the
        # path to the starting checkpoint.
        finetune_model_full_config = self.train_eval_unit.get_finetune_model_config()
        if finetune_model_full_config is not None:
            config.runner.train_eval_unit.model = finetune_model_full_config

        OmegaConf.save(config, os.path.join(checkpoint_location, "train_state.yaml"))

        # calls train_eval_unit.save_state
        state = {"unit_state": self.train_eval_unit.state_dict(), "config": config}
        dcp.save(state, checkpoint_id=checkpoint_location)

        # warning this can be VERY SLOW for large models, better not to do this at every checkpoint
        if (
            self.save_inference_ckpt
            and distutils.is_master()
            and os.path.exists(checkpoint_location)
        ):
            # TODO: create a protocol for this function, Runner are not suppose to know about functions of train_eval_unit
            convert_train_checkpoint_to_inference_checkpoint(
                checkpoint_location,
                os.path.join(checkpoint_location, "inference_ckpt.pt"),
            )

        logging.info(f"Saved dcp checkpoint to {checkpoint_location}")
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

        # HACK: we need to do this to trigger the loading of the scheduler before the loading the checkpoint
        # find a more elegant way to doing this, this currently breaks the train/eval interface
        self.train_eval_unit.load_scheduler(len(self.train_dataloader))

        state = {"unit_state": self.train_eval_unit.state_dict()}
        dcp.load(state_dict=state, checkpoint_id=checkpoint_to_load)
        self.train_eval_unit.load_state_dict(state["unit_state"])
        logging.info(f"Done loading checkpoint from {checkpoint_to_load}")
