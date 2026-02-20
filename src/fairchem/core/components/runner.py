"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABCMeta, abstractmethod
from typing import Any

from omegaconf import DictConfig

from fairchem.core.components.utils import ManagedAttribute


class Runner(metaclass=ABCMeta):
    """Represents an abstraction over things that run in a loop and can save/load state.

    ie: Trainers, Validators, Relaxation all fall in this category.

    Note:
        When running with the `fairchemv2` cli, the `job_config` and attribute is set at
        runtime to those given in the config file.

    Attributes:
        job_config (DictConfig): a managed attribute that gives access to the job config
    """

    job_config = ManagedAttribute(enforced_type=DictConfig)

    @abstractmethod
    def run(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_state(self, checkpoint_location: str | None) -> None:
        raise NotImplementedError


class MockRunner(Runner):
    """Used for testing"""

    def __init__(self, x: int, y: int, z: int, sleep_seconds: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.sleep_seconds = sleep_seconds
        self._loaded_state = None

    def run(self) -> Any:
        if self.sleep_seconds > 0:
            elapsed = 0.0
            while elapsed < self.sleep_seconds:
                time.sleep(0.1)
                elapsed += 0.1

        if self.x + self.y > 1000:
            raise ValueError("sum is greater than 1000!")
        return self.x + self.y

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        os.makedirs(checkpoint_location, exist_ok=True)
        state = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "is_preemption": is_preemption,
        }
        with open(os.path.join(checkpoint_location, "mock_state.json"), "w") as f:
            json.dump(state, f)
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        if checkpoint_location is None:
            return
        state_file = os.path.join(checkpoint_location, "mock_state.json")
        if os.path.exists(state_file):
            with open(state_file) as f:
                self._loaded_state = json.load(f)
                self.x = self._loaded_state["x"]
                self.y = self._loaded_state["y"]
                self.z = self._loaded_state["z"]
