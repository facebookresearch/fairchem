"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

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


class StopfairDetected(Exception):
    """
    Raised when a STOPFAIR sentinel file is detected.
    """


class PreemptableMixin(ABC):
    """
    Mixin that adds STOPFAIR preemption and atomic checkpointing to a runner.

    Intended to be combined with CalculateRunner (or any Runner subclass) via
    multiple inheritance.  The concrete class must inherit from Runner (for
    job_config) and implement the two abstract methods below.

    Subclasses implement:
    - save_simulation_state(checkpoint_dir, is_preemption): save domain-specific files
    - load_simulation_state(checkpoint_dir): load domain-specific files

    The mixin provides:
    - check_stopfair() / handle_stopfair(): STOPFAIR sentinel detection
    - save_state(): atomic checkpoint + portable_config + resume_config
    - load_state(): delegates to load_simulation_state()
    """

    @property
    def stopfair_path(self) -> Path:
        """
        Path to the STOPFAIR sentinel file.
        """
        return Path(self.job_config.metadata.checkpoint_dir).parent / "STOPFAIR"

    def check_stopfair(self) -> bool:
        """
        Check whether a STOPFAIR sentinel file exists.
        """
        return self.stopfair_path.exists()

    def handle_stopfair(self) -> None:
        """
        Save state to preemption dir, delete sentinel, raise StopfairDetected.
        """
        save_path = self.job_config.metadata.preemption_checkpoint_dir
        logging.info(f"STOPFAIR detected, saving state to {save_path}")
        if self.save_state(save_path, is_preemption=True):
            self.stopfair_path.unlink(missing_ok=True)
        raise StopfairDetected

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        """
        Atomically save checkpoint: domain state + preemption configs.

        Writes to a temporary directory first, then swaps it into place so
        that a crash mid-write never leaves a corrupt checkpoint.

        Args:
            checkpoint_location: Directory to save checkpoint files.
            is_preemption: Whether this save is due to preemption.

        Returns:
            True if state was successfully saved.
        """
        checkpoint_dir = Path(checkpoint_location)
        tmp_dir = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")

        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Subclass saves its domain-specific state
            self.save_simulation_state(tmp_dir, is_preemption)

            # Mixin saves portable + resume configs
            self._save_preemption_configs(tmp_dir, checkpoint_location)

            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            tmp_dir.rename(checkpoint_dir)
            return True
        except Exception as e:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            logging.exception(f"Failed to save checkpoint: {e}")
            return False

    @abstractmethod
    def save_simulation_state(self, checkpoint_dir: Path, is_preemption: bool) -> None:
        """
        Save domain-specific state files into checkpoint_dir.

        Args:
            checkpoint_dir: Directory to write state files into.
            is_preemption: Whether this save is due to preemption.
        """
        ...

    def _modify_resume_config(self, cfg: DictConfig) -> DictConfig:
        """
        Hook for subclasses to modify the config before saving resume/portable configs.

        Override this to strip or adjust runner-specific fields that should not
        appear in a resumed run (e.g. inline atoms data that will be loaded
        from the checkpoint instead).

        Args:
            cfg: The loaded OmegaConf config, already updated with runner_state_path.
        """
        return cfg

    def _save_preemption_configs(self, tmp_dir: Path, checkpoint_location: str) -> None:
        """
        Save resume_config.yaml and portable_config.yaml.
        """
        try:
            config_path = Path(self.job_config.metadata.config_path)
        except AttributeError:
            return
        if not config_path.is_file():
            return

        cfg = OmegaConf.load(config_path)
        cfg.job.runner_state_path = checkpoint_location
        cfg = self._modify_resume_config(cfg)

        # resume_config: same machine
        OmegaConf.save(cfg, tmp_dir / "resume_config.yaml")

        # portable_config: stripped of machine-specific fields
        portable_cfg = cfg.copy()
        portable_cfg.job.runner_state_path = "."
        if "run_dir" in portable_cfg.get("job", {}):
            run_dir = Path(portable_cfg.job.run_dir)
            portable_cfg.job.run_dir = run_dir.name
        if "metadata" in portable_cfg.get("job", {}):
            del portable_cfg.job.metadata
        if "timestamp_id" in portable_cfg.get("job", {}):
            del portable_cfg.job.timestamp_id
        OmegaConf.save(portable_cfg, tmp_dir / "portable_config.yaml")

    def load_state(self, checkpoint_location: str | None) -> None:
        """
        Load state from a checkpoint directory.

        Args:
            checkpoint_location: Directory containing checkpoint files, or None.
        """
        if checkpoint_location is None:
            return
        self.load_simulation_state(Path(checkpoint_location).absolute())

    @abstractmethod
    def load_simulation_state(self, checkpoint_dir: Path) -> None:
        """
        Load domain-specific state from checkpoint_dir.

        Args:
            checkpoint_dir: Directory containing checkpoint files.
        """
        ...
