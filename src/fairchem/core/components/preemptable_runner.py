"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import shutil
from abc import abstractmethod
from pathlib import Path

from omegaconf import OmegaConf

from fairchem.core.components.runner import Runner


class StopfairDetected(Exception):
    """
    Raised when a STOPFAIR sentinel file is detected.
    """


class PreemptableRunner(Runner):
    """
    Abstract base for runners with STOPFAIR + preemption support.

    Subclasses implement:
    - save_simulation_state(checkpoint_dir, is_preemption): save domain-specific files
    - load_simulation_state(checkpoint_dir): load domain-specific files

    The base class provides:
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

            # Base class saves portable + resume configs
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

    def _save_preemption_configs(self, tmp_dir: Path, checkpoint_location: str) -> None:
        """
        Save resume_config.yaml and portable_config.yaml.
        """
        config_path = Path(self.job_config.metadata.config_path)
        if not config_path.is_file():
            return

        cfg = OmegaConf.load(config_path)
        cfg.job.runner_state_path = checkpoint_location
        if "atoms" in cfg.get("runner", {}):
            del cfg.runner.atoms

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

    @abstractmethod
    def run(self):
        raise NotImplementedError
