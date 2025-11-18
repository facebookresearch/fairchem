"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import TYPE_CHECKING

import hydra
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError

from fairchem.core.launchers.api import (
    ALLOWED_TOP_LEVEL_KEYS,
    JobConfig,
    SchedulerType,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from fairchem.core.components.runner import Runner


# this effects the cli only since the actual job will be run in subprocesses or remoe
logging.basicConfig(level=logging.INFO)


def get_canonical_config(config: DictConfig) -> DictConfig:
    # manually initialize metadata, because OmegaConf currently doesn't call __post_init__ on dataclasses
    job = OmegaConf.to_object(config.job)
    job.__post_init__()
    config.job = job
    # check that each key other than the allowed top level keys are used in config
    # find all top level keys are not in the allowed set
    all_keys = set(config.keys()).difference(ALLOWED_TOP_LEVEL_KEYS)
    used_keys = set()
    for key in all_keys:
        # make a copy of all keys except the key in question
        copy_cfg = OmegaConf.create({k: v for k, v in config.items() if k != key})
        try:
            OmegaConf.resolve(copy_cfg)
        except InterpolationKeyError:
            # if this error is thrown, this means the key was actually required
            used_keys.add(key)

    unused_keys = all_keys.difference(used_keys)
    if unused_keys != set():
        raise ValueError(
            f"Found unused keys in the config: {unused_keys}, please remove them!, only keys other than {ALLOWED_TOP_LEVEL_KEYS} or ones that are used as variables are allowed."
        )

    # resolve the config to fully replace the variables and delete all top level keys except for the ALLOWED_TOP_LEVEL_KEYS
    OmegaConf.resolve(config)
    return OmegaConf.create(
        {k: v for k, v in config.items() if k in ALLOWED_TOP_LEVEL_KEYS}
    )


def get_hydra_config_from_yaml(
    config_yml: str, overrides_args: list[str]
) -> DictConfig:
    # Load the configuration from the file
    os.environ["HYDRA_FULL_ERROR"] = "1"
    config_directory = os.path.dirname(os.path.abspath(config_yml))
    config_name = os.path.basename(config_yml)
    hydra.initialize_config_dir(config_directory, version_base="1.1")
    cfg = hydra.compose(config_name=config_name, overrides=overrides_args)
    # merge default structured config with initialized job object
    cfg = OmegaConf.merge({"job": OmegaConf.structured(JobConfig)}, cfg)
    # canonicalize config (remove top level keys that just used replacing variables)
    return get_canonical_config(cfg)


def main(
    args: argparse.Namespace | None = None, override_args: list[str] | None = None
):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", type=str, required=True)
        args, override_args = parser.parse_known_args()

    cfg = get_hydra_config_from_yaml(args.config, override_args)
    log_dir = cfg.job.metadata.log_dir
    os.makedirs(cfg.job.run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    OmegaConf.save(cfg, cfg.job.metadata.config_path)
    logging.info(f"saved canonical config to {cfg.job.metadata.config_path}")

    scheduler_cfg = cfg.job.scheduler
    logging.info(f"Running fairchemv2 cli with {cfg}")
    if scheduler_cfg.mode == SchedulerType.SLURM:  # Run on cluster
        assert (
            os.getenv("SLURM_SUBMIT_HOST") is None
        ), "SLURM DID NOT SUBMIT JOB!! Please do not submit jobs from an active slurm job (srun or otherwise)"

        if scheduler_cfg.use_ray:
            logging.info("Lauching job on Ray + Slurm cluster")
            from fairchem.core.launchers import ray_on_slurm_launch

            ray_on_slurm_launch.ray_on_slurm_launch(cfg, log_dir)
        else:
            logging.info("Lauching job on directly on Slurm cluster")
            from fairchem.core.launchers import slurm_launch

            slurm_launch.slurm_launch(cfg, log_dir)
    elif scheduler_cfg.mode == SchedulerType.LOCAL:  # Run locally
        if scheduler_cfg.num_nodes > 1:
            cfg.job.scheduler.num_nodes = 1
            logging.warning(
                f"You cannot use more than one node (scheduler_cfg.num_nodes={scheduler_cfg.num_nodes}) in LOCAL mode, over-riding to 1 node"
            )
        # if using ray, then launch ray cluster locally
        if scheduler_cfg.use_ray:
            logging.info("Running in local mode with local ray cluster")
            # don't recursively instantiate the runner here to allow lazy instantiations in the runner
            # the hands all responsibility the user, ie they must initialize ray
            runner: Runner = hydra.utils.instantiate(cfg.runner, _recursive_=False)
            runner.run()
        else:
            from fairchem.core.launchers.slurm_launch import local_launch

            # else launch locally using torch elastic or local mode
            logging.info(
                f"Running in local mode with {scheduler_cfg.ranks_per_node} ranks using device_type:{cfg.job.device_type}"
            )
            local_launch(cfg, log_dir)
    else:
        raise ValueError(f"Unknown scheduler mode {scheduler_cfg.mode}")
