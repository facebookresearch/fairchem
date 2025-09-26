"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

SLURM job submission utilities for FastCSP.

This module provides utilities for submitting and managing parallel jobs
using the submitit library on SLURM-based clusters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import submitit
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger

if TYPE_CHECKING:
    from pathlib import Path


def submit_slurm_jobs(
    job_args: list[tuple[Callable, tuple[Any, ...], dict]],
    output_dir: Path,
    job_name: str = "fastcsp_job",
    **slurm_params: Any,
) -> list[submitit.Job]:
    """
    Submit a batch of jobs to SLURM using submitit with flexible parameter handling.

    This function provides a centralized way to submit parallel jobs with
    dynamic SLURM parameter configuration and consistent job management.

    Args:
        job_args: List of (function, args, kwargs) tuples for each job
        output_dir: Directory for SLURM log files
        **slurm_params: Any SLURM parameters supported by submitit.AutoExecutor

    Returns:
        List of submitit Job objects for monitoring

    Example:
        >>> job_args = [
        ...     (my_function, (arg1, arg2), {"kwarg": value}),
        ...     (my_function, (arg3, arg4), {"kwarg": value2}),
        ... ]
        >>> jobs = submit_slurm_jobs(
        ...     job_args,
        ...     Path("/tmp/logs"),
        ...     gpus_per_node=2,
        ... )
        >>> for job in jobs:
        ...     job.wait()  # Wait for completion
    """

    logger = get_central_logger()

    if not job_args:
        logger.info("No jobs to submit")
        return []

    # Set default parameters if not provided
    default_params = {
        "slurm_job_name": job_name,
        "cpus_per_task": 80,
        "mem_gb": 400,
        "timeout_min": 1000,
        "nodes": 1,
        "tasks_per_node": 1,
    }

    # Log user-requested SLURM parameters
    if slurm_params:
        logger.info(f"User-requested SLURM parameters: {slurm_params}")
    else:
        logger.info("No custom SLURM parameters provided, using defaults")

    # Merge defaults with provided parameters (provided params take precedence)
    executor_params = {**default_params, **slurm_params}

    # Log final SLURM parameters that will be used
    logger.info(f"Final SLURM parameters: {executor_params}")

    # Configure SLURM executor
    executor = submitit.AutoExecutor(folder=output_dir)
    executor.update_parameters(**executor_params)

    jobs = []
    with executor.batch():
        for func, args, kwargs in job_args:
            job = executor.submit(func, *args, **kwargs)
            jobs.append(job)

    if jobs:
        logger.info(f"Submitted {len(jobs)} jobs with job ID: {jobs[0].job_id}")

    return jobs


def wait_for_jobs(jobs: list[submitit.Job]) -> None:
    """
    Wait for all submitted jobs to complete execution.

    Args:
        jobs: List of submitit.Job objects to wait for completion

    Note:
        This function will block indefinitely until all jobs complete.
        Job failures are not explicitly handled - they will raise exceptions.
    """
    for job in jobs:
        job.wait()


def get_slurm_config(
    config: dict[str, Any], module_name: str, executor_type: str = "submit_slurm_jobs"
) -> dict[str, Any]:
    """
    Extract and prepare SLURM configuration for different FastCSP modules.

    This unified function handles SLURM configuration for all workflow stages
    with stage-specific defaults and automatic parameter handling.

    Args:
        config: Full configuration dictionary
        module_name: Name of the module ("genarris", "relax", "process_generated", "filter")
        executor_type: Type of executor ("submit_slurm_jobs" or "submitit_executor")

    Returns:
        dict: SLURM parameters formatted for the specified executor type
    """
    logger = get_central_logger()

    # Define module-specific defaults
    module_defaults = {
        "genarris": {
            "job-name": "genarris",
            "nodes": 1,
            "ntasks-per-node": 1,
            "time": 7200,
        },
        "process_generated": {
            "job-name": "process_genarris_outputs",
            "cpus_per_task": 80,
            "mem_gb": 400,
            "time": 1000,
        },
        "relax": {
            "job-name": "relax",
            "gpus_per_node": 1,
            "cpus_per_task": 10,
            "mem_gb": 50,
            "time": 1000,
        },
        "filter": {
            "job-name": "filter_and_deduplicate_structures",
            "cpus_per_task": 80,
            "mem_gb": 400,
            "time": 1000,
        },
        "evaluate": {
            "job-name": "eval",
            "cpus_per_task": 1,
            "mem_gb": 10,
            "time": 1000,
        },
    }

    if module_name not in module_defaults:
        raise ValueError(
            f"Unknown module: {module_name}. Supported: {list(module_defaults.keys())}"
        )

    # Extract module-specific SLURM config with fallback hierarchy
    module_config = config.get(module_name, {})
    module_slurm_config = module_config.get("slurm", {})

    # If no module-specific config, try general slurm config
    if not module_slurm_config:
        module_slurm_config = config.get("slurm", {})

    # Log user-requested SLURM configuration
    if module_slurm_config:
        logger.info(
            f"User-requested SLURM config for {module_name}: {module_slurm_config}"
        )

    # If still no config, use defaults
    if not module_slurm_config:
        module_slurm_config = module_defaults[module_name]
        logger.info(
            f"No SLURM configuration found for {module_name}, using default parameters: {module_slurm_config}"
        )

    normalized_config = {
        key.replace("-", "_"): value for key, value in module_slurm_config.items()
    }

    # Prepare parameters based on executor type
    if executor_type == "submitit_executor":
        executor_params = {}
        standard_flags = set()

        if module_name == "genarris":
            # Genarris-specific parameter mapping
            base_params = {
                "slurm_job_name": normalized_config.get("job_name", "genarris"),
                "nodes": normalized_config.get("nodes", 1),
                "tasks_per_node": normalized_config.get("ntasks_per_node", 1),
                "timeout_min": normalized_config.get("time", 7200),
                "slurm_use_srun": False,
                "cpus_per_task": normalized_config.get("cpus_per_task", 1),
            }
            standard_flags = {
                "job_name",
                "nodes",
                "ntasks_per_node",
                "time",
                "cpus_per_task",
            }

        elif module_name == "relax":
            # Relax-specific parameter mapping
            base_params = {
                "slurm_job_name": normalized_config.get("job_name", "relax"),
                "timeout_min": normalized_config.get("time", 1000),
                "gpus_per_node": normalized_config.get("gpus_per_node", 1),
                "cpus_per_task": normalized_config.get("cpus_per_task", 10),
                "mem_gb": normalized_config.get("mem_gb", 50),
            }
            standard_flags = {
                "job_name",
                "gpus_per_node",
                "cpus_per_task",
                "mem_gb",
                "time",
            }

        executor_params.update(base_params)

        # Handle additional SLURM flags with slurm_ prefix
        for key, value in normalized_config.items():
            if key not in standard_flags:
                executor_key = f"slurm_{key}"
                executor_params[executor_key] = value
                logger.debug(
                    f"Added additional SLURM parameter: {executor_key}={value}"
                )

        # Log final executor parameters
        logger.info(
            f"Final SLURM executor parameters for {module_name}: {executor_params}"
        )

        return executor_params

    elif executor_type == "submit_slurm_jobs":
        # For submit_slurm_jobs function (process_generated.py and filter.py)
        # Direct parameter mapping (already uses snake_case)
        slurm_params = {
            "job_name": normalized_config.get(
                "job_name", module_defaults[module_name]["job-name"]
            ),
            "cpus_per_task": normalized_config.get("cpus_per_task", 80),
            "mem_gb": normalized_config.get("mem_gb", 400),
            "timeout_min": normalized_config.get("time", 1000),
        }

        # Handle additional SLURM flags (already normalized to snake_case)
        standard_flags = {"job_name", "cpus_per_task", "mem_gb", "time"}
        for key, value in normalized_config.items():
            if key not in standard_flags:
                slurm_params[key] = value
                logger.debug(f"Added additional SLURM parameter: {key}={value}")

        # Log final SLURM parameters
        logger.info(f"Final SLURM parameters for {module_name}: {slurm_params}")

        return slurm_params

    else:
        raise ValueError(f"Unknown executor_type: {executor_type}")


def get_genarris_slurm_config(
    genarris_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Legacy wrapper for genarris SLURM configuration.

    Returns:
        tuple: (slurm_config_for_scripts, executor_params)
    """
    # Create a full config structure with the genarris config
    full_config = {"genarris": genarris_config}

    executor_params = get_slurm_config(full_config, "genarris", "submitit_executor")

    # Extract raw config for script generation
    genarris_slurm_config = genarris_config.get("slurm", {})
    if not genarris_slurm_config:
        genarris_slurm_config = {
            "job-name": "genarris",
            "nodes": 1,
            "ntasks-per-node": 1,
            "time": 7200,
        }

    return genarris_slurm_config, executor_params


def get_relax_slurm_config(
    relax_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Legacy wrapper for relax SLURM configuration.

    Returns:
        tuple: (slurm_config_for_scripts, executor_params)
    """
    # Create a full config structure with the relax config
    full_config = {"relax": relax_config}

    executor_params = get_slurm_config(full_config, "relax", "submitit_executor")

    # Extract raw config for potential script generation
    relax_slurm_config = relax_config.get("slurm", {})
    if not relax_slurm_config:
        relax_slurm_config = {
            "job-name": "relax",
            "gpus_per_node": 1,
            "cpus_per_task": 10,
            "mem_gb": 50,
            "time": 1000,
        }

    return relax_slurm_config, executor_params


def get_process_slurm_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Legacy wrapper for process_generated SLURM configuration.

    Returns:
        dict: SLURM parameters for submit_slurm_jobs function
    """
    return get_slurm_config(config, "process_generated", "submit_slurm_jobs")


def get_filter_slurm_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Legacy wrapper for filter SLURM configuration.

    Returns:
        dict: SLURM parameters for submit_slurm_jobs function
    """
    return get_slurm_config(config, "filter", "submit_slurm_jobs")


def get_eval_slurm_config(eval_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Get eval SLURM configuration from eval_config section.

    Args:
        eval_config: Evaluation configuration dictionary (the 'evaluate' section).
                    If None, returns default parameters.

    Returns:
        dict: SLURM parameters for submit_slurm_jobs function
    """
    if eval_config is None:
        eval_config = {}

    # Create a config structure that get_slurm_config expects
    # by wrapping eval_config in an "evaluate" key
    full_config = {"evaluate": eval_config}

    return get_slurm_config(full_config, "evaluate", "submit_slurm_jobs")
