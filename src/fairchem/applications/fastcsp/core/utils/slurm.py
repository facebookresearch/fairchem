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

if TYPE_CHECKING:
    from pathlib import Path


def submit_slurm_jobs(
    job_args: list[tuple[Callable, tuple[Any, ...], dict]],
    job_name: str,
    output_dir: Path,
    partition: str = "ocp,learnaccel",
    cpus_per_task: int = 80,
    mem_gb: int = 400,
    timeout_min: int = 1000,
    nodes: int = 1,
    tasks_per_node: int = 1,
) -> list[submitit.Job]:
    """
    Submit a batch of jobs to SLURM using submitit.

    This function provides a centralized way to submit parallel jobs with
    consistent resource allocation and job management across FastCSP modules.

    Args:
        job_args: List of (function, args, kwargs) tuples for each job
        job_name: Name prefix for SLURM jobs
        output_dir: Directory for SLURM log files
        partition: SLURM partition(s) to use
        cpus_per_task: Number of CPU cores per task
        mem_gb: Memory allocation in GB
        timeout_min: Job timeout in minutes
        nodes: Number of nodes per job
        tasks_per_node: Number of tasks per node

    Returns:
        List of submitit Job objects for monitoring

    Example:
        >>> job_args = [
        ...     (my_function, (arg1, arg2), {"kwarg": value}),
        ...     (my_function, (arg3, arg4), {"kwarg": value2}),
        ... ]
        >>> jobs = submit_slurm_jobs(job_args, "my_job", Path("/tmp/logs"))
        >>> for job in jobs:
        ...     job.wait()  # Wait for completion
    """
    if not job_args:
        print("No jobs to submit")
        return []

    # Configure SLURM executor
    executor = submitit.AutoExecutor(folder=output_dir)
    executor.update_parameters(
        slurm_job_name=job_name,
        timeout_min=timeout_min,
        slurm_partition=partition,
        cpus_per_task=cpus_per_task,
        tasks_per_node=tasks_per_node,
        nodes=nodes,
        mem_gb=mem_gb,
    )

    jobs = []
    with executor.batch():
        for func, args, kwargs in job_args:
            job = executor.submit(func, *args, **kwargs)
            jobs.append(job)

    if jobs:
        print(f"Submitted {len(jobs)} jobs: {jobs[0].job_id}")

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
