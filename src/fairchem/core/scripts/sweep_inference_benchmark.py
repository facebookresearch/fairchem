"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fairchem.core.launchers.api import JobConfig
from fairchem.core.launchers.slurm_launch import slurm_launch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: str, overrides: list[str] | None = None) -> DictConfig:
    """
    Load Hydra config from a YAML file with optional overrides.

    Args:
        config_path: Path to the config YAML file
        overrides: List of config overrides in key=value format

    Returns:
        DictConfig object with properly initialized JobConfig
    """
    # Use hydra to compose the config
    config_directory = str(Path(config_path).parent.absolute())
    config_name = Path(config_path).stem

    with hydra.initialize_config_dir(config_dir=config_directory, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides or [])

    # Merge with structured JobConfig to get proper defaults
    cfg = OmegaConf.merge({"job": OmegaConf.structured(JobConfig)}, cfg)

    # Initialize metadata by calling __post_init__
    job = OmegaConf.to_object(cfg.job)
    job.__post_init__()
    cfg.job = job

    return cfg


def calculate_node_gpu_distribution(num_gpus: int) -> tuple[int, int]:
    """
    Calculate the number of nodes and GPUs per node for a given total GPU count.

    Args:
        num_gpus: Total number of GPUs

    Returns:
        Tuple of (num_nodes, gpus_per_node)

    Raises:
        ValueError: If num_gpus is invalid (not 1-8 or a multiple of 8)
    """
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    if num_gpus <= 8:
        # Single node with partial GPUs
        return (1, num_gpus)
    if num_gpus % 8 == 0:
        # Multiple full nodes
        return (num_gpus // 8, 8)
    else:
        raise ValueError(
            f"num_gpus must be between 1-8 or a multiple of 8, got {num_gpus}"
        )


def wait_for_jobs(jobs: list, check_interval: int = 30) -> tuple[set[int], set[int]]:
    """
    Wait for all jobs to complete, periodically reporting status.

    Args:
        jobs: List of job objects with done(), results(), and job_id attributes
        check_interval: Seconds between status checks (default: 30)

    Returns:
        Tuple of (completed_job_indices, failed_job_indices)
    """
    completed_jobs: set[int] = set()
    failed_jobs: set[int] = set()

    with tqdm(total=len(jobs), desc="Jobs completed", unit="job") as pbar:
        while len(completed_jobs) + len(failed_jobs) < len(jobs):
            for i, job in enumerate(jobs):
                if i in completed_jobs or i in failed_jobs:
                    continue
                try:
                    # Check if job is done (non-blocking check)
                    if job.done():
                        try:
                            job.results()
                            completed_jobs.add(i)
                            pbar.update(1)
                            pbar.set_postfix(
                                completed=len(completed_jobs), failed=len(failed_jobs)
                            )
                            logging.info(f"Job {job.job_id} completed successfully")
                        except Exception as e:
                            failed_jobs.add(i)
                            pbar.update(1)
                            pbar.set_postfix(
                                completed=len(completed_jobs), failed=len(failed_jobs)
                            )
                            logging.error(f"Job {job.job_id} failed with error: {e}")
                except Exception as e:
                    # If done() check fails, mark as failed
                    failed_jobs.add(i)
                    pbar.update(1)
                    pbar.set_postfix(
                        completed=len(completed_jobs), failed=len(failed_jobs)
                    )
                    logging.error(f"Job {job.job_id} status check failed: {e}")

            num_waiting = len(jobs) - len(completed_jobs) - len(failed_jobs)
            if num_waiting > 0:
                time.sleep(check_interval)

    return completed_jobs, failed_jobs


def collect_and_aggregate_results(
    job_metadata: list[dict[str, Any]],
    timestamp: str,
    num_gpus_list: list[int],
    natoms_list: list[int],
    output_dir: str,
) -> dict[str, Any]:
    """
    Collect and aggregate benchmark results from completed jobs.

    Args:
        job_metadata: List of metadata dictionaries for each job
        timestamp: Timestamp string for the sweep
        num_gpus_list: List of GPU counts used in the sweep
        natoms_list: List of atom counts used in the sweep
        output_dir: Directory to save aggregated results

    Returns:
        Dictionary with aggregated results
    """
    logging.info("Collecting and aggregating results...")
    aggregated_results: dict[str, Any] = {
        "timestamp": timestamp,
        "num_gpus_list": num_gpus_list,
        "natoms_list": natoms_list,
        "configurations": [],
    }

    for metadata in job_metadata:
        # Results are saved in results_dir which is run_dir/timestamp_id/results/
        results_file = os.path.join(metadata["results_dir"], "benchmark_results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)

            # Add configuration metadata to results
            config_results = {
                "num_nodes": metadata["num_nodes"],
                "num_gpus": metadata["num_gpus"],
                "run_name": metadata["run_name"],
                "benchmark_data": results,
            }
            aggregated_results["configurations"].append(config_results)
            logging.info(
                f"Loaded results for {metadata['num_nodes']} nodes, {metadata['num_gpus']} GPUs from {results_file}"
            )
        else:
            logging.warning(
                f"Results file not found: {results_file} for {metadata['run_name']}"
            )

    # Save aggregated results
    output_file = os.path.join(output_dir, f"aggregated_results_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(aggregated_results, f, indent=2)

    logging.info(f"Saved aggregated results to {output_file}")

    return aggregated_results


def generate_plots(aggregated_results: dict[str, Any], output_dir: str) -> None:
    """
    Generate scaling plots from benchmark results.

    Args:
        aggregated_results: Aggregated benchmark results dictionary
        output_dir: Directory to save plots
    """
    if not aggregated_results["configurations"]:
        logging.warning("No configurations to plot")
        return

    # Organize data by model
    models_data = defaultdict(lambda: defaultdict(list))

    for config in aggregated_results["configurations"]:
        num_gpus = config["num_gpus"]
        benchmark_data = config["benchmark_data"]

        if "model_to_qps_data" not in benchmark_data:
            continue

        for model_name, data_points in benchmark_data["model_to_qps_data"].items():
            for natoms, ns_per_day in data_points:
                # Skip OOM entries
                if ns_per_day == "OOM" or isinstance(ns_per_day, str):
                    logging.info(
                        f"Skipping OOM data point: {model_name}, {natoms} atoms, {num_gpus} GPUs"
                    )
                    continue
                # Convert ns/day to QPS (queries per second)
                qps = ns_per_day * 1e6 / (24 * 3600)
                models_data[model_name][num_gpus].append((natoms, qps, ns_per_day))

    # Generate plots for each model
    for model_name, gpu_data in models_data.items():
        model_safe_name = model_name.replace("/", "_").replace(" ", "_")

        # Plot 1: Single GPU speed vs number of atoms
        single_gpu_data = gpu_data.get(1) or gpu_data.get(min(gpu_data.keys()))
        if single_gpu_data:
            plt.figure(figsize=(10, 6))
            natoms_list = [item[0] for item in single_gpu_data]
            qps_list = [item[1] for item in single_gpu_data]
            plt.plot(natoms_list, qps_list, "o-", linewidth=2, markersize=8)
            plt.xlabel("Number of Atoms", fontsize=12)
            plt.ylabel("QPS (Queries Per Second)", fontsize=12)
            plt.title(
                f"Single GPU Performance: {model_name}\n(GPUs: {min(gpu_data.keys())})",
                fontsize=14,
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_file = os.path.join(
                output_dir, f"{model_safe_name}_single_gpu_speed.png"
            )
            plt.savefig(plot_file, dpi=150)
            plt.close()
            logging.info(f"Saved single GPU plot: {plot_file}")

        # Plot 2: Weak scaling (atoms/GPU constant)
        # Group by atoms per GPU
        atoms_per_gpu_data = defaultdict(list)
        for num_gpus, data_points in gpu_data.items():
            for natoms, qps, ns_per_day in data_points:
                atoms_per_gpu = natoms / num_gpus
                atoms_per_gpu_data[atoms_per_gpu].append((num_gpus, qps, ns_per_day))

        if len(atoms_per_gpu_data) > 0:
            plt.figure(figsize=(10, 6))
            for atoms_per_gpu in sorted(atoms_per_gpu_data.keys()):
                data = sorted(atoms_per_gpu_data[atoms_per_gpu])
                gpus = [item[0] for item in data]
                qps = [item[1] for item in data]
                plt.plot(
                    gpus,
                    qps,
                    "o-",
                    linewidth=2,
                    markersize=8,
                    label=f"{int(atoms_per_gpu)} atoms/GPU",
                )
            plt.xlabel("Number of GPUs", fontsize=12)
            plt.ylabel("QPS (Queries Per Second)", fontsize=12)
            plt.title(f"Weak Scaling: {model_name}\n(Constant atoms/GPU)", fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"{model_safe_name}_weak_scaling.png")
            plt.savefig(plot_file, dpi=150)
            plt.close()
            logging.info(f"Saved weak scaling plot: {plot_file}")

        # Plot 3: Strong scaling (total atoms constant)
        # Group by total atoms
        total_atoms_data = defaultdict(list)
        for num_gpus, data_points in gpu_data.items():
            for natoms, qps, ns_per_day in data_points:
                total_atoms_data[natoms].append((num_gpus, qps, ns_per_day))

        if len(total_atoms_data) > 0:
            plt.figure(figsize=(10, 6))
            for total_atoms in sorted(total_atoms_data.keys()):
                data = sorted(total_atoms_data[total_atoms])
                gpus = [item[0] for item in data]
                qps = [item[1] for item in data]
                # Also plot ideal scaling for reference
                if len(gpus) > 0:
                    baseline_qps = qps[0]
                    baseline_gpus = gpus[0]
                    ideal_qps = [baseline_qps * (g / baseline_gpus) for g in gpus]
                    plt.plot(
                        gpus,
                        qps,
                        "o-",
                        linewidth=2,
                        markersize=8,
                        label=f"{int(total_atoms)} atoms",
                    )
                    plt.plot(
                        gpus,
                        ideal_qps,
                        "--",
                        linewidth=1,
                        alpha=0.5,
                        label=f"{int(total_atoms)} atoms (ideal)",
                    )
            plt.xlabel("Number of GPUs", fontsize=12)
            plt.ylabel("QPS (Queries Per Second)", fontsize=12)
            plt.title(
                f"Strong Scaling: {model_name}\n(Constant total atoms)", fontsize=14
            )
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_file = os.path.join(
                output_dir, f"{model_safe_name}_strong_scaling.png"
            )
            plt.savefig(plot_file, dpi=150)
            plt.close()
            logging.info(f"Saved strong scaling plot: {plot_file}")


def print_summary_table(aggregated_results: dict[str, Any]) -> None:
    """
    Print a summary table of benchmark results.

    Args:
        aggregated_results: Aggregated benchmark results dictionary
    """
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    for config in aggregated_results["configurations"]:
        print(f"\nNodes: {config['num_nodes']}, GPUs: {config['num_gpus']}")
        if "model_to_qps_data" in config["benchmark_data"]:
            for model_name, data_points in config["benchmark_data"][
                "model_to_qps_data"
            ].items():
                print(f"  Model: {model_name}")
                for natoms, ns_per_day in data_points:
                    if ns_per_day == "OOM" or isinstance(ns_per_day, str):
                        print(f"    {natoms} atoms: OOM")
                    else:
                        qps = ns_per_day * 1e6 / (24 * 3600)
                        print(
                            f"    {natoms} atoms: {ns_per_day:.2f} ns/day ({qps:.2f} QPS)"
                        )
    print("=" * 80)


def sweep_nodes_and_atoms(
    base_config_path: str,
    num_gpus_list: list[int],
    natoms_list: list[int],
    output_dir: str,
    base_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """
    Sweep over different numbers of GPUs and atoms, running inference benchmarks.

    Args:
        base_config_path: Path to base config YAML file
        num_gpus_list: List of total GPU counts to sweep over
        natoms_list: List of atom counts to benchmark
        output_dir: Directory to save results
        base_overrides: Base config overrides to apply to all runs

    Returns:
        Dictionary with aggregated results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_jobs = []
    job_metadata = []

    for num_gpus in num_gpus_list:
        # Calculate node and GPU distribution
        num_nodes, gpus_per_node = calculate_node_gpu_distribution(num_gpus)
        graph_parallel_size = num_gpus

        # Create unique run directory for this configuration
        run_name = f"benchmark_n{num_nodes}_g{num_gpus}_{timestamp}"
        run_dir = os.path.join(output_dir, run_name)

        # Build config overrides
        overrides = base_overrides.copy() if base_overrides else []
        overrides.extend(
            [
                "job=slurm",
                "job.scheduler.mode=SLURM",
                f"job.scheduler.num_nodes={num_nodes}",
                f"job.scheduler.ranks_per_node={gpus_per_node}",
                f"job.graph_parallel_group_size={graph_parallel_size}",
                f"job.run_name={run_name}",
                f"job.run_dir={run_dir}",
                f"runner.natoms_list=[{','.join(map(str, natoms_list))}]",
            ]
        )

        logging.info(
            f"Submitting benchmark for {num_nodes} nodes ({gpus_per_node} GPUs/node), total: {num_gpus} GPUs"
        )
        logging.info(f"Overrides: {overrides}")

        # Load config with overrides
        cfg = load_config(base_config_path, overrides)

        # Create necessary directories
        os.makedirs(run_dir, exist_ok=True)
        log_dir = cfg.job.metadata.log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Save the config for this run
        config_save_path = os.path.join(run_dir, "config.yaml")
        OmegaConf.save(cfg, config_save_path)
        logging.info(f"Saved config to {config_save_path}")

        # Submit job to SLURM
        jobs = slurm_launch(cfg, log_dir)

        all_jobs.extend(jobs)
        job_metadata.append(
            {
                "num_nodes": num_nodes,
                "num_gpus": num_gpus,
                "run_dir": run_dir,
                "run_name": run_name,
                "job_ids": [job.job_id for job in jobs],
                "natoms_list": natoms_list,
                "timestamp_id": cfg.job.timestamp_id,
                "results_dir": cfg.job.metadata.results_dir,
            }
        )

    # Wait for all jobs to complete
    wait_for_jobs(all_jobs)

    # Collect and aggregate results
    aggregated_results = collect_and_aggregate_results(
        job_metadata=job_metadata,
        timestamp=timestamp,
        num_gpus_list=num_gpus_list,
        natoms_list=natoms_list,
        output_dir=output_dir,
    )

    # Generate plots
    generate_plots(aggregated_results, output_dir)

    # Print summary table
    print_summary_table(aggregated_results)

    return aggregated_results


def main():
    parser = argparse.ArgumentParser(
        description="Sweep inference benchmark across different GPU configurations"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/uma/speed/uma-speed.yaml",
        required=True,
        help="Path to base config YAML file",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="List of total GPU counts to sweep over (must be 1-8 or multiples of 8, default: 8 16 32 64)",
    )
    parser.add_argument(
        "--natoms",
        type=int,
        nargs="+",
        default=[1000, 2000, 4000, 8000],
        help="List of atom counts to benchmark (default: 1000 2000 4000 8000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/checkpoint/ocp/shared/tmp/benchmark_sweep_results",
        help="Directory to save results (default: ./benchmark_sweep_results)",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        default=[],
        help="Additional config overrides in key=value format",
    )

    args = parser.parse_args()

    # Validate GPU counts
    for num_gpus in args.num_gpus:
        try:
            calculate_node_gpu_distribution(num_gpus)
        except ValueError as e:
            parser.error(str(e))

    logging.info("Starting inference benchmark sweep")
    logging.info(f"Config: {args.config}")
    logging.info(f"GPU counts: {args.num_gpus}")
    logging.info(f"Atom counts: {args.natoms}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Additional overrides: {args.overrides}")

    sweep_nodes_and_atoms(
        base_config_path=args.config,
        num_gpus_list=args.num_gpus,
        natoms_list=args.natoms,
        output_dir=args.output_dir,
        base_overrides=args.overrides,
    )

    logging.info("Benchmark sweep completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
