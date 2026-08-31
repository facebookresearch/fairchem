"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Benchmark the autobatching + Ray Serve batch-inference server.

This is the batch-server analogue of ``sweep_inference_benchmark.py``. Instead of
sweeping GPU counts on SLURM, it measures how the *local* Ray Serve batch server
plus probing-based autobatching (see ``examples/autobatch_threads.py``) scales as
the number of concurrent inference requests grows.

For each requested system size it:
  1. Builds an ``InferenceBatcher`` and calls ``auto_configure_batching`` once
     with a representative probe system.
  2. Measures a serial (unbatched) baseline over a bounded sample of systems.
  3. Sweeps a list of concurrency levels, submitting that many single-system
     requests concurrently through the batcher's executor and timing the wall
     clock.

Reported metrics per point: throughput (QPS = systems/second), speedup over the
serial baseline, and per-request latency (mean/median/p95). Results are written
to JSON and plotted (throughput, speedup, latency vs. concurrency).

Usage:
    python -m fairchem.core.scripts.benchmark_batch_server \\
        --model uma-s-1p1 --task omat \\
        --natoms 16 64 256 --num-requests 4 8 16 32 64 128 256 \\
        --output-dir ./autobatch_benchmark_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import as_completed
from datetime import datetime
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch
from ase.build import bulk

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.calculate._batch import AutobatchConfig, InferenceBatcher
from fairchem.core.datasets.atomic_data import AtomicData

if TYPE_CHECKING:
    from ase import Atoms

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# FCC metals whose ``bulk(el, cubic=True)`` cell holds 4 atoms, so cycling
# through them varies chemistry while keeping every system the same size.
FCC_ELEMENTS = ["Cu", "Ni", "Al", "Au", "Ag", "Pd", "Pt", "Pb"]


# ---------------------------------------------------------------------------
# Workload construction
# ---------------------------------------------------------------------------
def build_supercell(element: str, natoms_target: int) -> Atoms:
    """
    Build a near-cubic supercell of ``element`` with about ``natoms_target`` atoms.

    Args:
        element: Chemical symbol of an FCC metal (4 atoms per cubic cell).
        natoms_target: Desired number of atoms; the actual count is the nearest
            value reachable by an isotropic ``(n, n, n)`` repeat.

    Returns:
        An ASE ``Atoms`` supercell.
    """
    base = bulk(element, cubic=True)
    per_cell = len(base)
    reps = max(1, round((natoms_target / per_cell) ** (1 / 3)))
    return base.repeat((reps, reps, reps))


def build_workload(count: int, natoms_target: int) -> list[Atoms]:
    """
    Build ``count`` supercells of roughly ``natoms_target`` atoms each.

    Chemistry is cycled through :data:`FCC_ELEMENTS` for realism; because all of
    them are 4-atom cubic cells every system has the same atom count.

    Args:
        count: Number of systems to build.
        natoms_target: Approximate atom count per system.

    Returns:
        List of ASE ``Atoms`` objects.
    """
    return [
        build_supercell(FCC_ELEMENTS[i % len(FCC_ELEMENTS)], natoms_target)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Worker: kept at module scope so it stays picklable for process/ray backends
# ---------------------------------------------------------------------------
def timed_calculate(atoms: Atoms, task_name: str, predict_unit) -> dict[str, Any]:
    """
    Compute energy and forces for one system and time the call.

    Args:
        atoms: The system to evaluate.
        task_name: UMA task name (e.g. ``"omat"``).
        predict_unit: A predict unit or batch predict unit to run inference with.

    Returns:
        Dict with the per-request ``latency_s`` and the system's ``natoms``.
    """
    atoms = atoms.copy()
    atoms.calc = FAIRChemCalculator(predict_unit, task_name=task_name)
    t0 = time.perf_counter()
    atoms.get_potential_energy()
    atoms.get_forces()
    return {"latency_s": time.perf_counter() - t0, "natoms": len(atoms)}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _summarize(
    latencies: list[float], wall_time_s: float, count: int
) -> dict[str, float | int]:
    """
    Summarize a run into throughput and latency statistics.

    Args:
        latencies: Per-request wall-clock latencies in seconds.
        wall_time_s: Total wall-clock time for the whole run in seconds.
        count: Number of requests in the run.

    Returns:
        Dict with ``qps``, ``wall_time_s``, ``count`` and latency stats.
    """
    p95 = (
        statistics.quantiles(latencies, n=20)[-1]
        if len(latencies) > 1
        else latencies[0]
    )
    return {
        "qps": count / wall_time_s if wall_time_s > 0 else float("inf"),
        "wall_time_s": wall_time_s,
        "count": count,
        "mean_latency_s": statistics.fmean(latencies),
        "median_latency_s": statistics.median(latencies),
        "p95_latency_s": p95,
    }


def _collect(futures: list) -> list[dict[str, Any]]:
    """
    Collect results from submitted futures.

    Args:
        futures: Futures returned by ``executor.submit``.

    Returns:
        List of worker result dicts.
    """
    return [f.result() for f in as_completed(futures)]


def run_serial(
    workload: list[Atoms], task_name: str, predict_unit
) -> dict[str, float | int]:
    """Run predictions one at a time and return summary statistics."""
    t0 = time.perf_counter()
    latencies = [
        timed_calculate(atoms, task_name, predict_unit)["latency_s"]
        for atoms in workload
    ]
    return _summarize(latencies, time.perf_counter() - t0, len(workload))


def run_batched(
    workload: list[Atoms],
    task_name: str,
    batcher: InferenceBatcher,
) -> dict[str, float | int]:
    """Submit all systems concurrently through the batcher and summarize."""
    t0 = time.perf_counter()
    futures = [
        batcher.executor.submit(
            timed_calculate, atoms, task_name, batcher.batch_predict_unit
        )
        for atoms in workload
    ]
    results = _collect(futures)
    wall_time = time.perf_counter() - t0
    return _summarize([r["latency_s"] for r in results], wall_time, len(workload))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep_batch_server(
    model_name: str,
    task_name: str,
    natoms_list: list[int],
    num_requests_list: list[int],
    output_dir: str,
    serial_samples: int = 8,
    warmup_requests: int = 4,
    concurrency_backend: str = "threads",
    num_replicas: int = 1,
    min_batch_size: int = 512,
    max_batch_size_cap: int = 4096,
    run_name: str | None = None,
) -> dict[str, Any]:
    """
    Sweep concurrency x system size against the batch server.

    Args:
        model_name: Pretrained UMA model name.
        task_name: UMA task name used for all systems.
        natoms_list: Per-system atom counts to benchmark.
        num_requests_list: Concurrency levels (number of concurrent requests).
        output_dir: Directory to write results and plots to.
        serial_samples: Number of systems used for the serial baseline.
        warmup_requests: Number of warmup requests before each timed run.
        concurrency_backend: Concurrency backend for submitting requests. Only
            ``"threads"`` is supported.
        num_replicas: Number of batch-server replicas.
        min_batch_size: Autobatch minimum batch size (atoms).
        max_batch_size_cap: Autobatch maximum batch size cap (atoms).
        run_name: Optional label included in plot titles and output paths.

    Returns:
        Aggregated results dictionary (also written to JSON).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predict_unit = pretrained_mlip.get_predict_unit(model_name, device=device)

    autobatch_config = AutobatchConfig(
        min_batch_size=min_batch_size,
        max_batch_size_cap=max_batch_size_cap,
        probe_steps=3,
        warmup_steps=2,
        backoff_factor=0.9,
    )
    max_requests = max(num_requests_list)
    aggregated: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": model_name,
        "task": task_name,
        "device": device,
        "concurrency_backend": concurrency_backend,
        "num_replicas": num_replicas,
        "natoms_list": natoms_list,
        "num_requests_list": num_requests_list,
        "results": [],
    }

    for natoms_target in natoms_list:
        probe_atoms = build_supercell(FCC_ELEMENTS[0], natoms_target)
        natoms_actual = len(probe_atoms)
        logging.info(
            f"=== System size ~{natoms_target} atoms (actual {natoms_actual}) ==="
        )

        with InferenceBatcher(
            predict_unit=predict_unit,
            split_oom_batch=True,
            num_replicas=num_replicas,
            concurrency_backend=concurrency_backend,
            concurrency_backend_options={"max_workers": max_requests},
            deployment_name=f"benchmark-server-n{natoms_target}",
        ) as batcher:
            probe = [AtomicData.from_ase(probe_atoms, task_name=task_name)]
            autobatch = batcher.auto_configure_batching(probe, config=autobatch_config)
            logging.info(
                f"Autobatch: max_batch_size={autobatch.max_batch_size}, "
                f"timeout={autobatch.batch_wait_timeout_s:.4f}s, "
                f"median_latency={autobatch.median_latency_s:.4f}s"
            )

            # Warmup the server so probe/JIT costs don't pollute timings.
            if warmup_requests > 0:
                run_batched(
                    build_workload(warmup_requests, natoms_target),
                    task_name,
                    batcher,
                )

            # Serial baseline (bounded sample); QPS is ~constant in the workload
            # size, so a small sample suffices as the speedup reference.
            serial = run_serial(
                build_workload(serial_samples, natoms_target),
                task_name,
                batcher.predict_unit,
            )
            logging.info(
                f"Serial baseline: {serial['qps']:.2f} QPS, "
                f"mean_latency={serial['mean_latency_s']:.4f}s"
            )

            concurrency = []
            for num_requests in num_requests_list:
                batched = run_batched(
                    build_workload(num_requests, natoms_target),
                    task_name,
                    batcher,
                )
                speedup = (
                    batched["qps"] / serial["qps"]
                    if serial["qps"] > 0
                    else float("inf")
                )
                concurrency.append(
                    {
                        "num_requests": num_requests,
                        "batched": batched,
                        "speedup": speedup,
                    }
                )
                logging.info(
                    f"  n={num_requests:>4}: {batched['qps']:.2f} QPS "
                    f"({speedup:.2f}x), mean_latency={batched['mean_latency_s']:.4f}s"
                )

        aggregated["results"].append(
            {
                "natoms_target": natoms_target,
                "natoms_actual": natoms_actual,
                "autobatch": {
                    "max_batch_size": autobatch.max_batch_size,
                    "batch_wait_timeout_s": autobatch.batch_wait_timeout_s,
                    "median_latency_s": autobatch.median_latency_s,
                },
                "serial": serial,
                "concurrency": concurrency,
            }
        )

    return _save_results(aggregated, output_dir, run_name)


# ---------------------------------------------------------------------------
# Output: JSON, plots, summary
# ---------------------------------------------------------------------------
def _save_results(
    aggregated: dict[str, Any], output_dir: str, run_name: str | None
) -> dict[str, Any]:
    """Write aggregated results to JSON and return them."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = aggregated["timestamp"]
    output_file = os.path.join(output_dir, f"aggregated_results_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    logging.info(f"Saved aggregated results to {output_file}")
    return aggregated


def generate_plots(
    aggregated: dict[str, Any], output_dir: str, run_name: str | None = None
) -> None:
    """
    Generate throughput, speedup, and latency plots vs. concurrency.

    Args:
        aggregated: Aggregated results from :func:`sweep_batch_server`.
        output_dir: Directory to save plots to.
        run_name: Optional label included in all plot titles.
    """
    if not aggregated["results"]:
        logging.warning("No results to plot")
        return

    title_prefix = f"{run_name}\n" if run_name else ""
    model = aggregated["model"]
    backend = aggregated["concurrency_backend"]

    # (ylabel, filename suffix, title, extractor) for the concurrency plots.
    def _batched_metric(key):
        return lambda res: [c["batched"][key] for c in res["concurrency"]]

    def _speedup(res):
        return [c["speedup"] for c in res["concurrency"]]

    plots = [
        ("QPS (systems/second)", "throughput", "Throughput", _batched_metric("qps")),
        ("Speedup over serial (x)", "speedup", "Speedup", _speedup),
        ("Wall time (s)", "wall_time", "Wall time", _batched_metric("wall_time_s")),
        (
            "Mean latency (s)",
            "latency_mean",
            "Mean request latency",
            _batched_metric("mean_latency_s"),
        ),
        (
            "Median latency (s)",
            "latency_median",
            "Median request latency",
            _batched_metric("median_latency_s"),
        ),
        (
            "p95 latency (s)",
            "latency_p95",
            "p95 request latency",
            _batched_metric("p95_latency_s"),
        ),
    ]

    for ylabel, suffix, title, extractor in plots:
        plt.figure(figsize=(10, 6))
        for res in aggregated["results"]:
            x = [c["num_requests"] for c in res["concurrency"]]
            plt.plot(
                x,
                extractor(res),
                "o-",
                linewidth=2,
                markersize=8,
                label=f"{res['natoms_actual']} atoms",
            )
        plt.xlabel("Concurrent requests", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xscale("log", base=2)
        plt.title(f"{title_prefix}{title}: {model}\n(backend: {backend})", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"batch_server_{suffix}.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()
        logging.info(f"Saved plot: {plot_file}")


def print_summary_table(aggregated: dict[str, Any]) -> None:
    """Print a summary table of the benchmark results."""
    print("\n" + "=" * 80)
    print("BATCH SERVER BENCHMARK SUMMARY")
    print(
        f"  model={aggregated['model']}  task={aggregated['task']}  "
        f"device={aggregated['device']}  backend={aggregated['concurrency_backend']}"
    )
    print("=" * 80)
    for res in aggregated["results"]:
        ab = res["autobatch"]
        s = res["serial"]
        print(
            f"\n{res['natoms_actual']} atoms  "
            f"(serial baseline: {s['qps']:.2f} QPS, "
            f"mean={s['mean_latency_s']:.4f}s, median={s['median_latency_s']:.4f}s, "
            f"p95={s['p95_latency_s']:.4f}s; "
            f"autobatch max_batch_size={ab['max_batch_size']}, "
            f"timeout={ab['batch_wait_timeout_s']:.4f}s)"
        )
        print(
            f"  {'requests':>9}  {'QPS':>10}  {'speedup':>8}  {'wall(s)':>9}  "
            f"{'mean(s)':>9}  {'median(s)':>10}  {'p95(s)':>9}"
        )
        print(
            f"  {'-' * 9}  {'-' * 10}  {'-' * 8}  {'-' * 9}  "
            f"{'-' * 9}  {'-' * 10}  {'-' * 9}"
        )
        for c in res["concurrency"]:
            b = c["batched"]
            print(
                f"  {c['num_requests']:>9}  {b['qps']:>10.2f}  "
                f"{c['speedup']:>7.2f}x  {b['wall_time_s']:>9.4f}  "
                f"{b['mean_latency_s']:>9.4f}  {b['median_latency_s']:>10.4f}  "
                f"{b['p95_latency_s']:>9.4f}"
            )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the autobatching + batch-inference server."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pretrained UMA model name (default: smallest available UMA model).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="omat",
        help="UMA task name for all systems (default: omat).",
    )
    parser.add_argument(
        "--natoms",
        type=int,
        nargs="+",
        default=[16, 64, 256],
        help=(
            "Per-system atom counts to benchmark (default: 16 64 256). Smaller "
            "systems under-utilize the GPU per forward pass, so batching them "
            "yields the largest speedups; larger systems saturate the GPU and "
            "show diminishing returns."
        ),
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256],
        help=(
            "Concurrency levels to sweep (default: 4 8 16 32 64 128 256). "
            "Speedup rises with concurrency until batches fill the GPU, so the "
            "sweep must reach high concurrency to observe the plateau. Below ~4 "
            "requests a batch can't fill and batching is a net loss, so the "
            "sweep starts there. Values must stay below the server's "
            "max_ongoing_requests (300)."
        ),
    )
    parser.add_argument(
        "--serial-samples",
        type=int,
        default=8,
        help="Number of systems used for the serial baseline (default: 8).",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=4,
        help="Warmup requests before each timed run (default: 4).",
    )
    parser.add_argument(
        "--concurrency-backend",
        type=str,
        choices=["threads"],
        default="threads",
        help="Concurrency backend for submitting requests (default: threads).",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=1,
        help="Number of batch-server replicas (default: 1).",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=512,
        help="Autobatch minimum batch size in atoms (default: 512).",
    )
    parser.add_argument(
        "--max-batch-size-cap",
        type=int,
        default=2048,
        help=(
            "Autobatch maximum batch size cap in atoms (default: 2048). The "
            "probe repeats the representative system up to this many atoms, so "
            "with small systems a large cap explodes the probe into thousands "
            "of tiny graphs (very slow). 2048 keeps probing bounded while still "
            "fitting enough small systems per batch to reveal real speedups."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./autobatch_benchmark_results",
        help="Directory to save results and plots.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional label included in plot titles and output directory.",
    )
    args = parser.parse_args()

    # Initialize Ray early so setup_batch_predict_server reuses this instance.
    # Use a writable temp dir on a large filesystem to avoid object-store
    # spill failures on small /dev/shm or /var/tmp mounts.
    import ray

    ray_tmp = os.path.expanduser("~/.cache/ray_tmp")
    os.makedirs(ray_tmp, exist_ok=True)
    if not ray.is_initialized():
        ray.init(
            log_to_driver=False,
            logging_config=ray.LoggingConfig(log_level="WARNING"),
            _temp_dir=ray_tmp,
        )

    model_name = args.model
    if model_name is None:
        model_name = sorted(
            name for name in pretrained_mlip.available_models if "uma" in name
        )[0]

    # All outputs for this run live under a single timestamped directory.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{args.run_name}_{timestamp}" if args.run_name else timestamp
    run_dir = os.path.join(args.output_dir, run_dir_name)

    logging.info("Starting batch-server benchmark")
    logging.info(f"Model: {model_name}, task: {args.task}")
    logging.info(f"System sizes: {args.natoms}")
    logging.info(f"Concurrency levels: {args.num_requests}")
    logging.info(f"Backend: {args.concurrency_backend}, replicas: {args.num_replicas}")

    aggregated = sweep_batch_server(
        model_name=model_name,
        task_name=args.task,
        natoms_list=args.natoms,
        num_requests_list=args.num_requests,
        output_dir=run_dir,
        serial_samples=args.serial_samples,
        warmup_requests=args.warmup_requests,
        concurrency_backend=args.concurrency_backend,
        num_replicas=args.num_replicas,
        min_batch_size=args.min_batch_size,
        max_batch_size_cap=args.max_batch_size_cap,
        run_name=args.run_name,
    )

    generate_plots(aggregated, run_dir, run_name=args.run_name)
    print_summary_table(aggregated)

    logging.info("Batch-server benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
