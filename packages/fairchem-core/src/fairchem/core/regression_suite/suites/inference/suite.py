"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.regression_suite.spec.plan import precision_rotation_sweep
from fairchem.core.regression_suite.spec.schema import (
    CaseSpec,
    HistogramSpec,
    ObservableSpec,
    PerformanceMetricSpec,
)
from fairchem.core.regression_suite.spec.structures import frame_identities
from fairchem.core.regression_suite.spec.transforms import seeded_transform
from fairchem.core.regression_suite.suites.base import (
    STATIC_PREDICTION_OBSERVABLES,
    ObservableDescription,
    Suite,
    static_predictions,
)

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.regression_suite.spec.schema import SuiteCase


_FCC_CARBON_ATOM_COUNTS = (50, 100, 500, 1000)
# Saved performance sidecars record this protocol; comparisons require equality.
_BENCHMARK_PROTOCOL = {
    "name": "prepared-nve-steady-state",
    "warmup_steps": 5,
    "measured_steps": 25,
    "timestep_fs": 1.0,
    "synchronization": "cuda-before-and-after",
    "compilation_included": False,
    "model_loading_included": False,
    "memory_measurement": "torch-cuda-peak-allocated-after-warmup",
}


def _benchmark_nve(
    atoms,
    calculator,
    *,
    steps: int,
    warmup_steps: int,
    timestep_fs: float,
    velocity_seed: int,
    device: torch.device,
) -> tuple[float, float, float | None]:
    """
    Time only prepared Velocity-Verlet steps on a private structure copy.
    """
    benchmark_atoms = atoms.copy()
    benchmark_atoms.calc = calculator
    MaxwellBoltzmannDistribution(
        benchmark_atoms,
        temperature_K=300.0,
        force_temp=True,
        rng=np.random.RandomState(velocity_seed),
    )
    Stationary(benchmark_atoms)
    dynamics = VelocityVerlet(benchmark_atoms, timestep_fs * units.fs)
    if warmup_steps:
        dynamics.run(warmup_steps)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    dynamics.run(steps)
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if elapsed <= 0.0:
        raise ValueError("inference benchmark duration must be positive")
    milliseconds_per_step = 1000.0 * elapsed / steps
    nanoseconds_per_day = steps / elapsed * timestep_fs * 24.0 * 3600.0 / 1.0e6
    peak_memory = None
    if is_cuda:
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024.0**2
    return milliseconds_per_step, nanoseconds_per_day, peak_memory


@functools.cache
def source_frames() -> tuple[Atoms, ...]:
    """
    Generate the plan's sampled FCC carbon cells without changing the global RNG.
    """
    state = np.random.get_state()
    np.random.seed(42)
    try:
        structures = tuple(
            get_fcc_crystal_by_num_atoms(count) for count in _FCC_CARBON_ATOM_COUNTS
        )
    finally:
        np.random.set_state(state)
    for count, atoms in zip(_FCC_CARBON_ATOM_COUNTS, structures):
        # The plan hashes were frozen from extxyz positions with eight decimals.
        atoms.positions = np.round(atoms.positions, 8)
        atoms.info["structure_id"] = f"fcc-carbon-{count}"
    return structures


class InferenceSuite(Suite):
    """
    Check static inference outputs and report prepared NVE throughput.
    """

    name = "inference"
    explanation = (
        (
            "What it checks",
            "Whether large periodic cells give the same energies, forces, and "
            "stresses as FP64, and how fast the candidate runs them. Correctness "
            "is graded. Speed and memory are only reported. They never fail "
            "the suite.",
        ),
        (
            "The cases",
            "The structures are carbon crystals with 50, 100, 500, and 1,000 "
            "atoms. Each is cut from a face-centered cubic carbon lattice with "
            "a lattice constant of 3.8 Å. For correctness, each "
            "cell runs in its original orientation and in eight random "
            "rotations, which gives 36 cases. For speed, each cell runs in four "
            "other random rotations, which gives 16 timed runs.",
        ),
        (
            "What is measured",
            "Correctness cases record energy, forces, and stress. Timed runs "
            "give the atoms random velocities at 300 K and run constant-energy "
            "MD with a 1 fs step. After 5 warm-up steps, the next 25 steps are "
            "timed. Each timed run records milliseconds per step, nanoseconds "
            "of simulation per day, and peak GPU memory. Compilation and model "
            "loading are not timed.",
        ),
        (
            "How it is graded",
            "Energy, forces, and stress are graded exactly like "
            "prediction-fidelity. A cluster is one cell in one orientation, so "
            "there are 36. Each cluster's normalized differences, with a floor, "
            "are averaged. The median and 95th percentile must stay within twice "
            "the baseline's values. A quantity passes if either its normalized "
            "or its absolute differences do.",
        ),
        (
            "How speed is reported",
            "Each timed number is compared with the same runs of the baseline "
            "at FP64, FP32, and TF32, grouped by atom count. A factor above one "
            "is better, meaning faster, more simulation per day, or less "
            "memory. The summary line uses FP32. Speed is only compared when "
            "the hardware and timing protocol match. The timing histograms show "
            "the log ratio to the FP64 reference's timings.",
        ),
    )
    observable_descriptions: ClassVar[dict[str, ObservableDescription]] = {
        "energy": ObservableDescription(
            "Total energy of the carbon cell, in eV",
            "one carbon cell in one orientation (correctness cases only)",
        ),
        "forces": ObservableDescription(
            "Force on each atom along x, y, and z, in eV/Å",
            "one carbon cell in one orientation (correctness cases only)",
        ),
        "stress": ObservableDescription(
            "The six stress components, in eV/Å³",
            "one carbon cell in one orientation (correctness cases only)",
        ),
        "performance_ms_per_step": ObservableDescription(
            "Average time of one MD step over 25 timed steps after 5 warm-up steps, in ms",
            "one carbon cell in one timing rotation",
            "speed, not accuracy; compared in the performance table",
        ),
        "performance_ns_per_day": ObservableDescription(
            "Simulated nanoseconds per day of running at the measured speed, with 1 fs steps",
            "one carbon cell in one timing rotation",
            "speed, not accuracy; compared in the performance table",
        ),
        "performance_peak_gpu_memory": ObservableDescription(
            "Peak GPU memory PyTorch allocated during the timed steps, in MB of 1,048,576 bytes",
            "one carbon cell in one timing rotation (GPU runs only)",
            "memory, not accuracy; compared in the performance table",
        ),
    }
    plan_hash = "b1ff179f9c535a6b1bb48b01764ad167374343e9b382ad5dec9d2f466e035ae9"
    smoke_case_ids = ("inference/omat/fcc-carbon-100",)
    observables: ClassVar[dict[str, ObservableSpec]] = {
        **STATIC_PREDICTION_OBSERVABLES,
        "performance_ms_per_step": ObservableSpec(
            units="ms/step",
            comparison="log_ratio",
            histogram=HistogramSpec(1e-6, 10.0),
            report_only=True,
        ),
        "performance_ns_per_day": ObservableSpec(
            units="ns/day",
            comparison="log_ratio",
            histogram=HistogramSpec(1e-6, 10.0),
            report_only=True,
        ),
        "performance_peak_gpu_memory": ObservableSpec(
            units="MB",
            comparison="log_ratio",
            histogram=HistogramSpec(1e-6, 10.0),
            report_only=True,
        ),
    }
    generated_frames = staticmethod(source_frames)
    performance_protocol = _BENCHMARK_PROTOCOL
    performance_metrics: ClassVar[dict[str, PerformanceMetricSpec]] = {
        f"performance_{name}": PerformanceMetricSpec(units, objective, ("atom_count",))
        for name, units, objective in (
            ("ms_per_step", "ms/step", "minimize"),
            ("ns_per_day", "ns/day", "maximize"),
            ("peak_gpu_memory", "MB", "minimize"),
        )
    }

    def cases(self) -> list[CaseSpec]:
        structures = frame_identities(source_frames())
        cases = []
        for index, (structure_id, structure_hash) in enumerate(structures):
            cases.extend(
                precision_rotation_sweep(
                    case_id=f"inference/omat/{structure_id}",
                    metadata={"kind": "correctness", "structure_id": structure_id},
                    runtime_seed=64 + index,
                    sample_seed=220_042 + 10_000 * index,
                    structure="fcc_carbon.extxyz",
                    structure_hash=structure_hash,
                    structure_index=index,
                    task_name="omat",
                )
            )
        for index, (structure_id, structure_hash) in enumerate(structures):
            cases.extend(
                CaseSpec(
                    case_id=f"inference/performance/{structure_id}/rotation-{rotation:02d}",
                    metadata={
                        "kind": "performance",
                        "steps": 25,
                        "structure_id": structure_id,
                        "velocity_seed": 42 + index,
                        "warmup_steps": 5,
                    },
                    runtime_seed=42,
                    structure="fcc_carbon.extxyz",
                    structure_hash=structure_hash,
                    structure_index=index,
                    task_name="omat",
                    transform=seeded_transform(
                        43 + 10_000 * index + rotation,
                        transform_id=f"performance-rotation-{rotation:02d}",
                    ),
                )
                for rotation in range(4)
            )
        return cases

    def evaluate(self, calculator, case: SuiteCase) -> dict[str, np.ndarray]:
        metadata = case.spec.metadata
        if metadata["kind"] == "correctness":
            return static_predictions(calculator, case)
        milliseconds, nanoseconds, peak_memory = _benchmark_nve(
            case.transformed_atoms,
            calculator,
            steps=int(metadata["steps"]),
            warmup_steps=int(metadata["warmup_steps"]),
            timestep_fs=_BENCHMARK_PROTOCOL["timestep_fs"],
            velocity_seed=int(metadata["velocity_seed"]),
            device=torch.device(calculator.predictor.device),
        )
        measurements = {
            "performance_ms_per_step": np.asarray(milliseconds),
            "performance_ns_per_day": np.asarray(nanoseconds),
        }
        if peak_memory is not None:
            measurements["performance_peak_gpu_memory"] = np.asarray(peak_memory)
        return measurements
