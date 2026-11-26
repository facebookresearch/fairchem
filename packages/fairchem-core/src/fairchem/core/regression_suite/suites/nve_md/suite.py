"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# ``MeasurementRecord.values`` is a NumPy array, not a pandas object.
# ruff: noqa: PD011
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, NamedTuple

import ase.io
import numpy as np
from ase import units
from ase.md.verlet import VelocityVerlet

from fairchem.core.components.calculate.simulation_tools.initialization import (
    initialize_momenta,
)
from fairchem.core.regression_suite.spec.plan import unrotated_case
from fairchem.core.regression_suite.spec.schema import HistogramSpec, ObservableSpec
from fairchem.core.regression_suite.spec.structures import atoms_hash
from fairchem.core.regression_suite.suites.base import (
    GroupValue,
    ObservableDescription,
    Suite,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fairchem.core.regression_suite.spec.schema import (
        CaseSpec,
        MeasurementRecord,
        SuiteCase,
    )


class _Dataset(NamedTuple):
    structure_pattern: str
    task_name: str
    timestep_fs: float


class _System(NamedTuple):
    dataset: str
    name: str
    # Temperature of the starting velocities.
    kelvin: float
    steps: int


DATASETS = {
    "md22": _Dataset("md22/md22_{}.xyz.xz", task_name="omol", timestep_fs=1.0),
    "tm23": _Dataset("tm23/{}_melt.xyz.xz", task_name="omat", timestep_fs=5.0),
}

SYSTEMS = (
    _System("md22", "AT-AT", kelvin=500.0, steps=2_000),
    _System("md22", "AT-AT-CG-CG", kelvin=500.0, steps=2_000),
    _System("md22", "Ac-Ala3-NHMe", kelvin=500.0, steps=2_000),
    _System("md22", "DHA", kelvin=500.0, steps=10_000),
    _System("md22", "buckyball-catcher", kelvin=400.0, steps=2_000),
    _System("md22", "double-walled_nanotube", kelvin=400.0, steps=1_000),
    _System("md22", "stachyose", kelvin=500.0, steps=1_000),
    _System("tm23", "Cu", kelvin=1697.5, steps=2_000),
    _System("tm23", "Fe", kelvin=2263.75, steps=1_000),
    _System("tm23", "Au", kelvin=1671.25, steps=3_000),
)

# Independent velocity seeds per system.
_SEEDS_PER_SYSTEM = 20

# MD steps between recorded frames.
SAMPLE_INTERVAL = 20

# Gate 1 smooths each seed's gap to FP64 over this many frames. Gates 2 and 3
# smooth each energy curve over about this long, at most a quarter of the run.
GAP_SMOOTHING_FRAMES = 5
DRIFT_SMOOTHING_PS = 1.0

# Each system's three gates: how far it is from the FP64 run of the same seed,
# how far its smoothed energy moves from its start, and how far its seeds
# spread apart. Each is a distance from ideal behaviour, graded as an absolute
# difference from zero.
GATES = {
    "fp64_gap": ObservableSpec(
        units="meV/atom", comparison="absolute", histogram=HistogramSpec(1e-14, 1e2)
    ),
    "drift": ObservableSpec(
        units="meV/atom", comparison="absolute", histogram=HistogramSpec(1e-8, 1e2)
    ),
    "spread": ObservableSpec(
        units="(meV/atom)^2", comparison="absolute", histogram=HistogramSpec(1e-10, 1e2)
    ),
}


def gate_observable(system: str, gate: str) -> str:
    """
    Name the observable that holds one of a system's gates.
    """
    return f"{system}_{gate}"


def drift_window(system: _System) -> int:
    """
    Return how many frames gates 2 and 3 average each energy curve over.
    """
    frame_ps = SAMPLE_INTERVAL * DATASETS[system.dataset].timestep_fs / 1000.0
    frames = system.steps // SAMPLE_INTERVAL + 1
    return max(
        GAP_SMOOTHING_FRAMES, min(round(DRIFT_SMOOTHING_PS / frame_ps), frames // 4)
    )


def _moving_average(curves: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width) / width
    return np.stack([np.convolve(curve, kernel, mode="valid") for curve in curves])


def gate_curves(
    run: np.ndarray, reference: np.ndarray, window: int
) -> dict[str, np.ndarray]:
    """
    Return one system's gate curves over time, per seed and across seeds.

    Args:
        run: One energy-deviation series per seed, one row each.
        reference: The FP64 reference series of the same seeds, in order.
        window: Frames that gates 2 and 3 average each curve over.

    Returns:
        ``fp64_gap``: the median over seeds of each smoothed gap to FP64.
        ``drift``: the median over seeds of how far each smoothed curve has
        moved from its first value. ``spread``: the variance over seeds of
        that movement. The ``_seeds`` entries hold the per-seed curves.
    """
    gaps = _moving_average(np.abs(run - reference), GAP_SMOOTHING_FRAMES)
    smoothed = _moving_average(run, window)
    moved = smoothed - smoothed[:, :1]
    return {
        "fp64_gap_seeds": gaps,
        "drift_seeds": np.abs(moved),
        "fp64_gap": np.median(gaps, axis=0),
        "drift": np.median(np.abs(moved), axis=0),
        "spread": moved.var(axis=0),
    }


def _series_by_system(
    records: Sequence[MeasurementRecord],
) -> dict[str, dict[str, np.ndarray]]:
    series: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        if record.observable == "energy_deviation":
            system = record.case_id.split("/")[2]
            series.setdefault(system, {})[record.case_id] = record.values
    return series


def _deviation_rms(deviation: np.ndarray) -> float:
    """
    Return the RMS of an energy deviation, which starts at zero.
    """
    return float(np.sqrt(np.mean(np.square(deviation))))


def atom_count(system: _System) -> int:
    """
    Return the number of atoms in a system's starting structure.
    """
    pattern = DATASETS[system.dataset].structure_pattern
    return len(
        ase.io.read(Path(__file__).parent / "structures" / pattern.format(system.name))
    )


_GATE_DESCRIPTIONS = {
    "fp64_gap": "gate 1, the time average of the median over seeds of the "
    "smoothed gap to FP64 for the same seed, in meV/atom",
    "drift": "gate 2, the time average of the median over seeds of how far "
    "the smoothed energy has moved from its start, in meV/atom",
    "spread": "gate 3, the time average of how much that movement varies "
    "across seeds, in (meV/atom)^2",
}


def _observable_descriptions() -> dict[str, ObservableDescription]:
    seed = "one seed of one system"
    gated = "graded through the three gates instead"
    return {
        **{
            gate_observable(system.name, gate): ObservableDescription(
                f"{system.name}'s {text}", "the whole system"
            )
            for system in SYSTEMS
            for gate, text in _GATE_DESCRIPTIONS.items()
        },
        "energy_deviation_rms": ObservableDescription(
            "RMS over the whole run of the energy per atom measured from the "
            "first recorded frame, in meV/atom",
            seed,
            gated,
        ),
        "energy_drift": ObservableDescription(
            "Drift rate of the energy per atom, from its first and last 20-frame "
            "averages after the first tenth of the run, in meV/atom/ps",
            seed,
            gated,
        ),
        "energy_deviation": ObservableDescription(
            "Energy per atom at every recorded frame, measured from the first "
            "frame, in meV/atom",
            seed,
            gated,
        ),
        "temperature": ObservableDescription(
            "Temperature at every recorded frame, in K", seed, gated
        ),
    }


_SAFETY_FACTOR = 4.0


def _explanation() -> tuple[tuple[str, str], ...]:
    molecules = sum(system.dataset == "md22" for system in SYSTEMS)
    steps = sorted(system.steps for system in SYSTEMS)
    return (
        (
            "What it checks",
            "Whether simulations keep their total energy constant, as they "
            "should with no thermostat, and how closely the candidate follows "
            f"FP64. It runs {molecules} MD22 molecules and "
            f"{len(SYSTEMS) - molecules} TM23 metal melts.",
        ),
        (
            "The runs",
            f"Each system is simulated from {_SEEDS_PER_SYSTEM} different random "
            f"starting velocities, called seeds, for {steps[0]:,} to "
            f"{steps[-1]:,} MD steps. FP64, FP32, TF32, and the candidate start "
            "each seed from exactly the same atoms and velocities. The total "
            f"energy per atom is recorded every {SAMPLE_INTERVAL} steps, in meV.",
        ),
        (
            "Gate 1, matches FP64",
            "For every seed, how far the candidate's energy is from the FP64 "
            "run's energy at each recorded step, averaged over "
            f"{GAP_SMOOTHING_FRAMES} recorded steps. Thin lines are single seeds "
            "and the thick line is the median of the seeds. Lower precision "
            "rounds the forces a little differently, so its run slowly leaves "
            "FP64's path. This gate tells precisions apart best on short runs.",
        ),
        (
            "Gate 2, stays near its start",
            "Each seed's own energy, averaged over about "
            f"{DRIFT_SMOOTHING_PS:g} ps (a quarter of the run if that is "
            "shorter), measured from its first averaged value. Conserved energy "
            "stays near zero. A leak or wandering energy makes the line rise. "
            "The thick line is the median of the seeds.",
        ),
        (
            "Gate 3, seeds stay together",
            "The same curves as gate 2, but measuring how much the seeds "
            "disagree at each time, as their variance. If the energy wanders "
            "randomly, the seeds spread further apart as time goes on. This is "
            "where TF32-like precision shows on long runs.",
        ),
        (
            "How it is graded",
            "Each gate's number is the average height of its thick line over "
            f"the run. A precision's limit is {_SAFETY_FACTOR:g}x the number "
            "the baseline's run at that precision gets. Where two precisions "
            "get numbers too close to tell apart, the gate gives them one "
            "shared limit and only passes or fails there. Each gate table says "
            "which precisions that gate can tell apart.",
        ),
    )


class NVEMDSuite(Suite):
    """
    Check energy conservation in fixed NVE trajectories against FP64 with
    three gates per system.

    Each gate is the time average of a curve computed across a system's seeds.
    Where the calibration runs of two precisions give values the limits cannot
    tell apart, the gate shares one limit between them and only passes or
    fails there.
    """

    name = "nve-md"
    plan_hash = "4dc278b1ac629f1d59ce8045733c0ab988965f05c163b642bcbef56947d44e82"
    smoke_case_ids = tuple(
        f"nve-md/{system.dataset}/{system.name}/seed-0" for system in SYSTEMS
    )
    slurm_case_parallel = True
    observables: ClassVar[dict[str, ObservableSpec]] = {
        **{
            gate_observable(system.name, gate): spec
            for system in SYSTEMS
            for gate, spec in GATES.items()
        },
        "energy_deviation_rms": ObservableSpec(
            units="meV/atom",
            comparison="log_ratio",
            histogram=HistogramSpec(1e-16, 1e4),
            floor=0.02,
            report_only=True,
        ),
        "energy_drift": ObservableSpec(
            units="meV/atom/ps",
            comparison="absolute",
            histogram=HistogramSpec(1e-12, 100.0),
            report_only=True,
        ),
        "energy_deviation": ObservableSpec(
            units="meV/atom",
            comparison="absolute",
            histogram=HistogramSpec(1e-12, 1e4),
            report_only=True,
        ),
        "temperature": ObservableSpec(
            units="K",
            comparison="absolute",
            histogram=HistogramSpec(1e-8, 1e6),
            report_only=True,
        ),
    }
    decision_statistics = ("mean_absolute",)
    safety_factor = _SAFETY_FACTOR
    merge_unresolved_tiers = True
    explanation = _explanation()
    observable_descriptions = _observable_descriptions()

    def group_values(
        self,
        reference: Sequence[MeasurementRecord],
        records: Sequence[MeasurementRecord],
    ) -> dict[str, dict[str, GroupValue]]:
        reference_series = _series_by_system(reference)
        values = {}
        for system_name, run_series in _series_by_system(records).items():
            system = next(s for s in SYSTEMS if s.name == system_name)
            case_ids = sorted(run_series)
            curves = gate_curves(
                np.stack([run_series[case_id] for case_id in case_ids]),
                np.stack([reference_series[system_name][c] for c in case_ids]),
                drift_window(system),
            )
            cluster = f"nve-md/{system.dataset}/{system.name}"
            for gate in GATES:
                values[gate_observable(system.name, gate)] = {
                    cluster: GroupValue(float(curves[gate].mean()), len(case_ids))
                }
        return values

    def cases(self) -> list[CaseSpec]:
        cases = []
        for index, system in enumerate(SYSTEMS):
            dataset = DATASETS[system.dataset]
            structure = dataset.structure_pattern.format(system.name)
            structure_hash = atoms_hash(
                ase.io.read(Path(__file__).parent / "structures" / structure)
            )
            for seed_index in range(_SEEDS_PER_SYSTEM):
                seed = 42 + index + 100 * seed_index
                case_id = f"nve-md/{system.dataset}/{system.name}/seed-{seed_index}"
                cases.append(
                    unrotated_case(
                        case_id=case_id,
                        metadata={
                            "dataset": system.dataset,
                            "drift_window": 20,
                            "equilibration_fraction": 0.1,
                            "integrator": "velocity_verlet",
                            "sample_interval": SAMPLE_INTERVAL,
                            "steps": system.steps,
                            "system_name": system.name,
                            "temperature_k": system.kelvin,
                            "timestep_fs": dataset.timestep_fs,
                        },
                        runtime_seed=seed,
                        sample_seed=seed,
                        structure=structure,
                        structure_hash=structure_hash,
                        structure_index=0,
                        task_name=dataset.task_name,
                    )
                )
        return cases

    def evaluate(self, calculator, case: SuiteCase) -> dict[str, np.ndarray]:
        metadata = case.spec.metadata
        atoms = case.transformed_atoms.copy()
        atoms.calc = calculator
        timestep_fs = float(metadata["timestep_fs"])
        initialize_momenta(
            atoms,
            temperature_K=float(metadata["temperature_k"]),
            seed=case.spec.runtime_seed,
            remove_center_of_mass_momentum=False,
        )
        dynamics = VelocityVerlet(atoms, timestep_fs * units.fs)
        total_energies: list[float] = []
        temperatures: list[float] = []
        times_ps: list[float] = []

        def record() -> None:
            total_energies.append(float(atoms.get_total_energy()))
            temperatures.append(float(atoms.get_temperature()))
            times_ps.append(dynamics.get_number_of_steps() * timestep_fs / 1000.0)

        interval = int(metadata["sample_interval"])
        dynamics.attach(record, interval=interval)
        dynamics.run(int(metadata["steps"]))

        # Drift in meV/ps between the first and last moving averages of the
        # total energy after the equilibration fraction is discarded, over the
        # time between the ends of those two windows.
        energies = np.asarray(total_energies, dtype=np.float64)
        window = int(metadata["drift_window"])
        start = int(len(energies) * float(metadata["equilibration_fraction"]))
        smoothed = _moving_average(energies[None, start:], window)[0]
        drift = (
            1000.0
            * float(abs(smoothed[-1] - smoothed[0]))
            / float(times_ps[-1] - times_ps[start + window - 1])
        )
        # Energy deviation from the first frame in meV/atom; the gates are
        # computed from it across each system's seeds.
        deviation = 1000.0 * (energies - energies[0]) / len(atoms)
        return {
            "energy_deviation_rms": np.asarray(_deviation_rms(deviation)),
            "energy_drift": np.asarray(drift / len(atoms)),
            "energy_deviation": deviation,
            "temperature": np.asarray(temperatures, dtype=np.float64),
        }
