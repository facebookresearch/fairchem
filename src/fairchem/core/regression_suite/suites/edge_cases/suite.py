"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import ase.io
import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    force_temperature,
)

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2
from fairchem.core.regression_suite.spec.plan import (
    precision_rotation_sweep,
    unrotated_case,
)
from fairchem.core.regression_suite.spec.schema import HistogramSpec, ObservableSpec
from fairchem.core.regression_suite.spec.structures import frame_identities
from fairchem.core.regression_suite.suites.base import (
    STATIC_PREDICTION_OBSERVABLES,
    ObservableDescription,
    Suite,
    static_predictions,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from fairchem.core.regression_suite.spec.schema import CaseSpec, SuiteCase


_OMAT = {"structure": "omat.extxyz", "task_name": "omat"}


def _npt(*phases: tuple[str, float]) -> dict:
    return {
        "ensemble": "npt",
        "phases": [
            {"name": name, "pressure_bar": pressure_bar, "steps": 100}
            for name, pressure_bar in phases
        ],
        "temperature_k": 3_000.0,
        "timestep_fs": 0.5,
        "taut_fs": 25.0,
        "taup_fs": 25.0,
        "compressibility_per_bar": 7.0e-7,
        "sample_interval": 5,
    }


def _rattled_fcc(cells: int, symbol: str, lattice_constant: float, seed: int) -> Atoms:
    atoms = bulk(symbol, "fcc", a=lattice_constant, cubic=True).repeat((cells,) * 3)
    rattle = np.random.default_rng(seed).normal(0.0, 0.05, (len(atoms), 3))
    # The plan hashes were frozen from extxyz positions with eight decimals.
    atoms.positions = np.round(atoms.positions + rattle, 8)
    return atoms


# UMA keeps at most 300 neighbors per atom within its 6 A cutoff. Above the
# cap, the default (non-strict) mask keeps every neighbor within 0.01 A of the
# 301st nearest, so an over-cap atom keeps 301 neighbors unless ties add more.
NEIGHBOR_CAP = 300
NEIGHBOR_CAP_PROBES = (299, 300, 301, 320)
_CAP_CORE_SHELLS = (2.0, 2.9, 3.8, 4.6)
_CAP_EXTRA_RADIUS, _CAP_EXTRA_SPACING = 5.4, 0.02
_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def _sphere_points(count: int, radius: float) -> np.ndarray:
    """
    Spread ``count`` points quasi-uniformly over a sphere (Fibonacci lattice).
    """
    k = np.arange(count) + 0.5
    z = 1.0 - 2.0 * k / count
    rho = np.sqrt(1.0 - z**2)
    angle = _GOLDEN_ANGLE * k
    return radius * np.column_stack([rho * np.cos(angle), rho * np.sin(angle), z])


def _neighbor_cap_cluster(neighbors: int) -> Atoms:
    """
    Build a hydrogen cluster whose central atom has exactly ``neighbors``
    neighbors within the cutoff while every other atom stays below the cap.

    Up to the cap, neighbors fill concentric shells from 2.0 to 4.6 A, weighted
    by shell area (atoms stay at least 0.8 A apart). Starting the shells at
    2.0 A hides the far side of the cluster from every other atom, so only the
    central atom tests the cap. Neighbors beyond the cap sit at distinct radii
    from 5.4 A, 0.02 A apart, so the 0.01 A degeneracy tolerance never merges
    them and the mask keeps exactly the nearest 301. The 24 A box keeps
    periodic images beyond the cutoff.
    """
    core = min(neighbors, NEIGHBOR_CAP)
    weights = np.square(_CAP_CORE_SHELLS) / np.square(_CAP_CORE_SHELLS).sum()
    counts = np.floor(weights * core).astype(int)
    counts[-1] += core - counts.sum()
    extra = neighbors - core
    extra_radii = _CAP_EXTRA_RADIUS + _CAP_EXTRA_SPACING * np.arange(extra)
    positions = np.vstack(
        [
            np.zeros((1, 3)),
            *(_sphere_points(c, r) for c, r in zip(counts, _CAP_CORE_SHELLS)),
            _sphere_points(extra, 1.0) * extra_radii[:, None],
        ]
    )
    box = 24.0
    return Atoms(
        f"H{neighbors + 1}",
        positions=np.round(positions + box / 2, 8),
        cell=[box] * 3,
    )


@functools.cache
def source_frames() -> tuple[Atoms, ...]:
    """
    Return the plan's source structures in order: seven generated graph-shape
    probes, three MD structures, then the neighbor-cap probes.
    """
    probes = {
        "bulk_32": _rattled_fcc(2, "Cu", 3.61, 1),
        "bulk_108_tail": _rattled_fcc(3, "Al", 4.05, 2),
        "dimer_near_cutoff": Atoms(
            "CC", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 5.9]], cell=[20.0] * 3
        ),
        "isolated_atom": Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=[20.0] * 3),
        "bulk_256": _rattled_fcc(4, "Cu", 3.61, 3),
        "graph_churn_sparse": _rattled_fcc(2, "Cu", 5.0, 4),
        "graph_churn_dense": _rattled_fcc(2, "Cu", 3.0, 4),
    }
    for structure_id, atoms in probes.items():
        atoms.pbc = True
        atoms.info = {"charge": 0, "spin": 0, "structure_id": structure_id}
    # The first MD structure is the phonon suite's mp-17048 crystal.
    suites = Path(__file__).parents[1]
    npt = ase.io.read(suites / "phonon_crystal/structures/mdr_phonon_10.extxyz.xz", 0)
    npt.info = {"structure_id": "npt_mp_17048"}
    md_frames = ase.io.read(suites / "edge_cases/structures/omat.extxyz.xz", ":")
    cap_probes = [_neighbor_cap_cluster(n) for n in NEIGHBOR_CAP_PROBES]
    for neighbors, atoms in zip(NEIGHBOR_CAP_PROBES, cap_probes):
        atoms.pbc = True
        atoms.info = {
            "charge": 0,
            "spin": 0,
            "structure_id": f"neighbor_cap_{neighbors}",
        }
    return (*probes.values(), npt, *md_frames, *cap_probes)


def _edge_count(atoms) -> int:
    data = AtomicData.from_ase(atoms, task_name="omat")
    edge_index, _cell_offsets, _neighbors = radius_graph_pbc_v2(
        data,
        radius=6.0,
        max_num_neighbors_threshold=NEIGHBOR_CAP,
    )
    return int(edge_index.shape[1])


def _sample(atoms, samples: dict[str, list[float]]) -> None:
    samples["energy"].append(float(atoms.get_potential_energy()))
    samples["volume"].append(float(atoms.get_volume()))
    samples["temperature"].append(float(atoms.get_temperature()))
    samples["edge_count"].append(_edge_count(atoms))


def _run_sampled(
    dynamics, atoms, steps: int, interval: int, samples: dict[str, list[float]]
) -> None:
    for _ in range(steps):
        dynamics.run(1)
        if dynamics.get_number_of_steps() % interval == 0:
            _sample(atoms, samples)


def _trajectory(
    samples: dict[str, list[float]], prefix: str, names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    trajectory = {
        f"{prefix}_{name}": np.asarray(samples[name], dtype=np.float64)
        for name in names
    }
    trajectory["trajectory_edge_count"] = np.asarray(
        samples["edge_count"], dtype=np.float64
    )
    return trajectory


def _run_npt(calculator, case: SuiteCase) -> dict[str, np.ndarray]:
    metadata = case.spec.metadata
    atoms = case.transformed_atoms.copy()
    atoms.calc = calculator
    temperature_k = float(metadata["temperature_k"])
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=temperature_k,
        force_temp=True,
        rng=np.random.RandomState(case.spec.runtime_seed),
    )
    Stationary(atoms)
    samples = defaultdict(list)
    _sample(atoms, samples)
    for phase in metadata["phases"]:
        dynamics = NPTBerendsen(
            atoms,
            timestep=float(metadata["timestep_fs"]) * units.fs,
            temperature_K=temperature_k,
            pressure_au=float(phase["pressure_bar"]) * units.bar,
            taut=float(metadata["taut_fs"]) * units.fs,
            taup=float(metadata["taup_fs"]) * units.fs,
            compressibility_au=(float(metadata["compressibility_per_bar"]) / units.bar),
            fixcm=True,
        )
        _run_sampled(
            dynamics,
            atoms,
            int(phase["steps"]),
            int(metadata["sample_interval"]),
            samples,
        )
    return _trajectory(samples, "trajectory", ("energy", "volume", "temperature"))


def _run_nvt(calculator, case: SuiteCase) -> dict[str, np.ndarray]:
    metadata = case.spec.metadata
    atoms = case.transformed_atoms.copy()
    atoms.calc = calculator
    temperature_k = float(metadata["temperature_k"])
    atoms.set_velocities(atoms.get_center_of_mass() - atoms.positions)
    force_temperature(atoms, temperature_k, unit="K")
    samples = defaultdict(list)
    _sample(atoms, samples)
    dynamics = Langevin(
        atoms,
        timestep=float(metadata["timestep_fs"]) * units.fs,
        temperature_K=temperature_k,
        friction=float(metadata["friction_per_fs"]) / units.fs,
        rng=np.random.RandomState(case.spec.runtime_seed + 1),
    )
    _run_sampled(
        dynamics,
        atoms,
        int(metadata["steps"]),
        int(metadata["sample_interval"]),
        samples,
    )
    # The million-kelvin implosion is chaotic: rounding differences grow so
    # quickly that paired trajectory errors vary more than twofold between
    # identical FP32 runs. It must complete with finite values, but its paired
    # differences are reported rather than used for qualification.
    return _trajectory(samples, "dense_transient", ("energy", "temperature"))


class EdgeCasesSuite(Suite):
    """
    Exercise difficult graph shapes and sparse-dense-sparse calculator reuse.
    """

    name = "edge-cases"
    explanation = (
        (
            "What it checks",
            "Whether the candidate stays correct on inputs that ordinary "
            "structures rarely reach. UMA only looks at neighbors within 6 Å "
            "and keeps at most 300 per atom. Some fast backends size their "
            "memory on the first call. So this suite probes empty and crowded "
            "neighborhoods, the 300-neighbor cap, a graph that changes size "
            "between calls, and molecular dynamics (MD) under extreme pressure.",
        ),
        (
            "The static probes",
            "Eleven structures run in their original orientation and in eight "
            "random rotations, which gives 99 cases. Seven test graph shapes. "
            "One is a lone carbon atom with no neighbors. One is two carbons "
            "5.9 Å apart, just inside the cutoff. Three are randomly shaken "
            "copper and aluminum crystals of 32, 108, and 256 atoms. Two are a "
            "sparse and a dense copper cell. Four are hydrogen clusters whose "
            "central atom has 299, 300, 301, or 320 neighbors.",
        ),
        (
            "Sequences and MD runs",
            "Three cases reuse one calculator on the sparse cell, then the dense "
            "cell, then the sparse cell again. Four MD runs use the 28-atom "
            "crystal mp-17048. One squeezes it at 200 GPa and 3,000 K. One "
            "shrinks a copy to 0.70 of its length, then pulls it apart at the "
            "same strength. One cycles squeeze, pull, squeeze. Each phase is "
            "100 steps. The last starts a 112-atom block of the shrunk copy at "
            "a million kelvin, with atoms moving inward.",
        ),
        (
            "What is measured",
            "Static cases record energy, forces, and stress. They are compared "
            "with FP64 as normalized differences with a floor, like "
            "prediction-fidelity. The neighbor-cap clusters are graded as "
            "separate force and stress quantities, so the looser limits of the "
            "other probes cannot hide a cap mistake. MD runs record energy, "
            "volume, and temperature every 5 steps. Volume and temperature are "
            "compared by "
            "the size of the log of their ratio to FP64, which is about the "
            "relative difference.",
        ),
        (
            "How it is graded",
            "A cluster is one structure in one orientation, one step of the "
            "sequence, or one whole MD run. Each cluster's differences are "
            "averaged. The median and 95th percentile must stay within twice "
            "the baseline's values. Neighbor-cap energy, the count of neighbor "
            "pairs along each MD run, and the million-kelvin run are "
            "report-only. That run is chaotic, so it only has to finish. Only "
            "three MD runs are graded, so their median and 95th percentile come "
            "from three numbers. Any value that is not a finite number fails "
            "the run.",
        ),
    )
    observable_descriptions: ClassVar[dict[str, ObservableDescription]] = {
        "energy": ObservableDescription(
            "Total energy, in eV",
            "one graph-shape probe in one orientation, or one step of the sparse, dense, sparse sequence",
        ),
        "forces": ObservableDescription(
            "Force on each atom along x, y, and z, in eV/Å",
            "one graph-shape probe in one orientation, or one step of the sparse, dense, sparse sequence",
        ),
        "stress": ObservableDescription(
            "The six stress components, in eV/Å³",
            "one graph-shape probe in one orientation, or one step of the sparse, dense, sparse sequence",
        ),
        "neighbor_cap_energy": ObservableDescription(
            "Total energy of hydrogen clusters whose central atom has 299 to 320 neighbors, in eV",
            "one neighbor-cap probe in one orientation",
            "one energy per case moves no more than rounding does",
        ),
        "neighbor_cap_forces": ObservableDescription(
            "Force on each atom along x, y, and z, in eV/Å, for the same hydrogen clusters",
            "one neighbor-cap probe in one orientation",
        ),
        "neighbor_cap_stress": ObservableDescription(
            "The six stress components, in eV/Å³, for the same hydrogen clusters",
            "one neighbor-cap probe in one orientation",
        ),
        "trajectory_energy": ObservableDescription(
            "Potential energy every 5 steps of MD at 3,000 K under 200 GPa of squeezing or pulling, in eV",
            "one whole MD run (contraction, expansion, or pressure cycle)",
        ),
        "trajectory_volume": ObservableDescription(
            "Cell volume every 5 steps of the same runs, in Å³",
            "one whole MD run (contraction, expansion, or pressure cycle)",
        ),
        "trajectory_temperature": ObservableDescription(
            "Temperature every 5 steps of the same runs, in K",
            "one whole MD run (contraction, expansion, or pressure cycle)",
        ),
        "dense_transient_energy": ObservableDescription(
            "Potential energy every 5 steps of a run started at a million kelvin, in eV",
            "the whole million-kelvin run",
            "chaotic run, repeat runs differ by over 2x",
        ),
        "dense_transient_temperature": ObservableDescription(
            "Temperature every 5 steps of the million-kelvin run, in K",
            "the whole million-kelvin run",
            "chaotic run, repeat runs differ by over 2x",
        ),
        "trajectory_edge_count": ObservableDescription(
            "Neighbor links within 6 Å every 5 steps, each pair counted both ways and at most 300 per atom",
            "one whole MD run (all four runs)",
            "diagnostic count from the suite's own neighbor search",
        ),
    }
    plan_hash = "06d72c6ef18d396bdd6ed58bf9b4bbf5d6063de9b2103df75634d9d84d4e3b69"
    smoke_case_ids = (
        "edge_cases/omat/dimer_near_cutoff",
        "edge_cases/omat/neighbor_cap_320",
        "edge_cases/dynamics/npt-contraction",
    )
    observables: ClassVar[dict[str, ObservableSpec]] = {
        **STATIC_PREDICTION_OBSERVABLES,
        "neighbor_cap_energy": ObservableSpec(
            units="eV",
            comparison="relative_with_floor",
            histogram=HistogramSpec(1e-16, 1e4),
            floor=1e-8,
            report_only=True,
        ),
        "neighbor_cap_forces": ObservableSpec(
            units="eV/A",
            comparison="relative_with_floor",
            histogram=HistogramSpec(1e-16, 1e4),
            floor=1e-6,
        ),
        "neighbor_cap_stress": ObservableSpec(
            units="eV/A^3",
            comparison="relative_with_floor",
            histogram=HistogramSpec(1e-16, 1e4),
            floor=1e-8,
        ),
        "trajectory_energy": ObservableSpec(
            units="eV",
            comparison="relative_with_floor",
            histogram=HistogramSpec(1e-16, 1e4),
            floor=1e-8,
        ),
        "trajectory_volume": ObservableSpec(
            units="A^3", comparison="log_ratio", histogram=HistogramSpec(1e-16, 1e4)
        ),
        "trajectory_temperature": ObservableSpec(
            units="K", comparison="log_ratio", histogram=HistogramSpec(1e-16, 1e4)
        ),
        "dense_transient_energy": ObservableSpec(
            units="eV",
            comparison="relative_with_floor",
            histogram=HistogramSpec(1e-16, 1e4),
            floor=1e-8,
            report_only=True,
        ),
        "dense_transient_temperature": ObservableSpec(
            units="K",
            comparison="log_ratio",
            histogram=HistogramSpec(1e-16, 1e4),
            report_only=True,
        ),
        "trajectory_edge_count": ObservableSpec(
            units="edges",
            comparison="absolute",
            histogram=HistogramSpec(1e-16, 1e6),
            report_only=True,
        ),
    }
    generated_frames = staticmethod(source_frames)

    def cases(self) -> list[CaseSpec]:
        structures = frame_identities(source_frames())
        cases = []
        # Static probes: seven graph shapes, then (after the three MD structures) the
        # neighbor-cap clusters.
        for index in (*range(7), *range(10, len(structures))):
            structure_id, structure_hash = structures[index]
            cases.extend(
                precision_rotation_sweep(
                    case_id=f"edge_cases/omat/{structure_id}",
                    metadata={
                        "kind": "neighbor_cap" if index >= 10 else "correctness",
                        "structure_id": structure_id,
                    },
                    runtime_seed=57 + index,
                    sample_seed=150_042 + 10_000 * index,
                    structure_hash=structure_hash,
                    structure_index=index,
                    **_OMAT,
                )
            )
        for step, structure_index in enumerate((5, 6, 5)):
            structure_id, structure_hash = structures[structure_index]
            cases.append(
                unrotated_case(
                    case_id=f"edge_cases/graph_churn/step-{step:02d}-{structure_id}",
                    metadata={
                        "kind": "graph_churn",
                        "sequence_step": step,
                        "structure_id": structure_id,
                    },
                    runtime_seed=142,
                    structure_hash=structure_hash,
                    structure_index=structure_index,
                    transform_id=f"graph-churn-step-{step:02d}",
                    **_OMAT,
                )
            )
        dynamic_cases = (
            ("npt-contraction", 7, _npt(("contract", 2_000_000.0))),
            ("npt-expansion", 8, _npt(("expand", -2_000_000.0))),
            (
                "npt-cycle",
                7,
                _npt(
                    ("compress", 2_000_000.0),
                    ("expand", -2_000_000.0),
                    ("recompress", 2_000_000.0),
                ),
            ),
            (
                "nvt-dense-transient",
                9,
                {
                    "ensemble": "nvt",
                    "temperature_k": 1_000_000.0,
                    "timestep_fs": 0.5,
                    "friction_per_fs": 0.002,
                    "steps": 200,
                    "sample_interval": 5,
                    "velocity_profile": "inward_center_of_mass",
                },
            ),
        )
        for protocol, structure_index, protocol_metadata in dynamic_cases:
            structure_id, structure_hash = structures[structure_index]
            cases.append(
                unrotated_case(
                    case_id=f"edge_cases/dynamics/{protocol}",
                    metadata={
                        "kind": "dynamics",
                        "protocol": protocol,
                        "structure_id": structure_id,
                        **protocol_metadata,
                    },
                    runtime_seed=42,
                    structure_hash=structure_hash,
                    structure_index=structure_index,
                    transform_id=f"{protocol}-identity",
                    **_OMAT,
                )
            )
        return cases

    def predictor_group(self, case: CaseSpec) -> Hashable:
        # The sparse-dense-sparse sequence must reuse one calculator; its
        # structures share a composition, so a merged MOLE model serves all.
        if case.metadata["kind"] == "graph_churn":
            return "graph-churn"
        return super().predictor_group(case)

    def evaluate(self, calculator, case: SuiteCase) -> dict[str, np.ndarray]:
        kind = case.spec.metadata.get("kind")
        if kind == "neighbor_cap":
            # Separate observables keep the looser limits of the other probes
            # from masking a mishandled cap, which shifts forces and stress far
            # more than FP32 rounding does. Energy is report-only: one scalar
            # per case moves only about as much as FP32 backends differ.
            return {
                f"neighbor_cap_{name}": values
                for name, values in static_predictions(calculator, case).items()
            }
        if kind != "dynamics":
            return static_predictions(calculator, case)
        if case.spec.metadata["ensemble"] == "npt":
            return _run_npt(calculator, case)
        return _run_nvt(calculator, case)
