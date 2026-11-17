"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from ase import Atoms, units
from ase.io.jsonio import decode
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import Stationary, ZeroRotation, thermalize_momenta

from fairchem.core.components.calculate._calculate_runner import CalculateRunner

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


_AMU_PER_A3_TO_G_CM3 = 1.6605390666
_INPUT_SCHEMA_VERSION = 1


def _system_seed(seed: int, system_id: str) -> int:
    """Derive a stable, distinct NumPy seed for one liquid system."""
    digest = hashlib.sha256(f"{seed}:{system_id}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="little")


def load_liquid_density_inputs(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate the liquid-density JSON input catalog."""
    with Path(path).open() as handle:
        payload = json.load(handle)

    if payload.get("schema_version") != _INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"liquid-density input schema_version must be {_INPUT_SCHEMA_VERSION}"
        )
    raw_systems = payload.get("systems")
    if not isinstance(raw_systems, list) or not raw_systems:
        raise ValueError("liquid-density inputs must contain a non-empty systems list")

    systems: dict[str, dict[str, Any]] = {}
    for entry in raw_systems:
        system_id = _validate_input_entry(entry)
        if system_id in systems:
            raise ValueError(f"duplicate liquid-density system_id: {system_id}")
        systems[system_id] = entry
    return dict(sorted(systems.items()))


def atoms_from_liquid_density_entry(entry: dict[str, Any]) -> Atoms:
    """Decode a detached ASE Atoms instance from a validated catalog entry."""
    atoms = decode(entry["atoms"])
    if not isinstance(atoms, Atoms):
        raise TypeError("liquid-density atoms payload did not decode to ase.Atoms")
    return atoms


def _validate_input_entry(entry: Any) -> str:
    if not isinstance(entry, dict):
        raise ValueError("each liquid-density system must be an object")
    required = {
        "system_id",
        "atoms",
        "smiles",
        "number_of_molecules",
        "number_of_atoms",
        "experimental_temperature_K",
        "experimental_density_g_cm3",
    }
    missing = required - set(entry)
    if missing:
        raise ValueError(f"liquid-density system missing fields: {sorted(missing)}")

    system_id = entry["system_id"]
    if not isinstance(system_id, str) or not system_id:
        raise ValueError("liquid-density system_id must be a non-empty string")
    smiles = entry["smiles"]
    if not isinstance(smiles, str) or not smiles or "." in smiles:
        raise ValueError(f"{system_id}: smiles must describe one molecule")

    number_of_molecules = entry["number_of_molecules"]
    number_of_atoms = entry["number_of_atoms"]
    if (
        not isinstance(number_of_molecules, int)
        or isinstance(number_of_molecules, bool)
        or number_of_molecules <= 0
    ):
        raise ValueError(f"{system_id}: number_of_molecules must be positive")
    if (
        not isinstance(number_of_atoms, int)
        or isinstance(number_of_atoms, bool)
        or number_of_atoms <= 0
    ):
        raise ValueError(f"{system_id}: number_of_atoms must be positive")

    temperature = float(entry["experimental_temperature_K"])
    density = float(entry["experimental_density_g_cm3"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"{system_id}: experimental temperature must be positive")
    if not math.isfinite(density) or density <= 0:
        raise ValueError(f"{system_id}: experimental density must be positive")

    atoms = atoms_from_liquid_density_entry(entry)
    if len(atoms) != number_of_atoms:
        raise ValueError(
            f"{system_id}: number_of_atoms={number_of_atoms} does not match "
            f"decoded structure ({len(atoms)})"
        )
    metadata_checks = {
        "smiles": smiles,
        "number_of_molecules": number_of_molecules,
        "experimental_temperature(K)": temperature,
        "experimental_density(g/cc)": density,
    }
    for key, expected in metadata_checks.items():
        if key not in atoms.info:
            continue
        actual = atoms.info[key]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{system_id}: atoms.info[{key!r}] disagrees")
        elif actual != expected:
            raise ValueError(f"{system_id}: atoms.info[{key!r}] disagrees")
    return system_id


class LiquidDensityRunner(CalculateRunner):
    """Run long NPT liquid-density simulations over an array-job catalog."""

    result_glob_pattern: ClassVar[str] = "liquid_density_*-*.json.gz"

    def __init__(
        self,
        calculator: Calculator,
        input_data_path: str,
        timestep_fs: float = 1.0,
        total_time_ps: float = 1000.0,
        equilibration_fraction: float = 0.3,
        sample_interval: int = 100,
        thermostat_damping_steps: int = 100,
        barostat_damping_steps: int = 1000,
        pressure_bar: float = 1.01325,
        seed: int = 42,
    ):
        self.systems = load_liquid_density_inputs(input_data_path)
        self.system_ids = list(self.systems)
        self.timestep_fs = float(timestep_fs)
        self.total_time_ps = float(total_time_ps)
        self.equilibration_fraction = float(equilibration_fraction)
        self.sample_interval = sample_interval
        self.thermostat_damping_steps = thermostat_damping_steps
        self.barostat_damping_steps = barostat_damping_steps
        self.pressure_bar = float(pressure_bar)
        self.seed = seed
        self.total_steps = self._validate_protocol()
        super().__init__(calculator=calculator, input_data=self.system_ids)

    def _validate_protocol(self) -> int:
        if not math.isfinite(self.timestep_fs) or self.timestep_fs <= 0:
            raise ValueError("timestep_fs must be positive")
        if not math.isfinite(self.total_time_ps) or self.total_time_ps <= 0:
            raise ValueError("total_time_ps must be positive")
        if not 0 <= self.equilibration_fraction < 1:
            raise ValueError("equilibration_fraction must be in [0, 1)")
        for name, value in (
            ("sample_interval", self.sample_interval),
            ("thermostat_damping_steps", self.thermostat_damping_steps),
            ("barostat_damping_steps", self.barostat_damping_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.pressure_bar) or self.pressure_bar <= 0:
            raise ValueError("pressure_bar must be positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")

        raw_steps = self.total_time_ps * 1000.0 / self.timestep_fs
        total_steps = round(raw_steps)
        if not math.isclose(raw_steps, total_steps, rel_tol=0, abs_tol=1e-9):
            raise ValueError(
                "total_time_ps must be an integral number of timestep_fs steps"
            )
        if total_steps <= 0:
            raise ValueError("liquid-density protocol must contain at least one step")
        return total_steps

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> list[dict[str, Any]]:
        if not 0 <= job_num < num_jobs:
            raise ValueError("job_num must be in [0, num_jobs)")
        if num_jobs > len(self.system_ids):
            raise ValueError("num_jobs cannot exceed the number of liquid systems")

        chunk = np.array_split(self.system_ids, num_jobs)[job_num]
        return [self._run_system(system_id) for system_id in chunk]

    def _run_system(self, system_id: str) -> dict[str, Any]:
        entry = self.systems[system_id]
        common = {
            "system_id": system_id,
            "smiles": entry["smiles"],
            "number_of_molecules": entry["number_of_molecules"],
            "number_of_atoms": entry["number_of_atoms"],
            "target_temperature_K": float(entry["experimental_temperature_K"]),
            "requested_time_ps": self.total_time_ps,
            "timestep_fs": self.timestep_fs,
        }
        try:
            atoms = atoms_from_liquid_density_entry(entry)
            atoms.calc = self.calculator
            rng = np.random.RandomState(_system_seed(self.seed, system_id))
            temperature = common["target_temperature_K"]
            thermalize_momenta(
                atoms,
                temperature_K=temperature,
                exact_temperature=True,
                rng=rng,
            )
            Stationary(atoms)
            ZeroRotation(atoms)

            timestep = self.timestep_fs * units.fs
            dynamics = IsotropicMTKNPT(
                atoms,
                timestep=timestep,
                temperature_K=temperature,
                pressure_au=self.pressure_bar * units.bar,
                tdamp=self.thermostat_damping_steps * timestep,
                pdamp=self.barostat_damping_steps * timestep,
            )
            equilibration_steps = round(self.equilibration_fraction * self.total_steps)
            production_steps = self.total_steps - equilibration_steps
            dynamics.run(equilibration_steps)

            mass_amu = float(np.sum(atoms.get_masses()))
            densities: list[float] = []
            temperatures: list[float] = []

            def record() -> None:
                try:
                    volume = float(atoms.get_volume())
                except (ValueError, RuntimeError) as exc:
                    raise ValueError(f"{system_id}: invalid cell volume") from exc
                if not math.isfinite(volume) or volume <= 0:
                    raise ValueError(f"{system_id}: invalid cell volume {volume}")
                densities.append(mass_amu / volume * _AMU_PER_A3_TO_G_CM3)
                temperatures.append(float(atoms.get_temperature()))

            record()
            dynamics.attach(record, interval=self.sample_interval)
            dynamics.run(production_steps)
            if not densities:
                raise RuntimeError("no production density samples were collected")

            return {
                **common,
                "status": "ok",
                "predicted_density_g_cm3": float(np.mean(densities)),
                "density_std_g_cm3": float(np.std(densities)),
                "mean_temperature_K": float(np.mean(temperatures)),
                "completed_time_ps": dynamics.get_time() / (1000.0 * units.fs),
                "num_samples": len(densities),
                "error": None,
                "traceback": None,
            }
        except Exception as exc:
            return {
                **common,
                "status": "failed",
                "predicted_density_g_cm3": None,
                "density_std_g_cm3": None,
                "mean_temperature_K": None,
                "completed_time_ps": None,
                "num_samples": 0,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    def write_results(
        self,
        results: list[dict[str, Any]],
        results_dir: str,
        job_num: int = 0,
        num_jobs: int = 1,
    ) -> None:
        os.makedirs(results_dir, exist_ok=True)
        path = Path(results_dir) / f"liquid_density_{num_jobs}-{job_num}.json.gz"
        with gzip.open(path, "wt") as handle:
            json.dump(results, handle)

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        # Mid-trajectory MTK state is not checkpointed. A preempted array element
        # is intentionally restarted from its initial structure and deterministic seed.
        return False
