"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import units
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.verlet import VelocityVerlet
from monty.json import jsanitize

if TYPE_CHECKING:
    from ase import Atoms
    from ase.md.md import MolecularDynamics


class Thermostat(ABC):
    """
    Abstract base class defining the interface for MD thermostats.

    All thermostats must implement three methods:
        - build: construct an ASE MolecularDynamics integrator
        - save_state: serialize thermostat-specific state for checkpointing
        - restore_state: restore thermostat state from a checkpoint
    """

    @abstractmethod
    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        """
        Build and return an ASE MolecularDynamics integrator.

        Args:
            atoms: The atomic system to simulate.
            timestep_fs: Integration timestep in femtoseconds.

        Returns:
            An ASE MolecularDynamics object.
        """

    @abstractmethod
    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        """
        Save thermostat-specific state for checkpointing.

        Args:
            dyn: The active MolecularDynamics integrator.

        Returns:
            A JSON-serializable dict of thermostat state.
        """

    @abstractmethod
    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        """
        Restore thermostat state from a checkpoint.

        Args:
            dyn: The active MolecularDynamics integrator.
            state: The state dict previously returned by save_state.
        """


@dataclass
class VelocityVerletThermostat(Thermostat):
    """
    NVE dynamics (no thermostat).
    """

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return VelocityVerlet(atoms=atoms, timestep=timestep_fs * units.fs)

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        return {"class_name": "VelocityVerletThermostat"}

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        pass


@dataclass
class NoseHooverNVT(Thermostat):
    """
    Nose-Hoover chain NVT thermostat.
    """

    temperature_K: float  # Kelvin
    tdamp_fs: float  # femtoseconds

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return NoseHooverChainNVT(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            tdamp=self.tdamp_fs * units.fs,
        )

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        thermostat = dyn._thermostat
        return jsanitize(
            {
                "class_name": "NoseHooverNVT",
                "eta": thermostat._eta,
                "p_eta": thermostat._p_eta,
            }
        )

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        thermostat = dyn._thermostat
        thermostat._eta = np.array(state["eta"])
        thermostat._p_eta = np.array(state["p_eta"])


@dataclass
class BussiThermostat(Thermostat):
    """
    Bussi stochastic velocity rescaling thermostat.
    """

    temperature_K: float  # Kelvin
    taut_fs: float  # femtoseconds

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return Bussi(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            taut=self.taut_fs * units.fs,
        )

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        rng_state = dyn.rng.get_state()
        return jsanitize(
            {
                "class_name": "BussiThermostat",
                "rng_state": {
                    "algorithm": rng_state[0],
                    "keys": rng_state[1],
                    "pos": rng_state[2],
                    "has_gauss": rng_state[3],
                    "cached_gaussian": rng_state[4],
                },
                "transferred_energy": dyn.transferred_energy,
            }
        )

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        rng = state["rng_state"]
        dyn.rng.set_state(
            (
                rng["algorithm"],
                np.array(rng["keys"], dtype=np.uint32),
                rng["pos"],
                rng["has_gauss"],
                rng["cached_gaussian"],
            )
        )
        if "transferred_energy" in state:
            dyn.transferred_energy = state["transferred_energy"]


@dataclass
class LangevinThermostat(Thermostat):
    """
    Langevin stochastic dynamics thermostat.
    """

    temperature_K: float  # Kelvin
    friction_per_fs: float  # 1/femtoseconds

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return Langevin(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            friction=self.friction_per_fs / units.fs,
        )

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        rng_state = dyn.rng.get_state()
        return jsanitize(
            {
                "class_name": "LangevinThermostat",
                "rng_state": {
                    "algorithm": rng_state[0],
                    "keys": rng_state[1],
                    "pos": rng_state[2],
                    "has_gauss": rng_state[3],
                    "cached_gaussian": rng_state[4],
                },
            }
        )

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        rng = state["rng_state"]
        dyn.rng.set_state(
            (
                rng["algorithm"],
                np.array(rng["keys"], dtype=np.uint32),
                rng["pos"],
                rng["has_gauss"],
                rng["cached_gaussian"],
            )
        )


@dataclass
class BerendsenNPT(Thermostat):
    """
    Berendsen NPT thermostat/barostat for constant pressure simulations.
    """

    temperature_K: float
    pressure_bar: float = 1.0
    taut_fs: float = 5.0
    taup_fs: float = 500.0
    compressibility_bar: float = 5e-7

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return NPTBerendsen(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            pressure_au=self.pressure_bar * units.bar,
            taut=self.taut_fs * units.fs,
            taup=self.taup_fs * units.fs,
            compressibility_au=self.compressibility_bar / units.bar,
        )

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        return {"class_name": "BerendsenNPT"}

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        pass
