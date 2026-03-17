"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import units
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.verlet import VelocityVerlet

if TYPE_CHECKING:
    from ase import Atoms
    from ase.md.md import MolecularDynamics


@dataclass
class VelocityVerletThermostat:
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
class NoseHooverNVT:
    """
    Nose-Hoover chain NVT thermostat.
    """

    temperature_K: float
    tdamp_fs: float

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return NoseHooverChainNVT(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            tdamp=self.tdamp_fs * units.fs,
        )

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        thermostat = dyn._thermostat
        return {
            "class_name": "NoseHooverNVT",
            "eta": thermostat._eta.tolist(),
            "p_eta": thermostat._p_eta.tolist(),
        }

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        thermostat = dyn._thermostat
        thermostat._eta = np.array(state["eta"])
        thermostat._p_eta = np.array(state["p_eta"])


@dataclass
class BussiThermostat:
    """
    Bussi stochastic velocity rescaling thermostat.
    """

    temperature_K: float
    taut_fs: float

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return Bussi(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            taut=self.taut_fs * units.fs,
        )

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        rng_state = dyn.rng.get_state()
        return {
            "class_name": "BussiThermostat",
            "rng_state": {
                "algorithm": rng_state[0],
                "keys": rng_state[1].tolist(),
                "pos": int(rng_state[2]),
                "has_gauss": int(rng_state[3]),
                "cached_gaussian": float(rng_state[4]),
            },
            "transferred_energy": float(dyn.transferred_energy),
        }

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
class LangevinThermostat:
    """
    Langevin stochastic dynamics thermostat.
    """

    temperature_K: float
    friction_per_fs: float

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return Langevin(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            friction=self.friction_per_fs / units.fs,
        )

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        rng_state = dyn.rng.get_state()
        return {
            "class_name": "LangevinThermostat",
            "rng_state": {
                "algorithm": rng_state[0],
                "keys": rng_state[1].tolist(),
                "pos": int(rng_state[2]),
                "has_gauss": int(rng_state[3]),
                "cached_gaussian": float(rng_state[4]),
            },
        }

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
