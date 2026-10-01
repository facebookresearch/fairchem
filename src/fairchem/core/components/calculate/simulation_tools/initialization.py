"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

if TYPE_CHECKING:
    from ase import Atoms


def initialize_momenta(
    atoms: Atoms,
    temperature_K: float,
    seed: int,
    *,
    remove_center_of_mass_momentum: bool = True,
) -> None:
    """
    Initialize Maxwell-Boltzmann momenta reproducibly.

    Args:
        atoms: Structure whose momenta will be replaced.
        temperature_K: Target initialization temperature.
        seed: NumPy random seed recorded with the run.
        remove_center_of_mass_momentum: Remove net linear momentum when true.
    """

    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    if remove_center_of_mass_momentum:
        Stationary(atoms)
    atoms.info["velocity_seed"] = seed
    atoms.info["initial_temperature_K"] = temperature_K
