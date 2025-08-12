from __future__ import annotations

import tempfile
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.build import molecule
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet

from fairchem.core import FAIRChemCalculator, pretrained_mlip

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    "atoms, dynamics, device, steps, tol",
    [
        (
            molecule("H2O"),
            partial(VelocityVerlet, timestep=1 * units.fs),
            "cuda",
            10_000,
            1e-4,
        ),
        (
            molecule("H2O"),
            partial(
                Langevin,
                timestep=0.1 * units.fs,
                temperature_K=400,
                friction=0.001 / units.fs,
            ),
            "cuda",
            10_000,
            1e-4,
        ),
        (
            molecule("H2O"),
            partial(VelocityVerlet, timestep=1 * units.fs),
            "cpu",
            1_000,
            1e-4,
        ),
        (
            molecule("H2O"),
            partial(
                Langevin,
                timestep=0.1 * units.fs,
                temperature_K=400,
                friction=0.001 / units.fs,
            ),
            "cpu",
            1_000,
            1e-4,
        ),
    ],
)
def test_energy_conservation(
    atoms: Atoms,
    dynamics: Callable,
    device: str,
    steps: int,
    tol: float,
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
    calc = FAIRChemCalculator(predictor, task_name="omol")

    atoms.calc = calc

    dyn = dynamics(atoms)
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        trajectory = Trajectory(temp_file.name, "w", atoms)
        dyn.attach(trajectory.write, interval=1)
        dyn.run(steps=steps)

        traj = Trajectory(temp_file.name, "r")
        energies = np.asarray([atoms.get_potential_energy() for atoms in traj])
    assert (
        np.var(energies) < tol
    ), f"Potential energy should be conserved, variance is {np.var(energies)=}. tolerance is {tol=}"
