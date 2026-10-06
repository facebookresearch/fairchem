"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  Deterministic one-step ASE/LAMMPS NVE integration parity.
Models: uma-s-1p1.
CI:     test_lammps_gpu.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.lammps import convert
from ase.io import write
from ase.md.verlet import VelocityVerlet
from fairchem.lammps.lammps_fc import run_lammps_with_fairchem

from fairchem.core import FAIRChemCalculator
from tests.conftest import get_predict_unit_for_test

pytest.importorskip("lammps")

pytestmark = [
    pytest.mark.gpu(),
    pytest.mark.pretrained("uma-s-1p1"),
]


def test_one_step_nve_matches_ase(pretrained_checkpoint, tmp_path):
    """
    Compare one velocity-Verlet step from an identical deterministic state.

    NPT and Langevin trajectories are not compared because ASE and LAMMPS use
    different thermostat, barostat, and random-number implementations.
    """
    timestep_fs = 0.5
    initial = bulk("C", "fcc", a=3.8, cubic=True)
    initial.set_velocities(
        np.array(
            [
                [-0.012, 0.004, 0.008],
                [0.006, -0.010, 0.002],
                [0.009, 0.007, -0.011],
                [-0.003, -0.001, 0.001],
            ]
        )
    )

    predictor = get_predict_unit_for_test(pretrained_checkpoint, device="cuda")
    ase_atoms = initial.copy()
    ase_atoms.calc = FAIRChemCalculator(predictor, task_name="omat")
    VelocityVerlet(ase_atoms, timestep=timestep_fs * units.fs).run(1)

    data_path = tmp_path / "system.data"
    input_path = tmp_path / "one_step_nve.in"
    write(
        data_path,
        initial,
        format="lammps-data",
        atom_style="atomic",
        masses=True,
        velocities=True,
        units="metal",
    )
    input_path.write_text(
        "\n".join(
            [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                "atom_modify sort 0 0.0",
                f"read_data {data_path}",
                f"timestep {timestep_fs / 1000.0}",
                "fix integrator all nve",
                "thermo_style custom step pe ke etotal",
                "run 1",
                "",
            ]
        )
    )

    lmp = run_lammps_with_fairchem(
        predictor,
        str(input_path),
        "omat",
        cmdargs=["-nocite", "-log", "none", "-screen", "none"],
    )
    try:
        atom_count = int(lmp.get_natoms())
        atom_ids = lmp.numpy.extract_atom("id")[:atom_count].copy()
        order = np.argsort(atom_ids)
        assert np.array_equal(atom_ids[order], np.arange(1, atom_count + 1))

        lammps_positions = convert(
            lmp.numpy.extract_atom("x")[:atom_count].copy()[order],
            "distance",
            "metal",
            "ASE",
        )
        lammps_velocities = convert(
            lmp.numpy.extract_atom("v")[:atom_count].copy()[order],
            "velocity",
            "metal",
            "ASE",
        )
        lammps_forces = convert(
            lmp.numpy.extract_atom("f")[:atom_count].copy()[order],
            "force",
            "metal",
            "ASE",
        )
        lammps_potential_energy = convert(
            lmp.get_thermo("pe"), "energy", "metal", "ASE"
        )
        lammps_kinetic_energy = convert(lmp.get_thermo("ke"), "energy", "metal", "ASE")
    finally:
        lmp.close()

    np.testing.assert_allclose(
        lammps_positions, ase_atoms.get_positions(), rtol=0.0, atol=2.0e-6
    )
    np.testing.assert_allclose(
        lammps_velocities, ase_atoms.get_velocities(), rtol=0.0, atol=2.0e-6
    )
    np.testing.assert_allclose(
        lammps_forces, ase_atoms.get_forces(), rtol=0.0, atol=1.0e-5
    )
    assert lammps_potential_energy == pytest.approx(
        ase_atoms.get_potential_energy(), abs=1.0e-5
    )
    assert lammps_kinetic_energy == pytest.approx(
        ase_atoms.get_kinetic_energy(), abs=1.0e-7
    )
