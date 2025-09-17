from __future__ import annotations

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from fairchem.lammps.lammps_fc import run_lammps_with_fairchem

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate import pretrained_mlip


def run_ase_langevin():
    atoms = bulk("C", "fcc", a=3.567, cubic=True)
    atoms = atoms.repeat((2, 2, 2))
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
    atoms.calc = FAIRChemCalculator(predictor, task_name="omat")
    initial_temperature_K = 300.0
    np.random.seed(12345)
    MaxwellBoltzmannDistribution(atoms, initial_temperature_K * units.kB)
    dyn = Langevin(
        atoms,
        timestep=1 * units.fs,
        temperature_K=300,
        friction=0.1 / units.fs,
    )

    def print_thermo(a=atoms):
        """Function to print thermo info to stdout."""
        ekin = a.get_kinetic_energy()
        epot = a.get_potential_energy()
        etot = ekin + epot
        temp = ekin / (1.5 * units.kB) / len(a)
        print(
            f"Step: {dyn.get_number_of_steps()}, Temp: {temp:.2f} K, "
            f"Ekin: {ekin:.4f} eV, Epot: {epot:.4f} eV, Etot: {etot:.4f} eV"
        )

    dyn.attach(print_thermo, interval=1)  # Print thermo every 1000 steps
    dyn.run(100)
    # return the kin and pot energy for comparison
    return atoms.get_kinetic_energy(), atoms.get_potential_energy()


def run_ase_nve():
    atoms = bulk("C", "fcc", a=3.567, cubic=True)
    atoms = atoms.repeat((2, 2, 2))
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
    atoms.calc = FAIRChemCalculator(predictor, task_name="omat")
    initial_temperature_K = 300.0
    np.random.seed(12345)
    MaxwellBoltzmannDistribution(atoms, initial_temperature_K * units.kB)
    dyn = VelocityVerlet(
        atoms, timestep=units.fs, trajectory="nve.traj", logfile="nve.log"
    )

    def print_thermo(a=atoms):
        """Function to print thermo info to stdout."""
        ekin = a.get_kinetic_energy()
        epot = a.get_potential_energy()
        etot = ekin + epot
        temp = ekin / (1.5 * units.kB) / len(a)
        print(
            f"Step: {dyn.get_number_of_steps()}, Temp: {temp:.2f} K, "
            f"Ekin: {ekin:.4f} eV, Epot: {epot:.4f} eV, Etot: {etot:.4f} eV"
        )

    dyn.attach(print_thermo, interval=1)  # Print thermo every 1000 steps
    dyn.run(100)
    # return the kin and pot energy for comparison
    return atoms.get_kinetic_energy(), atoms.get_potential_energy()


def run_lammps(input_file):
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
    lmp = run_lammps_with_fairchem(predictor, input_file, "omat")
    return lmp.last_thermo()["KinEng"], lmp.last_thermo()["PotEng"]


@pytest.mark.gpu()
def test_ase_vs_lammps_nve():
    ase_kinetic, ase_pot = run_ase_nve()
    lammps_kinetic, lammps_pot = run_lammps("tests/lammps/lammps_nve.file")
    assert np.isclose(ase_kinetic, lammps_kinetic, rtol=0.1)
    assert np.isclose(ase_pot, lammps_pot, rtol=0.1)


@pytest.mark.xfail(
    reason="This is more demo purposes, need to configure the right parameters for ASE langevin to match lammps"
)
@pytest.mark.gpu()
def test_ase_vs_lammps_langevin():
    ase_kinetic, ase_pot = run_ase_langevin()
    lammps_kinetic, lammps_pot = run_lammps("tests/lammps/lammps_langevin.file")
    assert np.isclose(ase_kinetic, lammps_kinetic, rtol=1e-4)
    assert np.isclose(ase_pot, lammps_pot, rtol=1e-4)
