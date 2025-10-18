from __future__ import annotations

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate import pretrained_mlip
from fairchem.lammps.lammps_fc import run_lammps_with_fairchem


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


def run_ase_npt():
    """Run ASE NPT-like using a Berendsen barostat approximation via NPT wrapper.

    ASE doesn't provide a direct NPT integrator in the core; here we mimic
    an NPT run by coupling to a thermostat and using the `Parrinello-Rahman`
    style barostat if available in user's setup. For portability in tests we
    instead run VelocityVerlet with a simple rescaling of the cell using the
    `ase.constraints` is out of scope — this is a lightweight smoke test to
    exercise the predictor through an NPT LAMMPS run for comparison.
    """
    atoms = bulk("C", "fcc", a=3.567, cubic=True)
    atoms = atoms.repeat((2, 2, 2))
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
    atoms.calc = FAIRChemCalculator(predictor, task_name="omat")
    initial_temperature_K = 300.0
    np.random.seed(12345)
    MaxwellBoltzmannDistribution(atoms, temperature_K=initial_temperature_K)
    # Use ASE's NPT integrator which couples Nose-Hoover thermostat and
    # barostat (Parrinello-Rahman style) and updates the cell. We pick
    # thermostat/barostat time constants that map to LAMMPS fix npt's
    # Tdamp/Pdamp (units: ps here for LAMMPS). ASE's API expects time in
    # fs via ase.units, so use 0.1 ps = 100 fs as the thermostat time constant.
    tdamp = 0.1  # ps (thermostat damping time for LAMMPS mapping)
    pdamp = 1.0  # ps (barostat damping time for LAMMPS mapping)

    # Convert ps -> fs for ASE NPT ttime/pfactor which expect time in fs units
    tdamp_fs = tdamp * 1000.0 * units.fs
    pdamp_fs = pdamp * 1000.0 * units.fs

    # ASE NPT takes timestep in ASE units (seconds via units.fs) and temperature_K
    # externalstress is pressure in eV/Å^3 or a scalar (here 0 means 0 pressure)
    dyn = IsotropicMTKNPT(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=300,
        pressure_au=0.0 * units.bar,
        tdamp=tdamp_fs,
        pdamp=pdamp_fs,
    )

    def print_thermo(a=atoms):
        ekin = a.get_kinetic_energy()
        epot = a.get_potential_energy()
        etot = ekin + epot
        temp = ekin / (1.5 * units.kB) / len(a)
        vol = a.get_volume()
        print(
            f"Step: {dyn.get_number_of_steps()}, Temp: {temp:.2f} K, "
            f"Ekin: {ekin:.4f} eV, Epot: {epot:.4f} eV, Etot: {etot:.4f} eV, Vol: {vol:.4f} Å^3"
        )

    dyn.attach(print_thermo, interval=1)
    dyn.run(100)
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


@pytest.mark.gpu()
def test_ase_vs_lammps_npt():
    ase_kinetic, ase_pot = run_ase_npt()
    lammps_kinetic, lammps_pot = run_lammps("tests/lammps/lammps_npt.file")
    assert np.isclose(ase_kinetic, lammps_kinetic, rtol=0.5)
    assert np.isclose(ase_pot, lammps_pot, rtol=0.5)


@pytest.mark.xfail(
    reason="This is more demo purposes, need to configure the right parameters for ASE langevin to match lammps"
)
@pytest.mark.gpu()
def test_ase_vs_lammps_langevin():
    ase_kinetic, ase_pot = run_ase_langevin()
    lammps_kinetic, lammps_pot = run_lammps("tests/lammps/lammps_langevin.file")
    assert np.isclose(ase_kinetic, lammps_kinetic, rtol=1e-4)
    assert np.isclose(ase_pot, lammps_pot, rtol=1e-4)
