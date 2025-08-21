"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.build import add_adsorbate, bulk, fcc111, molecule
from ase.md.langevin import Langevin
from ase.optimize import BFGS

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate.ase_calculator import (
    AllZeroUnitCellError,
    MixedPBCError,
)
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings, UMATask
from fairchem.core.units.mlip_unit.predict import ParallelMLIPPredictUnit

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit import MLIPPredictUnit

from fairchem.core.calculate import pretrained_mlip


@pytest.fixture(scope="module", params=pretrained_mlip.available_models)
def mlip_predict_unit(request) -> MLIPPredictUnit:
    return pretrained_mlip.get_predict_unit(request.param)


@pytest.fixture(scope="module")
def all_calculators(mlip_predict_unit):
    """Generate calculators for all available datasets in the mlip predict unit"""

    def _calc_generator():
        for dataset in mlip_predict_unit.dataset_to_tasks:
            # check that all single task models load without specifying task name
            task_name = dataset if len(mlip_predict_unit.dataset_to_tasks) > 1 else None
            yield FAIRChemCalculator(mlip_predict_unit, task_name=task_name)

    return _calc_generator


@pytest.fixture(scope="module")
def omol_calculators(request):
    def _calc_generator():
        for model_name in pretrained_mlip.available_models:
            predict_unit = pretrained_mlip.get_predict_unit(model_name)
            if "omol" in predict_unit.dataset_to_tasks:
                yield FAIRChemCalculator(predict_unit, task_name="omol")

    return _calc_generator


@pytest.fixture()
def slab_atoms() -> Atoms:
    atoms = fcc111("Pt", size=(2, 2, 5), vacuum=10.0, periodic=True)
    add_adsorbate(atoms, "O", height=1.2, position="fcc")
    atoms.pbc = True
    return atoms


@pytest.fixture()
def bulk_atoms() -> Atoms:
    return bulk("Fe", "bcc", a=2.87).repeat((2, 2, 2))


@pytest.fixture()
def aperiodic_atoms() -> Atoms:
    return molecule("H2O")


@pytest.fixture()
def periodic_h2o_atoms() -> Atoms:
    """Create a periodic box of H2O molecules."""
    atoms = molecule("H2O")
    atoms.set_cell([100.0, 100.0, 100.0])  # Define a cubic cell
    atoms.set_pbc(True)  # Enable periodic boundary conditions
    atoms = atoms.repeat((2, 2, 2))  # Create a 2x2x2 periodic box
    return atoms


@pytest.fixture()
def large_bulk_atoms() -> Atoms:
    """Create a bulk system with approximately 1000 atoms."""
    return bulk("Fe", "bcc", a=2.87).repeat((10, 10, 10))  # 10x10x10 unit cell


def test_calculator_from_checkpoint():
    calc = FAIRChemCalculator.from_model_checkpoint(
        pretrained_mlip.available_models[0], task_name="omol"
    )
    assert "energy" in calc.implemented_properties
    assert "forces" in calc.implemented_properties


def test_calculator_with_task_names_matches_uma_task(aperiodic_atoms):
    calc_omol = FAIRChemCalculator.from_model_checkpoint(
        pretrained_mlip.available_models[0], task_name="omol"
    )
    calc_omol_uma_task = FAIRChemCalculator.from_model_checkpoint(
        pretrained_mlip.available_models[0], task_name=UMATask.OMOL
    )
    calculators = [calc_omol, calc_omol_uma_task]
    energies = []
    for calc in calculators:
        atoms = aperiodic_atoms
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        energies.append(energy)
    np.testing.assert_allclose(energies[0], energies[1])


def test_no_task_name_single_task():
    for model_name in pretrained_mlip.available_models:
        predict_unit = pretrained_mlip.get_predict_unit(model_name)
        datasets = list(predict_unit.dataset_to_tasks.keys())
        if len(datasets) == 1:
            calc = FAIRChemCalculator(predict_unit)
            assert calc.task_name == datasets[0]


def test_calculator_unknown_task_raises_error():
    with pytest.raises(AssertionError):
        FAIRChemCalculator.from_model_checkpoint(
            pretrained_mlip.available_models[0], task_name="ommmmmol"
        )


@pytest.mark.gpu()
def test_calculator_setup(all_calculators):
    for calc in all_calculators():
        implemented_properties = ["energy", "forces"]
        datasets = list(calc.predictor.dataset_to_tasks.keys())

        # all conservative UMA checkpoints should support E/F/S!
        if (
            not calc.predictor.direct_forces
            and len(datasets) > 1
            or calc.task_name != "omol"
        ):
            print(len(datasets), calc.task_name)
            implemented_properties.append("stress")

        assert all(
            prop in calc.implemented_properties for prop in implemented_properties
        )


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "atoms_fixture",
    [
        "slab_atoms",
        "bulk_atoms",
        "aperiodic_atoms",
        "periodic_h2o_atoms",
    ],
)
def test_energy_calculation(request, atoms_fixture, all_calculators):
    for calc in all_calculators():
        atoms = request.getfixturevalue(atoms_fixture)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)


@pytest.mark.gpu()
def test_relaxation_final_energy(slab_atoms, mlip_predict_unit):
    datasets = list(mlip_predict_unit.dataset_to_tasks.keys())
    calc = FAIRChemCalculator(
        mlip_predict_unit,
        task_name=datasets[0],
    )

    slab_atoms.calc = calc
    initial_energy = slab_atoms.get_potential_energy()
    assert isinstance(initial_energy, float)

    opt = BFGS(slab_atoms)
    opt.run(fmax=0.05, steps=100)
    final_energy = slab_atoms.get_potential_energy()
    assert isinstance(final_energy, float)


@pytest.mark.gpu()
@pytest.mark.parametrize("inference_settings", ["default", "turbo"])
def test_calculator_configurations(inference_settings, slab_atoms):
    # turbo mode requires compilation and needs to reset here
    if inference_settings == "turbo":
        torch.compiler.reset()

    predict_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1", inference_settings=inference_settings
    )
    datasets = list(predict_unit.dataset_to_tasks.keys())
    calc = FAIRChemCalculator(
        predict_unit,
        task_name=datasets[0],
    )
    slab_atoms.calc = calc
    assert predict_unit.model.module.otf_graph is True
    # Test energy calculation
    energy = slab_atoms.get_potential_energy()
    assert isinstance(energy, float)

    forces = slab_atoms.get_forces()
    assert isinstance(forces, np.ndarray)

    if "stress" in calc.implemented_properties:
        stress = slab_atoms.get_stress()
        assert isinstance(stress, np.ndarray)


@pytest.mark.gpu()
def test_large_bulk_system(large_bulk_atoms):
    """Test a bulk system with 1000 atoms using the small model."""
    predict_unit = pretrained_mlip.get_predict_unit("uma-s-1", device="cuda")
    calc = FAIRChemCalculator(predict_unit, task_name="omat")
    large_bulk_atoms.calc = calc

    # Test energy calculation
    energy = large_bulk_atoms.get_potential_energy()
    assert isinstance(energy, float)

    # Test forces calculation
    forces = large_bulk_atoms.get_forces()
    assert isinstance(forces, np.ndarray)


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "pbc",
    [
        (True, True, True),
        (False, False, False),
        (True, False, True),
        (False, True, False),
        (True, True, False),
    ],
)
def test_mixed_pbc_behavior(pbc, aperiodic_atoms, all_calculators):
    """Test guess_pbc behavior"""
    pbc = np.array(pbc)
    aperiodic_atoms.pbc = pbc
    if np.all(pbc):
        aperiodic_atoms.cell = [100.0, 100.0, 100.0]

    for calc in all_calculators():
        if np.any(aperiodic_atoms.pbc) and not np.all(aperiodic_atoms.pbc):
            with pytest.raises(MixedPBCError):
                aperiodic_atoms.calc = calc
                aperiodic_atoms.get_potential_energy()
        else:
            aperiodic_atoms.calc = calc
            energy = aperiodic_atoms.get_potential_energy()
            assert isinstance(energy, float)


@pytest.mark.gpu()
def test_error_for_pbc_with_zero_cell(aperiodic_atoms, all_calculators):
    """Test error raised when pbc=True but atoms.cell is zero."""
    aperiodic_atoms.pbc = True  # Set PBC to True

    for calc in all_calculators():
        with pytest.raises(AllZeroUnitCellError):
            aperiodic_atoms.calc = calc
            aperiodic_atoms.get_potential_energy()


@pytest.mark.gpu()
def test_omol_missing_spin_charge_logs_warning(
    periodic_h2o_atoms, omol_calculators, caplog
):
    """Test that missing spin/charge in atoms.info logs a warning when task_name='omol'."""

    for calc in omol_calculators():
        periodic_h2o_atoms.calc = calc

        with caplog.at_level(logging.WARNING):
            _ = periodic_h2o_atoms.get_potential_energy()

        assert "charge is not set in atoms.info" in caplog.text
        assert "spin multiplicity is not set in atoms.info" in caplog.text


@pytest.mark.gpu()
def test_omol_energy_diff_for_charge_and_spin(aperiodic_atoms, omol_calculators):
    """Test that energy differs for H2O molecule with different charge and spin_multiplicity."""

    for calc in omol_calculators():
        # Test all combinations of charge and spin
        charges = [0, 1, -1]
        spins = [0, 1, 2]
        energy_results = {}

        for charge in charges:
            for spin in spins:
                aperiodic_atoms.info["charge"] = charge
                aperiodic_atoms.info["spin"] = spin
                aperiodic_atoms.calc = calc
                energy = aperiodic_atoms.get_potential_energy()
                energy_results[(charge, spin)] = energy

        # Ensure all combinations produce unique energies
        energy_values = list(energy_results.values())
        assert len(energy_values) == len(
            set(energy_values)
        ), "Energy values are not unique for different charge/spin combinations"


def test_single_atom_systems():
    """Test a system with a single atom. Single atoms do not currently use the model."""
    predict_unit = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")

    for at_num in range(1, 84):
        atom = Atoms([at_num], positions=[(0.0, 0.0, 0.0)])
        atom.info["charge"] = 0
        atom.info["spin"] = 3

        for task_name in ("omat", "omol", "oc20"):
            calc = FAIRChemCalculator(predict_unit, task_name=task_name)
            atom.calc = calc
            # Test energy calculation
            energy = atom.get_potential_energy()
            assert isinstance(energy, float)

            # Test forces are 0.0
            forces = atom.get_forces()
            assert (forces == 0.0).all()


def test_single_atom_system_errors():
    """Test that a charged system with a single atom does not work."""
    predict_unit = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
    calc = FAIRChemCalculator(predict_unit, task_name="omol")

    atom = Atoms("C", positions=[(0.0, 0.0, 0.0)])
    atom.calc = calc
    atom.info["charge"] = -1
    atom.info["spin"] = 4

    with pytest.raises(ValueError):
        atom.get_potential_energy()


@pytest.mark.gpu()
@pytest.mark.skip(
    reason="the wigner matrices should be dependent on the RNG, but the energies"
    "are not actually different using the above seed setting code."
)
def test_random_seed_final_energy():
    seeds = [100, 200, 300, 200]
    results_by_seed = {}

    calc = FAIRChemCalculator(
        pretrained_mlip.get_predict_unit("uma-s-1"),
        task_name="omat",
    )

    for seed in seeds:
        calc.predictor.seed(seed)
        atoms = bulk("Cu").repeat(2)  # recreate atoms to avoid caching previous result
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        if seed in results_by_seed:
            assert results_by_seed[seed] == energy
        else:
            results_by_seed[seed] = energy

    for seed_a in set(seeds):
        for seed_b in set(seeds) - {seed_a}:
            assert results_by_seed[seed_a] != results_by_seed[seed_b]


def run_md_simulation(calc, steps: int = 10):
    atoms = molecule("H2O")
    atoms.calc = calc

    dyn = Langevin(
        atoms,
        timestep=0.1 * units.fs,
        temperature_K=400,
        friction=0.001 / units.fs,
    )
    dyn.run(steps=10)
    expected_energy = -2079.86
    assert np.allclose(atoms.get_potential_energy(), expected_energy, atol=1e-4)


def test_simple_md():
    inference_settings = InferenceSettings(
        tf32=True,
        merge_mole=True,
        wigner_cuda=False,
        compile=False,
        activation_checkpointing=False,
        internal_graph_gen_version=2,
        external_graph_gen=False,
    )
    predictor = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device="cpu", inference_settings=inference_settings
    )
    calc = FAIRChemCalculator(predictor, task_name="omol")
    run_md_simulation(calc, steps=10)


def test_parallel_md():
    inference_settings = InferenceSettings(
        tf32=True,
        merge_mole=True,
        wigner_cuda=False,
        compile=False,
        activation_checkpointing=False,
        internal_graph_gen_version=2,
        external_graph_gen=False,
    )
    model_path = pretrained_mlip.pretrained_checkpoint_path_from_name("uma-s-1p1")
    predictor = ParallelMLIPPredictUnit(
        inference_model_path=model_path,
        device="cpu",
        inference_settings=inference_settings,
        server_config={"workers": 2},
    )

    calc = FAIRChemCalculator(predictor, task_name="omol")
    run_md_simulation(calc, steps=10)
    predictor.cleanup()
