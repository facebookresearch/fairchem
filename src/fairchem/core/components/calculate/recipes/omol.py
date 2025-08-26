"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

OMol Evaluation Recipes
========================

This module provides evaluation recipes for various molecular property tasks proposed in the OMol25 paper.

The module includes functions for:
- Geometry optimizations and conformer generation
- Protonation state energetics
- Ionization energies and electron affinities
- Spin gap calculations
- Protein-ligand interactions
- Ligand strain energies
- Distance scaling behavior
- Single-point energy and force calculations

Each function follows a consistent pattern of taking input data and any ASE calculator,
performing the required calculations, and returning results in a standardized format
suitable for downstream evaluation on the OMol leaderboard - #TODO: add link.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

from fairchem.data.omol.orca.calc import TIGHT_OPT_PARAMETERS
from pymatgen.io.ase import MSONAtoms
from tqdm import tqdm


def relax_job(
    atoms: Atoms, calculator: Calculator, opt_flags: dict[str, Any]
) -> dict[str, Any]:
    """
    Perform a geometry optimization job on an atomic structure.

    This function optimizes the geometry of an atomic structure using the provided
    calculator and optimization parameters. It captures both the initial and final
    states (energy, forces, and atomic positions) for comparison.

    Args:
        atoms: ASE Atoms object representing the initial structure
        calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations
        opt_flags (dict): Dictionary containing optimization parameters including:
            - optimizer: ASE optimizer class (e.g., BFGS, FIRE)
            - optimizer_kwargs: Additional kwargs for the optimizer
            - fmax: Force convergence criterion (eV/Å)
            - max_steps: Maximum number of optimization steps

    Returns:
        dict: Results organized in the following form -
        {
            "initial": {
                "atoms": MSONAtoms dictionary of initial structure,
                "energy": Initial total energy (eV),
                "forces": Initial forces as a list (eV/Å),
            },
            "final": {
                "atoms": MSONAtoms dictionary of optimized structure,
                "energy": Final total energy (eV),
                "forces": Final forces as a list (eV/Å),
            }
        }

    Note:
        If optimization fails, the function logs the error and returns the last
        valid state rather than crashing.
    """
    atoms.calc = calculator
    initial_energy = atoms.get_potential_energy()
    initial_forces = atoms.get_forces()
    initial_atoms = atoms.copy()

    try:
        dyn = opt_flags["optimizer"](atoms, **opt_flags["optimizer_kwargs"])
        dyn.run(fmax=opt_flags["fmax"], steps=opt_flags["max_steps"])
    except Exception as e:
        # atoms are updated in place, so no actual change needed.
        logging.info(f"Optimization failed, using last valid step. {e}")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    result = {
        "initial": {
            "atoms": MSONAtoms(initial_atoms).as_dict(),
            "energy": initial_energy,
            "forces": initial_forces.tolist(),
        },
        "final": {
            "atoms": MSONAtoms(atoms).as_dict(),
            "energy": energy,
            "forces": forces.tolist(),
        },
    }
    return result


def single_point_job(atoms: Atoms, calculator: Calculator) -> dict[str, Any]:
    """
    Perform a single-point energy and force calculation.

    This function calculates the energy and forces for a given atomic structure.

    Args:
        atoms: ASE Atoms object representing the structure
        calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

    Returns:
        dict: Results organized in the following form -
        {
            "atoms": MSONAtoms dictionary of the structure,
            "energy": Total energy (eV),
            "forces": Forces as a list (eV/Å),
        }
    """
    # clear cache, especially for ieea + spin-gap
    calculator.reset()
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms.calc = None
    result = {
        "atoms": MSONAtoms(atoms).as_dict(),
        "energy": energy,
        "forces": forces.tolist(),
    }
    return result


def conformers(input_data: dict[str, Any], calculator: Calculator) -> dict[str, Any]:
    """
    Calculate conformer energies and geometries.

    This function performs geometry optimizations on molecular conformers.

    Args:
        input_data (dict): Input data organized by molecule families, where each
            entry contains conformer information with initial and final structures
        calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

    Returns:
        dict: Results organized in the following form -
        {
            "molecule_family_1": {
                "conformer_id_1": {
                    "initial": {
                        "atoms": MSONAtoms dictionary of initial structure,
                        "energy": Initial total energy (eV),
                        "forces": Initial forces as a list (eV/Å),
                    },
                    "final": {
                        "atoms": MSONAtoms dictionary of optimized structure,
                        "energy": Final total energy (eV),
                        "forces": Final forces as a list (eV/Å),
                    },
                },
                "conformer_id_2": { ... },
                ...
            },
            "molecule_family_2": { ... },
            ...
        }
    """
    all_results = {}
    for molecule_family in tqdm(input_data):
        conformer_results = {}
        conformers = input_data[molecule_family]
        for conformer in conformers:
            sid = conformer["sid"]
            initial_atoms = conformer["initial_atoms"]
            results = relax_job(initial_atoms, calculator, TIGHT_OPT_PARAMETERS)
            conformer_results[sid] = results

        all_results[molecule_family] = conformer_results
    return all_results


def protonation(input_data: dict[str, Any], calculator: Calculator) -> dict[str, Any]:
    """
    Calculate protonation state energies and geometries.

    This function calculates the energies and geometries of different protonation
    states of molecules.

    Args:
        input_data (dict): Input data organized by molecule families, where each
            entry contains different protonation states with initial and final structures
        calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

    Returns:
        dict: Results organized in the following form -
        {
            "molecule_family_1": {
                "protonation_state_1": {
                    "initial": {
                        "atoms": MSONAtoms dictionary of initial structure,
                        "energy": Initial total energy (eV),
                        "forces": Initial forces as a list (eV/Å),
                    },
                    "final": {
                        "atoms": MSONAtoms dictionary of optimized structure,
                        "energy": Final total energy (eV),
                        "forces": Final forces as a list (eV/Å),
                    },
                },
                "protonation_state_2": { ... },
                ...
            },
            "molecule_family_2": { ... },
            ...
        }
    """
    all_results = {}
    for molecule_family in tqdm(input_data):
        state_results = {}
        states = input_data[molecule_family]
        for state in states:
            initial_atoms = states[state]["initial_atoms"]
            results = relax_job(initial_atoms, calculator, TIGHT_OPT_PARAMETERS)

            state_results[state] = results

        all_results[molecule_family] = state_results
    return all_results


def ieea(input_data: dict[str, Any], calculator: Calculator) -> dict[str, Any]:
    """
    Calculate unoptimized ionization energies and electron affinities.

    This function performs single-point calculations on structures at different
    charge states to evaluate ionization energies (IE) and electron affinities (EA).
    No geometry optimization is performed, testing the MLIP's ability to predict
    energetics of charged species at fixed geometries.

    Args:
        input_data (dict): Input data organized by system identifier, with each
            entry containing structures at different charge and spin states
        calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

    Returns:
        dict: Results organized in the following form -
        {
            "identifier_1": {
                "charge_state_1": {
                    "spin_state_1": {
                        "atoms": MSONAtoms dictionary of the structure,
                        "energy": Total energy (eV),
                        "forces": Forces as a list (eV/Å),
                    },
                    "spin_state_2": { ... },
                    ...
                },
                "charge_state_2": { ... },
                ...
            },
            "identifier_2": { ... },
        }
    """
    all_results = {}
    for identifier in tqdm(input_data):
        molecule_results = defaultdict(dict)
        for charge in input_data[identifier]:
            for spin, entry in input_data[identifier][charge].items():
                atoms = entry["atoms"]
                results = single_point_job(atoms, calculator)

                molecule_results[charge][spin] = results

        all_results[identifier] = molecule_results
    return all_results


def spin_gap(input_data: dict[str, Any], calculator: Calculator) -> dict[str, Any]:
    """
    Calculate unoptimized spin gap energies.

    This function performs single-point calculations on structures at different
    spin states to evaluate spin gaps (energy differences between different
    spin multiplicities). No geometry optimization is performed.

    Args:
        input_data (dict): Input data organized by system identifier, with each
            entry containing structures at different spin states
        calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

    Returns:
        dict: Results organized in the following form -
        {
            "identifier_1": {
                "spin_state_1": {
                    "atoms": MSONAtoms dictionary of the structure,
                    "energy": Total energy (eV),
                    "forces": Forces as a list (eV/Å),
                },
                "spin_state_2": { ... },
            },
            "identifier_2": { ... },
        }
    """
    all_results = {}
    for identifier in tqdm(input_data):
        molecule_results = {}
        for spin, entry in input_data[identifier].items():
            atoms = entry["atoms"]
            results = single_point_job(atoms, calculator)
            molecule_results[spin] = results

        all_results[identifier] = molecule_results
    return all_results


def ligand_pocket(input_data: dict[str, Any], calculator: Calculator) -> dict[str, Any]:
    """
    Calculate protein-ligand interaction energies and forces.

    This function performs single-point calculations on protein-ligand systems,
    calculating energies and forces for the complex and individual components
    (ligand, pocket, ligand_pocket). This enables evaluation of interaction
    energies and binding affinity predictions.

    Args:
        input_data (dict): Input data organized by system identifier, with each
            entry containing ASE Atoms objects for ligand, pocket, and complex
        calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

    Returns:
        dict: Results organized in the following form -
        {
            "identifier_1": {
                "ligand": {
                    "atoms": MSONAtoms dictionary of the structure,
                    "energy": Total energy (eV),
                    "forces": Forces as a list (eV/Å),
                },
                "pocket": {
                    "atoms": MSONAtoms dictionary of the structure,
                    "energy": Total energy (eV),
                    "forces": Forces as a list (eV/Å),
                },
                "ligand_pocket": {
                    "atoms": MSONAtoms dictionary of the structure,
                    "energy": Total energy (eV),
                    "forces": Forces as a list (eV/Å),
                },
            },
            "identifier_2": { ... },
        }
    """
    all_results = {}
    for identifier, entry in tqdm(input_data.items(), total=len(input_data)):
        complex_results = {}
        for mol_type in ["ligand", "pocket", "ligand_pocket"]:
            atoms = entry[mol_type]
            results = single_point_job(atoms, calculator)
            complex_results[mol_type] = results

        all_results[identifier] = complex_results
    return all_results


def ligand_strain(input_data: dict[str, Any], calculator: Calculator) -> dict[str, Any]:
    """
    Calculate ligand strain energies in protein-bound conformations.

    This function calculates strain energies by comparing the energy of a ligand
    in its bioactive (protein-bound) conformation with its global minimum energy
    conformation in the gas phase.


    Args:
        input_data (dict): Input data organized by system identifier, with each
            entry containing:
            - bioactive_conf: Ligand in bioactive conformation
            - conformers: List of (initial, final) conformer pairs for gas phase
        calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

    Returns:
        dict: Results organized in the following form -
        {
            "identifier_1": {
                "bioactive": {
                    "atoms": MSONAtoms dictionary of bioactive conformation,
                    "energy": Total energy (eV),
                    "forces": Forces as a list (eV/Å),
                },
                "gas_phase": {
                    "0": {  # conformer index
                        "initial": {
                            "atoms": MSONAtoms dictionary of initial structure,
                            "energy": Initial total energy (eV),
                            "forces": Initial forces as a list (eV/Å),
                        },
                        "final": {
                            "atoms": MSONAtoms dictionary of optimized structure,
                            "energy": Final total energy (eV),
                            "forces": Final forces as a list (eV/Å),
                        }
                    },
                    "1": { ... },
                    ...
                },
            },
            "identifier_2": { ...
            }
        }
    """
    all_results = {}
    for identifier, ligand_system in tqdm(input_data.items()):
        complex_results = {}
        # Bioactive part
        bioactive = ligand_system["bioactive_conf"]
        results = single_point_job(bioactive, calculator)

        complex_results["bioactive"] = results

        # Gas-phase conformers parts
        conformer_prediction = {}
        for idx, initial_atoms in enumerate(ligand_system["conformers"]):
            results = relax_job(initial_atoms, calculator, TIGHT_OPT_PARAMETERS)

            conformer_prediction[idx] = results
        complex_results["gas_phase"] = conformer_prediction

        all_results[identifier] = complex_results
    return all_results


def distance_scaling(
    input_data: dict[str, Any], calculator: Calculator
) -> dict[str, Any]:
    """
    Calculate energies and forces at different inter-molecular distances.

    This function performs single-point calculations on molecular systems where
    inter-molecular distances have been systematically varied. This tests the
    MLIP's ability to capture both short-range repulsion and long-range attraction
    in potential energy surfaces.

    Args:
        input_data (dict): Input data organized by domain type (vertical), then
            by system identifier, then by distance scale factor, containing
            ASE Atoms objects at different inter-molecular separations
        calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

    Returns:
        dict: Results organized in the following form -
        {
            "vertical_1": {
                "identifier_1": {
                    "short_range_scaled_complex_X": {
                        "atoms": MSONAtoms dictionary of the structure,
                        "energy": Total energy (eV),
                        "forces": Forces as a list (eV/Å),
                    },
                    "short_range_scaled_complex_Y": { ... },
                    "long_range_scaled_complex_Z": { ... },
                    ...
                },
                "identifier_2": { ... },
                ...
            },
            "vertical_2": { ... },
            ...
    """
    all_results = {}
    for vertical, systems in input_data.items():
        species_results = defaultdict(dict)
        for identifier, structures in tqdm(systems.items()):
            for scale, scale_structure in structures.items():
                results = single_point_job(scale_structure, calculator)
                species_results[identifier][scale] = results

        all_results[vertical] = species_results
    return all_results


def singlepoint(input_data: dict[str, Any], calculator: Calculator) -> dict[str, Any]:
    """
    Perform general single-point energy and force calculations.

    This is a general-purpose function for performing single-point calculations
    on arbitrary molecular structures.

    Args:
        input_data (dict): Input data organized by system identifier, with each
            entry containing an ASE Atoms object
        calculator: ASE calculator object (e.g., FAIRChemCalculator) to use for energy/force calculations

    Returns:
        dict: Results organized in the following form -
        {
            "identifier_1": {
                "atoms": MSONAtoms dictionary of the structure,
                "energy": Total energy (eV),
                "forces": Forces as a list (eV/Å),
            },
            "identifier_2": { ... },
        }
    """
    all_results = {}
    for identifier, atoms in tqdm(input_data.items(), total=len(input_data)):
        results = single_point_job(atoms, calculator)
        all_results[identifier] = results
    return all_results
