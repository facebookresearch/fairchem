"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from fairchem.data.omol.omdata.orca.calc import TIGHT_OPT_PARAMETERS
from pymatgen.io.ase import MSONAtoms
from tqdm import tqdm


def relax_job(atoms, calc, opt_flags):
    atoms.calc = calc
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


def single_point_job(atoms, calc):
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms.calc = None
    result = {
        "atoms": MSONAtoms(atoms).as_dict(),
        "energy": energy,
        "forces": forces.tolist(),
    }
    return result


def geom_conformers(input_data, calculator, use_relaxed):
    all_results = defaultdict(dict)
    for molecule_family in tqdm(input_data):
        conformer_prediction = {}
        conformer_target = {}
        conformers = input_data[molecule_family]
        for conformer in conformers:
            sid = conformer["sid"]
            initial_atoms = conformer["initial_atoms"]
            final_atoms = conformer["final_atoms"]

            dft_results = {
                "initial": {
                    "atoms": MSONAtoms(initial_atoms).as_dict(),
                    "energy": initial_atoms.get_potential_energy(),
                    "forces": initial_atoms.get_forces().tolist(),
                },
                "final": {
                    "atoms": MSONAtoms(final_atoms).as_dict(),
                    "energy": final_atoms.get_potential_energy(),
                    "forces": final_atoms.get_forces().tolist(),
                },
            }

            if use_relaxed:
                initial_atoms = final_atoms

            results = relax_job(initial_atoms, calculator, TIGHT_OPT_PARAMETERS)

            conformer_prediction[sid] = results
            conformer_target[sid] = dft_results

        all_results[molecule_family]["prediction"] = conformer_prediction
        all_results[molecule_family]["target"] = conformer_target
    return all_results


def protonation(input_data, calculator, use_relaxed):
    all_results = defaultdict(dict)
    for molecule_family in tqdm(input_data):
        state_prediction = {}
        state_target = {}
        states = input_data[molecule_family]
        for state in states:
            initial_atoms = states[state]["initial_atoms"]
            final_atoms = states[state]["final_atoms"]

            dft_results = {
                "initial": {
                    "atoms": MSONAtoms(initial_atoms).as_dict(),
                    "energy": initial_atoms.get_potential_energy(),
                    "forces": initial_atoms.get_forces().tolist(),
                },
                "final": {
                    "atoms": MSONAtoms(final_atoms).as_dict(),
                    "energy": final_atoms.get_potential_energy(),
                    "forces": final_atoms.get_forces().tolist(),
                },
            }

            if use_relaxed:
                initial_atoms = final_atoms

            results = relax_job(initial_atoms, calculator, TIGHT_OPT_PARAMETERS)

            state_prediction[state] = results
            state_target[state] = dft_results

        all_results[molecule_family]["prediction"] = state_prediction
        all_results[molecule_family]["target"] = state_target
    return all_results


def unoptimized_ieea(input_data, calculator):
    all_results = defaultdict(dict)
    for identifier in tqdm(input_data):
        prediction = defaultdict(dict)
        target = defaultdict(dict)
        for charge in input_data[identifier]:
            for spin, entry in input_data[identifier][charge].items():
                atoms = entry["atoms"]
                dft_results = {
                    "atoms": MSONAtoms(atoms).as_dict(),
                    "energy": atoms.get_potential_energy(),
                    "forces": atoms.get_forces().tolist(),
                }

                results = single_point_job(atoms, calculator)

                prediction[charge][spin] = results
                target[charge][spin] = dft_results

        all_results[identifier]["prediction"] = prediction
        all_results[identifier]["target"] = target
    return all_results


def unoptimized_spin_gap(input_data, calculator):
    all_results = defaultdict(dict)
    for identifier in tqdm(input_data):
        prediction = defaultdict(dict)
        target = defaultdict(dict)
        for spin, entry in input_data[identifier].items():
            atoms = entry["atoms"]
            dft_results = {
                "atoms": MSONAtoms(atoms).as_dict(),
                "energy": atoms.get_potential_energy(),
                "forces": atoms.get_forces().tolist(),
            }

            results = single_point_job(atoms, calculator)

            prediction[spin] = results
            target[spin] = dft_results

        all_results[identifier]["prediction"] = prediction
        all_results[identifier]["target"] = target
    return all_results


def pdb_pocket(input_data, calculator):
    all_results = defaultdict(dict)
    for identifier, entry in tqdm(input_data.items(), total=len(input_data)):
        prediction = {}
        target = {}
        for mol_type in ["ligand", "pocket", "ligand_pocket"]:
            atoms = entry[mol_type]
            dft_results = {
                "atoms": MSONAtoms(atoms).as_dict(),
                "energy": atoms.get_potential_energy(),
                "forces": atoms.get_forces().tolist(),
            }

            results = single_point_job(atoms, calculator)

            prediction[mol_type] = results
            target[mol_type] = dft_results

        all_results[identifier]["prediction"] = prediction
        all_results[identifier]["target"] = target
    return all_results


def ligand_strain(input_data, calculator):
    all_results = defaultdict(dict)
    for identifier, ligand_system in tqdm(input_data.items()):
        prediction = {}
        target = {}

        # Bioactive part
        bioactive = ligand_system["bioactive_conf"]
        dft_results = {
            "atoms": MSONAtoms(bioactive).as_dict(),
            "energy": bioactive.get_potential_energy(),
            "forces": bioactive.get_forces().tolist(),
        }

        results = single_point_job(bioactive, calculator)

        prediction["bioactive"] = results
        target["bioactive"] = dft_results

        # Gas-phase conformers parts
        conformer_prediction = {}
        conformer_target = {}
        for idx, (initial_atoms, final_atoms) in enumerate(ligand_system["conformers"]):
            dft_results = {
                "initial": {
                    "atoms": MSONAtoms(initial_atoms).as_dict(),
                    "energy": initial_atoms.get_potential_energy(),
                    "forces": initial_atoms.get_forces().tolist(),
                },
                "final": {
                    "atoms": MSONAtoms(final_atoms).as_dict(),
                    "energy": final_atoms.get_potential_energy(),
                    "forces": final_atoms.get_forces().tolist(),
                },
            }

            results = relax_job(initial_atoms, calculator, TIGHT_OPT_PARAMETERS)

            conformer_prediction[idx] = results
            conformer_target[idx] = dft_results
        prediction["gas_phase"] = conformer_prediction
        target["gas_phase"] = conformer_target

        all_results[identifier]["prediction"] = prediction
        all_results[identifier]["target"] = target
    return all_results


def distance_scaling(input_data, calculator):
    all_results = defaultdict(dict)
    for vertical, systems in input_data.items():
        prediction = defaultdict(dict)
        target = defaultdict(dict)
        for identifier, structures in tqdm(systems.items()):
            for scale, scale_structure in structures.items():
                dft_results = {
                    "energy": scale_structure.get_potential_energy(),
                    "forces": scale_structure.get_forces().tolist(),
                }

                results = single_point_job(scale_structure, calculator)

                prediction[identifier][scale] = results
                target[identifier][scale] = dft_results

        all_results[vertical]["prediction"] = prediction
        all_results[vertical]["target"] = target

    return all_results


def singlepoint(input_data, calculator):
    all_results = defaultdict(dict)
    for identifier, atoms in tqdm(input_data.items(), total=len(input_data)):
        dft_results = {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces().tolist(),
        }

        results = single_point_job(atoms, calculator)

        all_results[identifier]["target"] = dft_results
        all_results[identifier]["prediction"] = results
    return all_results
