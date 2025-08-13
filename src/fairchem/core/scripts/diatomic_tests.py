from __future__ import annotations

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

# TODO these reference energies need to replaced if not using OMol25 or OMat24
mol_dict = {
    ("C", "C"): {
        "atoms_info": {"spin": 1, "charge": 0},
        "ref_energy": -2059.708257873304,
    },
    ("H", "H"): {
        "atoms_info": {"spin": 1, "charge": 0},
        "ref_energy": -26.89116811474675,
    },
    ("O", "O"): {
        "atoms_info": {"spin": 3, "charge": 0},
        "ref_energy": -4085.9570373686342,
    },
    ("N", "N"): {
        "atoms_info": {"spin": 1, "charge": 0},
        "ref_energy": -2971.0840536105143,
    },
    ("F", "F"): {
        "atoms_info": {"spin": 1, "charge": 0},
        "ref_energy": -5428.480814580473,
    },
    ("S", "S"): {
        "atoms_info": {"spin": 3, "charge": 0},
        "ref_energy": -21665.24472704541,
    },
}

mat_dict = {
    ("C", "C"): {"atoms_info": {}, "ref_energy": -2.55503062},
    ("O", "O"): {"atoms_info": {}, "ref_energy": -3.09594272},
    ("Pt", "Pt"): {"atoms_info": {}, "ref_energy": -1.00580368},
    ("Si", "Si"): {"atoms_info": {}, "ref_energy": -1.65238266},
    ("Cu", "Cu"): {"atoms_info": {}, "ref_energy": -0.46898334},
    ("Mg", "Mg"): {"atoms_info": {}, "ref_energy": -0.01886358},
    ("Fe", "O"): {"atoms_info": {}, "ref_energy": -4.85459983},
    ("Ca", "O"): {"atoms_info": {}, "ref_energy": -1.57365455},
}


def get_model(
    model_name,
    released,
):
    if not released:
        uma_pred = pretrained_mlip.load_predict_unit(
            model_name,
            device="cuda",
        )
    else:
        uma_pred = pretrained_mlip.get_predict_unit(model_name, device="cuda")
    return uma_pred


def compute_diatomic_curve(predictor, task, atom_1, atom_2, atoms_dict):
    # Get the atomic number
    atomic_number_1 = atomic_numbers[atom_1]
    atomic_number_2 = atomic_numbers[atom_2]

    rmin = 0.5 * (covalent_radii[atomic_number_1] + covalent_radii[atomic_number_2])
    rmax = 6.0

    distances = np.linspace(rmin, rmax, 100)  # Range of distances in Angstroms
    energies = []

    # Calculate energy for each distance
    atoms_list = []
    for d in distances:
        # Create the diatomic molecule
        dimer = Atoms([atom_1, atom_2], positions=[[0, 0, 0], [0, 0, d]], pbc=False)
        if atoms_dict["atoms_info"] is not None:
            dimer.info.update(atoms_dict["atoms_info"])

        atoms_list.append(dimer)

    if task == "omol":
        atomic_data_list = [
            AtomicData.from_ase(atoms, task_name="omol", r_data_keys=["spin", "charge"])
            for atoms in atoms_list
        ]
    else:
        atomic_data_list = [
            AtomicData.from_ase(
                atoms,
                task_name=task,
            )
            for atoms in atoms_list
        ]
    batch = atomicdata_list_to_batch(atomic_data_list)

    energies = predictor.predict(batch)["energy"]

    # reference energy is the sum of the isolated atomic energies
    relative_energies = energies.detach().cpu().numpy() - atoms_dict["ref_energy"]

    return distances, relative_energies.tolist(), atom_1, atom_2


def generate_plot(distances, energies, atom_1, atom_2, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(distances, energies, label=f"{atom_1}-{atom_2}")
    plt.xlabel("Distance (Å)")
    plt.ylabel("Relative Energy (eV)")
    plt.title("Diatomic Potential Energy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)


def get_relative_energies(energies):
    energies = np.array(energies)
    last_energy = energies[-1]
    rel_e = energies - last_energy
    return rel_e


def combined_plot(results_dict, save_path):
    plt.figure(figsize=(8, 6))
    for atom_type, data in results_dict.items():
        plt.plot(
            data["distances"],
            data["energies"],
            label=f"{atom_type[0]}-{atom_type[1]}",
        )
    plt.xlabel("Distance (Å)")
    plt.ylabel("Relative Energy (eV)")
    plt.title("Diatomic Potential Energy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument(
        "-m", "--model", type=str, help="name or path to model", required=True
    )
    parser.add_argument(
        "-r",
        "--released",
        action="store_true",
        help="include flag if the model has been released",
    )
    parser.add_argument(
        "-t", "--task", type=str, help="name of the task", required=True
    )
    parser.add_argument(
        "-s", "--save_dir", type=str, help="Directory to save the output", required=True
    )
    # Parse the arguments
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # make save dir if it does not exist
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save_name = os.path.basename(save_dir)

    results_dict = {}
    predictor = get_model(args.model, args.released)
    if args.task == "omol":
        diatomic_dict = mol_dict
    elif args.task == "omat":
        diatomic_dict = mat_dict

    for atom_type, atoms_dict in diatomic_dict.items():
        distances, energies, atom_1, atom_2 = compute_diatomic_curve(
            predictor, args.task, atom_type[0], atom_type[1], atoms_dict=atoms_dict
        )
        results_dict[atom_type] = {
            "distances": distances,
            "energies": energies,
        }

        generate_plot(
            distances,
            energies,
            atom_1,
            atom_2,
            save_path=os.path.join(
                save_dir, f"{atom_1}_{atom_2}_potential_energy_curve.png"
            ),
        )

    combined_plot(
        results_dict,
        save_path=os.path.join(save_dir, "all_potential_energy_curves.png"),
    )

    # save results_dict to a pickle file in save_dir
    with open(os.path.join(save_dir, "results.pkl"), "wb") as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    main()
