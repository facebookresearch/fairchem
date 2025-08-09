from __future__ import annotations

import argparse
import copy
import os
import random

import mendeleev
from architector.io_molecule import convert_io_molecule
from ase.io import write

from fairchem.core.datasets import AseDBDataset

random.seed(42)


open_shell_metals = [
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
]


def num_open_shell_metals(atoms):
    return len([atom for atom in atoms if atom.symbol in open_shell_metals])


def parse_sdf_metal_ox_state(input_path, filename):
    """
    Parse the oxidation states of metals in a given SDF file.

    Args:
        input_path (str): The path to the directory containing the SDF file.
        filename (str): The name of the SDF file.

    Returns:
        list: A list of tuples, where each tuple contains the metal symbol and its oxidation state.
    """
    metal_list = []
    with open(input_path + "/cod_complexes/" + filename) as f:
        for line in f.readlines():
            split_line = line.split()
            if len(split_line) > 7:  # This line contains an XYZ position of an atom
                if split_line[3] in open_shell_metals:  # This atom is a metal
                    if (
                        len(split_line) > 8
                    ):  # This metal has info that might be an oxidation state
                        for kk in range(8, len(split_line)):
                            if (
                                split_line[kk][0:4] == "CHG="
                            ):  # This info is an oxidation state
                                metal_list.append(
                                    [split_line[3], int(split_line[kk][4:])]
                                )
                                break
                        else:  # This metal has no oxidation state info, so we assume charge is 0
                            metal_list.append([split_line[3], 0])
                    else:  # This metal has no oxidation state info, so we assume charge is 0
                        metal_list.append([split_line[3], 0])

    return metal_list


def ie_ea_processing(atoms):
    """
    Process the atoms to add or remove an electron and return the structures to add.

    Args:
        atoms (ase.Atoms): The atoms object to process.

    Returns:
        list: A list of dictionaries, where each dictionary contains the charge and spin of the structure to add.
    """
    structs_to_add = []
    add_electron_charge = atoms.info["charge"] - 1
    remove_electron_charge = atoms.info["charge"] + 1
    if (
        atoms.info["spin"] == 1
    ):  # If the structure is a singlet, then both oxidized and reduced states are doublets
        add_electron_spin_multiplicity = [2]
        remove_electron_spin_multiplicity = [2]
    else:  # If the structure is not a singlet, then oxidized and reduced states can take two different spin multiplicities
        add_electron_spin_multiplicity = [
            atoms.info["spin"] + 1,
            atoms.info["spin"] - 1,
        ]
        remove_electron_spin_multiplicity = [
            atoms.info["spin"] + 1,
            atoms.info["spin"] - 1,
        ]

    for spin_multiplicity in add_electron_spin_multiplicity:
        structs_to_add.append(
            {
                "charge": add_electron_charge,
                "spin": spin_multiplicity,
            }
        )

    for spin_multiplicity in remove_electron_spin_multiplicity:
        structs_to_add.append(
            {
                "charge": remove_electron_charge,
                "spin": spin_multiplicity,
            }
        )

    return structs_to_add


def sample_structures_for_unoptimized_ie_ea(structures, number, idx_name):
    """
    Sample structures for unoptimized IE/EA.

    Args:
        structures (ase.Atoms): The structures to sample.
        number (int): The number of structures to sample.
        idx_name (str): The name of the index, distinguishing between electrolyte structures (e_idx),
                        metal complex structures from Architector (mca_idx),
                        and metal complex structures from COD (mcc_idx).

    Returns:
        list: A list of ASE atoms objects to save for calculations
    """
    sampled_structures = []
    viable_idxs = []
    for idx in range(len(structures)):
        if idx_name == "e_idx":
            if (
                num_open_shell_metals(structures.get_atoms(idx)) > 2
            ):  # Electrolytes are allowed to have one or two open-shell metals
                continue
            viable_idxs.append(idx)
        elif (
            idx_name == "mca_idx"
        ):  # Architector metal complexes are assumed to have only one open-shell metal
            viable_idxs.append(idx)
        elif idx_name == "mcc_idx":
            if (
                len(structures[idx].info["metal_list"]) > 1
            ):  # COD metal complexes are allowed to have one open-shell metal
                continue
            viable_idxs.append(idx)
    for idx in random.sample(viable_idxs, number):
        if idx_name in ["e_idx", "mca_idx"]:
            structs_to_add = ie_ea_processing(structures.get_atoms(idx))
            tmp_atoms = structures.get_atoms(idx)
        elif idx_name == "mcc_idx":
            structs_to_add = ie_ea_processing(structures[idx])
            tmp_atoms = structures[idx]
        tmp_atoms.info["idx_name"] = idx_name
        tmp_atoms.info["idx"] = idx
        sampled_structures.append(tmp_atoms)  # Add the original structure
        for struct in structs_to_add:
            new_atoms = copy.deepcopy(tmp_atoms)
            new_atoms.info["spin"] = struct["spin"]
            new_atoms.info["charge"] = struct["charge"]
            sampled_structures.append(new_atoms)
    return sampled_structures


def sample_structures_for_unoptimized_spin_gap(structures, number, idx_name):
    """
    Sample structures for unoptimized spin gap.

    Args:
        structures (ase.Atoms): The structures to sample.
        number (int): The number of structures to sample.
        idx_name (str): The name of the index, distinguishing between electrolyte structures (e_idx),
                        metal complex structures from Architector (mca_idx),
                        and metal complex structures from COD (mcc_idx).

    Returns:
        list: A list of ASE atoms objects to save for calculations
    """
    cached_unpaired_electrons = {}
    sampled_structures = []
    idx_list = list(range(len(structures)))
    random.shuffle(idx_list)
    samples_taken = 0
    for idx in idx_list:
        new_spins = []
        if idx_name in ["e_idx", "mca_idx"]:
            tmp_atoms = structures.get_atoms(idx)
            if (
                tmp_atoms.info["spin"] > 2
            ):  # Electrolytes and Architector metal complexes start as high spin and thus must be at least a triplet for a spin gap
                if idx_name == "e_idx":
                    if (
                        num_open_shell_metals(tmp_atoms) > 1
                    ):  # Electrolytes are allowed to have only one open-shell metal. Architector complexes are assumed to have only one open-shell metal
                        continue
                new_spin = (
                    tmp_atoms.info["spin"] - 2
                )  # Iteratively reduce the spin by two to find all viable spins
                while new_spin > 0:
                    new_spins.append(new_spin)
                    new_spin -= 2
        elif idx_name in ["mcc_idx"]:
            tmp_atoms = structures[idx]
            if (
                len(tmp_atoms.info["metal_list"]) > 1
            ):  # Only consider COD metal complexes with one open-shell metal
                continue
            max_num_unpaired_electrons = 0
            lanthanide_present = False
            for metal in tmp_atoms.info["metal_list"]:
                if metal[0] in [
                    "La",
                    "Ce",
                    "Pr",
                    "Nd",
                    "Pm",
                    "Sm",
                    "Eu",
                    "Gd",
                    "Tb",
                    "Dy",
                    "Ho",
                    "Er",
                    "Tm",
                ]:
                    lanthanide_present = True
                if (metal[0], metal[1]) not in cached_unpaired_electrons:
                    cached_unpaired_electrons[(metal[0], metal[1])] = (
                        mendeleev.element(metal[0])
                        .ec.ionize(metal[1])
                        .unpaired_electrons()
                    )
                max_num_unpaired_electrons += cached_unpaired_electrons[
                    (metal[0], metal[1])
                ]
            if max_num_unpaired_electrons != 0:
                new_spin = max_num_unpaired_electrons + 1
                while (
                    new_spin > 0
                ):  # Iteratively reduce the spin by two to find all viable spins
                    if new_spin != tmp_atoms.info["spin"]:
                        if (
                            new_spin < 12 or lanthanide_present
                        ):  # Only consider spins less than 12 or lanthanides
                            new_spins.append(new_spin)
                    new_spin -= 2

        if (
            new_spins != []
        ):  # If there are viable additional spin states, then this structure is viable for a spin gap
            samples_taken += 1
            tmp_atoms.info["idx_name"] = idx_name
            tmp_atoms.info["idx"] = idx
            sampled_structures.append(tmp_atoms)  # Add the original structure
            for new_spin in new_spins:
                assert new_spin != tmp_atoms.info["spin"]
                assert new_spin % 2 == tmp_atoms.info["spin"] % 2
                new_atoms = copy.deepcopy(tmp_atoms)
                new_atoms.info["spin"] = new_spin
                sampled_structures.append(new_atoms)
            if samples_taken == number:
                break
    return sampled_structures


def sample_structures_for_distance_scaling(structures, number, idx_name):
    """
    Sample structures for distance scaling.

    Args:
        structures (ase.Atoms): The structures to sample.
        number (int): The number of structures to sample.
        idx_name (str): The name of the index, distinguishing between electrolyte structures (e_idx),
                        electrolyte structures with one open-shell metal (eo_idx),
                        metal complex structures from Architector (mca_idx),
                        and metal complex structures from COD (mcc_idx).

    Returns:
        list: A list of ASE atoms objects to save for subsequent distance scaling
    """
    sampled_structures = []
    viable_idxs = []
    num_3A = 0
    num_5A = 0
    num_solvated = 0
    if idx_name in ["mca_idx", "mcc_idx"]:
        viable_idxs = list(
            range(len(structures))
        )  # All Architector metal complexes and COD metal complexes are viable
    else:
        for idx in range(len(structures)):
            if idx_name == "e_idx":
                tmp_atoms = structures.get_atoms(idx)
                if (
                    "3_shell" in tmp_atoms.info["source"]
                    or "3_group" in tmp_atoms.info["source"]
                ) and num_3A < 1500:
                    if num_open_shell_metals(tmp_atoms) == 0:
                        num_3A += 1
                        viable_idxs.append(idx)
                elif "5A" in tmp_atoms.info["source"] and num_5A < 500:
                    if num_open_shell_metals(tmp_atoms) == 0:
                        num_5A += 1
                        viable_idxs.append(idx)
                elif (
                    "solvated" in tmp_atoms.info["source"]
                    and "unsolvated" not in tmp_atoms.info["source"]
                    and num_solvated < 500
                ):
                    if num_open_shell_metals(tmp_atoms) == 0:
                        num_solvated += 1
                        viable_idxs.append(idx)
            elif idx_name == "eo_idx":
                tmp_atoms = structures.get_atoms(idx)
                if (
                    "3_shell" in tmp_atoms.info["source"]
                    or "3_group" in tmp_atoms.info["source"]
                    and num_3A < 1000
                ):
                    if num_open_shell_metals(tmp_atoms) == 1:
                        num_3A += 1
                        viable_idxs.append(idx)
                elif (
                    "solvated" in tmp_atoms.info["source"]
                    and "unsolvated" not in tmp_atoms.info["source"]
                    and num_solvated < 500
                ):
                    if num_open_shell_metals(tmp_atoms) == 1:
                        num_solvated += 1
                        viable_idxs.append(idx)
            elif idx_name == "b_idx":
                if (
                    "pdb_fragments" in structures.get_atoms(idx).info["source"]
                ):  # Must be a pdb fragment
                    viable_idxs.append(idx)
    print("num_3A", num_3A)
    print("num_5A", num_5A)
    print("num_solvated", num_solvated)
    print(len(viable_idxs), "viable structures for distance scaling from", idx_name)
    for idx in random.sample(viable_idxs, number):
        if idx_name == "mcc_idx":
            tmp_atoms = structures[idx]
            tmp_atoms.info["idx_name"] = idx_name
            tmp_atoms.info["idx"] = idx
            sampled_structures.append(tmp_atoms)
        else:
            tmp_atoms = structures.get_atoms(idx)
            tmp_atoms.info["idx_name"] = idx_name
            tmp_atoms.info["idx"] = idx
            sampled_structures.append(tmp_atoms)
    return sampled_structures


def main(args):
    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    elytes = AseDBDataset({"src": input_path + "/test_elytes_samples"})
    metal_complexes_arch = AseDBDataset(
        {"src": input_path + "/test_organometallics_samples"}
    )
    biomolecules = AseDBDataset({"src": input_path + "/test_biomolecules_samples"})

    metal_complexes_cod = []
    for filename in os.listdir(input_path + "/cod_complexes"):
        atoms = convert_io_molecule(input_path + "/cod_complexes/" + filename).ase_atoms
        atoms.info["charge"] = int(filename[:-4].split("_")[-2])
        atoms.info["spin"] = int(filename[:-4].split("_")[-1])
        atoms.info["source"] = filename
        metal_list = parse_sdf_metal_ox_state(input_path, filename)
        negative_or_zero_ox_found = False
        for metal in metal_list:
            if metal[1] <= 0:
                negative_or_zero_ox_found = True
                break
        if not negative_or_zero_ox_found:
            atoms.info["metal_list"] = metal_list
            metal_complexes_cod.append(atoms)

    distance_scaling_elytes = sample_structures_for_distance_scaling(
        elytes, 2500, "e_idx"
    )
    assert len(distance_scaling_elytes) == 2500
    distance_scaling_elytes_os = sample_structures_for_distance_scaling(
        elytes, 1000, "eo_idx"
    )
    assert len(distance_scaling_elytes_os) == 1000
    distance_scaling_metal_complexes_arch = sample_structures_for_distance_scaling(
        metal_complexes_arch, 1000, "mca_idx"
    )
    assert len(distance_scaling_metal_complexes_arch) == 1000
    distance_scaling_metal_complexes_cod = sample_structures_for_distance_scaling(
        metal_complexes_cod, 3000, "mcc_idx"
    )
    assert len(distance_scaling_metal_complexes_cod) == 3000
    distance_scaling_biomolecules = sample_structures_for_distance_scaling(
        biomolecules, 2000, "b_idx"
    )
    assert len(distance_scaling_biomolecules) == 2000

    for atoms in distance_scaling_elytes:
        write(
            os.path.join(
                output_path,
                "distance_scaling_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + ".traj",
            ),
            atoms,
        )
    for atoms in distance_scaling_elytes_os:
        write(
            os.path.join(
                output_path,
                "distance_scaling_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + ".traj",
            ),
            atoms,
        )
    for atoms in distance_scaling_metal_complexes_arch:
        write(
            os.path.join(
                output_path,
                "distance_scaling_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + ".traj",
            ),
            atoms,
        )
    for atoms in distance_scaling_metal_complexes_cod:
        write(
            os.path.join(
                output_path,
                "distance_scaling_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + ".traj",
            ),
            atoms,
        )
    for atoms in distance_scaling_biomolecules:
        write(
            os.path.join(
                output_path,
                "distance_scaling_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + ".traj",
            ),
            atoms,
        )

    unoptimized_ie_ea_structures_elytes = sample_structures_for_unoptimized_ie_ea(
        elytes, 4000, "e_idx"
    )
    print(
        len(unoptimized_ie_ea_structures_elytes),
        "elyte calcs to run for unoptimized IE/EA",
    )
    unoptimized_ie_ea_structures_metal_complexes_arch = (
        sample_structures_for_unoptimized_ie_ea(metal_complexes_arch, 1000, "mca_idx")
    )
    print(
        len(unoptimized_ie_ea_structures_metal_complexes_arch),
        "Architector metal complex calcs to run for unoptimized IE/EA",
    )
    unoptimized_ie_ea_structures_metal_complexes_cod = (
        sample_structures_for_unoptimized_ie_ea(metal_complexes_cod, 4000, "mcc_idx")
    )
    print(
        len(unoptimized_ie_ea_structures_metal_complexes_cod),
        "COD metal complex calcs to run for unoptimized IE/EA",
    )

    for atoms in unoptimized_ie_ea_structures_elytes:
        write(
            os.path.join(
                output_path,
                "ieea_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + "_"
                + str(atoms.info["charge"])
                + "_"
                + str(atoms.info["spin"])
                + ".traj",
            ),
            atoms,
        )
    for atoms in unoptimized_ie_ea_structures_metal_complexes_arch:
        write(
            os.path.join(
                output_path,
                "ieea_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + "_"
                + str(atoms.info["charge"])
                + "_"
                + str(atoms.info["spin"])
                + ".traj",
            ),
            atoms,
        )
    for atoms in unoptimized_ie_ea_structures_metal_complexes_cod:
        write(
            os.path.join(
                output_path,
                "ieea_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + "_"
                + str(atoms.info["charge"])
                + "_"
                + str(atoms.info["spin"])
                + ".traj",
            ),
            atoms,
        )

    unoptimized_spin_gap_structures_elytes = sample_structures_for_unoptimized_spin_gap(
        elytes, 1000, "e_idx"
    )
    print(
        len(unoptimized_spin_gap_structures_elytes),
        "elyte calcs to run for unoptimized spin gap",
    )
    unoptimized_spin_gap_structures_metal_complexes_arch = (
        sample_structures_for_unoptimized_spin_gap(
            metal_complexes_arch, 3000, "mca_idx"
        )
    )
    print(
        len(unoptimized_spin_gap_structures_metal_complexes_arch),
        "Architector metal complex calcs to run for unoptimized spin gap",
    )
    unoptimized_spin_gap_structures_metal_complexes_cod = (
        sample_structures_for_unoptimized_spin_gap(metal_complexes_cod, 5000, "mcc_idx")
    )
    print(
        len(unoptimized_spin_gap_structures_metal_complexes_cod),
        "COD metal complex calcs to run for unoptimized spin gap",
    )

    for atoms in unoptimized_spin_gap_structures_elytes:
        write(
            os.path.join(
                output_path,
                "spingap_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + "_"
                + str(atoms.info["charge"])
                + "_"
                + str(atoms.info["spin"])
                + ".traj",
            ),
            atoms,
        )
    for atoms in unoptimized_spin_gap_structures_metal_complexes_arch:
        write(
            os.path.join(
                output_path,
                "spingap_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + "_"
                + str(atoms.info["charge"])
                + "_"
                + str(atoms.info["spin"])
                + ".traj",
            ),
            atoms,
        )
    for atoms in unoptimized_spin_gap_structures_metal_complexes_cod:
        write(
            os.path.join(
                output_path,
                "spingap_"
                + atoms.info["idx_name"]
                + "_"
                + str(atoms.info["idx"])
                + "_"
                + str(atoms.info["charge"])
                + "_"
                + str(atoms.info["spin"])
                + ".traj",
            ),
            atoms,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=".")
    parser.add_argument("--output_path", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
