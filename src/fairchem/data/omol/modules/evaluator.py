from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
from ase.build import minimize_rotation_and_translation
from pymatgen.io.ase import MSONAtoms
from scipy.optimize import linear_sum_assignment


def rmsd(orca_atoms, mlip_atoms):
    """
    Calculate the RMSD between atoms optimized with ORCA and the MLIP,
    where we assume that ORCA atoms have sensible bonding.
    """
    orca_atoms = MSONAtoms.from_dict(orca_atoms)
    mlip_atoms = MSONAtoms.from_dict(mlip_atoms)

    minimize_rotation_and_translation(orca_atoms, mlip_atoms)
    return np.sqrt(
        np.mean(
            np.sum(
                (orca_atoms.get_positions() - mlip_atoms.get_positions()) ** 2, axis=1
            )
        )
    )


def interaction_energy_and_forces(results, supersystem):
    """
    For a supersystem (e.g. a protein-ligand complex), calculate the
    interaction energy and forces with each individual component (e.g.
    the protein and the ligand) in the complex.

    We assume that the supersystem is the sum of the individual components
    and all are present.

    `results` looks like:
    {
        "system_name": {
            "ligand_pocket": {
                "energy": float,
                "forces": list of floats,
                "atoms": dict (MSONAtoms format)
            },
            "ligand": {
                "energy": float,
                "forces": list of floats,
                "atoms": dict (MSONAtoms format)
            },
            "pocket": {
                "energy": float,
                "forces": list of floats,
                "atoms": dict (MSONAtoms format)
            }
        }
    }

    Args:
        results (dict): Results from ORCA or MLIP calculations.
        supersystem (str): The name of the supersystem (e.g. "ligand_pocket")

    Returns:
        interaction_energy, interaction_forces
    """
    ixn_energy = {}
    ixn_forces = {}
    for system_name, components in results.items():
        indices_found = set()
        ixn_energy[system_name] = components[supersystem]["energy"]
        ixn_forces[system_name] = np.array(components[supersystem]["forces"])

        # Make a map from atomic symbol/position to index in the supersystem
        # so that we can map the component atoms to the supersystem atoms
        supersystem_atoms = MSONAtoms.from_dict(components[supersystem]["atoms"])
        supersys_map = {
            (at.symbol, tuple(at.position)): at.index for at in supersystem_atoms
        }

        # Loop over the components that make up the supersystem
        for component_name, component_data in components.items():
            if component_name == supersystem:
                continue
            ixn_energy[system_name] -= component_data["energy"]
            component_atoms = MSONAtoms.from_dict(component_data["atoms"])
            for at in component_atoms:
                component_at_key = (at.symbol, tuple(at.position))
                supersystem_at_idx = supersys_map.get(component_at_key)
                if supersystem_at_idx is None:
                    raise ValueError(
                        f"Atom {at.symbol} at position {at.position} in "
                        f"component {component_name} not found in supersystem."
                    )
                indices_found.add(supersystem_at_idx)
                ixn_forces[system_name][supersystem_at_idx] -= np.array(
                    component_data["forces"]
                )[at.index]
        assert len(indices_found) == len(supersystem_atoms)

    return ixn_energy, ixn_forces


def spin_deltas(results):
    """
    Calculate deltaE and deltaF values for the spin gap evaluation task.

    `results` looks like:
    {
        "system_name": {
            "1": {
                "energy": float,
                "forces": list of floats,
                "atoms": dict (MSONAtoms format)
            },
            "3": {
                "energy": float,
                "forces": list of floats,
                "atoms": dict (MSONAtoms format)
            }
    }


    Args:
        results (dict): Results from ORCA or MLIP calculations performed at
        different spins.

    Returns:
        deltaE (dict), deltaF (dict)
    """
    deltaE = {}
    deltaF = {}
    for identifier, spin_states in results.items():
        deltaE[identifier] = {}
        deltaF[identifier] = {}
        max_spin = max(spin_states, key=lambda x: int(x))
        high_spin_data = spin_states[max_spin]
        for spin, spin_data in spin_states.items():
            if spin == max_spin:
                continue
            deltaE[identifier][spin] = high_spin_data["energy"] - spin_data["energy"]
            deltaF[identifier][spin] = np.array(high_spin_data["forces"]) - np.array(
                spin_data["forces"]
            )
    return deltaE, deltaF


def charge_deltas(results):
    """
    Calculate deltaE and deltaF values for adding and removing electrons

    Args:
        results (dict): Results from ORCA or MLIP calculations performed
        at different charges.

    Returns:
        deltaE (dict), deltaF (dict)
    """
    deltaE = {}
    deltaF = {}
    for identifier, charge_states in results.items():
        charges = sorted(charge_states, key=lambda x: int(x))
        assert charges == [str(i) for i in range(int(charges[0]), int(charges[0]) + 3)]
        deltaE[identifier] = {"add_electron": {}, "remove_electron": {}}
        deltaF[identifier] = {"add_electron": {}, "remove_electron": {}}

        # Verify that we have exactly one spin state for the base charge
        assert len(charge_states[charges[1]]) == 1
        base_charge_data = next(iter(charge_states[charges[1]].values()))
        orig_energy = base_charge_data["energy"]
        orig_forces = np.array(base_charge_data["forces"])
        for charge_val, tag in [
            (str(charges[0]), "add_electron"),
            (str(charges[2]), "remove_electron"),
        ]:
            for spin, spin_data in charge_states[charge_val].items():
                deltaE[identifier][tag][spin] = spin_data["energy"] - orig_energy
                deltaF[identifier][tag][spin] = (
                    np.array(spin_data["forces"]) - orig_forces
                )
    return deltaE, deltaF


def rmsd_mapping(structs0, structs1):
    """
    Map two conformer ensembles via linear sum assignment on an RMSD
    cost matrix.
    """
    cost_matrix = np.zeros(shape=(len(structs0), len(structs1)))
    for ii, key0 in enumerate(structs0):
        for jj, key1 in enumerate(structs1):
            cost_matrix[ii][jj] = rmsd(
                structs0[key0]["final"]["atoms"],
                structs1[key1]["final"]["atoms"],
            )
    row_ind, column_ind = linear_sum_assignment(cost_matrix)
    assert (row_ind == sorted(row_ind)).all()
    mapping = {}
    for ii, jj in zip(row_ind, column_ind):
        mapping[list(structs0)[ii]] = list(structs1)[jj]
    return mapping, cost_matrix[row_ind, column_ind]


def ligand_strain_processing(results):
    """
    Process results for the ligand strain evaluation task.
    Calculate the strain energy as the difference in energy between
    the global minimum and the loosely optimized ligand-in-pocket structure.
    Also save the global minimum structure for RMSD calculations.

    Args:
        results (dict): Results from ORCA or MLIP calculations.

    Returns:
        dict: Processed results for the ligand strain evaluation task.
    """
    processed_results = defaultdict(dict)
    for identifier, data in results.items():
        min_energy_struct = min(
            data["gas_phase"].values(),
            key=lambda x: x["final"]["energy"],
        )
        min_energy = min_energy_struct["final"]["energy"]
        processed_results[identifier]["global_min"] = min_energy_struct
        processed_results[identifier]["strain_energy"] = (
            data["bioactive"]["energy"] - min_energy
        )
    return processed_results


def get_one_prot_diff_name_pairs(names):
    """
    Get all pairs of names that have a charge difference of 1.

    Assumes that the names are in the format "name_charge_spin"
    """
    name_pairs = []
    for name0, name1 in itertools.combinations(names, 2):
        name0_charge = int(name0.split("_")[-2])
        name1_charge = int(name1.split("_")[-2])
        if abs(name0_charge - name1_charge) == 1:
            name_pairs.append((name0, name1))
    return name_pairs


def sr_or_lr(name):
    if "short_range" in name:
        return "sr"
    elif "long_range" in name:
        return "lr"
    else:
        raise NotImplementedError


# Evaluator functions


def ligand_pocket(orca_results, mlip_results):
    """
    Calculate error metrics for ligand pocket evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for ligand pocket evaluation task
    """
    energy_mae = 0
    forces_mae = 0
    interaction_energy_mae = 0
    interaction_forces_mae = 0
    orca_interaction_energy, orca_interaction_forces = interaction_energy_and_forces(
        orca_results, "ligand_pocket"
    )
    mlip_interaction_energy, mlip_interaction_forces = interaction_energy_and_forces(
        mlip_results, "ligand_pocket"
    )
    force_denom = 0
    ixn_force_denom = 0
    for identifier, orca_components in orca_results.items():
        assert len(orca_components) == 3
        assert len(mlip_results[identifier]) == 3
        for component_identifier, orca_data in orca_components.items():
            mlip_data = mlip_results[identifier][component_identifier]
            energy_mae += abs(orca_data["energy"] - mlip_data["energy"])
            forces_mae += np.sum(
                np.abs(np.array(orca_data["forces"]) - np.array(mlip_data["forces"]))
            )
            force_denom += 3 * len(orca_data["forces"])
        interaction_energy_mae += abs(
            orca_interaction_energy[identifier] - mlip_interaction_energy[identifier]
        )
        interaction_forces_mae += np.sum(
            np.abs(
                orca_interaction_forces[identifier]
                - mlip_interaction_forces[identifier]
            )
        )
        ixn_force_denom += 3 * len(orca_interaction_forces[identifier])

    results = {
        "energy_mae": energy_mae / (3 * len(orca_results)),
        "forces_mae": forces_mae / force_denom,
        "interaction_energy_mae": interaction_energy_mae / len(orca_results),
        "interaction_forces_mae": interaction_forces_mae / ixn_force_denom,
    }
    return results


def ligand_strain(orca_results, mlip_results):
    """
    Calculate error metrics for ligand strain evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for ligand strain evaluation task
    """
    processed_orca_results = ligand_strain_processing(orca_results)
    processed_mlip_results = ligand_strain_processing(mlip_results)
    strain_energy_mae = 0
    global_min_rmsd = 0
    for identifier in orca_results:
        strain_energy_mae += abs(
            processed_orca_results[identifier]["strain_energy"]
            - processed_mlip_results[identifier]["strain_energy"]
        )
        global_min_rmsd += rmsd(
            processed_orca_results[identifier]["global_min"]["final"]["atoms"],
            processed_mlip_results[identifier]["global_min"]["final"]["atoms"],
        )

    results = {
        "strain_energy_mae": strain_energy_mae / len(orca_results),
        "global_min_rmsd": global_min_rmsd / len(orca_results),
    }
    return results


def geom_conformers(orca_results, mlip_results):
    """
    Calculate error metrics for conformer evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for type1 conformer evaluation task
    """
    ensemble_rmsd = 0
    deltaE_mae = 0
    for family_identifier, orca_structs in orca_results.items():
        mlip_structs = mlip_results[family_identifier]
        mapping, cost_vector = rmsd_mapping(orca_structs, mlip_structs)
        ensemble_rmsd += cost_vector.mean()

        orca_min_energy_id, min_energy_struct = min(
            orca_structs.items(), key=lambda x: x[1]["final"]["energy"]
        )
        orca_min_energy = min_energy_struct["final"]["energy"]
        mlip_energy_of_orca_min = mlip_structs[mapping[orca_min_energy_id]]["final"][
            "energy"
        ]
        for conformer_identifier, orca_struct in orca_structs.items():
            if conformer_identifier == orca_min_energy_id:
                continue
            orca_deltaE = orca_struct["final"]["energy"] - orca_min_energy
            mlip_deltaE = (
                mlip_structs[mapping[conformer_identifier]]["final"]["energy"]
                - mlip_energy_of_orca_min
            )
            deltaE_mae += abs(orca_deltaE - mlip_deltaE) / (len(orca_structs) - 1)

    results = {
        "ensemble_rmsd": ensemble_rmsd / len(orca_results),
        "deltaE_mae": deltaE_mae / len(orca_results),
    }
    return results


def protonation_energies(orca_results, mlip_results):
    """
    Calculate error metrics for the type1 protonation energies evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for protonation energies evaluation task
    """
    deltaE_mae = 0
    tot_rmsd = 0
    for identifier, orca_structs in orca_results.items():
        mlip_structs = mlip_results[identifier]
        one_prot_diff_name_pairs = get_one_prot_diff_name_pairs(list(orca_structs))
        for name, orca_struct in orca_structs.items():
            tot_rmsd += rmsd(
                orca_struct["final"]["atoms"],
                mlip_structs[name]["final"]["atoms"],
            ) / len(orca_structs)
        for name0, name1 in one_prot_diff_name_pairs:
            orca_deltaE = (
                orca_structs[name0]["final"]["energy"]
                - orca_structs[name1]["final"]["energy"]
            )
            mlip_deltaE = (
                mlip_structs[name0]["final"]["energy"]
                - mlip_structs[name1]["final"]["energy"]
            )
            deltaE_mae += abs(orca_deltaE - mlip_deltaE) / len(one_prot_diff_name_pairs)

    results = {
        "deltaE_mae": deltaE_mae / len(orca_results),
        "rmsd": tot_rmsd / len(orca_results),
    }
    return results


def unoptimized_ie_ea(orca_results, mlip_results):
    """
    Calculate error metrics for unoptimized IE and EA calculations.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for unoptimized IE and EA calculations.
    """
    energy_mae = 0
    forces_mae = 0
    force_denom = 0
    deltaE_mae = 0
    deltaF_mae = 0
    deltaF_denom = 0
    orca_deltaE, orca_deltaF = charge_deltas(orca_results)
    mlip_deltaE, mlip_deltaF = charge_deltas(mlip_results)
    for identifier, orca_data in orca_results.items():
        mlip_data = mlip_results[identifier]
        num_samples = sum(len(structs) for structs in orca_data.values())
        num_deltas = sum(len(structs) for structs in orca_deltaE[identifier].values())
        assert num_samples - 1 == num_deltas
        for charge, orca_charge_states in orca_data.items():
            for spin, orca_spin_state in orca_charge_states.items():
                mlip_spin_state = mlip_data[charge][spin]
                energy_mae += (
                    abs(orca_spin_state["energy"] - mlip_spin_state["energy"])
                    / num_samples
                )
                forces_mae += np.sum(
                    np.abs(
                        np.array(orca_spin_state["forces"])
                        - np.array(mlip_spin_state["forces"])
                    )
                )
                force_denom += 3 * len(orca_spin_state["forces"])

        for tag in ["add_electron", "remove_electron"]:
            for spin in orca_deltaE[identifier][tag]:
                deltaE_mae += (
                    abs(
                        orca_deltaE[identifier][tag][spin]
                        - mlip_deltaE[identifier][tag][spin]
                    )
                    / num_deltas
                )
                deltaF_mae += np.sum(
                    np.abs(
                        orca_deltaF[identifier][tag][spin]
                        - mlip_deltaF[identifier][tag][spin]
                    )
                )
                deltaF_denom += 3 * len(orca_deltaF[identifier][tag][spin])

    results = {
        "energy_mae": energy_mae / len(orca_results),
        "forces_mae": forces_mae / force_denom,
        "deltaE_mae": deltaE_mae / len(orca_results),
        "deltaF_mae": deltaF_mae / deltaF_denom,
    }
    return results


def compute_distance_scaling_metrics(
    pes_curve, mlip_results, orca_min_point, orca_min_data
):
    """
    Compute metrics for distance scaling eval for a single PES curve

    :param pes_curve: specification of points on PES in the short or
                      long range regime only, also contains ORCA data
    :param mlip_results: results from all points alng
    :param orca_min_point: name of reference point for ddE
    :param orca_min_data: data for reference point for ddE
    :return: ddE, and ddF metrics for the given PES curve
    """
    ddEnergy_mae_system = 0
    ddForces_mae_system = 0
    # We need to keep track of the number of differences for normalization
    # this may or may not equal the total number of points depending on
    # where the reference is.
    n_deltas = 0
    n_deltas_atoms = 0

    mlip_ref_data = mlip_results[orca_min_point]
    orca_ref_forces = np.array(orca_min_data["forces"])
    mlip_ref_forces = np.array(mlip_ref_data["forces"])

    for pes_point, orca_data in pes_curve.items():
        # pes_point is a given point on the PES curve

        # We exclude the reference point DeltaDelatE since it is
        # definitionally zero.
        if pes_point == orca_min_point:
            continue

        mlip_data = mlip_results[pes_point]
        # Energy differences: We compute the DeltaDeltaE between ML and
        # DFT of the energy diff of each point to that reference point
        ml_forces = np.array(mlip_data["forces"])
        orca_forces = np.array(orca_data["forces"])

        orca_deltaE = orca_data["energy"] - orca_min_data["energy"]
        orca_deltaF = orca_forces - orca_ref_forces
        mlip_deltaE = mlip_data["energy"] - mlip_ref_data["energy"]
        mlip_deltaF = ml_forces - mlip_ref_forces

        ddEnergy_mae_system += abs(orca_deltaE - mlip_deltaE)
        ddForces_mae_system += np.sum(np.abs(orca_deltaF - mlip_deltaF))
        n_deltas += 1
        n_deltas_atoms += len(orca_deltaF)

    # Normalize MAEs due to number of points on PES curve
    ddEnergy_mae_system /= n_deltas
    ddForces_mae_system /= n_deltas_atoms

    return ddEnergy_mae_system, ddForces_mae_system


def distance_scaling(orca_results, mlip_results):
    """
    Calculate error metrics for distance scaling evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for distance scaling evaluation task
    """
    deltadeltaE_mae = {"sr": 0, "lr": 0}
    deltadeltaF_mae = {"sr": 0, "lr": 0}
    n_systems_with_sr = 0
    n_systems_with_lr = 0
    for vertical, samples in orca_results.items():
        # vertical is e.g. 'biomolecules', samples is a dict keyed on
        # identifiers with values being a given PES curve (i.e. many points)
        for identifier, orca_curve in samples.items():
            mlip_curve = mlip_results[vertical][identifier]
            sr_orca_curve = {k: v for k, v in orca_curve.items() if sr_or_lr(k) == "sr"}
            lr_orca_curve = {k: v for k, v in orca_curve.items() if sr_or_lr(k) == "lr"}
            if sr_orca_curve:
                orca_min_point, orca_min_data = min(
                    sr_orca_curve.items(), key=lambda x: x[1]["energy"]
                )
            else:
                orca_min_point, orca_min_data = min(
                    lr_orca_curve.items(), key=lambda x: x[1]["energy"]
                )
            # We require at least 2 points for short-range because, if there
            # is only 1, it won't actually contribute to the metrics since
            # that one point is the reference. Because we require that there
            # are at least 5 points on the PES, if there are >0 in LR, then
            # they will contribute so no need to adjust there.
            if len(sr_orca_curve) == 1:
                sr_orca_curve = {}
            if len(lr_orca_curve) == 1 and not sr_orca_curve:
                lr_orca_curve = {}

            n_systems_with_sr += len(sr_orca_curve)
            n_systems_with_lr += len(lr_orca_curve)
            for pes_curve, range_label in zip(
                (sr_orca_curve, lr_orca_curve), ("sr", "lr")
            ):
                if not pes_curve:
                    continue

                (
                    ddEnergy_mae_system,
                    ddForces_mae_system,
                ) = compute_distance_scaling_metrics(
                    pes_curve, mlip_curve, orca_min_point, orca_min_data
                )

                # Package in overall metric
                deltadeltaE_mae[range_label] += ddEnergy_mae_system
                deltadeltaF_mae[range_label] += ddForces_mae_system

    results = {
        "sr_ddE_mae": deltadeltaE_mae["sr"] / n_systems_with_sr,
        "sr_ddF_mae": deltadeltaF_mae["sr"] / n_systems_with_sr,
        "lr_ddE_mae": deltadeltaE_mae["lr"] / n_systems_with_lr,
        "lr_ddF_mae": deltadeltaF_mae["lr"] / n_systems_with_lr,
    }
    return results


def unoptimized_spin_gap(orca_results, mlip_results):
    """
    Calculate error metrics for unoptimized spin gap evaluation task.

    Args:
        orca_results (dict): Results from ORCA calculations.
        mlip_results (dict): Results from MLIP calculations.

    Returns:
        dict: Error metrics for unoptimized spin gap evaluation task
    """
    energy_mae = 0
    forces_mae = 0
    deltaE_mae = 0
    deltaF_mae = 0
    force_denom = 0
    deltaF_denom = 0
    orca_deltaE, orca_deltaF = spin_deltas(orca_results)
    mlip_deltaE, mlip_deltaF = spin_deltas(mlip_results)
    for identifier, orca_data in orca_results.items():
        mlip_data = mlip_results[identifier]
        spins = sorted(orca_data, key=lambda x: int(x), reverse=True)
        for spin in spins:
            energy_mae += abs(
                orca_data[spin]["energy"] - mlip_data[spin]["energy"]
            ) / len(spins)
            forces_mae += np.sum(
                np.abs(
                    np.array(orca_data[spin]["forces"])
                    - np.array(mlip_data[spin]["forces"])
                )
            )
            force_denom += 3 * len(orca_data[spin]["forces"])
            if spin == spins[0]:
                continue
            deltaE_mae += abs(
                orca_deltaE[identifier][spin] - mlip_deltaE[identifier][spin]
            ) / (len(spins) - 1)
            deltaF_mae += np.sum(
                np.abs(orca_deltaF[identifier][spin] - mlip_deltaF[identifier][spin])
            )
            deltaF_denom += 3 * len(orca_deltaF[identifier][spin])

    results = {
        "energy_mae": energy_mae / len(orca_results),
        "forces_mae": forces_mae / force_denom,
        "deltaE_mae": deltaE_mae / len(orca_results),
        "deltaF_mae": deltaF_mae / deltaF_denom,
    }
    return results


def singlepoint(orca_results, mlip_results):
    energy_mae = 0
    energy_per_atom_mae = 0
    forces_mae = 0
    natoms = 0
    for identifier, orca_data in orca_results.items():
        mlip_data = mlip_results[identifier]
        _energy_mae = np.abs(orca_data["energy"] - mlip_data["energy"])
        _natoms = len(orca_data["forces"])
        forces_mae += np.sum(
            np.abs(np.array(orca_data["forces"]) - np.array(mlip_data["forces"]))
        )
        natoms += _natoms
        energy_mae += _energy_mae
        energy_per_atom_mae += _energy_mae / _natoms

    metrics = {
        "energy_mae": energy_mae / len(orca_results),
        "energy_per_atom_mae": energy_per_atom_mae / len(orca_results),
        "forces_mae": forces_mae / (3 * natoms),
    }

    return metrics
