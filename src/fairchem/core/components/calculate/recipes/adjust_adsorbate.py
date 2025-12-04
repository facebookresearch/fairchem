"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from ase.geometry import cellpar_to_cell


def get_middle_mol(atoms):
    """
    Returns atomic positions of the adsorbate molecule nearest the center
    of a 2x2x2 MOF supercell.

    Args:
        atoms: ASE Atoms object containing a 2x2x2 supercell of a unit cell
               with one adsorbate molecule.

    Returns:
        List of atomic positions for the center adsorbate molecule.
    """
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    symbols = atoms.get_chemical_symbols()

    center_of_unit_cell = 0.5 * (cell[0] + cell[1] + cell[2])

    # Distances from each atom to the center
    distances_to_center = np.linalg.norm(pos - center_of_unit_cell, axis=1)
    closest_atom_index = np.argmin(distances_to_center)
    target_point = pos[closest_atom_index]

    # Distances from each atom to the closest atom
    distances_to_target = np.linalg.norm(pos - target_point, axis=1)
    closest_atom_indices = np.argsort(distances_to_target)[:3]

    # Skip the center atom itself
    idxs = [closest_atom_index, closest_atom_indices[1], closest_atom_indices[2]]

    # Organize atoms based on type (CO2 or H2O)
    if 'C' in symbols:  # CO2
        if symbols[idxs[1]] == 'C':
            idxs[1], idxs[0] = idxs[0], idxs[1]
        elif symbols[idxs[2]] == 'C':
            idxs[2], idxs[0] = idxs[0], idxs[2]
    else:  # H2O
        if symbols[idxs[0]] == 'O':
            idxs[2], idxs[0] = idxs[0], idxs[2]
        elif symbols[idxs[1]] == 'O':
            idxs[2], idxs[1] = idxs[1], idxs[2]

    middle_pos = [list(pos[idx]) for idx in idxs]

    return middle_pos


def get_large_adsorbate(ads):
    """
    Creates an ASE Atoms object for a lone adsorbate molecule centered
    in a large unit cell (20x20x20 Å).

    Args:
        ads: ASE Atoms object containing an adsorbate molecule in a MOF unit cell.

    Returns:
        ASE Atoms object containing the centered adsorbate molecule.
    """
    ads_big = ads.repeat((2, 2, 2))
    pos_mid = get_middle_mol(ads_big)
    ads.positions = pos_mid

    contains_carbon = any(atom.symbol == 'C' for atom in ads)

    # Set atomic numbers
    if contains_carbon:  # CO2
        ads.set_atomic_numbers([6, 8, 8])
    else:  # H2O
        ads.set_atomic_numbers([1, 1, 8])

    pos_orig = ads.positions.copy()

    # Delete side atoms
    if contains_carbon:
        for i in range(len(ads) - 1, -1, -1):
            if ads[i].symbol == 'O':
                del ads[i]
    else:
        for i in range(len(ads) - 1, -1, -1):
            if ads[i].symbol == 'H':
                del ads[i]

    # Change unit cell
    new_cell = cellpar_to_cell([20, 20, 20, 90, 90, 90])
    ads.set_cell(new_cell, scale_atoms=True)
    ads.wrap()
    ads.center()

    # Re-add side atoms
    if contains_carbon:  # CO2
        vec = ads.positions[0] - pos_orig[0]
        ads.append('O')
        ads[-1].position = pos_orig[1] + vec
        ads.append('O')
        ads[-1].position = pos_orig[2] + vec
    else:  # H2O
        vec = ads.positions[0] - pos_orig[2]
        ads.append('H')
        ads[-1].position = pos_orig[0] + vec
        ads.append('H')
        ads[-1].position = pos_orig[1] + vec

    return ads


def shift_adsorbate(ads, new_cell):
    """
    Shifts an adsorbate molecule to a new MOF unit cell while preserving
    its relative position.

    Only compatible with CO2 and H2O adsorbates.

    Args:
        ads: ASE Atoms object containing an adsorbate molecule in the original MOF unit cell.
        new_cell: ASE unit cell for desired MOF geometry.

    Returns:
        ASE Atoms object containing the adsorbate molecule in the new unit cell.
    """
    ads_big = ads.repeat((2, 2, 2))
    pos_mid = get_middle_mol(ads_big)
    ads.positions = pos_mid

    contains_carbon = any(atom.symbol == 'C' for atom in ads)

    if contains_carbon:  # CO2
        ads.set_atomic_numbers([6, 8, 8])
    else:  # H2O
        ads.set_atomic_numbers([1, 1, 8])

    pos_orig = ads.positions.copy()

    # Delete side atoms
    if contains_carbon:
        for i in range(len(ads) - 1, -1, -1):
            if ads[i].symbol == 'O':
                del ads[i]
    else:
        for i in range(len(ads) - 1, -1, -1):
            if ads[i].symbol == 'H':
                del ads[i]

    ads.set_cell(new_cell, scale_atoms=True)
    ads.wrap()

    # Re-add side atoms
    if contains_carbon:  # CO2
        vec = ads.positions[0] - pos_orig[0]
        ads.append('O')
        ads[-1].position = pos_orig[1] + vec
        ads.append('O')
        ads[-1].position = pos_orig[2] + vec
    else:  # H2O
        vec = ads.positions[0] - pos_orig[2]
        ads.append('H')
        ads[-1].position = pos_orig[0] + vec
        ads.append('H')
        ads[-1].position = pos_orig[1] + vec

    return ads
