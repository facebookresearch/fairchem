"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Helper scripts to run phonon calculations

- Compute phonon frequencies at commensurate points
- Compute thermal properties with Fourier interpolation
- Optionally compute and plot band-structures and DOS

Needs phonopy installed
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ase.units
import numpy as np
from ase import Atoms
from ase.build.supercells import make_supercell
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from monty.dev import requires

try:
    from pymatgen.core import Structure
    from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

    pmg_installed = True
except ImportError:
    pmg_installed = False

try:
    from phonopy import Phonopy
    from phonopy.harmonic.dynmat_to_fc import get_commensurate_points

    phonopy_installed = True
except ImportError:
    phonopy_installed = False


if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike, NDArray
    from phonopy.structure.atoms import PhonopyAtoms


THz_to_K = ase.units._hplanck * 1e12 / ase.units._k


@requires(phonopy_installed, message="Requires `phonopy` to be installed")
@requires(pmg_installed, message="Requires `pymatgen` to be installed")
def run_mdr_phonon_benchmark(
    mdr_phonon: Phonopy,
    calculator: Calculator,
    displacement: float = 0.01,
    run_relax: bool = True,
    fix_symm_relax: bool = False,
    symprec: int = 1e-4,
    symmetrize_fc: bool = False,
) -> dict:
    """Run a phonon calculation for a single datapoint of the MDR PBE dataset

    Properties computed for benchmark:
        - maximum frequency from phonon frequencies computed at supercell commensurate points
        - vibrational free energy, entropy and heat capacity computed with a [20, 20, 20] mesh

    Args:
        mdr_phonon: the baseline MDR Phonopy object
        calculator: an Ase Calculator
        displacement: displacement step to compute forces (A)
        run_relax: run a structural relaxation
        fix_symm_relax: wether to fix symmetry in relaxation
        symprec: symmetry precision used by phonopy
        symmetrize_fc: symmetrize force constants

    Returns:
        dict: dictionary of computed properties
    """

    if run_relax:
        # relax the primitive cell instead of the unitcell for efficiency
        primcell = get_pmg_structure(mdr_phonon.primitive).to_ase_atoms()

        if fix_symm_relax:
            primcell.set_constraint(FixSymmetry(primcell))

        primcell.calc = calculator
        opt = FIRE(FrechetCellFilter(primcell), logfile=None)
        opt.run(fmax=0.005, steps=500)
        natoms = len(primcell.positions)
        final_energy_per_atom = primcell.get_potential_energy() / natoms
        final_volume_per_atom = primcell.get_volume() / natoms
        if mdr_phonon.primitive_matrix is not None:
            P = np.asarray(np.linalg.inv(mdr_phonon.primitive_matrix.T), dtype=np.intc)
            unitcell = make_supercell(primcell, P)
        else:  # assume prim is the same as unit
            # can always check for good measure
            # assert np.allclose(mdr_phonon.unitcell, mdr_phonon.primitive.cell)
            unitcell = primcell
    else:
        unitcell = mdr_phonon.unitcell
        final_energy_per_atom = np.nan
        final_volume_per_atom = np.nan

    phonon = get_phonopy_object(
        unitcell,
        displacement=displacement,
        supercell_matrix=mdr_phonon.supercell_matrix,
        primitive_matrix=mdr_phonon.primitive_matrix,
        symprec=symprec,
    )
    produce_force_constants(phonon, calculator, symmetrize=symmetrize_fc)

    results = {
        "frequencies": calculate_phonon_frequencies(phonon) * THz_to_K,
        "energy_per_atom": final_energy_per_atom,
        "volume_per_atom": final_volume_per_atom,
        **calculate_thermal_properties(phonon, t_step=75, t_max=600, t_min=0),
    }

    return results


def get_phonopy_object(
    atoms: PhonopyAtoms | Atoms | Structure,
    displacement: float = 0.01,
    supercell_matrix: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
    primitive_matrix: ArrayLike | None = None,
    symprec: int = 1e-5,
    **phonopy_kwargs,
) -> Phonopy:
    """Create a Phonopy api object from ase Atoms.

    Args:
        atoms: Phonopy atoms, ASE atoms object or a pmg Structure
        displacement: displacement step to compute forces (A)
        supercell_matrix: transformation matrix to super cell from unit cell.
        primitive_matrix: transformation matrix to primitive cell from unit cell.
        symprec: symmetry precision
        phonopy_kwargs: additional keyword arguments to initialize Phonopy API object
    Returns:
        Phonopy: api object
    """
    if isinstance(atoms, Atoms):
        atoms = Structure.from_ase_atoms(atoms)

    if isinstance(atoms, Structure):
        atoms = get_phonopy_structure(atoms)

    supercell_matrix = np.ascontiguousarray(supercell_matrix, dtype=int)
    phonon = Phonopy(
        atoms,
        supercell_matrix,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
        **phonopy_kwargs,
    )
    phonon.generate_displacements(distance=displacement)
    return phonon


def produce_force_constants(
    phonon: Phonopy, calculator: Calculator, symmetrize: bool = False
) -> None:
    """Run force calculations and produce force constants with Phonopy

    Args:
        phonon: a Phonopy API object
        calculator: an ASE Calculator
        symmetrize: symmetrize force constants
    """

    phonon.forces = [
        calculator.get_forces(get_pmg_structure(supercell).to_ase_atoms())
        for supercell in phonon.supercells_with_displacements
    ]
    phonon.produce_force_constants()

    if symmetrize:
        phonon.symmetrize_force_constants()
        phonon.symmetrize_force_constants_by_space_group()


def calculate_phonon_frequencies(
    phonon: Phonopy, qpoints: ArrayLike | None = None
) -> NDArray:
    """Calculate phonon frequencies at a given set of qpoints.

    Args:
        phonon: a Phonopy api object with displacements generated
        qpoints: ndarray of qpoints to calculate phonon frequencies at. If none are given, the supercell commensurate
            points will be used

    Returns:
        NDArray: ndarray of phonon frequencies in THz, (qpoints, frequencies)
    """
    if qpoints is None:
        qpoints = get_commensurate_points(phonon.supercell_matrix)

    frequencies = np.stack([phonon.get_frequencies(q) for q in qpoints])

    return frequencies


def calculate_thermal_properties(
    phonon: Phonopy, t_min, t_max, t_step, mesh: ArrayLike = (20, 20, 20)
) -> dict[str, float]:
    """Calculate thermal properties from initialized phonopy object

    Thermal properties include: vibrational free energy, entropy and heat capacity

    Args:
        phonon: a Phonopy api object with displacements generated
        t_min: minimum temperature
        t_max: max temperature
        t_step: temperature step between min and max
        mesh: qpoint mesh to compute properties using Fourier interpolation

    Returns:
        dict: dictionary of computed properties
    """

    phonon.run_mesh(mesh)
    phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    return phonon.get_thermal_properties_dict()
