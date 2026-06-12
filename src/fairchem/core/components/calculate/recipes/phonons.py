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

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Sequence

import ase
import ase.units
import numpy as np
from ase import Atoms
from ase.build.supercells import make_supercell
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from monty.dev import requires

from fairchem.core.components.calculate.recipes.relax import relax_atoms
from fairchem.core.components.calculate.recipes.utils import scale_atoms

try:
    from pymatgen.core import Structure
    from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

    pmg_installed = True
except ImportError:
    pmg_installed = False

try:
    from phonopy import Phonopy, PhonopyQHA
    from phonopy.harmonic.dynmat_to_fc import get_commensurate_points

    phonopy_installed = True
except ImportError:
    phonopy_installed = False


if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from numpy.typing import ArrayLike, NDArray
    from phonopy.structure.atoms import PhonopyAtoms


THz_to_K = ase.units._hplanck * 1e12 / ase.units._k
kJmol2eV = ase.units.kJ / (ase.units.eV * ase.units.mol)


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
            P = np.asarray(np.linalg.inv(mdr_phonon.primitive_matrix.T), dtype=np.int32)
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
    """
    Calculate phonon frequencies at a given set of qpoints.

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


def calculate_total_dos(
    phonon: Phonopy,
    mesh: ArrayLike = (20, 20, 20),
    sigma: float | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    freq_pitch: float | None = None,
    use_tetrahedron_method: bool = True,
) -> dict[str, NDArray]:
    """
    Calculate the total phonon density of states.

    This is a standalone function for users who want DOS independently of thermal
    properties. Runs its own mesh integration and returns the DOS dict.

    Args:
        phonon: a Phonopy api object with force constants already computed
        mesh: qpoint mesh for Brillouin zone sampling
        sigma: smearing width for Gaussian broadening (ignored if use_tetrahedron_method=True)
        freq_min: minimum frequency for DOS (THz)
        freq_max: maximum frequency for DOS (THz)
        freq_pitch: frequency step for DOS (THz)
        use_tetrahedron_method: use tetrahedron method for DOS integration (default True)

    Returns:
        dict with keys 'frequency_points' (THz) and 'total_dos' (states/THz)
    """
    phonon.run_mesh(mesh)
    phonon.run_total_dos(
        sigma=sigma,
        freq_min=freq_min,
        freq_max=freq_max,
        freq_pitch=freq_pitch,
        use_tetrahedron_method=use_tetrahedron_method,
    )
    return phonon.get_total_dos_dict()


def calculate_thermal_properties(
    phonon: Phonopy,
    t_min,
    t_max,
    t_step,
    mesh: ArrayLike = (20, 20, 20),
    return_dos: bool = False,
) -> dict[str, float]:
    """
    Calculate thermal properties from initialized phonopy object.

    Thermal properties include: vibrational free energy, entropy and heat capacity.
    Optionally also returns the total phonon DOS.

    Args:
        phonon: a Phonopy api object with displacements generated
        t_min: minimum temperature
        t_max: max temperature
        t_step: temperature step between min and max
        mesh: qpoint mesh to compute properties using Fourier interpolation
        return_dos: if True, also compute and return the total DOS

    Returns:
        dict: dictionary of computed properties. Always includes 'temperatures',
            'free_energy', 'entropy', 'heat_capacity'. If return_dos is True, also
            includes 'frequency_points' and 'total_dos'.
    """
    phonon.run_mesh(mesh)
    phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    result = phonon.get_thermal_properties_dict()

    if return_dos:
        phonon.run_total_dos()
        result.update(phonon.get_total_dos_dict())

    return result


def calculate_vibrational_thermo(
    atoms,
    quasiharmonic: bool = True,
    *,
    atom_disp: float = 0.01,
    scale_factors: Sequence[float] = tuple(np.arange(0.94, 1.06, 0.02)),
    min_lengths: float | tuple[float, float, float] | None = 15.0,
    supercell_matrix: ArrayLike | None = None,
    relax_initial: bool = False,
    relax_scaled: bool = True,
    t_step: float = 10,
    t_max: float = 500,
    t_min: float = 0,
    fmax: float = 0.01,
    max_steps: int = 500,
    optimizer: str = "FIRE",
    eos: str = "vinet",
    pressure: float | None = None,
    compute_dos: bool = False,
) -> dict[str, ArrayLike]:
    """
    Calculate vibrational thermodynamic properties using either anharmonic (QHA) or harmonic (phonon) approach.

    Parameters:
    -----------
    atoms : Atoms object
        Atomic structure for which to calculate vibrational properties
    quasiharmonic: bool, default=True
        Whether to use QHA or harmonic (phonon) approach
    atom_disp : float, default=0.01
        Atomic displacement for finite difference calculations
    scale_factors : Sequence[float]
        Volume scale factors for QHA
    supercell_matrix : ArrayLike, default=None
        Matrix defining the supercell for phonon calculations
    min_lengths : float, default=15
        Minimum lattice vector length used to determine optimal supercell size
    relax_initial : bool, default=False
        Whether to relax the structure before calculations
    relax_scaled : bool, default=True
        If true will relax atomic positions and cell shape at constant volume of each scaled structure
    t_step : float, default=10
        Temperature step size in K
    t_max : float, default=500
        Maximum temperature in K
    t_min : float, default=0
        Minimum temperature in K
    fmax : float, default=0.01
        Force convergence criterion for structure optimization
    max_steps : int, default=500
        Maximum number of relaxation steps
    optimizer : str, default="FIRE"
        Optimization algorithm for structure relaxation
    eos : str, default="vinet"
        Equation of state to use for QHA calculations
    pressure : float | None, default=None
        External pressure for calculations
    compute_dos : bool, default=False
        If True, also compute and return the total phonon density of states.
        The DOS is temperature-independent in the harmonic approximation, so
        a single DOS is returned valid for all temperatures. In QHA mode, the
        DOS is computed at the equilibrium volume (scale_factor=1.0).

    Returns:
    --------
    dict[str, ArrayLike]
        Dictionary containing vibrational free energy corrections. If compute_dos
        is True, also includes 'frequency_points' and 'total_dos'.
    """
    calculator = atoms.calc

    if supercell_matrix is None and min_lengths is not None:
        supercell_matrix = np.diag(
            np.round(np.ceil(min_lengths / atoms.cell.lengths()))
        )

    if quasiharmonic:
        return _calc_nvt_npt(
            atoms,
            calculator,
            scale_factors=scale_factors,
            supercell_matrix=supercell_matrix,
            atom_disp=atom_disp,
            relax_initial=relax_initial,
            relax_scaled=relax_scaled,
            t_step=t_step,
            t_max=t_max,
            t_min=t_min,
            fmax=fmax,
            max_steps=max_steps,
            optimizer=optimizer,
            eos=eos,
            pressure=pressure,
            return_dos=compute_dos,
        )
    else:
        return _calc_nvt(
            atoms,
            calculator,
            supercell_matrix=supercell_matrix,
            atom_disp=atom_disp,
            relax_initial=relax_initial,
            t_step=t_step,
            t_max=t_max,
            t_min=t_min,
            fmax=fmax,
            max_steps=max_steps,
            optimizer=optimizer,
            return_dos=compute_dos,
        )


def _run_phonon_thermal(
    atoms,
    calculator,
    supercell_matrix,
    atom_disp,
    t_step,
    t_max,
    t_min,
    return_dos: bool = False,
):
    phonon = get_phonopy_object(
        atoms, displacement=atom_disp, supercell_matrix=supercell_matrix
    )
    produce_force_constants(phonon, calculator)
    return calculate_thermal_properties(
        phonon, t_min=t_min, t_max=t_max, t_step=t_step, return_dos=return_dos
    )


def _calc_nvt_npt(
    atoms,
    calculator,
    *,
    scale_factors,
    supercell_matrix,
    atom_disp,
    relax_initial,
    relax_scaled,
    t_step,
    t_max,
    t_min,
    fmax,
    max_steps,
    optimizer,
    eos,
    pressure,
    return_dos: bool = False,
) -> dict:
    if relax_initial:
        atoms.calc = calculator
        relax_atoms(
            atoms,
            steps=max_steps,
            fmax=fmax,
            optimizer_cls=getattr(ase.optimize, optimizer),
            cell_filter_cls=FrechetCellFilter,
        )

    if not np.isclose(scale_factors, 1).any():
        raise ValueError(
            "Cannot calculate harmonic properties if scale_factors does not include 1"
        )

    temperatures = np.arange(t_min, t_max + t_step, t_step)

    volumes = []
    electronic_energies = []
    free_energies = []
    entropies = []
    heat_capacities = []
    harmonic_properties = {}

    for scale_factor in scale_factors:
        scaled_atoms = scale_atoms(atoms, scale_factor)

        if relax_scaled:
            scaled_atoms.calc = calculator
            relax_atoms(
                scaled_atoms,
                steps=max_steps,
                fmax=fmax,
                optimizer_cls=getattr(ase.optimize, optimizer),
                cell_filter_cls=partial(FrechetCellFilter, constant_volume=True),
            )

        scaled_atoms.calc = calculator
        volumes.append(scaled_atoms.get_volume())
        electronic_energy = scaled_atoms.get_potential_energy()
        electronic_energies.append(electronic_energy)

        thermal = _run_phonon_thermal(
            scaled_atoms,
            calculator,
            supercell_matrix,
            atom_disp,
            t_step,
            t_max,
            t_min,
            return_dos=(return_dos and np.isclose(scale_factor, 1.0)),
        )
        free_energies.append(thermal["free_energy"])
        entropies.append(thermal["entropy"])
        heat_capacities.append(thermal["heat_capacity"])

        if np.isclose(scale_factor, 1.0):
            harmonic_properties = deepcopy(thermal)
            harmonic_properties["free_energy"] = (
                kJmol2eV * harmonic_properties["free_energy"] + electronic_energy
            )
            harmonic_properties["entropy"] = harmonic_properties["entropy"] * (
                kJmol2eV / 1000
            )
            harmonic_properties["heat_capacity"] = harmonic_properties[
                "heat_capacity"
            ] * (kJmol2eV / 1000)

    qha = PhonopyQHA(
        volumes=volumes,
        electronic_energies=electronic_energies,
        temperatures=temperatures,
        free_energy=np.transpose(free_energies),
        entropy=np.transpose(entropies),
        cv=np.transpose(heat_capacities),
        eos=eos,
        t_max=t_max,
        pressure=pressure,
    )

    # PhonopyQHA uses centered finite differences over the input temperature
    # grid (see ``_set_thermal_expansion`` etc. in ``phonopy/qha/core.py``), so
    # the QHA-derived arrays drop the high-temperature endpoint and are one
    # element shorter than the input grid. Truncate the harmonic arrays to the
    # same length so every value in the returned dict is on a single
    # consistent temperature axis.
    qha_len = len(qha.thermal_expansion)
    for key in ("temperatures", "free_energy", "entropy", "heat_capacity"):
        if key in harmonic_properties:
            harmonic_properties[key] = harmonic_properties[key][:qha_len]

    anharmonic_properties = {
        "temperatures": temperatures[:qha_len],
        "thermal_expansion_coefficients": qha.thermal_expansion,
        "gibbs_free_energies": qha.gibbs_temperature,
        "bulk_modulus_P": qha.bulk_modulus_temperature,
        "heat_capacity_P": qha.heat_capacity_P_polyfit,
        "gruneisen_parameters": qha.gruneisen_temperature,
        "volume_temperature": qha.volume_temperature,
    }

    return anharmonic_properties | harmonic_properties


def _calc_nvt(
    atoms,
    calculator,
    *,
    supercell_matrix,
    atom_disp,
    relax_initial,
    t_step,
    t_max,
    t_min,
    fmax,
    max_steps,
    optimizer,
    return_dos: bool = False,
) -> dict:
    if relax_initial:
        atoms.calc = calculator
        relax_atoms(
            atoms,
            steps=max_steps,
            fmax=fmax,
            optimizer_cls=getattr(ase.optimize, optimizer),
            cell_filter_cls=FrechetCellFilter,
        )

    thermal = _run_phonon_thermal(
        atoms,
        calculator,
        supercell_matrix,
        atom_disp,
        t_step,
        t_max,
        t_min,
        return_dos=return_dos,
    )
    thermal["free_energy"] = (
        thermal["free_energy"] * kJmol2eV
    )  # kJ/mol → eV; entropy/Cv stay in J/K/mol
    return thermal
