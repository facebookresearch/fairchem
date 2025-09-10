"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Functions to run elasticity calculations using ASE + pymatgen

- Compute elasticity tensor
- Compute Bulk modulus
- Compute Shear modulus
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monty.dev import requires

from fairchem.core.components.calculate.recipes.relax import (
    relax_atoms,
)

try:
    from pymatgen.analysis.elasticity import DeformedStructureSet, ElasticTensor, Strain
    from pymatgen.io.ase import AseAtomsAdaptor

    pmg_installed = True
except ImportError:
    pmg_installed = False

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator


@requires(pmg_installed, message="Requires `pymatgen` to be installed")
def calculate_elasticity(
    atoms: Atoms,
    calculator: Calculator,
    norm_strains: Sequence[float] | float = (-0.01, -0.005, 0.005, 0.01),
    shear_strains: Sequence[float] | float = (-0.06, -0.03, 0.03, 0.06),
    relax_initial: bool = True,
    relax_strained: bool = True,
    use_equilibrium_stress: bool = True,
    **relax_kwargs,
) -> dict[str, Any]:
    """Calculate elastic tensor, bulk, shear moduli following MP workflow

    Will not run a relaxation. We do that outside in order to be able to have more control.

    Args:
        atoms: ASE atoms object
        calculator: an ASE Calculator
        norm_strains: sequence of normal strains
        shear_strains: sequence of shear strains
        relax_initial: relax the initial structure. Default is True
        relax_strained: relax the atomic positions of strained structure. Default True.
        use_equilibrium_stress: use equilibrium stress in calculation. For relaxed structures this
            should be very small

    Returns:
        dict of elasticity results
    """
    atoms.calc = calculator
    if relax_initial:
        atoms = relax_atoms(atoms, **relax_kwargs)

    eq_stress = atoms.get_stress(voigt=False) if use_equilibrium_stress else None

    deformed_structure_set = DeformedStructureSet(
        AseAtomsAdaptor.get_structure(atoms),
        norm_strains,
        shear_strains,
    )

    strains = [
        Strain.from_deformation(deformation)
        for deformation in deformed_structure_set.deformations
    ]

    stresses = []
    for deformed_structure in deformed_structure_set:
        atoms = deformed_structure.to_ase_atoms()
        atoms.calc = calculator

        if relax_strained is True:
            relax_kwargs.update({"cell_filter_cls": None})
            atoms = relax_atoms(atoms, **relax_kwargs)

        stresses.append(atoms.get_stress(voigt=False))

    elastic_tensor = ElasticTensor.from_independent_strains(
        strains=strains, stresses=stresses, eq_stress=eq_stress
    )

    # results are in eV/A^3
    results = {
        "elastic_tensor": elastic_tensor,
        "shear_modulus_vrh": elastic_tensor.g_vrh,
        "bulk_modulus_vrh": elastic_tensor.k_vrh,
    }

    return results
