"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatgen.io.ase import MSONAtoms

from fairchem.core.components.calculate.recipes.relax import (
    relax_atoms,
)

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize.optimize import Optimizer


def adsorb_atoms(
    adslab_atoms: Atoms,
    calculator: Calculator,
    optimizer_cls: Optimizer,
    steps: int = 500,
    fmax: float = 0.02,
    relax_surface: bool = False,
    save_relaxed_atoms: bool = False,
) -> Atoms:
    """
    Simple helper function to run relaxations and compute the adsorption energy
    of a given adsorbate+surface atoms object.

    Args:
        atoms: ASE atoms object
        calculator: ASE calculator
        optimizer_cls: ASE optimizer. Default LBFGS
        steps: max number of relaxation steps
        fmax: force convergence threshold
        relax_surface: Whether to relax the bare surface
        save_relaxed_atoms: Whether to save the relaxed atoms
    Returns:
        dict of adsorption results
    """
    sid = adslab_atoms.info["sid"]
    gas_reference_energy = adslab_atoms.info["gas_ref"]
    # Compute DFT adsorption energy
    dft_relaxed_adslab_energy = adslab_atoms.info["dft_relaxed_adslab_energy"]
    dft_relaxed_slab_energy = adslab_atoms.info["dft_relaxed_slab_energy"]
    dft_adsorption_energy = (
        dft_relaxed_adslab_energy - dft_relaxed_slab_energy - gas_reference_energy
    )

    # Relax provided adslab system
    adslab_atoms.calc = calculator
    relaxed_adslab_atoms = relax_atoms(
        adslab_atoms,
        fmax=fmax,
        steps=steps,
        optimizer_cls=optimizer_cls,
    )
    relaxed_adslab_atoms_energy = relaxed_adslab_atoms.get_potential_energy()

    results = {
        "sid": sid,
        "direct": relaxed_adslab_atoms_energy,
        "target": dft_adsorption_energy,
    }

    # Compute adsorption energy using DFT slab energies
    hybrid_adsorption_energy = (
        relaxed_adslab_atoms_energy - dft_relaxed_slab_energy - gas_reference_energy
    )
    results["hybrid"] = hybrid_adsorption_energy

    # Compute adsorption energy using ML relaxed slabs
    if relax_surface:
        slab_atoms = adslab_atoms.info["initial_slab_atoms"]
        slab_atoms.calc = calculator
        relaxed_slab_atoms = relax_atoms(
            slab_atoms,
            fmax=fmax,
            steps=steps,
            optimizer_cls=optimizer_cls,
        )
        pred_adsorption_energy = (
            relaxed_adslab_atoms_energy
            - relaxed_slab_atoms.get_potential_energy()
            - gas_reference_energy
        )
        results["full"] = pred_adsorption_energy

    if save_relaxed_atoms:
        results["relaxed_adslab_atoms"] = MSONAtoms(relaxed_adslab_atoms).as_dict()
        if relax_surface:
            results["relaxed_slab_atoms"] = MSONAtoms(relaxed_slab_atoms).as_dict()

    return results
