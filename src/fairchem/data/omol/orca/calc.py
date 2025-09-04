"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import re
from enum import Enum
from shutil import which

from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from sella import Sella

# ECP sizes taken from Table 6.5 in the Orca 5.0.3 manual
ECP_SIZE = {
    **{i: 28 for i in range(37, 55)},
    **{i: 46 for i in range(55, 58)},
    **{i: 28 for i in range(58, 72)},
    **{i: 60 for i in range(72, 87)},
}
BASIS_DICT = {
    "H": 9,
    "He": 9,
    "Li": 17,
    "Be": 22,
    "B": 37,
    "C": 37,
    "N": 37,
    "O": 40,
    "F": 40,
    "Ne": 40,
    "Na": 35,
    "Mg": 35,
    "Al": 43,
    "Si": 43,
    "P": 43,
    "S": 46,
    "Cl": 46,
    "Ar": 46,
    "K": 36,
    "Ca": 36,
    "Sc": 48,
    "Ti": 48,
    "V": 48,
    "Cr": 48,
    "Mn": 48,
    "Fe": 48,
    "Co": 48,
    "Ni": 48,
    "Cu": 48,
    "Zn": 51,
    "Ga": 54,
    "Ge": 54,
    "As": 54,
    "Se": 57,
    "Br": 57,
    "Kr": 57,
    "Rb": 33,
    "Sr": 33,
    "Y": 40,
    "Zr": 40,
    "Nb": 40,
    "Mo": 40,
    "Tc": 40,
    "Ru": 40,
    "Rh": 40,
    "Pd": 40,
    "Ag": 40,
    "Cd": 40,
    "In": 56,
    "Sn": 56,
    "Sb": 56,
    "Te": 59,
    "I": 59,
    "Xe": 59,
    "Cs": 32,
    "Ba": 40,
    "La": 43,
    "Ce": 105,
    "Pr": 105,
    "Nd": 98,
    "Pm": 98,
    "Sm": 98,
    "Eu": 93,
    "Gd": 98,
    "Tb": 98,
    "Dy": 98,
    "Ho": 98,
    "Er": 101,
    "Tm": 101,
    "Yb": 96,
    "Lu": 96,
    "Hf": 43,
    "Ta": 43,
    "W": 43,
    "Re": 43,
    "Os": 43,
    "Ir": 43,
    "Pt": 43,
    "Au": 43,
    "Hg": 46,
    "Tl": 56,
    "Pb": 56,
    "Bi": 56,
    "Po": 59,
    "At": 59,
    "Rn": 59,
}


ORCA_FUNCTIONAL = "wB97M-V"
ORCA_BASIS = "def2-TZVPD"
ORCA_SIMPLE_INPUT = [
    "EnGrad",
    "RIJCOSX",
    "def2/J",
    "NoUseSym",
    "DIIS",
    "NOSOSCF",
    "NormalConv",
    "DEFGRID3",
    "ALLPOP",
]
ORCA_BLOCKS = [
    "%scf Convergence Tight maxiter 300 end",
    "%elprop Dipole true Quadrupole true end",
    "%output Print[P_ReducedOrbPopMO_L] 1 Print[P_ReducedOrbPopMO_M] 1 Print[P_BondOrder_L] 1 Print[P_BondOrder_M] 1 Print[P_Fockian] 1 Print[P_OrbEn] 2 end",
    '%basis GTOName "def2-tzvpd.bas" end',
    "%scf THRESH 1e-12 TCUT 1e-13 end",
]
NBO_FLAGS = '%nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end'
ORCA_ASE_SIMPLE_INPUT = " ".join([ORCA_FUNCTIONAL] + [ORCA_BASIS] + ORCA_SIMPLE_INPUT)
LOOSE_OPT_PARAMETERS = {
    "optimizer": Sella,
    "store_intermediate_results": True,
    "fmax": 0.1,
    "max_steps": 100,
    "optimizer_kwargs": {
        "order": 0,
        "internal": True,
    },
}
OPT_PARAMETERS = {
    "optimizer": Sella,
    "store_intermediate_results": True,
    "fmax": 0.05,
    "max_steps": 100,
    "optimizer_kwargs": {
        "order": 0,
        "internal": True,
    },
}
TIGHT_OPT_PARAMETERS = {
    "optimizer": Sella,
    "store_intermediate_results": True,
    "fmax": 0.01,
    "max_steps": 100,
    "optimizer_kwargs": {
        "order": 0,
        "internal": True,
    },
}


class Vertical(Enum):
    Default = "default"
    MetalOrganics = "metal-organics"
    Oss = "open-shell-singlet"


def get_symm_break_block(atoms: Atoms, charge: int) -> str:
    """
    Determine the ORCA Rotate block needed to break symmetry in a singlet

    This is determined by taking the sum of atomic numbers less any charge (because
    electrons are negatively charged) and removing any electrons that are in an ECP
    and dividing by 2. This gives the number of occupied orbitals, but since ORCA is
    0-indexed, it gives the index of the LUMO.

    We use a rotation angle of 20 degrees or about a 12% mixture of LUMO into HOMO.
    This is somewhat arbitrary but similar to the default setting in Q-Chem, and seemed
    to perform well in tests of open-shell singlets.
    """
    n_electrons = sum(atoms.get_atomic_numbers()) - charge
    ecp_electrons = sum(
        ECP_SIZE.get(at_num, 0) for at_num in atoms.get_atomic_numbers()
    )
    n_electrons -= ecp_electrons
    lumo = n_electrons // 2
    return f"%scf rotate {{{lumo-1}, {lumo}, 20, 1, 1}} end end"


def get_n_basis(atoms: Atoms) -> int:
    """
    Get the number of basis functions that will be used for the given input.

    We assume our basis is def2-tzvpd. The number of basis functions is used
    to estimate the memory requirments of a given job.

    :param atoms: atoms to compute the number of basis functions of
    :return: number of basis functions as printed by Orca
    """
    nbasis = 0
    for elt in atoms.get_chemical_symbols():
        nbasis += BASIS_DICT[elt]
    return nbasis


def get_mem_estimate(
    atoms: Atoms, vertical: Enum = Vertical.Default, mult: int = 1
) -> int:
    """
    Get an estimate of the memory requirement for given input in MB.

    If the estimate is less than 1000MB, we return 1000MB.

    :param atoms: atoms to compute the number of basis functions of
    :param vertical: Which vertical this is for (all metal-organics are
                     UKS, as are all regular open-shell calcs)
    :param mult: spin multiplicity of input
    :return: estimated (upper-bound) to the memory requirement of this Orca job
    """
    nbasis = get_n_basis(atoms)
    if vertical == Vertical.Default and mult == 1:
        # Default RKS scaling as determined by PDB-ligand pockets in Orca6
        a = 0.0076739752343756434
        b = 361.4745947062764
    else:
        # Default UKS scaling as determined by metal-organics in Orca5
        a = 0.016460518374501867
        b = -320.38502508802776
    mem_est = max(a * nbasis**1.5 + b, 1000)
    return mem_est


def write_orca_inputs(
    atoms: Atoms,
    output_directory,
    charge: int = 0,
    mult: int = 1,
    nbo: bool = True,
    orcasimpleinput: str = ORCA_ASE_SIMPLE_INPUT,
    orcablocks: str = " ".join(ORCA_BLOCKS),
    vertical: Enum = Vertical.Default,
    scf_MaxIter: int = None,
):
    """
    One-off method to be used if you wanted to write inputs for an arbitrary
    system. Primarily used for debugging.
    """

    MyOrcaProfile = OrcaProfile([which("orca")])

    # Include estimate of memory needs
    mem_est = get_mem_estimate(atoms, vertical, mult)
    orcablocks += f" %maxcore {mem_est}"
    if not nbo:
        orcasimpleinput += " NONBO NONPA"
    else:
        orcablocks += f" {NBO_FLAGS}"

    if vertical in {Vertical.MetalOrganics, Vertical.Oss} and mult == 1:
        orcasimpleinput += " UKS"
        orcablocks += f" {get_symm_break_block(atoms, charge)}"

    if scf_MaxIter:
        if "maxiter" in orcablocks:
            orcablocks = re.sub(r"maxiter \d+", f"maxiter {scf_MaxIter}", orcablocks)
        else:
            orcablocks += f" %scf MaxIter {scf_MaxIter} end"

    calc = ORCA(
        charge=charge,
        mult=mult,
        profile=MyOrcaProfile,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        directory=output_directory,
    )
    calc.write_inputfiles(atoms, ["energy", "forces"])
