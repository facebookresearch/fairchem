"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet

from fairchem.data.omat.vasp.sets import OMat24StaticSet

OMAT24_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OMat24Compatibility.yaml")


class OMat24Compatibility(MaterialsProject2020Compatibility):
    """Exact same as MaterialsProject2020Compatibility but with different defaults.
    
    See documentation of MaterialsProject2020Compatibility for more details:
        https://pymatgen.org/pymatgen.entries.html
    """

    def __init__(
        self,
        compat_type: str = "Advanced",
        correct_peroxide: bool = True,
        strict_anions: Literal["require_exact", "require_bound", "no_check"] = "require_bound",
        check_potcar: bool = False,
        check_potcar_hash: bool = False,
        config_file: str | None = None,
    ) -> None:

        if config_file is None:
            config_file = OMAT24_CONFIG_FILE
    
        super().__init__(
            compat_type=compat_type,
            correct_peroxide=correct_peroxide,
            strict_anions=strict_anions,
            check_potcar=check_potcar,
            check_potcar_hash=check_potcar_hash,
            config_file=config_file,
        )


def generate_cse_parameters(input_set: VaspInputSet) -> dict:
    """Generate parameters for a ComputedStructureEntry from a VASP input set in order"""

    parameters = {
        "potcar_spec": input_set.potcar.spec,
        "potcar_symbols": input_set.potcar.symbols,
        "hubbards": {},
    }
    if "LDAUU" in input_set.incar:
        parameters["hubbards"] = dict(
            zip(input_set.poscar.site_symbols, input_set.incar["LDAUU"], strict=False)
        )

    parameters["is_hubbard"] = (
        input_set.incar.get("LDAU", False) and sum(parameters["hubbards"].values()) > 0
    )

    if parameters["is_hubbard"]:
        parameters["run_type"] = "GGA+U"
    else:
        parameters["run_type"] = "GGA"

    return parameters


def generate_computed_structure_entry(
    structure: Structure,
    total_energy: float,
    correction_type: Literal["MP2020", "OMat24"] = "OMat24",
    check_potcar: bool = False,
) -> ComputedStructureEntry:
    # Make a ComputedStructureEntry without the correction
    if correction_type == "MP2020":
        input_set = MPRelaxSet(structure)
        compatibility = MaterialsProject2020Compatibility(check_potcar=check_potcar)
    elif correction_type == "OMat24":
        input_set = OMat24StaticSet(structure)
        compatibility = OMat24Compatibility(check_potcar=check_potcar)
    else:
        raise ValueError(
            f"{correction_type} is not a valid correction type. Choose from OMat24 or MP2020"
        )

    oxidation_states = structure.composition.oxi_state_guesses()
    oxidation_states = {} if len(oxidation_states) == 0 else oxidation_states[0]

    cse_parameters = generate_cse_parameters(input_set)
    cse = ComputedStructureEntry(
        structure=structure,
        energy=total_energy,
        parameters=cse_parameters,
        data=dict(oxidation_states=oxidation_states),  # noqa
    )

    compatibility.process_entry(cse, clean=True, inplace=True)
    return cse


def apply_mp_style_corrections(
    energy: float, atoms: Atoms, correction_type: Literal["MP2020", "OMat24"] = "OMat24"
) -> float:
    """Applies Materials Project style energy corrections to an ASE Atoms object

    Args:
        energy: The uncorrected energy to be corrected.
        atoms: ASE Atoms object for which to apply the corrections.

    Returns:
        Corrected energy.
    """

    structure = AseAtomsAdaptor.get_structure(atoms)
    cse = generate_computed_structure_entry(
        structure, energy, correction_type=correction_type
    )
    
    return cse.energy
