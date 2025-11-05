"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from typing import Literal

from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet

from pymatgen.io.vasp.inputs import PmgVaspPspDirError

from fairchem.data.omat.vasp.sets import OMat24StaticSet

OMAT24_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OMat24Compatibility.yaml")


class OMat24Compatibility(MaterialsProject2020Compatibility):
    """Exact same as MaterialsProject2020Compatibility but with different defaults.
    
    See documentation of MaterialsProject2020Compatibility for more details:
        https://pymatgen.org/pymatgen.entries.html
    """

    def __init__(
        self,
        compat_type: Literal["GGA", "Advanced"] = "Advanced",
        correct_peroxide: bool = True,
        strict_anions: Literal["require_exact", "require_bound", "no_check"] = "require_bound",
        check_potcar: bool = True,
        check_potcar_hash: bool = False,
        config_file: str | None = None,
    ) -> None:
                """
        Args:
            compat_type: Two options, GGA or Advanced. GGA means all GGA+U
                entries are excluded. Advanced means the GGA/GGA+U mixing scheme
                of Jain et al. (see References) is implemented. In this case,
                entries which are supposed to be calculated in GGA+U (i.e.,
                transition metal oxides and fluorides) will have the corresponding
                GGA entries excluded. For example, Fe oxides should
                have a U value under the Advanced scheme. An Fe oxide run in GGA
                will therefore be excluded.

                To use the "Advanced" type, Entry.parameters must contain a "hubbards"
                key which is a dict of all non-zero Hubbard U values used in the
                calculation. For example, if you ran a Fe2O3 calculation with
                Materials Project parameters, this would look like
                entry.parameters["hubbards"] = {"Fe": 5.3}. If the "hubbards" key
                is missing, a GGA run is assumed. Entries obtained from the
                MaterialsProject database will automatically have these fields
                populated. Default: "Advanced"
            correct_peroxide: Specify whether peroxide/superoxide/ozonide
                corrections are to be applied or not. If false, all oxygen-containing
                compounds are assigned the 'oxide' correction. Default: True
            strict_anions: only apply the anion corrections to anions. The option
                "require_exact" will only apply anion corrections in cases where the
                anion oxidation state is between the oxidation states used
                in the experimental fitting data. The option "require_bound" will
                define an anion as any species with an oxidation state value of <= -1.
                This prevents the anion correction from being applied to unrealistic
                hypothetical structures containing large proportions of very electronegative
                elements, thus artificially over-stabilizing the compound. Set to "no_check"
                to restore the original behavior described in the associated publication. Default: True
            check_potcar (bool): Check that the POTCARs used in the calculation are consistent
                with the Materials Project parameters. False bypasses this check altogether. Default: True
                Can also be disabled globally by running `pmg config --add PMG_POTCAR_CHECKS false`.
            check_potcar_hash (bool): Use potcar hash to verify POTCAR settings are
                consistent with MPRelaxSet. If False, only the POTCAR symbols will
                be used. Default: False
            config_file (Path): Path to the selected compatibility.yaml config file.
                If None, defaults to `MP2020Compatibility.yaml` distributed with
                pymatgen.

        References:
            Wang, A., Kingsbury, R., McDermott, M., Horton, M., Jain. A., Ong, S.P.,
                Dwaraknath, S., Persson, K. A framework for quantifying uncertainty
                in DFT energy corrections. Scientific Reports 11: 15496, 2021.
                https://doi.org/10.1038/s41598-021-94550-5

            Jain, A. et al. Formation enthalpies by mixing GGA and GGA + U calculations.
                Phys. Rev. B - Condens. Matter Mater. Phys. 84, 1-10 (2011).
        """
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

    parameters = {"hubbards": {}}
    try:
        parameters.update(
            {
                "potcar_spec": input_set.potcar.spec,
                "potcar_symbols": input_set.potcar.symbols,
            }
        )
    except PmgVaspPspDirError:
        pass

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
    check_potcar: bool = True,
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
    energy: float, atoms: Atoms, correction_type: Literal["MP2020", "OMat24"] = "OMat24", check_potcar: bool = False
) -> float:
    """Applies Materials Project style energy corrections to an ASE Atoms object

    Args:
        energy: The uncorrected energy to be corrected.
        atoms: ASE Atoms object for which to apply the corrections.
        correction_type: Type of corrections to apply: MP2020 or OMat24.
        check_potcar: Whether to check POTCAR consistency when applying corrections.

    Returns:
        Corrected energy.
    """

    structure = AseAtomsAdaptor.get_structure(atoms)
    cse = generate_computed_structure_entry(
        structure, energy, correction_type=correction_type, check_potcar=check_potcar
    )
    
    return cse.energy
