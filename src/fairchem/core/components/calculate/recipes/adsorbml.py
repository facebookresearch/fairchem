"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Optional

from ase.atoms import Atoms
from fairchem.data.oc.core.adsorbate import Adsorbate
from fairchem.data.oc.core.multi_adsorbate_slab_config import (
    MultipleAdsorbateSlabConfig,
)
from fairchem.data.oc.core.slab import Slab
from fairchem.data.oc.utils import DetectTrajAnomaly

if TYPE_CHECKING:
    from collections.abc import Callable

    from ase.optimize import Optimizer


def relax_job(initial_atoms, calc, optimizer_cls, fmax, steps):
    atoms = initial_atoms.copy()
    atoms.calc = calc
    dyn = optimizer_cls(atoms)
    dyn.run(fmax=fmax, steps=steps)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms.calc = None
    result = {
        "input_atoms": {"atoms": initial_atoms},
        "atoms": atoms,
        "results": {
            "energy": energy,
            "forces": forces,
        },
    }
    return result


def adsorb_ml_pipeline(
    slab: Slab,
    adsorbates_kwargs: dict[str, Any],
    multiple_adsorbate_slab_config_kwargs: dict[str, Any],
    ml_slab_adslab_relax_job: Callable[..., Any],
    reference_ml_energies: bool = True,
    atomic_reference_energies: Optional[dict] = None,
    relaxed_slab_atoms: Atoms = None,
    place_on_relaxed_slab: bool = False,
):
    """
    Run a machine learning-based pipeline for adsorbate-slab systems.

    1. Relax slab using ML
    2. Generate trial adsorbate-slab configurations for the relaxed slab
    3. Relax adsorbate-slab configurations using ML
    4. Validate slab and adsorbate-slab configurations (check for anomalies like dissociations))
    5. Reference the energies to gas phase if needed (eg using a total energy ML model)

    Parameters
    ----------
    slab : Slab
        The slab structure to which adsorbates will be added.
    adsorbates_kwargs : dict[str,Any]
        Keyword arguments for generating adsorbate configurations.
    multiple_adsorbate_slab_config_kwargs : dict[str, Any]
        Keyword arguments for generating multiple adsorbate-slab configurations.
    ml_slab_adslab_relax_job : Job
        Job for relaxing slab and adsorbate-slab configurations using ML.
    reference_ml_energies: bool, optional
        Whether to reference ML energies to gas phase, by default False.
    atomic_reference_energies : AtomicReferenceEnergies, optional
        Atomic reference energies for referencing, by default None.
    relaxed_slab_atoms: ase.Atoms, optional
        DFT Relaxed slab atoms for anomaly detection for adsorption energy models, by default None.
    place_on_relaxed_slab: bool, optional
        Whether to place adsorbates on the relaxed slab or initial unrelaxed slab, by default False.

    Returns
    -------
    dict
        Dictionary containing the slab, ML-relaxed adsorbate-slab configurations,
        detected anomalies.
    """

    # only run slab relaxation if total energy model or placing on relaxed slab
    if place_on_relaxed_slab or reference_ml_energies:
        ml_relaxed_slab_result = ml_slab_adslab_relax_job(slab.atoms)

    unrelaxed_adslab_configurations = ocp_adslab_generator(
        ml_relaxed_slab_result["atoms"] if place_on_relaxed_slab else slab.atoms,
        adsorbates_kwargs,
        multiple_adsorbate_slab_config_kwargs,
    )

    ml_relaxed_configurations = [
        ml_slab_adslab_relax_job(adslab_configuration)
        for adslab_configuration in unrelaxed_adslab_configurations
    ]

    if reference_ml_energies:
        assert (
            atomic_reference_energies is not None
        ), "Missing atomic reference energies"

        ml_relaxed_configurations = reference_adslab_energies(
            ml_relaxed_configurations,
            ml_relaxed_slab_result,
            atomic_energies=atomic_reference_energies,
        )

    adslab_anomalies_list = [
        detect_anomaly(
            relaxed_result["input_atoms"]["atoms"],
            relaxed_result["atoms"],
            ml_relaxed_slab_result["atoms"]
            if reference_ml_energies
            else relaxed_slab_atoms,
        )
        for relaxed_result in ml_relaxed_configurations
    ]

    top_candidates = filter_sort_select_adslabs(
        adslab_results=ml_relaxed_configurations,
        adslab_anomalies_list=adslab_anomalies_list,
    )

    return {
        "slab": slab.get_metadata_dict(),
        "adslabs": top_candidates,
        "adslab_anomalies": adslab_anomalies_list,
    }


def ocp_adslab_generator(
    slab: Slab | Atoms,
    adsorbates_kwargs: list[dict[str, Any]] | None = None,
    multiple_adsorbate_slab_config_kwargs: dict[str, Any] | None = None,
) -> list[Atoms]:
    """
    Generate adsorbate-slab configurations.

    Parameters
    ----------
    slab : Slab | Atoms
        The slab structure.
    adsorbates_kwargs : list[dict[str,Any]], optional
        List of keyword arguments for generating adsorbates, by default None.
    multiple_adsorbate_slab_config_kwargs : dict[str,Any], optional
        Keyword arguments for generating multiple adsorbate-slab configurations, by default None.

    Returns
    -------
    list[Atoms]
        List of generated adsorbate-slab configurations.
    """
    adsorbates = [
        Adsorbate(**adsorbate_kwargs) for adsorbate_kwargs in adsorbates_kwargs
    ]

    if isinstance(slab, Atoms):
        slab = Slab(slab_atoms=slab)

    if multiple_adsorbate_slab_config_kwargs is None:
        multiple_adsorbate_slab_config_kwargs = {}

    adslabs = MultipleAdsorbateSlabConfig(
        copy.deepcopy(slab), adsorbates, **multiple_adsorbate_slab_config_kwargs
    )

    atoms_list = adslabs.atoms_list
    for atoms in atoms_list:
        atoms.pbc = True

    return adslabs.atoms_list


def reference_adslab_energies(
    adslab_results: list[dict],
    slab_result: dict,
    atomic_energies: dict,
) -> list[dict]:
    """
    Reference adsorbate-slab energies to atomic and slab energies.

    Parameters
    ----------
    adslab_results : list[dict[str, Any]]
        List of adsorbate-slab results.
    slab_result : dict
        Result of the slab calculation.
    atomic_energies : AtomicReferenceEnergies | None
        Dictionary of atomic energies.

    Returns
    -------
    list[dict[str, Any]]
        List of adsorbate-slab results with referenced energies.
    """
    slab_energy = slab_result["results"]["energy"]

    for adslab_result in adslab_results:
        adslab_result["results"]["referenced_adsorption_energy"] = {
            "atomic_energies": atomic_energies,
            "slab_energy": slab_energy,
            "adslab_energy": adslab_result["results"]["energy"],
            "gas_reactant_energy": sum(
                [
                    atomic_energies[atom.symbol]
                    for atom in adslab_result["atoms"][
                        adslab_result["atoms"].get_tags() == 2
                    ]  # all adsorbate tagged with tag=2!
                ]
            ),
            "adsorption_energy": adslab_result["results"]["energy"]
            - slab_energy
            - sum(
                [
                    atomic_energies[atom.symbol]
                    for atom in adslab_result["atoms"][
                        adslab_result["atoms"].get_tags() == 2
                    ]  # all adsorbate tagged with tag=2!
                ]
            ),
        }

    return adslab_results


def filter_sort_select_adslabs(
    adslab_results: list[dict], adslab_anomalies_list: list[list[str]]
) -> list[dict]:
    """
    Filter, sort, and select adsorbate-slab configurations based on anomalies and energy.

    Parameters
    ----------
    adslab_results : list[dict]
        List of adsorbate-slab results.
    adslab_anomalies_list : list[list[str]]
        List of detected anomalies for each adsorbate-slab configuration.

    Returns
    -------
    list[dict]
        Sorted list of adsorbate-slab configurations without anomalies.
    """
    for adslab_result, adslab_anomalies in zip(
        adslab_results, adslab_anomalies_list, strict=True
    ):
        adslab_result["results"]["adslab_anomalies"] = adslab_anomalies

    adslabs_no_anomalies = [
        adslab_result
        for adslab_result in adslab_results
        if len(adslab_result["results"]["adslab_anomalies"]) == 0
    ]

    return sorted(adslabs_no_anomalies, key=lambda x: x["results"]["energy"])


def detect_anomaly(
    initial_atoms: Atoms,
    final_atoms: Atoms,
    final_slab_atoms: Atoms,
) -> list[
    Literal[
        "adsorbate_dissociated",
        "adsorbate_desorbed",
        "surface_changed",
        "adsorbate_intercalated",
    ]
]:
    """
    Detect anomalies between initial and final atomic structures.

    Parameters
    ----------
    initial_atoms : Atoms
        Initial atomic structure.
    final_atoms : Atoms
        Final atomic structure.

    Returns
    -------
    list[Literal["adsorbate_dissociated", "adsorbate_desorbed", "surface_changed", "adsorbate_intercalated"]]
        List of detected anomalies.
    """
    atom_tags = initial_atoms.get_tags()

    detector = DetectTrajAnomaly(
        initial_atoms,
        final_atoms,
        atoms_tag=atom_tags,
        final_slab_atoms=final_slab_atoms,
    )
    anomalies = []
    if detector.is_adsorbate_dissociated():
        anomalies.append("adsorbate_dissociated")
    if detector.is_adsorbate_desorbed():
        anomalies.append("adsorbate_desorbed")
    if detector.has_surface_changed():
        anomalies.append("surface_changed")
    if detector.is_adsorbate_intercalated():
        anomalies.append("adsorbate_intercalated")
    return anomalies


# Run the AdsorbML workflow for a given slab and adsorbate
def run_adsorbml(
    slab,
    adsorbate,
    calculator,
    optimizer_cls: Optimizer,
    fmax: float = 0.02,
    steps: int = 300,
    num_placements: int = 100,
    reference_ml_energies: bool = True,
    relaxed_slab_atoms: Atoms = None,
    place_on_relaxed_slab: bool = False,
):
    """
    Run the AdsorbML pipeline for a given slab and adsorbate using a pretrained ML model.
    Parameters
    ----------
    slab : ase.Atoms
        The clean slab structure to which the adsorbate will be added.
    adsorbate : str
        A string identifier for the adsorbate from the database (e.g., '*O').
    reference_ml_energies : bool, optional
        If True, assumes the model is a total energy model and references energies
        to gas phase and bare slab, by default True since the default model is a total energy model.
    num_placements : int, optional
        Number of initial adsorbate placements to generate for relaxation, by default 100.
    fmax: float, default 0.02.
        Relaxation force convergence threshold
    steps: int, default 300
        Max number of relaxation steps
    relaxed_slab_atoms : ase.Atoms, optional
        DFT Relaxed slab atoms for anomaly detection for adsorption energy models, by default None.
    place_on_relaxed_slab : bool, optional
        Whether to place adsorbates on the relaxed slab or initial unrelaxed slab, by default False.
    Returns
    -------
    dict
        Dictionary containing the ML-relaxed slab, adsorbate-slab configurations,
        energies, and validation results (matching the AdsorbMLSchema format).
    """

    # if we are using a total energy model, we need to set the DFT atomic reference energies
    # obtained from the supplementary information of the OC20 paper
    atomic_reference_energies = {
        "H": -3.477,  # eV
        "O": -7.204,  # eV
        "C": -7.282,  # eV
        "N": -8.083,  # eV
    }

    ml_relax_job = partial(
        relax_job, calc=calculator, optimizer_cls=optimizer_cls, fmax=fmax, steps=steps
    )

    outputs = adsorb_ml_pipeline(
        slab=slab,
        adsorbates_kwargs=[{"adsorbate_smiles_from_db": adsorbate}],
        multiple_adsorbate_slab_config_kwargs={"num_configurations": num_placements},
        ml_slab_adslab_relax_job=ml_relax_job,
        reference_ml_energies=reference_ml_energies,
        atomic_reference_energies=atomic_reference_energies,
        relaxed_slab_atoms=relaxed_slab_atoms,
        place_on_relaxed_slab=place_on_relaxed_slab,
    )
    return outputs
