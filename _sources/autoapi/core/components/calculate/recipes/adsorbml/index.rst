core.components.calculate.recipes.adsorbml
==========================================

.. py:module:: core.components.calculate.recipes.adsorbml

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.components.calculate.recipes.adsorbml.data_oc_installed


Functions
---------

.. autoapisummary::

   core.components.calculate.recipes.adsorbml.relax_job
   core.components.calculate.recipes.adsorbml.adsorb_ml_pipeline
   core.components.calculate.recipes.adsorbml.ocp_adslab_generator
   core.components.calculate.recipes.adsorbml.reference_adslab_energies
   core.components.calculate.recipes.adsorbml.filter_sort_select_adslabs
   core.components.calculate.recipes.adsorbml.detect_anomaly
   core.components.calculate.recipes.adsorbml.run_adsorbml


Module Contents
---------------

.. py:data:: data_oc_installed
   :value: True


.. py:function:: relax_job(initial_atoms, calc, optimizer_cls, fmax, steps)

.. py:function:: adsorb_ml_pipeline(slab: fairchem.data.oc.core.slab.Slab, adsorbates_kwargs: dict[str, Any], multiple_adsorbate_slab_config_kwargs: dict[str, Any], ml_slab_adslab_relax_job: collections.abc.Callable[Ellipsis, Any], reference_ml_energies: bool = True, atomic_reference_energies: Optional[dict] = None, relaxed_slab_atoms: ase.atoms.Atoms = None, place_on_relaxed_slab: bool = False)

   Run a machine learning-based pipeline for adsorbate-slab systems.

   1. Relax slab using ML
   2. Generate trial adsorbate-slab configurations for the relaxed slab
   3. Relax adsorbate-slab configurations using ML
   4. Validate slab and adsorbate-slab configurations (check for anomalies like dissociations))
   5. Reference the energies to gas phase if needed (eg using a total energy ML model)

   :param slab: The slab structure to which adsorbates will be added.
   :type slab: Slab
   :param adsorbates_kwargs: Keyword arguments for generating adsorbate configurations.
   :type adsorbates_kwargs: dict[str,Any]
   :param multiple_adsorbate_slab_config_kwargs: Keyword arguments for generating multiple adsorbate-slab configurations.
   :type multiple_adsorbate_slab_config_kwargs: dict[str, Any]
   :param ml_slab_adslab_relax_job: Job for relaxing slab and adsorbate-slab configurations using ML.
   :type ml_slab_adslab_relax_job: Job
   :param reference_ml_energies: Whether to reference ML energies to gas phase, by default False.
   :type reference_ml_energies: bool, optional
   :param atomic_reference_energies: Atomic reference energies for referencing, by default None.
   :type atomic_reference_energies: AtomicReferenceEnergies, optional
   :param relaxed_slab_atoms: DFT Relaxed slab atoms for anomaly detection for adsorption energy models, by default None.
   :type relaxed_slab_atoms: ase.Atoms, optional
   :param place_on_relaxed_slab: Whether to place adsorbates on the relaxed slab or initial unrelaxed slab, by default False.
   :type place_on_relaxed_slab: bool, optional

   :returns: Dictionary containing the slab, ML-relaxed adsorbate-slab configurations,
             detected anomalies.
   :rtype: dict


.. py:function:: ocp_adslab_generator(slab: fairchem.data.oc.core.slab.Slab | ase.atoms.Atoms, adsorbates_kwargs: list[dict[str, Any]] | None = None, multiple_adsorbate_slab_config_kwargs: dict[str, Any] | None = None) -> list[ase.atoms.Atoms]

   Generate adsorbate-slab configurations.

   :param slab: The slab structure.
   :type slab: Slab | Atoms
   :param adsorbates_kwargs: List of keyword arguments for generating adsorbates, by default None.
   :type adsorbates_kwargs: list[dict[str,Any]], optional
   :param multiple_adsorbate_slab_config_kwargs: Keyword arguments for generating multiple adsorbate-slab configurations, by default None.
   :type multiple_adsorbate_slab_config_kwargs: dict[str,Any], optional

   :returns: List of generated adsorbate-slab configurations.
   :rtype: list[Atoms]


.. py:function:: reference_adslab_energies(adslab_results: list[dict], slab_result: dict, atomic_energies: dict) -> list[dict]

   Reference adsorbate-slab energies to atomic and slab energies.

   :param adslab_results: List of adsorbate-slab results.
   :type adslab_results: list[dict[str, Any]]
   :param slab_result: Result of the slab calculation.
   :type slab_result: dict
   :param atomic_energies: Dictionary of atomic energies.
   :type atomic_energies: AtomicReferenceEnergies | None

   :returns: List of adsorbate-slab results with referenced energies.
   :rtype: list[dict[str, Any]]


.. py:function:: filter_sort_select_adslabs(adslab_results: list[dict], adslab_anomalies_list: list[list[str]]) -> list[dict]

   Filter, sort, and select adsorbate-slab configurations based on anomalies and energy.

   :param adslab_results: List of adsorbate-slab results.
   :type adslab_results: list[dict]
   :param adslab_anomalies_list: List of detected anomalies for each adsorbate-slab configuration.
   :type adslab_anomalies_list: list[list[str]]

   :returns: Sorted list of adsorbate-slab configurations without anomalies.
   :rtype: list[dict]


.. py:function:: detect_anomaly(initial_atoms: ase.atoms.Atoms, final_atoms: ase.atoms.Atoms, final_slab_atoms: ase.atoms.Atoms) -> list[Literal['adsorbate_dissociated', 'adsorbate_desorbed', 'surface_changed', 'adsorbate_intercalated']]

   Detect anomalies between initial and final atomic structures.

   :param initial_atoms: Initial atomic structure.
   :type initial_atoms: Atoms
   :param final_atoms: Final atomic structure.
   :type final_atoms: Atoms

   :returns: List of detected anomalies.
   :rtype: list[Literal["adsorbate_dissociated", "adsorbate_desorbed", "surface_changed", "adsorbate_intercalated"]]


.. py:function:: run_adsorbml(slab, adsorbate, calculator, optimizer_cls: ase.optimize.Optimizer, fmax: float = 0.02, steps: int = 300, num_placements: int = 100, reference_ml_energies: bool = True, relaxed_slab_atoms: ase.atoms.Atoms = None, place_on_relaxed_slab: bool = False)

   Run the AdsorbML pipeline for a given slab and adsorbate using a pretrained ML model.
   :param slab: The clean slab structure to which the adsorbate will be added.
   :type slab: ase.Atoms
   :param adsorbate: A string identifier for the adsorbate from the database (e.g., '*O').
   :type adsorbate: str
   :param reference_ml_energies: If True, assumes the model is a total energy model and references energies
                                 to gas phase and bare slab, by default True since the default model is a total energy model.
   :type reference_ml_energies: bool, optional
   :param num_placements: Number of initial adsorbate placements to generate for relaxation, by default 100.
   :type num_placements: int, optional
   :param fmax: Relaxation force convergence threshold
   :type fmax: float, default 0.02.
   :param steps: Max number of relaxation steps
   :type steps: int, default 300
   :param relaxed_slab_atoms: DFT Relaxed slab atoms for anomaly detection for adsorption energy models, by default None.
   :type relaxed_slab_atoms: ase.Atoms, optional
   :param place_on_relaxed_slab: Whether to place adsorbates on the relaxed slab or initial unrelaxed slab, by default False.
   :type place_on_relaxed_slab: bool, optional

   :returns: Dictionary containing the ML-relaxed slab, adsorbate-slab configurations,
             energies, and validation results (matching the AdsorbMLSchema format).
   :rtype: dict


