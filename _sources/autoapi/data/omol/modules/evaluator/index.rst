data.omol.modules.evaluator
===========================

.. py:module:: data.omol.modules.evaluator


Functions
---------

.. autoapisummary::

   data.omol.modules.evaluator.rmsd
   data.omol.modules.evaluator.interaction_energy_and_forces
   data.omol.modules.evaluator.spin_deltas
   data.omol.modules.evaluator.charge_deltas
   data.omol.modules.evaluator.rmsd_mapping
   data.omol.modules.evaluator.ligand_strain_processing
   data.omol.modules.evaluator.get_one_prot_diff_name_pairs
   data.omol.modules.evaluator.sr_or_lr
   data.omol.modules.evaluator.ligand_pocket
   data.omol.modules.evaluator.ligand_strain
   data.omol.modules.evaluator.geom_conformers
   data.omol.modules.evaluator.protonation_energies
   data.omol.modules.evaluator.unoptimized_ie_ea
   data.omol.modules.evaluator.compute_distance_scaling_metrics
   data.omol.modules.evaluator.distance_scaling
   data.omol.modules.evaluator.unoptimized_spin_gap
   data.omol.modules.evaluator.singlepoint


Module Contents
---------------

.. py:function:: rmsd(orca_atoms, mlip_atoms)

   Calculate the RMSD between atoms optimized with ORCA and the MLIP,
   where we assume that ORCA atoms have sensible bonding.


.. py:function:: interaction_energy_and_forces(results, supersystem)

   For a supersystem (e.g. a protein-ligand complex), calculate the
   interaction energy and forces with each individual component (e.g.
   the protein and the ligand) in the complex.

   We assume that the supersystem is the sum of the individual components
   and all are present.

   `results` looks like:
   {
       "system_name": {
           "ligand_pocket": {
               "energy": float,
               "forces": list of floats,
               "atoms": dict (MSONAtoms format)
           },
           "ligand": {
               "energy": float,
               "forces": list of floats,
               "atoms": dict (MSONAtoms format)
           },
           "pocket": {
               "energy": float,
               "forces": list of floats,
               "atoms": dict (MSONAtoms format)
           }
       }
   }

   :param results: Results from ORCA or MLIP calculations.
   :type results: dict
   :param supersystem: The name of the supersystem (e.g. "ligand_pocket")
   :type supersystem: str

   :returns: interaction_energy, interaction_forces


.. py:function:: spin_deltas(results)

   Calculate deltaE and deltaF values for the spin gap evaluation task.

   `results` looks like:
   {
       "system_name": {
           "1": {
               "energy": float,
               "forces": list of floats,
               "atoms": dict (MSONAtoms format)
           },
           "3": {
               "energy": float,
               "forces": list of floats,
               "atoms": dict (MSONAtoms format)
           }
   }


   :param results: Results from ORCA or MLIP calculations performed at
   :type results: dict
   :param different spins.:

   :returns: deltaE (dict), deltaF (dict)


.. py:function:: charge_deltas(results)

   Calculate deltaE and deltaF values for adding and removing electrons

   :param results: Results from ORCA or MLIP calculations performed
   :type results: dict
   :param at different charges.:

   :returns: deltaE (dict), deltaF (dict)


.. py:function:: rmsd_mapping(structs0, structs1)

   Map two conformer ensembles via linear sum assignment on an RMSD
   cost matrix.


.. py:function:: ligand_strain_processing(results)

   Process results for the ligand strain evaluation task.
   Calculate the strain energy as the difference in energy between
   the global minimum and the loosely optimized ligand-in-pocket structure.
   Also save the global minimum structure for RMSD calculations.

   :param results: Results from ORCA or MLIP calculations.
   :type results: dict

   :returns: Processed results for the ligand strain evaluation task.
   :rtype: dict


.. py:function:: get_one_prot_diff_name_pairs(names)

   Get all pairs of names that have a charge difference of 1.

   Assumes that the names are in the format "name_charge_spin"


.. py:function:: sr_or_lr(name)

.. py:function:: ligand_pocket(orca_results, mlip_results)

   Calculate error metrics for ligand pocket evaluation task.

   :param orca_results: Results from ORCA calculations.
   :type orca_results: dict
   :param mlip_results: Results from MLIP calculations.
   :type mlip_results: dict

   :returns: Error metrics for ligand pocket evaluation task
   :rtype: dict


.. py:function:: ligand_strain(orca_results, mlip_results)

   Calculate error metrics for ligand strain evaluation task.

   :param orca_results: Results from ORCA calculations.
   :type orca_results: dict
   :param mlip_results: Results from MLIP calculations.
   :type mlip_results: dict

   :returns: Error metrics for ligand strain evaluation task
   :rtype: dict


.. py:function:: geom_conformers(orca_results, mlip_results)

   Calculate error metrics for conformer evaluation task.

   :param orca_results: Results from ORCA calculations.
   :type orca_results: dict
   :param mlip_results: Results from MLIP calculations.
   :type mlip_results: dict

   :returns: Error metrics for type1 conformer evaluation task
   :rtype: dict


.. py:function:: protonation_energies(orca_results, mlip_results)

   Calculate error metrics for the type1 protonation energies evaluation task.

   :param orca_results: Results from ORCA calculations.
   :type orca_results: dict
   :param mlip_results: Results from MLIP calculations.
   :type mlip_results: dict

   :returns: Error metrics for protonation energies evaluation task
   :rtype: dict


.. py:function:: unoptimized_ie_ea(orca_results, mlip_results)

   Calculate error metrics for unoptimized IE and EA calculations.

   :param orca_results: Results from ORCA calculations.
   :type orca_results: dict
   :param mlip_results: Results from MLIP calculations.
   :type mlip_results: dict

   :returns: Error metrics for unoptimized IE and EA calculations.
   :rtype: dict


.. py:function:: compute_distance_scaling_metrics(pes_curve, mlip_results, orca_min_point, orca_min_data)

   Compute metrics for distance scaling eval for a single PES curve

   :param pes_curve: specification of points on PES in the short or
                     long range regime only, also contains ORCA data
   :param mlip_results: results from all points alng
   :param orca_min_point: name of reference point for ddE
   :param orca_min_data: data for reference point for ddE
   :return: ddE, and ddF metrics for the given PES curve


.. py:function:: distance_scaling(orca_results, mlip_results)

   Calculate error metrics for distance scaling evaluation task.

   :param orca_results: Results from ORCA calculations.
   :type orca_results: dict
   :param mlip_results: Results from MLIP calculations.
   :type mlip_results: dict

   :returns: Error metrics for distance scaling evaluation task
   :rtype: dict


.. py:function:: unoptimized_spin_gap(orca_results, mlip_results)

   Calculate error metrics for unoptimized spin gap evaluation task.

   :param orca_results: Results from ORCA calculations.
   :type orca_results: dict
   :param mlip_results: Results from MLIP calculations.
   :type mlip_results: dict

   :returns: Error metrics for unoptimized spin gap evaluation task
   :rtype: dict


.. py:function:: singlepoint(orca_results, mlip_results)

