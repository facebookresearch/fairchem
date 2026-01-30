adsorbml.scripts.dense_eval
===========================

.. py:module:: adsorbml.scripts.dense_eval

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   adsorbml.scripts.dense_eval.SUCCESS_THRESHOLD
   adsorbml.scripts.dense_eval.parser


Functions
---------

.. autoapisummary::

   adsorbml.scripts.dense_eval.is_successful
   adsorbml.scripts.dense_eval.compute_hybrid_success
   adsorbml.scripts.dense_eval.compute_valid_ml_success
   adsorbml.scripts.dense_eval.get_dft_data
   adsorbml.scripts.dense_eval.get_dft_compute
   adsorbml.scripts.dense_eval.filter_ml_data


Module Contents
---------------

.. py:data:: SUCCESS_THRESHOLD
   :value: 0.1


.. py:function:: is_successful(best_ml_dft_energy, best_dft_energy)

   Computes the success rate given the best ML+DFT energy and the best ground
   truth DFT energy.


   success_parity: The standard definition for success, where ML needs to be
   within the SUCCESS_THRESHOLD, or lower, of the DFT energy.

   success_much_better: A system in which the ML energy is predicted to be
   much lower (less than the SUCCESS_THRESHOLD) of the DFT energy.


.. py:function:: compute_hybrid_success(ml_data, dft_data, k)

   Computes AdsorbML success rates at varying top-k values.
   Here, results are generated for the hybrid method, where the top-k ML
   energies are used to to run DFT on the corresponding ML structures. The
   resulting energies are then compared to the ground truth DFT energies.

   Return success rates and DFT compute usage at varying k.


.. py:function:: compute_valid_ml_success(ml_data, dft_data)

   Computes validated ML success rates.
   Here, results are generated only from ML. DFT single-points are used to
   validate whether the ML energy is within 0.1eV of the DFT energy of the
   predicted structure. If valid, the ML energy is compared to the ground
   truth DFT energy, otherwise it is discarded.

   Return validated ML success rates.


.. py:function:: get_dft_data(targets)

   Organizes the released target mapping for evaluation lookup.

   oc20dense_targets.pkl:
       ['system_id 1': [('config_id 1', dft_adsorption_energy), ('config_id 2', dft_adsorption_energy)], `system_id 2]

   Returns: Dict:
       {
          'system_id 1': {'config_id 1': dft_ads_energy, 'config_id 2': dft_ads_energy},
          'system_id 2': {'config_id 1': dft_ads_energy, 'config_id 2': dft_ads_energy},
          ...
       }


.. py:function:: get_dft_compute(counts)

   Calculates the total DFT compute associated with establishing a ground
   truth using the released DFT timings: oc20dense_compute.pkl.

   Compute is measured in the total number of self-consistent steps (SC). The
   total number of ionic steps is also included for reference.


.. py:function:: filter_ml_data(ml_data, dft_data)

   For ML systems in which no configurations made it through the physical
   constraint checks, set energies to an arbitrarily high value to ensure
   a failure case in evaluation.


.. py:data:: parser

