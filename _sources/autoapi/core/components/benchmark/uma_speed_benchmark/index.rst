core.components.benchmark.uma_speed_benchmark
=============================================

.. py:module:: core.components.benchmark.uma_speed_benchmark

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.benchmark.uma_speed_benchmark.InferenceBenchRunner


Functions
---------

.. autoapisummary::

   core.components.benchmark.uma_speed_benchmark.seed_everywhere
   core.components.benchmark.uma_speed_benchmark.ase_to_graph
   core.components.benchmark.uma_speed_benchmark.get_fcc_carbon_xtal
   core.components.benchmark.uma_speed_benchmark.get_qps
   core.components.benchmark.uma_speed_benchmark.trace_handler
   core.components.benchmark.uma_speed_benchmark.make_profile


Module Contents
---------------

.. py:function:: seed_everywhere(seed)

.. py:function:: ase_to_graph(atoms, neighbors: int, cutoff: float, external_graph=True)

.. py:function:: get_fcc_carbon_xtal(neighbors: int, radius: float, num_atoms: int, lattice_constant: float = 3.8, external_graph: bool = True)

.. py:function:: get_qps(data, predictor, warmups: int = 10, timeiters: int = 100)

.. py:function:: trace_handler(p, name, save_loc)

.. py:function:: make_profile(data, predictor, name, save_loc)

.. py:class:: InferenceBenchRunner(run_dir_root, natoms_list: list[int], model_checkpoints: dict[str, str], timeiters: int = 10, seed: int = 1, device='cuda', overrides: dict | None = None, inference_settings: fairchem.core.units.mlip_unit.api.inference.InferenceSettings = inference_settings_default(), generate_traces: bool = False)

   Bases: :py:obj:`fairchem.core.components.runner.Runner`


   Represents an abstraction over things that run in a loop and can save/load state.

   ie: Trainers, Validators, Relaxation all fall in this category.

   .. note::

      When running with the `fairchemv2` cli, the `job_config` and attribute is set at
      runtime to those given in the config file.

   .. attribute:: job_config

      a managed attribute that gives access to the job config

      :type: DictConfig


   .. py:attribute:: natoms_list


   .. py:attribute:: device


   .. py:attribute:: seed


   .. py:attribute:: timeiters


   .. py:attribute:: model_checkpoints


   .. py:attribute:: run_dir


   .. py:attribute:: overrides


   .. py:attribute:: inference_settings


   .. py:attribute:: generate_traces


   .. py:method:: run() -> None


   .. py:method:: save_state(_)


   .. py:method:: load_state(_)


