core.components.benchmark._single.uma_speed_benchmark
=====================================================

.. py:module:: core.components.benchmark._single.uma_speed_benchmark

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.components.benchmark._single.uma_speed_benchmark.InferenceBenchRunner


Functions
---------

.. autoapisummary::

   core.components.benchmark._single.uma_speed_benchmark.seed_everywhere
   core.components.benchmark._single.uma_speed_benchmark.ase_to_graph
   core.components.benchmark._single.uma_speed_benchmark.get_qps
   core.components.benchmark._single.uma_speed_benchmark.trace_handler
   core.components.benchmark._single.uma_speed_benchmark.make_profile


Module Contents
---------------

.. py:function:: seed_everywhere(seed)

.. py:function:: ase_to_graph(atoms, neighbors: int, cutoff: float, external_graph=True, dataset_name='omat')

.. py:function:: get_qps(data, predictor, warmups: int = 10, timeiters: int = 10, repeats: int = 5)

.. py:function:: trace_handler(p, name, save_loc)

.. py:function:: make_profile(data, predictor, name, save_loc)

.. py:class:: InferenceBenchRunner(model_checkpoints: dict[str, str], natoms_list: list[int] | None = None, input_system: dict | None = None, timeiters: int = 10, repeats: int = 5, seed: int = 1, device='cuda', overrides: dict | None = None, inference_settings: fairchem.core.units.mlip_unit.api.inference.InferenceSettings = inference_settings_default(), generate_traces: bool = False, expand_supercells: int | None = None, dataset_name: str = 'omat')

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


   .. py:attribute:: input_system


   .. py:attribute:: device


   .. py:attribute:: seed


   .. py:attribute:: timeiters


   .. py:attribute:: model_checkpoints


   .. py:attribute:: overrides


   .. py:attribute:: inference_settings


   .. py:attribute:: generate_traces


   .. py:attribute:: expand_supercells


   .. py:attribute:: dataset_name


   .. py:attribute:: repeats


   .. py:method:: run() -> None


   .. py:method:: save_state(_)


   .. py:method:: load_state(_)


