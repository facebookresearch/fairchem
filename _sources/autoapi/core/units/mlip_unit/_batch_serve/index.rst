core.units.mlip_unit._batch_serve
=================================

.. py:module:: core.units.mlip_unit._batch_serve

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.units.mlip_unit._batch_serve.BatchPredictServer


Functions
---------

.. autoapisummary::

   core.units.mlip_unit._batch_serve.setup_batch_predict_server


Module Contents
---------------

.. py:class:: BatchPredictServer(predict_unit_ref, max_batch_size: int, batch_wait_timeout_s: float, split_oom_batch: bool = True)

   Ray Serve deployment that batches incoming inference requests.


   .. py:attribute:: predict_unit


   .. py:attribute:: split_oom_batch


   .. py:method:: configure_batching(max_batch_size: int = 32, batch_wait_timeout_s: float = 0.05)


   .. py:method:: get_predict_unit_attribute(attribute_name: str) -> Any


   .. py:method:: predict(data_list: list[fairchem.core.datasets.atomic_data.AtomicData], undo_element_references: bool = True) -> list[dict]
      :async:


      Process a batch of AtomicData objects.

      :param data_list: List of AtomicData objects (automatically batched by Ray Serve)

      :returns: List of prediction dictionaries, one per input



   .. py:method:: __call__(data: fairchem.core.datasets.atomic_data.AtomicData, undo_element_references: bool = True) -> dict
      :async:


      Main entry point for inference requests.

      :param data: Single AtomicData object

      :returns: Prediction dictionary for this system



   .. py:method:: _split_predictions(predictions: dict, batch: fairchem.core.datasets.atomic_data.AtomicData) -> list[dict]

      Split batched predictions back into individual system predictions.

      :param batch_predictions: Dictionary of batched prediction tensors
      :param batch: The batched AtomicData used for inference

      :returns: List of prediction dictionaries, one per system



.. py:function:: setup_batch_predict_server(predict_unit: fairchem.core.units.mlip_unit.MLIPPredictUnit, max_batch_size: int = 32, batch_wait_timeout_s: float = 0.1, split_oom_batch: bool = True, num_replicas: int = 1, ray_actor_options: dict | None = None, deployment_name: str = 'predict-server', route_prefix: str = '/predict') -> ray.serve.handle.DeploymentHandle

   Set up and deploy a BatchPredictServer for batched inference.

   :param predict_unit: An MLIPPredictUnit instance to use for batched inference
   :param max_batch_size: Maximum number of systems per batch.
   :param batch_wait_timeout_s: Maximum wait time before processing partial batch.
   :param split_oom_batch: Whether to split batches that cause OOM errors.
   :param num_replicas: Number of deployment replicas for scaling.
   :param ray_actor_options: Additional Ray actor options (e.g., {"num_gpus": 1, "num_cpus": 4})
   :param deployment_name: Name for the Ray Serve deployment.
   :param route_prefix: HTTP route prefix for the deployment.

   :returns: Ray Serve deployment handle that can be used to initialize BatchServerPredictUnit


