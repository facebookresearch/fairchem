core.units.mlip_unit.inference.inference_server_ray
===================================================

.. py:module:: core.units.mlip_unit.inference.inference_server_ray

.. autoapi-nested-parse::

   Sequential request server with parallel model execution
   Usage: python server.py --workers 4 --port 8000



Classes
-------

.. autoapisummary::

   core.units.mlip_unit.inference.inference_server_ray.MLIPWorker
   core.units.mlip_unit.inference.inference_server_ray.MLIPInferenceServerWebSocket


Functions
---------

.. autoapisummary::

   core.units.mlip_unit.inference.inference_server_ray.main


Module Contents
---------------

.. py:class:: MLIPWorker(worker_id: int, world_size: int, master_port: int, predictor_config: dict)

   .. py:attribute:: worker_id


   .. py:attribute:: predict_unit


   .. py:method:: _distributed_setup(worker_id: int, master_port: int, world_size: int, device: str)


   .. py:method:: predict(data: bytes)


.. py:class:: MLIPInferenceServerWebSocket(predictor_config: dict, port=8001, num_workers=1)

   .. py:attribute:: host
      :value: 'localhost'



   .. py:attribute:: port


   .. py:attribute:: num_workers


   .. py:attribute:: predictor_config


   .. py:attribute:: master_pg_port


   .. py:attribute:: workers


   .. py:method:: _setup_signal_handlers()

      Set up signal handlers for graceful shutdown



   .. py:method:: handler(websocket)
      :async:



   .. py:method:: start()
      :async:



   .. py:method:: run()

      Run the server (blocking)



   .. py:method:: shutdown()

      Shutdown the server and clean up Ray resources



.. py:function:: main(cfg: omegaconf.DictConfig)

