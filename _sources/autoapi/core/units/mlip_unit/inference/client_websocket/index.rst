core.units.mlip_unit.inference.client_websocket
===============================================

.. py:module:: core.units.mlip_unit.inference.client_websocket


Classes
-------

.. autoapisummary::

   core.units.mlip_unit.inference.client_websocket.AsyncMLIPInferenceWebSocketClient
   core.units.mlip_unit.inference.client_websocket.SyncMLIPInferenceWebSocketClient


Module Contents
---------------

.. py:class:: AsyncMLIPInferenceWebSocketClient(host, port)

   .. py:attribute:: uri


   .. py:attribute:: websocket
      :value: None



   .. py:method:: connect()
      :async:



   .. py:method:: close()
      :async:



   .. py:method:: call(atomic_data)
      :async:



.. py:class:: SyncMLIPInferenceWebSocketClient(host, port)

   .. py:attribute:: uri


   .. py:attribute:: ws
      :value: None



   .. py:method:: connect()


   .. py:method:: call(atomic_data)


   .. py:method:: __del__()


   .. py:method:: close()


