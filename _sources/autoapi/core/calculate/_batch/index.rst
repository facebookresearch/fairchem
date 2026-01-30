core.calculate._batch
=====================

.. py:module:: core.calculate._batch

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.calculate._batch.ExecutorProtocol
   core.calculate._batch.InferenceBatcher


Functions
---------

.. autoapisummary::

   core.calculate._batch._get_concurrency_backend


Module Contents
---------------

.. py:class:: ExecutorProtocol

   Bases: :py:obj:`Protocol`


   Base class for protocol classes.

   Protocol classes are defined as::

       class Proto(Protocol):
           def meth(self) -> int:
               ...

   Such classes are primarily used with static type checkers that recognize
   structural subtyping (static duck-typing).

   For example::

       class C:
           def meth(self) -> int:
               return 0

       def func(x: Proto) -> int:
           return x.meth()

       func(C())  # Passes static type check

   See PEP 544 for details. Protocol classes decorated with
   @typing.runtime_checkable act as simple-minded runtime protocols that check
   only the presence of given attributes, ignoring their type signatures.
   Protocol classes can be generic, they are defined as::

       class GenProto[T](Protocol):
           def meth(self) -> T:
               ...


   .. py:method:: submit(fn, *args, **kwargs)


   .. py:method:: map(fn, *iterables, **kwargs)


   .. py:method:: shutdown(wait: bool = True)


.. py:function:: _get_concurrency_backend(backend: Literal['threads'], options: dict) -> ExecutorProtocol

   Get a backend to run ASE calculations concurrently.


.. py:class:: InferenceBatcher(predict_unit: fairchem.core.units.mlip_unit.predict.MLIPPredictUnit, max_batch_size: int = 512, batch_wait_timeout_s: float = 0.1, num_replicas: int = 1, concurrency_backend: Literal['threads'] = 'threads', concurrency_backend_options: dict | None = None, ray_actor_options: dict | None = None)

   Batches incoming inference requests.


   .. py:attribute:: predict_unit


   .. py:attribute:: max_batch_size


   .. py:attribute:: batch_wait_timeout_s


   .. py:attribute:: num_replicas


   .. py:attribute:: predict_server_handle


   .. py:attribute:: executor
      :type:  ExecutorProtocol


   .. py:method:: __enter__()


   .. py:method:: __exit__(exc_type, exc_val, exc_tb)


   .. py:property:: batch_predict_unit
      :type: fairchem.core.units.mlip_unit.predict.BatchServerPredictUnit



   .. py:method:: shutdown(wait: bool = True) -> None

      Shutdown the executor.

      :param wait: If True, wait for pending tasks to complete before returning.



   .. py:method:: __del__()

      Cleanup on deletion.



