core.models.escaip.utils.stochastic_depth
=========================================

.. py:module:: core.models.escaip.utils.stochastic_depth

.. autoapi-nested-parse::

   Modified from https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py



Classes
-------

.. autoapisummary::

   core.models.escaip.utils.stochastic_depth.StochasticDepth
   core.models.escaip.utils.stochastic_depth.SkipStochasticDepth


Functions
---------

.. autoapisummary::

   core.models.escaip.utils.stochastic_depth.stochastic_depth_2d
   core.models.escaip.utils.stochastic_depth.stochastic_depth_3d


Module Contents
---------------

.. py:function:: stochastic_depth_2d(input: torch.Tensor, batch: torch.Tensor, p: float, training: bool = True) -> torch.Tensor

   Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
   <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
   branches of residual architectures.

   :param input: The input tensor or arbitrary dimensions
                 with the first one being its node dimension.
   :type input: Tensor[num_nodes, ...]
   :param batch: The batch tensor of the input tensor.
   :type batch: LongTensor[num_nodes]
   :param p: probability of the input to be zeroed.
   :type p: float
   :param training: apply stochastic depth if is ``True``. Default: ``True``

   :returns: The randomly zeroed tensor.
   :rtype: Tensor[N, ...]


.. py:function:: stochastic_depth_3d(input: torch.Tensor, batch: torch.Tensor, p: float, training: bool = True) -> torch.Tensor

.. py:class:: StochasticDepth(p: float)

   Bases: :py:obj:`torch.nn.Module`


   Stochastic Depth for graph features.


   .. py:attribute:: p


   .. py:method:: forward(node_features, edge_features, node_batch)


   .. py:method:: __repr__() -> str


.. py:class:: SkipStochasticDepth(*args, **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   Skip Stochastic Depth for graph features.


   .. py:method:: forward(node_features, edge_features, _)


   .. py:method:: __repr__() -> str


