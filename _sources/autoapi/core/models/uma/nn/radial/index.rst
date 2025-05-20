core.models.uma.nn.radial
=========================

.. py:module:: core.models.uma.nn.radial

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.uma.nn.radial.PolynomialEnvelope
   core.models.uma.nn.radial.GaussianSmearing
   core.models.uma.nn.radial.RadialMLP


Functions
---------

.. autoapisummary::

   core.models.uma.nn.radial.gaussian


Module Contents
---------------

.. py:function:: gaussian(x: torch.Tensor, mean, std) -> torch.Tensor

.. py:class:: PolynomialEnvelope(exponent: int = 5)

   Bases: :py:obj:`torch.nn.Module`


   Polynomial envelope function that ensures a smooth cutoff.


   .. py:attribute:: p
      :type:  float


   .. py:attribute:: a
      :type:  float


   .. py:attribute:: b
      :type:  float


   .. py:attribute:: c
      :type:  float


   .. py:method:: forward(d_scaled: torch.Tensor) -> torch.Tensor


.. py:class:: GaussianSmearing(start: float = -5.0, stop: float = 5.0, num_gaussians: int = 50, basis_width_scalar: float = 1.0)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: num_output


   .. py:attribute:: coeff


   .. py:method:: forward(dist) -> torch.Tensor


.. py:class:: RadialMLP(channels_list)

   Bases: :py:obj:`torch.nn.Module`


   Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels


   .. py:attribute:: net


   .. py:method:: forward(inputs: torch.Tensor) -> torch.Tensor


