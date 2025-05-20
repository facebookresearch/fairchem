core.models.uma.nn.mole
=======================

.. py:module:: core.models.uma.nn.mole

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core.models.uma.nn.mole.fairchem_cpp_found
   core.models.uma.nn.mole.fairchem_cpp_found


Classes
-------

.. autoapisummary::

   core.models.uma.nn.mole.MOLEGlobals
   core.models.uma.nn.mole.MOLEDGL
   core.models.uma.nn.mole.MOLE


Functions
---------

.. autoapisummary::

   core.models.uma.nn.mole._softmax
   core.models.uma.nn.mole._pnorm
   core.models.uma.nn.mole.norm_str_to_fn
   core.models.uma.nn.mole.init_linear


Module Contents
---------------

.. py:data:: fairchem_cpp_found
   :value: False


.. py:data:: fairchem_cpp_found
   :value: True


.. py:function:: _softmax(x)

.. py:function:: _pnorm(x)

.. py:function:: norm_str_to_fn(act)

.. py:class:: MOLEGlobals

   .. py:attribute:: expert_mixing_coefficients
      :type:  torch.Tensor


   .. py:attribute:: mole_sizes
      :type:  torch.Tensor


.. py:function:: init_linear(num_experts, use_bias, out_features, in_features)

.. py:class:: MOLEDGL(num_experts, in_features, out_features, global_mole_tensors, bias: bool)

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


   .. py:attribute:: num_experts


   .. py:attribute:: in_features


   .. py:attribute:: out_features


   .. py:attribute:: global_mole_tensors


   .. py:method:: forward(x)


.. py:class:: MOLE(num_experts, in_features, out_features, global_mole_tensors: MOLEGlobals, bias: bool)

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


   .. py:attribute:: num_experts


   .. py:attribute:: in_features


   .. py:attribute:: out_features


   .. py:attribute:: global_mole_tensors


   .. py:method:: merged_linear_layer()


   .. py:method:: forward(x)


