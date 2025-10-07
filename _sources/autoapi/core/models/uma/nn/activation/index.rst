core.models.uma.nn.activation
=============================

.. py:module:: core.models.uma.nn.activation

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.uma.nn.activation.ScaledSiLU
   core.models.uma.nn.activation.ScaledSwiGLU
   core.models.uma.nn.activation.SwiGLU
   core.models.uma.nn.activation.SmoothLeakyReLU
   core.models.uma.nn.activation.ScaledSmoothLeakyReLU
   core.models.uma.nn.activation.ScaledSigmoid
   core.models.uma.nn.activation.GateActivation
   core.models.uma.nn.activation.S2Activation
   core.models.uma.nn.activation.SeparableS2Activation
   core.models.uma.nn.activation.S2Activation_M
   core.models.uma.nn.activation.SeparableS2Activation_M


Module Contents
---------------

.. py:class:: ScaledSiLU(inplace: bool = False)

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


   .. py:attribute:: inplace


   .. py:attribute:: scale_factor
      :value: 1.6791767923989418



   .. py:method:: forward(inputs)


   .. py:method:: extra_repr()

      Return the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: ScaledSwiGLU(in_channels: int, out_channels: int, bias: bool = True)

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


   .. py:attribute:: in_channels


   .. py:attribute:: out_channels


   .. py:attribute:: w


   .. py:attribute:: act


   .. py:method:: forward(inputs)


.. py:class:: SwiGLU(in_channels: int, out_channels: int, bias: bool = True)

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


   .. py:attribute:: in_channels


   .. py:attribute:: out_channels


   .. py:attribute:: w


   .. py:attribute:: act


   .. py:method:: forward(inputs)


.. py:class:: SmoothLeakyReLU(negative_slope: float = 0.2)

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


   .. py:attribute:: alpha


   .. py:method:: forward(x)


   .. py:method:: extra_repr()

      Return the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: ScaledSmoothLeakyReLU

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


   .. py:attribute:: act


   .. py:attribute:: scale_factor
      :value: 1.531320475574866



   .. py:method:: forward(x)


   .. py:method:: extra_repr()

      Return the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: ScaledSigmoid

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


   .. py:attribute:: scale_factor
      :value: 1.8467055342154763



   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


.. py:class:: GateActivation(lmax: int, mmax: int, num_channels: int, m_prime: bool = False)

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


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: num_channels


   .. py:attribute:: m_prime


   .. py:attribute:: scalar_act


   .. py:attribute:: gate_act


   .. py:method:: forward(gating_scalars, input_tensors)

      `gating_scalars`: shape [N, lmax * num_channels]
      `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]



.. py:class:: S2Activation(lmax: int, mmax: int, SO3_grid)

   Bases: :py:obj:`torch.nn.Module`


   Assume we only have one resolution


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: act


   .. py:attribute:: SO3_grid


   .. py:method:: forward(inputs)


.. py:class:: SeparableS2Activation(lmax: int, mmax: int, SO3_grid)

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


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: scalar_act


   .. py:attribute:: s2_act


   .. py:method:: forward(input_scalars, input_tensors)


.. py:class:: S2Activation_M(lmax: int, mmax: int, SO3_grid, to_m: torch.Tensor)

   Bases: :py:obj:`torch.nn.Module`


   Assume we only have one resolution


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: act


   .. py:attribute:: SO3_grid


   .. py:method:: forward(inputs)


.. py:class:: SeparableS2Activation_M(lmax: int, mmax: int, SO3_grid, to_m)

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


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: scalar_act


   .. py:attribute:: s2_act


   .. py:method:: forward(input_scalars, input_tensors)


