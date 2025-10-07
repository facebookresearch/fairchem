core.models.uma.nn.dropout
==========================

.. py:module:: core.models.uma.nn.dropout

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.

   Add `extra_repr` into DropPath implemented by timm
   for displaying more info.



Classes
-------

.. autoapisummary::

   core.models.uma.nn.dropout.DropPath
   core.models.uma.nn.dropout.GraphDropPath
   core.models.uma.nn.dropout.EquivariantDropout
   core.models.uma.nn.dropout.EquivariantScalarsDropout
   core.models.uma.nn.dropout.EquivariantDropoutArraySphericalHarmonics


Functions
---------

.. autoapisummary::

   core.models.uma.nn.dropout.drop_path


Module Contents
---------------

.. py:function:: drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor

   Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
   This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
   the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
   See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
   changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
   'survival rate' as the argument.


.. py:class:: DropPath(drop_prob: float)

   Bases: :py:obj:`torch.nn.Module`


   Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).


   .. py:attribute:: drop_prob


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


   .. py:method:: extra_repr() -> str

      Return the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: GraphDropPath(drop_prob: float)

   Bases: :py:obj:`torch.nn.Module`


   Consider batch for graph data when dropping paths.


   .. py:attribute:: drop_prob


   .. py:method:: forward(x: torch.Tensor, batch) -> torch.Tensor


   .. py:method:: extra_repr() -> str

      Return the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: EquivariantDropout(irreps, drop_prob: float)

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


   .. py:attribute:: irreps


   .. py:attribute:: num_irreps


   .. py:attribute:: drop_prob


   .. py:attribute:: drop


   .. py:attribute:: mul


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


.. py:class:: EquivariantScalarsDropout(irreps, drop_prob: float)

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


   .. py:attribute:: irreps


   .. py:attribute:: drop_prob


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


   .. py:method:: extra_repr() -> str

      Return the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: EquivariantDropoutArraySphericalHarmonics(drop_prob: float, drop_graph: bool = False)

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


   .. py:attribute:: drop_prob


   .. py:attribute:: drop


   .. py:attribute:: drop_graph


   .. py:method:: forward(x: torch.Tensor, batch=None) -> torch.Tensor


   .. py:method:: extra_repr() -> str

      Return the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



