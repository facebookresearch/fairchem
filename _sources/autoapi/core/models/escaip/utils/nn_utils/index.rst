core.models.escaip.utils.nn_utils
=================================

.. py:module:: core.models.escaip.utils.nn_utils


Classes
-------

.. autoapisummary::

   core.models.escaip.utils.nn_utils.Activation
   core.models.escaip.utils.nn_utils.SquaredReLU
   core.models.escaip.utils.nn_utils.StarReLU
   core.models.escaip.utils.nn_utils.SmeLU
   core.models.escaip.utils.nn_utils.NormalizationType
   core.models.escaip.utils.nn_utils.Skip


Functions
---------

.. autoapisummary::

   core.models.escaip.utils.nn_utils.build_activation
   core.models.escaip.utils.nn_utils.get_linear
   core.models.escaip.utils.nn_utils.get_feedforward
   core.models.escaip.utils.nn_utils.no_weight_decay
   core.models.escaip.utils.nn_utils.init_linear_weights
   core.models.escaip.utils.nn_utils.get_normalization_layer


Module Contents
---------------

.. py:class:: Activation

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`


   str(object='') -> str
   str(bytes_or_buffer[, encoding[, errors]]) -> str

   Create a new string object from the given object. If encoding or
   errors is specified, then the object must expose a data buffer
   that will be decoded using the given encoding and error handler.
   Otherwise, returns the result of object.__str__() (if defined)
   or repr(object).
   encoding defaults to sys.getdefaultencoding().
   errors defaults to 'strict'.


   .. py:attribute:: SquaredReLU
      :value: 'squared_relu'



   .. py:attribute:: GeLU
      :value: 'gelu'



   .. py:attribute:: LeakyReLU
      :value: 'leaky_relu'



   .. py:attribute:: ReLU
      :value: 'relu'



   .. py:attribute:: SmeLU
      :value: 'smelu'



   .. py:attribute:: StarReLU
      :value: 'star_relu'



.. py:class:: SquaredReLU

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


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


.. py:class:: StarReLU

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


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


.. py:class:: SmeLU(beta: float = 2.0)

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


   .. py:attribute:: beta


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


.. py:function:: build_activation(activation: Optional[Activation])

.. py:function:: get_linear(in_features: int, out_features: int, bias: bool = False, activation: Activation | None = None, dropout: float = 0.0)

   Build a linear layer with optional activation and dropout.


.. py:function:: get_feedforward(hidden_dim: int, activation: Activation | None, hidden_layer_multiplier: int, bias: bool = False, dropout: float = 0.0)

   Build a feedforward layer with optional activation function.


.. py:function:: no_weight_decay(model)

.. py:function:: init_linear_weights(module, gain=1.0)

.. py:class:: NormalizationType

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`


   str(object='') -> str
   str(bytes_or_buffer[, encoding[, errors]]) -> str

   Create a new string object from the given object. If encoding or
   errors is specified, then the object must expose a data buffer
   that will be decoded using the given encoding and error handler.
   Otherwise, returns the result of object.__str__() (if defined)
   or repr(object).
   encoding defaults to sys.getdefaultencoding().
   errors defaults to 'strict'.


   .. py:attribute:: LayerNorm
      :value: 'layernorm'



   .. py:attribute:: Skip
      :value: 'skip'



   .. py:attribute:: RMSNorm
      :value: 'rmsnorm'



.. py:class:: Skip(*_, **__)

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


   .. py:method:: forward(x, **_)


.. py:function:: get_normalization_layer(normalization_type: NormalizationType)

