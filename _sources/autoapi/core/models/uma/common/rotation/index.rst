core.models.uma.common.rotation
===============================

.. py:module:: core.models.uma.common.rotation

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.uma.common.rotation.Safeacos
   core.models.uma.common.rotation.Safeatan2


Functions
---------

.. autoapisummary::

   core.models.uma.common.rotation.init_edge_rot_euler_angles
   core.models.uma.common.rotation.wigner_D
   core.models.uma.common.rotation._z_rot_mat
   core.models.uma.common.rotation.eulers_to_wigner


Module Contents
---------------

.. py:class:: Safeacos(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)


   .. py:method:: forward(ctx, x)
      :staticmethod:


      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.



   .. py:method:: backward(ctx, grad_output)
      :staticmethod:


      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: Safeatan2(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`


   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)


   .. py:method:: forward(ctx, y, x)
      :staticmethod:


      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.



   .. py:method:: backward(ctx, grad_output)
      :staticmethod:


      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:function:: init_edge_rot_euler_angles(edge_distance_vec)

.. py:function:: wigner_D(lv: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, _Jd: list[torch.Tensor]) -> torch.Tensor

.. py:function:: _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor

.. py:function:: eulers_to_wigner(eulers: torch.Tensor, start_lmax: int, end_lmax: int, Jd: list[torch.Tensor]) -> torch.Tensor

   set <rot_clip=True> to handle gradient instability when using gradient-based force/stress prediction.


